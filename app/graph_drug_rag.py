import argparse
import os
from typing import List, TypedDict, Literal, Any, Dict

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

from embedding_utils import get_embedding_model
from custom_pgvector import CustomPGVector
from db_utils import make_conn_str

_LLM_INSTANCE: ChatOllama | None = None
_COMPILED_GRAPH = None

class RAGState(TypedDict, total=False):
    """그래프 상태 정의"""
    question: str
    k: int
    collection_name: str
    in_domain: bool
    retrieved_docs: List[Document]
    context: str
    answer: str
    citations: List[Dict[str, Any]]


def get_llm() -> ChatOllama:
    """ChatOllama LLM을 초기화"""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        model = os.getenv("OLLAMA_MODEL")
        temperature = float(os.getenv("GEN_TEMPERATURE", "0.2"))
        _LLM_INSTANCE = ChatOllama(model=model, temperature=temperature)
    return _LLM_INSTANCE


def get_vectorstore(collection_name: str) -> CustomPGVector:
    """pgvector 컬렉션을 VectorStore로 감싼 객체를 생성"""
    embedding_model = get_embedding_model()
    return CustomPGVector(
            conn_str=make_conn_str(),
            embedding_fn=embedding_model,
            table=collection_name,
        )

def build_prompt() -> ChatPromptTemplate:
    """의약품 도메인에 맞춘 시스템 프롬프트"""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "너는 한국어 의약품 정보 안내 챗봇이야. "
                    "아래 CONTEXT(근거)를 바탕으로만 정확하고 간결하게 답하고, "
                    "모르는 것은 모른다고 말해. "
                    "효능/용법, 사용 전 주의사항, 이상반응, 보관법 등은 반드시 정확한 표현을 사용해.\n\n"
                    "FORMAT 지침:\n"
                    "- 핵심 요약 3~5줄\n"
                    "- 필요한 경우 목록으로 정리\n"
                    "- 출처 제품명을 '근거' 섹션에 함께 표기"
                ),
            ),
            ("human", "질문: {question}\n\nCONTEXT:\n{context}\n\n한국어로 답변해줘."),
        ]
    )


def build_guard_prompt() -> ChatPromptTemplate:
    """
    주제 연관성(의약품 도메인) 판별 프롬프트.
    - 'YES' 또는 'NO'로만 답하도록 강제.
    - YES: 약, 복약, 효능, 용법/용량, 이상반응, 상호작용, 보관, 금기, 성분, 제형 등과 관련.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "너는 입력 문장이 '의약품/복약/약물정보' 도메인과 관련 있는지 판별하는 분류기야. "
                    "질문에 실제 의약품 이름, 의약품 브랜드명, 약물 성분, 복약 행위, 의약적 처치 등 구체적 약물 맥락이 명시된 경우에만 'YES'를 출력해. "
                    "가상인물·외계인·음악·예술 등 약물과 무관한 소재는 모두 'NO'라고 판별해. "
                    "관련 있으면 'YES', 없으면 'NO' 라는 한 단어만 출력해."
                ),
            ),
            (
                "human",
                (
                    "판별 기준 예시:\n"
                    "- YES: 약 이름/제품명/성분/효능/용법/용량/상호작용/보관/금기/주의/부작용 등과 명시적으로 연결된 질문\n"
                    "- YES: 사용자가 현재 증상을 설명하며 복용 할 수 있는약을 추천해(알려)달라고 하는 질문\n"
                    "- YES: 약 이름을 명시하며 그 약에 대해서 알려달라고 하는 질문\n"
                    "- NO: 일반 상식, 시사, 주식, 스포츠, 법률(약과 무관), 농담, 음악, 종교, 가상/외계인 소재 등\n"
                    "예시 판단:\n"
                    "- 질문: '지르텍에 대해서 알려줘' → YES\n"
                    "- 질문: '타이레놀 500mg을 복용했는데 발열이 계속돼요. 부작용인가요?' → YES\n"
                    "- 질문: '바흐의 녹턴 교향곡이 외계인에게 주는 증상은?' → NO\n\n"
                    "입력: {question}\n"
                    "정답(YES/NO)만 출력:"
                ),
            ),
        ]
    )


def node_guard(state: RAGState) -> RAGState:
    """사용자 질문이 의약품 도메인과 관련 있는지 LLM으로 판별"""
    llm = get_llm()
    guard_chain = build_guard_prompt() | llm | StrOutputParser()
    result = guard_chain.invoke({"question": state["question"]}).strip().upper()
    state["in_domain"] = (result == "YES")
    return state


def node_retrieve(state: RAGState) -> RAGState:
    """유사도 검색으로 문서 청크를 가져오는 함수"""
    collection = state["collection_name"]
    k = state.get("k", 5)
    vectorstore = get_vectorstore(collection)
    docs_and_scores = vectorstore.similarity_search_with_score(state["question"], k=k)
    docs: List[Document] = [d for d, _ in docs_and_scores]

    context_lines: List[str] = []
    citations: List[Dict[str, Any]] = []
    for doc, score in docs_and_scores:
        meta = doc.metadata or {}
        product = meta.get("제품명") or meta.get("title") or meta.get("product") or "알 수 없는 제품"
        snippet = (doc.page_content or "")[:300].replace("\n", " ")
        context_lines.append(f"[제품명: {product}] {doc.page_content}")
        citations.append({"제품명": product, "score": float(score), "snippet": snippet})

    state["retrieved_docs"] = docs
    state["context"] = "\n\n".join(context_lines)
    state["citations"] = citations
    return state


def node_generate(state: RAGState) -> RAGState:
    """LLM으로 최종 답변을 생성"""
    llm = get_llm()
    prompt = build_prompt()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": state["question"], "context": state.get("context", "")})
    state["answer"] = answer
    return state


def node_fallback(state: RAGState) -> RAGState:
    """도메인과 관련 없을 때의 안내 메시지"""
    state["answer"] = (
        "이 챗봇은 의약품 정보 전용입니다. 약 이름, 효능·용법, 상호작용, 이상반응, 보관법 등 "
        "의약품 관련 질문을 해주시면 근거에 기반하여 정확히 안내해드릴게요."
    )
    state["citations"] = []
    state["context"] = ""
    return state


def route_topic(state: RAGState) -> Literal["retrieve", "fallback"]:
    """in_domain state값에 따라 분기를 정하는 함수"""
    return "retrieve" if state.get("in_domain") else "fallback"


def build_graph():
    """그래프를 정의 하는 함수"""
    graph = StateGraph(RAGState)

    graph.add_node("guard", node_guard)        # 주제 연관성 판별
    graph.add_node("retrieve", node_retrieve)  # 연관 시 검색
    graph.add_node("generate", node_generate)  # 답변 생성
    graph.add_node("fallback", node_fallback)  # 비연관 시

    graph.set_entry_point("guard")
    # guard 노드를 지나 retrieve|fallback 둘 중 어떤 노드로 갈지 결정하는 분기 엣지
    graph.add_conditional_edges("guard", route_topic, {"retrieve": "retrieve", "fallback": "fallback"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def get_compiled_graph():
    """빌드된 그래프를 싱글턴으로 재사용"""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def warm_up_pipeline() -> None:
    """
    LangGraph와 LLM을 미리 준비해 첫 사용자 입력 전에 초기화 비용을 지불합니다.
    """
    get_llm()
    get_embedding_model()
    get_compiled_graph()


def run_once(question: str, collection_name: str = "drug_info", k: int = 4) -> Dict[str, Any]:
    """그래프를 한 번 실행하고 결과를 dict로 반환"""
    app = get_compiled_graph()
    initial: RAGState = {"question": question, "collection_name": collection_name, "k": k}
    final_state = app.invoke(initial)
    return {
        "question": final_state["question"],
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "in_domain": final_state.get("in_domain", False),
    }


def run(collection_name: str = "drug_info", k: int = 4, exit_words: tuple[str, ...] = ("quit", "exit", "bye")) -> None:
    """사용자가 종료 단어를 입력할 때까지 반복 실행하는 인터랙티브 루프"""
    app = get_compiled_graph()
    exit_words_lower = {word.lower() for word in exit_words}
    print(
        "💊 의약품 정보 RAG 챗봇입니다. 종료하려면 "
        f"{', '.join(exit_words)} 중 하나를 입력하세요."
    )

    while True:
        try:
            question = input("\n질문> ").strip()
        except EOFError:
            print("\n입력을 종료합니다.")
            break

        if not question:
            print("질문을 입력하거나 종료 명령을 입력하세요.")
            continue

        if question.lower() in exit_words_lower:
            print("채팅을 종료합니다.")
            break

        initial: RAGState = {
            "question": question,
            "collection_name": collection_name,
            "k": k,
        }
        final_state = app.invoke(initial)

        answer = final_state.get("answer", "")
        citations = final_state.get("citations", [])
        in_domain = final_state.get("in_domain", False)

        print("\n=== IN_DOMAIN ===\n", in_domain)
        print("\n=== ANSWER ===\n")
        print(answer or "❗ 답변을 생성하지 못했습니다.")
        if citations:
            print("\n=== CITATIONS ===")
            for idx, citation in enumerate(citations, start=1):
                product = citation.get("제품명") or citation.get("product_name") or "N/A"
                score = citation.get("score")
                snippet = citation.get("snippet", "")
                if score is not None:
                    print(f"#{idx} 제품명: {product} | score={score:.4f}")
                else:
                    print(f"#{idx} 제품명: {product}")
                if snippet:
                    print(f"   snippet: {snippet}")
        else:
            print("\n=== CITATIONS ===\n(없음)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default="drug_info", help="pgvector 컬렉션명")
    p.add_argument("--k", type=int, default=5, help="검색 상위 k")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    run(args.collection, args.k)


if __name__ == "__main__":
    main()
