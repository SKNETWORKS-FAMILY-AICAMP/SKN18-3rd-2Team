"""
콘솔 RAG: LangChain 임베딩 + 기존 SQL 검색 + LangChain PromptTemplate 기반 프롬프트 구성.
"""
import os
from typing import List, Sequence, Tuple

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from common import search_similar

RowType = Tuple[float, int, str, str, str, str]


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 정확한 한국어 면접 코치입니다. 제공된 근거를 활용해 질문에 답변하세요.\n\n"
            "[근거 Q/A]\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def build_prompt_inputs(rows: Sequence[RowType], user_query: str) -> dict[str, str]:
    """
    검색된 Q/A 행들을 LangChain PromptTemplate에 바로 넣을 수 있는 dict로 변환한다.
    - top_n: 환경변수 RAG_MAX_ITEMS (기본 5)
    - 각 항목은 "#번호 (대분류/중분류) Q: ... A: ..." 포맷
    - 너무 긴 항목은 RAG_MAX_CHARS 기준으로 잘라낸다.
    """
    top_n = int(os.getenv("RAG_MAX_ITEMS", "5"))
    max_chars = int(os.getenv("RAG_MAX_CHARS", "1200"))

    formatted_blocks: List[str] = []
    for rank, (_score, _id, big, mid, question, answer) in enumerate(rows[:top_n], start=1):
        block = f"#{rank} ({big}/{mid})\nQ: {question}\nA: {answer}"
        if len(block) > max_chars:
            block = block[:max_chars] + " …(생략)"
        formatted_blocks.append(block)

    context_text = "\n\n".join(formatted_blocks)
    question_text = (
        f"질문: {user_query}\n\n"
        "요청:\n"
        "- 답변은 간결하고 명확하게 작성\n"
        "- 추측은 피하고 근거에서 확인 가능한 내용만 사용\n"
        "- 답변 끝에 참고 근거 번호를 '참고: #1, #2' 형태로 명시"
    )
    return {"context": context_text, "question": question_text}


def build_chain():
    temperature = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    model_name = os.getenv("OLLAMA_MODEL")
    base_url = os.getenv("OLLAMA_HOST")
    llm = ChatOllama(model=model_name, temperature=temperature, base_url=base_url)
    return PROMPT_TEMPLATE | llm | StrOutputParser()


def chat_loop():
    chain = build_chain()
    print("💬 면접 Q/A 챗봇입니다. 종료하려면 'quit', 'exit', 'bye' 중 하나를 입력하세요.")
    while True:
        try:
            user_query = input("\n질문> ").strip()
        except EOFError:
            print("\n입력을 종료합니다.")
            break
        if not user_query:
            print("질문을 입력하거나 종료 명령을 입력하세요.")
            continue
        if user_query.lower() in {"quit", "exit", "bye"}:
            print("안녕히 가세요!")
            break

        rows = search_similar(user_query, k=8)
        if not rows:
            print("❗ 관련 항목이 없습니다. 질문을 바꿔보세요.")
            continue

        prompt_inputs = build_prompt_inputs(rows, user_query)
        answer = chain.invoke(prompt_inputs)

        print("\n=== 답변 ===\n")
        print(answer)
        print("\n--- 참고한 Q/A ---")
        for idx, (score, _id, big, mid, question, _answer) in enumerate(rows[:5], start=1):
            preview = question[:80] + ("..." if len(question) > 80 else "")
            print(f"#{idx} [{score:.3f}] ({big}/{mid}) Q: {preview}")


def main():
    load_dotenv()
    chat_loop()


if __name__ == "__main__":
    main()
