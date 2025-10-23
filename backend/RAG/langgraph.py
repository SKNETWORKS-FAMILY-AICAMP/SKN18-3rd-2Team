"""
단순화된 약품 정보 시스템
백터DB 검색 → 답변 생성
"""

from .model import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from vectordb.customPGVector import CustomPGVector
from .model import embeddings_model



# 상태 정의
class MedicineSelfRAGState(TypedDict):
    """단순화된 약품 정보 시스템 상태"""
    
    # 입력 정보
    question: str  # 사용자 질문
    is_medicine_related: bool  # 약/증상 관련 여부
    question_type: str  # 질문 유형 (symptom, medicine_info, side_effect)
    
    # vectordb 검색 관련
    contents: List[str]  # 검색된 문서 내용들
    sources: List[Dict[str, Any]]  # 검색된 문서의 메타데이터
    
    # 최종 결과
    final_answer: str  # 최종 답변

################### Nodes ###################
### 백터디비 조회 노드

def get_vectorstore(collection_name: str) -> CustomPGVector:
    """pgvector 컬렉션을 VectorStore로 감싼 객체를 생성"""
    embedding_model = embeddings_model
    return CustomPGVector(
            conn_str="postgresql://admin:admin123@localhost:55432/vectordb",
            embedding_fn=embedding_model,
            table="qa_embedding",
        )


### 백터디비 검색 노드
def search_vectordb_node(state: MedicineSelfRAGState) -> MedicineSelfRAGState:
    """백터디비에서 관련 문서 검색"""
    vectorstore = get_vectorstore("medicine_collection")
    question = state["question"]
    
    # 1. 일반 유사도 검색
    docs = vectorstore.similarity_search(question, k=6)
    
    # 2. 제품명이 질문에 포함된 경우 우선순위 부여
    product_matches = []
    other_docs = []
    
    for doc in docs:
        if doc.metadata and "제품명" in doc.metadata:
            product_name = doc.metadata["제품명"]
            # 질문에 제품명이 포함되어 있으면 우선순위
            if product_name in question or any(word in product_name for word in question.split()):
                product_matches.append(doc)
            else:
                other_docs.append(doc)
        else:
            other_docs.append(doc)
    
    # 3. 제품명 매칭 문서를 앞에 배치
    final_docs = product_matches + other_docs[:4-len(product_matches)]
    
    print(f"[검색 결과] 총 {len(final_docs)}개 문서 발견")
    if product_matches:
        print(f"[제품명 매칭] {len(product_matches)}개 문서 발견")
    
    return {
        "contents": [doc.page_content for doc in final_docs],
        "sources": [doc.metadata for doc in final_docs]
    }

### 에이전트 함수
def question_type_agent(state: MedicineSelfRAGState) -> MedicineSelfRAGState:
    """질문이 약/증상 관련인지 판단하고 질문 유형을 분류"""
    question = state["question"]
    
    # 더 간단하고 명확한 프롬프트
    type_message = [
        SystemMessage(content="""질문을 분석하여 다음 중 하나로 분류하세요:

1. symptom - 증상에 대한 약 추천 요청
2. medicine_info - 특정 약의 정보 요청  
3. side_effect - 약 복용 후 부작용 관련
5. no - 약품/의료와 무관한 질문

답변은 반드시 다음 형식으로만 하세요:
- symptom
- medicine_info  
- side_effect

- no"""),
        HumanMessage(content=question)
    ]
    
    type_response = llm.invoke(type_message)
    response_text = type_response.content if hasattr(type_response, 'content') else str(type_response)
    response_text = response_text.strip().lower()
    
    # 응답에서 질문 유형 추출
    if "symptom" in response_text:
        question_type = "symptom"
        is_medicine_related = True
    elif "medicine_info" in response_text:
        question_type = "medicine_info"
        is_medicine_related = True
    elif "side_effect" in response_text:
        question_type = "side_effect"
        is_medicine_related = True
    elif "no" in response_text:
        question_type = "general"
        is_medicine_related = False
    else:
        # LLM이 예상 형식으로 응답하지 않은 경우 기본값 (벡터 검색으로 처리)
        question_type = "general"
        is_medicine_related = True
    
    print(f"[1단계] 약/증상 관련: {is_medicine_related}")
    print(f"[질문 유형] {question_type}")
    print(f"[LLM 응답] {response_text.strip()}")
    
    return {
        **state,
        "is_medicine_related": is_medicine_related,
        "question_type": question_type
    }



def generate_answer_agent(state: MedicineSelfRAGState) -> MedicineSelfRAGState:
    """검색된 문서를 바탕으로 답변 생성"""
    question = state["question"]
    contents = state.get("contents", [])
    
    if not contents:
        return {
            **state,
            "final_answer": "죄송합니다. 관련 정보를 찾을 수 없습니다."
        }
    
    # 검색된 문서들을 하나의 컨텍스트로 결합
    context = "\n\n".join(contents)
    
    # 답변 생성
    answer_message = [
        SystemMessage(content="""
            검색된 약품 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.
            약품명, 효능, 용법, 주의사항 등을 포함하여 답변하세요.
            의학적 조언이 아닌 일반적인 정보 제공임을 명시하세요.
            검색된 약품 정보가 아니면 답을 생성하지 마세요
        """),
        HumanMessage(content=f"""
        질문: {question}
        
        관련 정보:
        {context}
        
        위 정보를 바탕으로 답변해주세요.
        """)
    ]
    
    answer_response = llm.invoke(answer_message)
    # LLM 응답 처리
    final_answer = answer_response.content if hasattr(answer_response, 'content') else str(answer_response)
    final_answer = final_answer.strip()
    
    print(f"[답변 생성] 완료")
    
    return {
        **state,
        "final_answer": final_answer
    }



def non_medicine_response_agent(state: MedicineSelfRAGState) -> MedicineSelfRAGState:
    """약/증상 관련이 아닌 질문에 대한 응답"""
    return {
        **state,
        "final_answer": "죄송합니다. 이 시스템은 약품 정보에 대해서만 답변할 수 있습니다. 약품이나 의료 증상에 관한 질문을 해주세요."
    }


# -----------------------------------------------
# (2) 증상 → 약 추천 + 약 정보 제공
# -----------------------------------------------
def symptom_to_medicine_agent(state: MedicineSelfRAGState) -> MedicineSelfRAGState:
    """
    사용자가 증상을 말하면 약을 추천하고 그 약의 정보를 제공하는 노드
    """
    question = state["question"]
    
    # LangGraph 방식으로 직접 LLM 호출
    message = [
        SystemMessage(content="""사용자가 말한 증상에 맞는 일반의약품(OTC)을 추천하고,
추천된 약의 주요 성분, 효능, 복용법, 주의사항을 요약해줘.

출력 형식:
- 추천 약 이름:
- 주요 효능:
- 복용 방법:
- 주의사항:"""),
        HumanMessage(content=f"💬 사용자 증상: {question}")
    ]
    
    response = llm.invoke(message)
    answer = response.content if hasattr(response, 'content') else str(response)
    answer = answer.strip()
    
    print(f"[증상→약 추천] 완료")
    
    return {
        **state,
        "final_answer": answer
    }


# -----------------------------------------------
# (3) 약 이름 → 관련 정보 제공
# -----------------------------------------------
def medicine_info_agent(state: MedicineSelfRAGState) -> MedicineSelfRAGState:
    """
    사용자가 약 이름을 말하면 해당 약의 상세정보를 제공하는 노드
    """
    question = state["question"]
    
    # LangGraph 방식으로 직접 LLM 호출
    message = [
        SystemMessage(content="""아래 약 이름에 대한 정보를 상세히 요약해줘.
효능, 복용법, 부작용, 주의사항 등을 포함해줘.

출력 형식:
- 약 이름:
- 주요 효능:
- 복용 방법:
- 부작용 및 주의사항:"""),
        HumanMessage(content=f"💊 약 이름: {question}")
    ]
    
    response = llm.invoke(message)
    answer = response.content if hasattr(response, 'content') else str(response)
    answer = answer.strip()
    
    print(f"[약 정보 제공] 완료")
    
    return {
        **state,
        "final_answer": answer
    }


# -----------------------------------------------
# (4) 먹은 약 + 증상 → 부작용 원인 분석
# -----------------------------------------------
def side_effect_agent(state: MedicineSelfRAGState) -> MedicineSelfRAGState:
    """
    사용자가 복용한 약과 증상을 제시하면
    어떤 약에서 부작용이 나타났는지 분석하는 노드
    """
    question = state["question"]
    
    # LangGraph 방식으로 직접 LLM 호출
    message = [
        SystemMessage(content="""사용자가 복용한 약 목록과 나타난 증상을 바탕으로
어떤 약에서 부작용이 발생했을 가능성이 높은지 추론해줘.
각 약의 성분과 부작용 사례를 근거로 설명해줘.

출력 형식:
- 의심되는 약:
- 근거 설명:
- 권장 조치:"""),
        HumanMessage(content=f"💊 복용 약 목록과 증상: {question}")
    ]
    
    response = llm.invoke(message)
    answer = response.content if hasattr(response, 'content') else str(response)
    answer = answer.strip()
    
    print(f"[부작용 분석] 완료")
    
    return {
        **state,
        "final_answer": answer
    }



################### 워크플로우 구성 ###################

def route_question(state):
    """질문 유형에 따른 라우팅 함수"""
    if not state.get("is_medicine_related", False):
        return "non_medicine_response"
    
    question_type = state.get("question_type", "general")
    
    if question_type == "symptom":
        return "symptom_to_medicine"
    elif question_type == "medicine_info":
        return "medicine_info"
    elif question_type == "side_effect":
        return "side_effect"
    else:
        # 일반적인 약품 관련 질문은 벡터 검색 후 답변 생성
        return "search_vectordb"


def create_medicine_workflow():
    """약품 정보 워크플로우 생성"""
    
    # 그래프 생성
    workflow = StateGraph(MedicineSelfRAGState)
    
    # 노드 추가
    workflow.add_node("question_type", question_type_agent)
    workflow.add_node("search_vectordb", search_vectordb_node)
    workflow.add_node("generate_answer", generate_answer_agent)
    workflow.add_node("symptom_to_medicine", symptom_to_medicine_agent)
    workflow.add_node("medicine_info", medicine_info_agent)
    workflow.add_node("side_effect", side_effect_agent)
    workflow.add_node("non_medicine_response", non_medicine_response_agent)
    
    # 시작 엣지 추가
    workflow.set_entry_point("question_type")
    
    # 질문 유형에 따른 분기
    workflow.add_conditional_edges("question_type", route_question)
    
    # 벡터 검색 후 답변 생성 (일반적인 약품 질문용)
    workflow.add_edge("search_vectordb", "generate_answer")
    
    # 각 에이전트에서 직접 종료
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("symptom_to_medicine", END)
    workflow.add_edge("medicine_info", END)
    workflow.add_edge("side_effect", END)
    workflow.add_edge("non_medicine_response", END)
    
    return workflow.compile()

################### 메인 실행 함수 ###################

def run_medicine_rag(question: str):
    """약품 정보 RAG 시스템 실행"""
    
    try:
        # 워크플로우 생성
        workflow = create_medicine_workflow()
        
        # 초기 상태 설정
        initial_state = {
            "question": question,
            "is_medicine_related": False,
            "question_type": "general",
            "contents": [],
            "sources": [],
            "final_answer": ""
        }
        
        # 워크플로우 실행
        result = workflow.invoke(initial_state)
        
        return result
        
    except Exception as e:
        return {
            "final_answer": f"처리 중 오류가 발생했습니다: {str(e)}"
        }