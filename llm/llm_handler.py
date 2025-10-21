"""
멀티에이전트 + 랭그래프 기반 약품 정보 시스템
Ollama Gemma3:4b 모델 사용
"""
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import List, Dict, Any, Optional, TypedDict, Annotated
import json


# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    user_query: str
    context: str
    agent_type: str
    response: str
    error: str


# LLM 초기화
llm = Ollama(
    model="gemma2:4b",
    temperature=0.1,
    top_p=0.9,
    num_ctx=1024
)


# 에이전트별 프롬프트 템플릿
def get_agent_prompt(agent_type: str) -> str:
    """에이전트 타입별 프롬프트 반환"""
    prompts = {
        "query_analyzer": """
당신은 사용자 질문을 분석하는 전문가입니다.
사용자 질문을 분석하여 다음 중 하나로 분류하세요:
- "search": 약품 검색이 필요한 경우
- "question": 일반적인 질문인 경우  
- "recommendation": 약품 추천이 필요한 경우
- "interaction": 약물 상호작용 분석이 필요한 경우
- "usage": 사용법 안내가 필요한 경우

사용자 질문: {query}

분석 결과 (JSON 형태):
{{"type": "분류결과", "reasoning": "분석 근거"}}
""",
        
        "drug_expert": """
당신은 약품 정보 전문가입니다. 주어진 약품 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

약품 정보:
{context}

사용자 질문: {query}

답변 가이드라인:
1. 주어진 약품 정보를 바탕으로 답변하세요
2. 정확하지 않은 정보는 추측하지 마세요
3. 의학적 조언이 필요한 경우 의사나 약사와 상담하라고 안내하세요
4. 약품 사용 시 주의사항을 강조하세요
5. 한국어로 친근하고 이해하기 쉽게 답변하세요

답변:
""",
        
        "pharmacist": """
당신은 약사입니다. 사용자의 증상과 관련된 약품 정보를 바탕으로 적절한 약품을 추천해주세요.

사용자 증상: {query}

관련 약품 정보:
{context}

추천 가이드라인:
1. 증상에 맞는 약품을 추천하세요
2. 사용법과 주의사항을 안내하세요
3. 부작용이나 주의사항을 강조하세요
4. 의사 상담이 필요한 경우를 언급하세요
5. 한국어로 친근하게 답변하세요

추천:
""",
        
        "interaction_analyst": """
당신은 약사입니다. 여러 약품의 동시 복용 시 상호작용을 분석해주세요.

확인할 약품들: {query}

관련 약품 정보:
{context}

분석 가이드라인:
1. 각 약품의 주요 성분과 작용을 파악하세요
2. 상호작용 가능성을 분석하세요
3. 주의사항과 권장사항을 제시하세요
4. 의사나 약사 상담이 필요한 경우를 명시하세요
5. 한국어로 명확하게 답변하세요

분석 결과:
""",
        
        "usage_guide": """
당신은 약사입니다. 특정 약품의 사용법을 자세히 안내해주세요.

약품명: {query}

약품 정보:
{context}

안내 가이드라인:
1. 용법과 용량을 명확히 설명하세요
2. 복용 시 주의사항을 강조하세요
3. 부작용과 이상반응을 안내하세요
4. 보관법을 설명하세요
5. 의사 상담이 필요한 경우를 언급하세요
6. 한국어로 이해하기 쉽게 설명하세요

사용법 안내:
"""
    }
    return prompts.get(agent_type, "")


# 에이전트 함수들
def query_analyzer(state: AgentState) -> AgentState:
    """질문 분석 에이전트"""
    try:
        query = state["user_query"]
        prompt = get_agent_prompt("query_analyzer").format(query=query)
        
        response = llm.invoke(prompt)
        
        # JSON 파싱 시도
        try:
            analysis = json.loads(response)
            agent_type = analysis.get("type", "question")
        except:
            # JSON 파싱 실패 시 키워드 기반 분류
            if any(keyword in query.lower() for keyword in ["검색", "찾아", "어떤 약"]):
                agent_type = "search"
            elif any(keyword in query.lower() for keyword in ["추천", "어떤 약을", "증상"]):
                agent_type = "recommendation"
            elif any(keyword in query.lower() for keyword in ["상호작용", "함께", "동시"]):
                agent_type = "interaction"
            elif any(keyword in query.lower() for keyword in ["사용법", "복용법", "용법"]):
                agent_type = "usage"
            else:
                agent_type = "question"
        
        return {
            **state,
            "agent_type": agent_type,
            "messages": state["messages"] + [{"role": "assistant", "content": f"질문 분석 완료: {agent_type}"}]
        }
    except Exception as e:
        return {
            **state,
            "error": f"질문 분석 중 오류: {str(e)}",
            "agent_type": "question"
        }


def drug_expert(state: AgentState) -> AgentState:
    """약품 정보 전문가 에이전트"""
    try:
        query = state["user_query"]
        context = state["context"]
        
        prompt = get_agent_prompt("drug_expert").format(query=query, context=context)
        response = llm.invoke(prompt)
        
        return {
            **state,
            "response": response,
            "messages": state["messages"] + [{"role": "assistant", "content": response}]
        }
    except Exception as e:
        return {
            **state,
            "error": f"약품 전문가 응답 생성 중 오류: {str(e)}"
        }


def pharmacist(state: AgentState) -> AgentState:
    """약사 에이전트 (추천)"""
    try:
        query = state["user_query"]
        context = state["context"]
        
        prompt = get_agent_prompt("pharmacist").format(query=query, context=context)
        response = llm.invoke(prompt)
        
        return {
            **state,
            "response": response,
            "messages": state["messages"] + [{"role": "assistant", "content": response}]
        }
    except Exception as e:
        return {
            **state,
            "error": f"약사 추천 생성 중 오류: {str(e)}"
        }


def interaction_analyst(state: AgentState) -> AgentState:
    """상호작용 분석 에이전트"""
    try:
        query = state["user_query"]
        context = state["context"]
        
        prompt = get_agent_prompt("interaction_analyst").format(query=query, context=context)
        response = llm.invoke(prompt)
        
        return {
            **state,
            "response": response,
            "messages": state["messages"] + [{"role": "assistant", "content": response}]
        }
    except Exception as e:
        return {
            **state,
            "error": f"상호작용 분석 중 오류: {str(e)}"
        }


def usage_guide(state: AgentState) -> AgentState:
    """사용법 안내 에이전트"""
    try:
        query = state["user_query"]
        context = state["context"]
        
        prompt = get_agent_prompt("usage_guide").format(query=query, context=context)
        response = llm.invoke(prompt)
        
        return {
            **state,
            "response": response,
            "messages": state["messages"] + [{"role": "assistant", "content": response}]
        }
    except Exception as e:
        return {
            **state,
            "error": f"사용법 안내 생성 중 오류: {str(e)}"
        }


def error_handler(state: AgentState) -> AgentState:
    """오류 처리 에이전트"""
    error_msg = state.get("error", "알 수 없는 오류가 발생했습니다.")
    return {
        **state,
        "response": f"죄송합니다. {error_msg}",
        "messages": state["messages"] + [{"role": "assistant", "content": f"오류: {error_msg}"}]
    }


# 랭그래프 워크플로우 정의
def create_drug_workflow():
    """약품 정보 워크플로우 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("drug_expert", drug_expert)
    workflow.add_node("pharmacist", pharmacist)
    workflow.add_node("interaction_analyst", interaction_analyst)
    workflow.add_node("usage_guide", usage_guide)
    workflow.add_node("error_handler", error_handler)
    
    # 시작점 설정
    workflow.set_entry_point("query_analyzer")
    
    # 조건부 라우팅
    def route_after_analysis(state: AgentState) -> str:
        """분석 후 라우팅"""
        if state.get("error"):
            return "error_handler"
        
        agent_type = state.get("agent_type", "question")
        
        if agent_type == "search" or agent_type == "question":
            return "drug_expert"
        elif agent_type == "recommendation":
            return "pharmacist"
        elif agent_type == "interaction":
            return "interaction_analyst"
        elif agent_type == "usage":
            return "usage_guide"
        else:
            return "drug_expert"
    
    # 엣지 추가
    workflow.add_conditional_edges(
        "query_analyzer",
        route_after_analysis,
        {
            "drug_expert": "drug_expert",
            "pharmacist": "pharmacist", 
            "interaction_analyst": "interaction_analyst",
            "usage_guide": "usage_guide",
            "error_handler": "error_handler"
        }
    )
    
    # 모든 에이전트에서 종료
    workflow.add_edge("drug_expert", END)
    workflow.add_edge("pharmacist", END)
    workflow.add_edge("interaction_analyst", END)
    workflow.add_edge("usage_guide", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()


# 메인 함수들
def process_query(user_query: str, context: str = "") -> Dict[str, Any]:
    """사용자 질문 처리"""
    workflow = create_drug_workflow()
    
    initial_state = {
        "messages": [{"role": "user", "content": user_query}],
        "user_query": user_query,
        "context": context,
        "agent_type": "",
        "response": "",
        "error": ""
    }
    
    try:
        result = workflow.invoke(initial_state)
        return {
            "success": True,
            "response": result.get("response", ""),
            "agent_type": result.get("agent_type", ""),
            "error": result.get("error", "")
        }
    except Exception as e:
        return {
            "success": False,
            "response": "",
            "error": f"워크플로우 실행 중 오류: {str(e)}"
        }


def test_agents():
    """에이전트 시스템 테스트"""
    print("멀티에이전트 시스템 테스트 시작...")
    
    test_queries = [
        "두통에 좋은 약이 있나요?",
        "소화불량 증상에 어떤 약을 추천해주세요",
        "게보린정과 겔포스엠을 함께 복용해도 되나요?",
        "게보린정 사용법을 알려주세요"
    ]
    
    for query in test_queries:
        print(f"\n=== 질문: {query} ===")
        result = process_query(query, "테스트 컨텍스트")
        
        if result["success"]:
            print(f"답변: {result['response']}")
            print(f"에이전트 타입: {result['agent_type']}")
        else:
            print(f"오류: {result['error']}")
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    test_agents()
