"""
Multi-Agent RAG 시스템
의약품 정보 검색의 정확도 향상을 위한 다중 에이전트 구조
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import VectorDB
from src.embeddings import EmbeddingGenerator

@dataclass
class AgentResponse:
    """에이전트 응답 데이터 클래스"""
    agent_name: str
    response: str
    confidence: float
    reasoning: str
    sources: List[str]
    metadata: Dict[str, Any]

@dataclass
class QueryContext:
    """쿼리 컨텍스트 정보"""
    original_query: str
    processed_query: str
    query_type: str  # "symptom", "drug_name", "interaction", "dosage" etc.
    user_context: Dict[str, Any]  # age, conditions, etc.

class BaseAgent(ABC):
    """기본 에이전트 추상 클래스"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.db = VectorDB()
        self.embedding_generator = EmbeddingGenerator()
    
    @abstractmethod
    def can_handle(self, query_context: QueryContext) -> bool:
        """이 에이전트가 해당 쿼리를 처리할 수 있는지 판단"""
        pass
    
    @abstractmethod
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """쿼리 처리 및 응답 생성"""
        pass
    
    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[tuple]:
        """관련 문서 검색"""
        query_embedding = self.embedding_generator.generate_embedding(query, is_query=True)
        return self.db.similarity_search(query_embedding, limit=top_k)

class SymptomAnalysisAgent(BaseAgent):
    """증상 분석 전문 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="SymptomAnalysisAgent",
            description="증상을 분석하고 관련 의약품을 추천하는 에이전트"
        )
    
    def can_handle(self, query_context: QueryContext) -> bool:
        """증상 관련 쿼리인지 판단"""
        # TODO: 증상 키워드 분석 로직 구현
        symptom_keywords = ["아파", "통증", "열", "기침", "소화불량", "두통", "복통"]
        return any(keyword in query_context.processed_query for keyword in symptom_keywords)
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """증상 기반 의약품 추천"""
        query = query_context.processed_query
        
        # 증상별 키워드 매핑
        symptom_mapping = {
            "두통": ["두통", "머리", "아파", "편두통"],
            "감기": ["감기", "기침", "콧물", "목아픔", "인후통"],
            "소화불량": ["소화", "위", "속쓰림", "복통", "배아픔"],
            "해열": ["열", "발열", "체온", "해열"],
            "진통": ["아픔", "통증", "진통", "아프"]
        }
        
        # 관련 문서 검색
        relevant_docs = self.get_relevant_documents(query, top_k=5)
        
        if not relevant_docs:
            return AgentResponse(
                agent_name=self.name,
                response="해당 증상과 관련된 의약품 정보를 찾을 수 없습니다.",
                confidence=0.1,
                reasoning="데이터베이스에서 관련 문서를 찾지 못함",
                sources=[],
                metadata={"query_type": "symptom_analysis"}
            )
        
        # 증상 분석 및 추천
        detected_symptoms = []
        for symptom, keywords in symptom_mapping.items():
            if any(keyword in query for keyword in keywords):
                detected_symptoms.append(symptom)
        
        # 응답 생성
        if detected_symptoms:
            symptom_text = ", ".join(detected_symptoms)
            response = f"{symptom_text} 증상에 도움이 될 수 있는 의약품을 찾았습니다.\n\n"
        else:
            response = "증상 분석 결과 다음 의약품들이 도움이 될 수 있습니다.\n\n"
        
        # 상위 3개 제품 정보 포함
        for i, (doc_id, content, product_name, distance) in enumerate(relevant_docs[:3], 1):
            confidence_score = 1 - distance
            response += f"{i}. {product_name} (관련도: {confidence_score:.2f})\n"
            # 내용 요약 (첫 100자)
            summary = content[:100] + "..." if len(content) > 100 else content
            response += f"   {summary}\n\n"
        
        avg_confidence = sum(1 - doc[3] for doc in relevant_docs[:3]) / min(3, len(relevant_docs))
        
        return AgentResponse(
            agent_name=self.name,
            response=response.strip(),
            confidence=avg_confidence,
            reasoning=f"증상 키워드 분석 및 유사도 기반 검색 (검색된 문서: {len(relevant_docs)}개)",
            sources=[doc[2] for doc in relevant_docs[:3]],
            metadata={"query_type": "symptom_analysis", "detected_symptoms": detected_symptoms}
        )

class DrugInteractionAgent(BaseAgent):
    """약물 상호작용 분석 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="DrugInteractionAgent", 
            description="약물 간 상호작용을 분석하는 에이전트"
        )
    
    def can_handle(self, query_context: QueryContext) -> bool:
        """약물 상호작용 관련 쿼리인지 판단"""
        # TODO: 상호작용 키워드 분석 로직 구현
        interaction_keywords = ["함께", "같이", "병용", "상호작용", "금기"]
        return any(keyword in query_context.processed_query for keyword in interaction_keywords)
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """약물 상호작용 분석"""
        query = query_context.processed_query
        
        # 약물명 추출 시도
        common_drugs = [
            "아스피린", "타이레놀", "게보린", "낙센", "부루펜", "애드빌",
            "펜잘", "낙센", "이부프로펜", "아세트아미노펜", "디클로페낙"
        ]
        
        mentioned_drugs = [drug for drug in common_drugs if drug in query]
        
        # 상호작용 관련 키워드 강화 검색
        interaction_query = f"상호작용 병용 함께 금기 {query}"
        relevant_docs = self.get_relevant_documents(interaction_query, top_k=5)
        
        if not relevant_docs:
            return AgentResponse(
                agent_name=self.name,
                response="약물 상호작용 정보를 찾을 수 없습니다.",
                confidence=0.1,
                reasoning="상호작용 관련 문서를 찾지 못함",
                sources=[],
                metadata={"query_type": "drug_interaction"}
            )
        
        # 상호작용 정보 추출
        interaction_info = []
        warning_info = []
        
        for doc_id, content, product_name, distance in relevant_docs[:3]:
            confidence_score = 1 - distance
            
            sentences = content.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in ['상호작용', '병용', '함께', '동시']):
                    interaction_info.append({
                        'product': product_name,
                        'info': sentence.strip(),
                        'confidence': confidence_score
                    })
                elif any(keyword in sentence_lower for keyword in ['금기', '피해야', '주의']):
                    warning_info.append({
                        'product': product_name,
                        'info': sentence.strip(),
                        'confidence': confidence_score
                    })
        
        # 응답 생성
        response = "약물 상호작용 분석 결과:\n\n"
        
        if mentioned_drugs:
            response += f"🔍 언급된 약물: {', '.join(mentioned_drugs)}\n\n"
        
        if interaction_info:
            response += "🔸 상호작용 정보:\n"
            for info in interaction_info[:3]:  # 최대 3개
                response += f"   • {info['product']}: {info['info']}\n"
            response += "\n"
        
        if warning_info:
            response += "⚠️ 주의사항:\n"
            for info in warning_info[:3]:  # 최대 3개
                response += f"   • {info['product']}: {info['info']}\n"
            response += "\n"
        
        if not interaction_info and not warning_info:
            response += "구체적인 상호작용 정보:\n\n"
            for i, (doc_id, content, product_name, distance) in enumerate(relevant_docs[:2], 1):
                confidence_score = 1 - distance
                summary = content[:150] + "..." if len(content) > 150 else content
                response += f"{i}. {product_name} (관련도: {confidence_score:.2f})\n"
                response += f"   {summary}\n\n"
        
        response += "🚨 중요: 여러 약물을 함께 복용하기 전에 반드시 의사나 약사와 상담하세요."
        
        avg_confidence = sum(1 - doc[3] for doc in relevant_docs[:3]) / min(3, len(relevant_docs))
        
        return AgentResponse(
            agent_name=self.name,
            response=response.strip(),
            confidence=avg_confidence,
            reasoning="상호작용 키워드 기반 검색 및 약물명 분석",
            sources=[doc[2] for doc in relevant_docs[:3]],
            metadata={
                "query_type": "drug_interaction", 
                "mentioned_drugs": mentioned_drugs,
                "interaction_found": len(interaction_info),
                "warnings_found": len(warning_info)
            }
        )

class DosageAgent(BaseAgent):
    """용법/용량 전문 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="DosageAgent",
            description="약물의 용법과 용량 정보를 제공하는 에이전트"
        )
    
    def can_handle(self, query_context: QueryContext) -> bool:
        """용법/용량 관련 쿼리인지 판단"""
        # TODO: 용법/용량 키워드 분석 로직 구현
        dosage_keywords = ["용법", "용량", "복용법", "먹는법", "하루", "몇번", "얼마나"]
        return any(keyword in query_context.processed_query for keyword in dosage_keywords)
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """용법/용량 정보 제공"""
        query = query_context.processed_query
        
        # 용법/용량 관련 키워드 강화 검색
        dosage_query = f"용법 용량 복용법 {query}"
        relevant_docs = self.get_relevant_documents(dosage_query, top_k=5)
        
        if not relevant_docs:
            return AgentResponse(
                agent_name=self.name,
                response="해당 의약품의 용법/용량 정보를 찾을 수 없습니다.",
                confidence=0.1,
                reasoning="용법/용량 관련 문서를 찾지 못함",
                sources=[],
                metadata={"query_type": "dosage_info"}
            )
        
        # 용법/용량 정보 추출
        dosage_info = []
        for doc_id, content, product_name, distance in relevant_docs[:3]:
            confidence_score = 1 - distance
            
            # 용법/용량 관련 문장 추출
            sentences = content.split('.')
            dosage_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in ['용법', '용량', '복용', '투여', '하루', '회', '정', '캡슐', 'mg']):
                    dosage_sentences.append(sentence.strip())
            
            if dosage_sentences:
                dosage_text = '. '.join(dosage_sentences[:2])  # 최대 2문장
                dosage_info.append({
                    'product': product_name,
                    'dosage': dosage_text,
                    'confidence': confidence_score
                })
        
        # 응답 생성
        if dosage_info:
            response = "용법/용량 정보:\n\n"
            for i, info in enumerate(dosage_info, 1):
                response += f"{i}. {info['product']} (신뢰도: {info['confidence']:.2f})\n"
                response += f"   {info['dosage']}\n\n"
        else:
            # 일반적인 정보 제공
            response = "용법/용량 관련 정보:\n\n"
            for i, (doc_id, content, product_name, distance) in enumerate(relevant_docs[:2], 1):
                confidence_score = 1 - distance
                summary = content[:150] + "..." if len(content) > 150 else content
                response += f"{i}. {product_name} (관련도: {confidence_score:.2f})\n"
                response += f"   {summary}\n\n"
        
        response += "⚠️ 정확한 용법/용량은 의사나 약사와 상담하시기 바랍니다."
        
        avg_confidence = sum(1 - doc[3] for doc in relevant_docs[:3]) / min(3, len(relevant_docs))
        
        return AgentResponse(
            agent_name=self.name,
            response=response.strip(),
            confidence=avg_confidence,
            reasoning="용법/용량 키워드 기반 문서 검색 및 정보 추출",
            sources=[doc[2] for doc in relevant_docs[:3]],
            metadata={"query_type": "dosage_info", "found_dosage_info": len(dosage_info)}
        )

class SafetyAgent(BaseAgent):
    """안전성/주의사항 전문 에이전트"""
    
    def __init__(self):
        super().__init__(
            name="SafetyAgent",
            description="약물 안전성 및 주의사항을 분석하는 에이전트"
        )
    
    def can_handle(self, query_context: QueryContext) -> bool:
        """안전성 관련 쿼리인지 판단"""
        # TODO: 안전성 키워드 분석 로직 구현
        safety_keywords = ["주의", "부작용", "금기", "임신", "수유", "어린이", "노인"]
        return any(keyword in query_context.processed_query for keyword in safety_keywords)
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """안전성 정보 분석"""
        query = query_context.processed_query
        
        # 안전성 관련 키워드 강화 검색
        safety_query = f"주의사항 부작용 금기 안전성 {query}"
        relevant_docs = self.get_relevant_documents(safety_query, top_k=5)
        
        if not relevant_docs:
            return AgentResponse(
                agent_name=self.name,
                response="해당 의약품의 안전성 정보를 찾을 수 없습니다.",
                confidence=0.1,
                reasoning="안전성 관련 문서를 찾지 못함",
                sources=[],
                metadata={"query_type": "safety_info"}
            )
        
        # 안전성 정보 분류
        safety_categories = {
            "부작용": ["부작용", "이상반응", "부작용", "side effect"],
            "금기사항": ["금기", "금지", "피해야", "하지마", "contraindication"],
            "주의사항": ["주의", "조심", "warning", "caution"],
            "특수집단": ["임신", "수유", "어린이", "노인", "간장애", "신장애"]
        }
        
        # 안전성 정보 추출
        safety_info = {}
        for doc_id, content, product_name, distance in relevant_docs[:3]:
            confidence_score = 1 - distance
            
            sentences = content.split('.')
            for category, keywords in safety_categories.items():
                category_sentences = []
                for sentence in sentences:
                    if any(keyword in sentence for keyword in keywords):
                        category_sentences.append(sentence.strip())
                
                if category_sentences:
                    if category not in safety_info:
                        safety_info[category] = []
                    safety_info[category].append({
                        'product': product_name,
                        'info': '. '.join(category_sentences[:2]),  # 최대 2문장
                        'confidence': confidence_score
                    })
        
        # 응답 생성
        response = "안전성 및 주의사항 정보:\n\n"
        
        if safety_info:
            for category, items in safety_info.items():
                response += f"🔸 {category}:\n"
                for item in items[:2]:  # 카테고리당 최대 2개
                    response += f"   • {item['product']}: {item['info']}\n"
                response += "\n"
        else:
            # 일반적인 안전성 정보 제공
            response += "관련 안전성 정보:\n\n"
            for i, (doc_id, content, product_name, distance) in enumerate(relevant_docs[:2], 1):
                confidence_score = 1 - distance
                summary = content[:150] + "..." if len(content) > 150 else content
                response += f"{i}. {product_name} (관련도: {confidence_score:.2f})\n"
                response += f"   {summary}\n\n"
        
        response += "⚠️ 중요: 의약품 사용 전 반드시 의사나 약사와 상담하시기 바랍니다."
        
        avg_confidence = sum(1 - doc[3] for doc in relevant_docs[:3]) / min(3, len(relevant_docs))
        
        return AgentResponse(
            agent_name=self.name,
            response=response.strip(),
            confidence=avg_confidence,
            reasoning="안전성 키워드 기반 문서 검색 및 카테고리별 정보 분류",
            sources=[doc[2] for doc in relevant_docs[:3]],
            metadata={"query_type": "safety_info", "safety_categories": list(safety_info.keys())}
        )

class MultiAgentCoordinator:
    """다중 에이전트 조정자"""
    
    def __init__(self):
        self.agents: List[BaseAgent] = [
            SymptomAnalysisAgent(),
            DrugInteractionAgent(), 
            DosageAgent(),
            SafetyAgent()
        ]
        self.llm_client = None  # TODO: LLM 클라이언트 연결
    
    def analyze_query(self, query: str, user_context: Dict[str, Any] = None) -> QueryContext:
        """쿼리 분석 및 컨텍스트 생성"""
        # TODO: 쿼리 분석 로직 구현
        return QueryContext(
            original_query=query,
            processed_query=query.lower(),
            query_type="general",
            user_context=user_context or {}
        )
    
    def select_agents(self, query_context: QueryContext) -> List[BaseAgent]:
        """적절한 에이전트들 선택"""
        selected_agents = []
        for agent in self.agents:
            if agent.can_handle(query_context):
                selected_agents.append(agent)
        
        # 기본적으로 최소 1개 에이전트는 선택
        if not selected_agents:
            selected_agents = [self.agents[0]]  # SymptomAnalysisAgent를 기본으로
        
        return selected_agents
    
    def aggregate_responses(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """에이전트 응답들을 종합"""
        # TODO: 응답 종합 로직 구현
        if not responses:
            return {"error": "No agent responses"}
        
        # 신뢰도 기반 가중 평균 또는 투표 시스템
        best_response = max(responses, key=lambda r: r.confidence)
        
        return {
            "primary_response": best_response.response,
            "confidence": best_response.confidence,
            "contributing_agents": [r.agent_name for r in responses],
            "all_sources": list(set(sum([r.sources for r in responses], []))),
            "detailed_responses": [
                {
                    "agent": r.agent_name,
                    "response": r.response,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning
                } for r in responses
            ]
        }
    
    def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """메인 쿼리 처리 함수"""
        try:
            # 1. 쿼리 분석
            query_context = self.analyze_query(query, user_context)
            
            # 2. 적절한 에이전트 선택
            selected_agents = self.select_agents(query_context)
            
            # 3. 각 에이전트에서 응답 생성
            responses = []
            for agent in selected_agents:
                try:
                    response = agent.process_query(query_context)
                    responses.append(response)
                except Exception as e:
                    print(f"Agent {agent.name} error: {e}")
            
            # 4. 응답 종합
            final_result = self.aggregate_responses(responses)
            
            return final_result
            
        except Exception as e:
            return {"error": f"Multi-agent processing failed: {e}"}

# 사용 예시 및 테스트 함수
def test_multi_agent_system():
    """Multi-agent 시스템 테스트"""
    coordinator = MultiAgentCoordinator()
    
    test_queries = [
        "두통에 좋은 약이 있나요?",
        "게보린정과 타이레놀을 함께 먹어도 되나요?", 
        "아스피린은 하루에 몇 번 먹어야 하나요?",
        "임신 중에 먹으면 안 되는 약은 무엇인가요?"
    ]
    
    print("🤖 Multi-Agent RAG 시스템 테스트")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[테스트 {i}] {query}")
        result = coordinator.process_query(query)
        print(f"응답: {result.get('primary_response', 'No response')}")
        print(f"참여 에이전트: {result.get('contributing_agents', [])}")
        print(f"신뢰도: {result.get('confidence', 0):.2f}")
        print("-" * 30)

if __name__ == "__main__":
    test_multi_agent_system()