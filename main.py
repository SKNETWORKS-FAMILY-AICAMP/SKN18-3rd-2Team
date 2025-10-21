"""
멀티에이전트 + 랭그래프 기반 약품 정보 시스템
벡터 검색 + LLM 통합
"""
import os
import sys
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# 로컬 모듈 임포트
from retriever import VectorRetriever
from llm_handler import create_drug_workflow, AgentState


class DrugInfoSystem:
    """멀티에이전트 기반 약품 정보 검색 및 질의응답 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        print("멀티에이전트 시스템 초기화 중...")
        
        # 컴포넌트 초기화
        self.retriever = VectorRetriever()
        self.workflow = create_drug_workflow()  # 랭그래프 워크플로우 초기화
        
        print("시스템 초기화 완료!")
    
    def search_drugs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        약품 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        print(f"검색 중: '{query}'")
        results = self.retriever.hybrid_search(query, top_k)
        return results
    
    def ask_question(self, question: str, use_context: bool = True) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성 (랭그래프 기반 멀티에이전트)
        
        Args:
            question: 사용자 질문
            use_context: 컨텍스트 사용 여부
            
        Returns:
            에이전트 처리 결과
        """
        print(f"질문 처리 중: '{question}'")
        
        # 컨텍스트 수집
        context = ""
        if use_context:
            context = self.retriever.get_context_for_llm(question, top_k=3)
            if "관련 정보를 찾을 수 없습니다" in context:
                context = ""
        
        # 랭그래프 워크플로우 실행
        initial_state = {
            "messages": [{"role": "user", "content": question}],
            "user_query": question,
            "context": context,
            "agent_type": "",
            "response": "",
            "error": ""
        }
        
        try:
            result = self.workflow.invoke(initial_state)
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
    
    def recommend_drugs(self, symptoms: str) -> Dict[str, Any]:
        """
        증상에 따른 약품 추천 (랭그래프 기반)
        
        Args:
            symptoms: 사용자 증상
            
        Returns:
            추천 결과
        """
        print(f"약품 추천 중: '{symptoms}'")
        
        # 관련 약품 검색
        context = self.retriever.get_context_for_llm(symptoms, top_k=5)
        
        if "관련 정보를 찾을 수 없습니다" in context:
            return {
                "success": False,
                "response": "죄송합니다. 해당 증상과 관련된 약품 정보를 찾을 수 없습니다. 의사나 약사와 상담하시기 바랍니다.",
                "agent_type": "pharmacist"
            }
        
        # 랭그래프 워크플로우 실행
        initial_state = {
            "messages": [{"role": "user", "content": symptoms}],
            "user_query": symptoms,
            "context": context,
            "agent_type": "recommendation",  # 추천 타입으로 강제 설정
            "response": "",
            "error": ""
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            return {
                "success": True,
                "response": result.get("response", ""),
                "agent_type": result.get("agent_type", "pharmacist"),
                "error": result.get("error", "")
            }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "error": f"워크플로우 실행 중 오류: {str(e)}"
            }
    
    def analyze_drug_interaction(self, drug_names: List[str]) -> Dict[str, Any]:
        """
        약물 상호작용 분석 (랭그래프 기반)
        
        Args:
            drug_names: 확인할 약품명 리스트
            
        Returns:
            상호작용 분석 결과
        """
        print(f"약물 상호작용 분석 중: {', '.join(drug_names)}")
        
        # 각 약품에 대한 정보 수집
        all_context = []
        for drug_name in drug_names:
            results = self.retriever.search_by_product_name(drug_name)
            if results:
                for result in results:
                    all_context.append(f"약품명: {result['metadata'].get('제품명', 'Unknown')}\n{result['content']}")
        
        if not all_context:
            return {
                "success": False,
                "response": "죄송합니다. 해당 약품들의 정보를 찾을 수 없습니다.",
                "agent_type": "interaction_analyst"
            }
        
        context = "\n\n".join(all_context)
        query = f"{', '.join(drug_names)}의 상호작용을 분석해주세요"
        
        # 랭그래프 워크플로우 실행
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "user_query": query,
            "context": context,
            "agent_type": "interaction",  # 상호작용 타입으로 강제 설정
            "response": "",
            "error": ""
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            return {
                "success": True,
                "response": result.get("response", ""),
                "agent_type": result.get("agent_type", "interaction_analyst"),
                "error": result.get("error", "")
            }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "error": f"워크플로우 실행 중 오류: {str(e)}"
            }
    
    def get_drug_usage_guide(self, drug_name: str) -> Dict[str, Any]:
        """
        특정 약품의 사용법 안내 (랭그래프 기반)
        
        Args:
            drug_name: 약품명
            
        Returns:
            사용법 안내
        """
        print(f"사용법 안내 조회 중: '{drug_name}'")
        
        # 약품 정보 검색
        results = self.retriever.search_by_product_name(drug_name)
        
        if not results:
            return {
                "success": False,
                "response": f"죄송합니다. '{drug_name}'에 대한 정보를 찾을 수 없습니다.",
                "agent_type": "usage_guide"
            }
        
        # 가장 관련성 높은 결과 사용
        best_result = results[0]
        context = f"약품명: {best_result['metadata'].get('제품명', 'Unknown')}\n{best_result['content']}"
        
        query = f"{drug_name}의 사용법을 알려주세요"
        
        # 랭그래프 워크플로우 실행
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "user_query": query,
            "context": context,
            "agent_type": "usage",  # 사용법 타입으로 강제 설정
            "response": "",
            "error": ""
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            return {
                "success": True,
                "response": result.get("response", ""),
                "agent_type": result.get("agent_type", "usage_guide"),
                "error": result.get("error", "")
            }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "error": f"워크플로우 실행 중 오류: {str(e)}"
            }
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "="*60)
        print("멀티에이전트 약품 정보 검색 및 질의응답 시스템")
        print("="*60)
        print("명령어:")
        print("1. '검색: [검색어]' - 약품 검색")
        print("2. '질문: [질문]' - 일반 질문 (멀티에이전트 처리)")
        print("3. '추천: [증상]' - 약품 추천 (약사 에이전트)")
        print("4. '상호작용: [약품1, 약품2, ...]' - 약물 상호작용 분석")
        print("5. '사용법: [약품명]' - 사용법 안내")
        print("6. 'quit' 또는 'exit' - 종료")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n사용자: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("시스템을 종료합니다.")
                    break
                
                if not user_input:
                    continue
                
                # 명령어 파싱
                if user_input.startswith('검색:'):
                    query = user_input[3:].strip()
                    results = self.search_drugs(query, top_k=3)
                    
                    if results:
                        print(f"\n검색 결과 ({len(results)}개):")
                        for i, result in enumerate(results, 1):
                            product_name = result['metadata'].get('제품명', 'Unknown')
                            similarity = result['similarity']
                            content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                            print(f"\n{i}. {product_name} (유사도: {similarity:.3f})")
                            print(f"   {content}")
                    else:
                        print("검색 결과가 없습니다.")
                
                elif user_input.startswith('질문:'):
                    question = user_input[3:].strip()
                    result = self.ask_question(question)
                    
                    if result["success"]:
                        print(f"\n답변: {result['response']}")
                        print(f"처리 에이전트: {result['agent_type']}")
                    else:
                        print(f"오류: {result['error']}")
                
                elif user_input.startswith('추천:'):
                    symptoms = user_input[3:].strip()
                    result = self.recommend_drugs(symptoms)
                    
                    if result["success"]:
                        print(f"\n추천: {result['response']}")
                        print(f"처리 에이전트: {result['agent_type']}")
                    else:
                        print(f"오류: {result['error']}")
                
                elif user_input.startswith('상호작용:'):
                    drugs_input = user_input[5:].strip()
                    drug_names = [drug.strip() for drug in drugs_input.split(',')]
                    result = self.analyze_drug_interaction(drug_names)
                    
                    if result["success"]:
                        print(f"\n상호작용 분석: {result['response']}")
                        print(f"처리 에이전트: {result['agent_type']}")
                    else:
                        print(f"오류: {result['error']}")
                
                elif user_input.startswith('사용법:'):
                    drug_name = user_input[4:].strip()
                    result = self.get_drug_usage_guide(drug_name)
                    
                    if result["success"]:
                        print(f"\n사용법 안내: {result['response']}")
                        print(f"처리 에이전트: {result['agent_type']}")
                    else:
                        print(f"오류: {result['error']}")
                
                else:
                    # 기본적으로 질문으로 처리 (멀티에이전트)
                    result = self.ask_question(user_input)
                    
                    if result["success"]:
                        print(f"\n답변: {result['response']}")
                        print(f"처리 에이전트: {result['agent_type']}")
                    else:
                        print(f"오류: {result['error']}")
            
            except KeyboardInterrupt:
                print("\n시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"오류가 발생했습니다: {str(e)}")
    
    def close(self):
        """시스템 종료"""
        if hasattr(self, 'retriever'):
            self.retriever.close()
        print("시스템이 종료되었습니다.")


def main():
    """메인 함수"""
    print("멀티에이전트 약품 정보 검색 및 질의응답 시스템 시작")
    
    try:
        # 시스템 초기화
        system = DrugInfoSystem()
        
        # 대화형 모드 시작
        system.interactive_mode()
        
    except Exception as e:
        print(f"시스템 초기화 중 오류가 발생했습니다: {str(e)}")
        print("다음을 확인해주세요:")
        print("1. PostgreSQL 데이터베이스가 실행 중인지")
        print("2. Ollama가 실행 중이고 gemma2:4b 모델이 설치되어 있는지")
        print("3. 필요한 Python 패키지들이 설치되어 있는지")
        print("4. langgraph 패키지가 설치되어 있는지")
    
    finally:
        if 'system' in locals():
            system.close()


def test_system():
    """시스템 테스트"""
    print("멀티에이전트 시스템 테스트 시작...")
    
    try:
        system = DrugInfoSystem()
        
        # 테스트 케이스들
        test_cases = [
            ("검색", "두통에 좋은 약"),
            ("질문", "게보린정은 어떤 약인가요?"),
            ("질문", "게보린정을 먹고 어지러운데 왜그런건가요?"),
            ("추천", "소화불량 증상"),
            ("사용법", "게보린정")
        ]
        
        for test_type, test_input in test_cases:
            print(f"\n=== {test_type} 테스트: {test_input} ===")
            
            if test_type == "검색":
                results = system.search_drugs(test_input, top_k=2)
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['metadata'].get('제품명', 'Unknown')} (유사도: {result['similarity']:.3f})")
            
            elif test_type == "질문":
                result = system.ask_question(test_input)
                if result["success"]:
                    print(f"답변: {result['response']}")
                    print(f"에이전트: {result['agent_type']}")
                else:
                    print(f"오류: {result['error']}")
            
            elif test_type == "추천":
                result = system.recommend_drugs(test_input)
                if result["success"]:
                    print(f"추천: {result['response']}")
                    print(f"에이전트: {result['agent_type']}")
                else:
                    print(f"오류: {result['error']}")
            
            elif test_type == "사용법":
                result = system.get_drug_usage_guide(test_input)
                if result["success"]:
                    print(f"사용법: {result['response']}")
                    print(f"에이전트: {result['agent_type']}")
                else:
                    print(f"오류: {result['error']}")
        
        print("\n테스트 완료!")
        system.close()
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_system()
    else:
        main()
