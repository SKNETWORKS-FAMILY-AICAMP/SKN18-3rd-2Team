"""
Multi-Agent RAG 시스템 테스트 및 실행 (로컬 LLM 사용)
"""
from src.multi_agent_system import MultiAgentCoordinator

import requests
import json

class EnhancedMultiAgentRAG:
    """로컬 LLM과 연동된 향상된 Multi-Agent RAG 시스템"""
    
    def __init__(self, use_local_llm: bool = False, llm_model: str = "gemma3:4b"):
        self.coordinator = MultiAgentCoordinator()
        self.use_local_llm = use_local_llm
        self.llm_model = llm_model
        self.local_llm = None
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_base_url = "http://localhost:11434"
        
        print(f"🚀 Enhanced Multi-Agent RAG 시스템 초기화 완료!")
        print(f"👥 활성 에이전트: {len(self.coordinator.agents)}개")
        print(f"🐳 Docker Ollama 모드: {llm_model}")
        
        # Docker Ollama 컨테이너 사용
    
    def query_with_llm_synthesis(self, query: str, user_context: dict = None) -> dict:
        """Multi-agent 결과를 LLM으로 종합하여 최종 답변 생성"""
        
        print(f"🔍 쿼리: {query}")
        
        # 1. Multi-agent 시스템으로 정보 수집
        agent_results = self.coordinator.process_query(query, user_context)
        
        if "error" in agent_results:
            return {"error": agent_results["error"]}
        
        # 2. LLM을 위한 프롬프트 구성
        prompt = self._create_synthesis_prompt(query, agent_results)
        
        # 3. LLM으로 최종 답변 생성
        llm_response = self._call_llm(prompt)
        
        return {
            "final_answer": llm_response,
            "agent_analysis": agent_results,
            "query": query,
            "model_used": self.llm_model
        }
    
    def _create_synthesis_prompt(self, query: str, agent_results: dict) -> str:
        """Multi-agent 결과를 바탕으로 LLM 프롬프트 생성"""
        
        detailed_responses = agent_results.get("detailed_responses", [])
        sources = agent_results.get("all_sources", [])
        
        prompt = f"""당신은 의약품 정보 전문가입니다. 여러 전문 에이전트들이 분석한 결과를 종합하여 사용자에게 정확하고 도움이 되는 답변을 제공해주세요.

사용자 질문: {query}

전문 에이전트 분석 결과:
"""
        
        for response in detailed_responses:
            prompt += f"""
[{response['agent']}] (신뢰도: {response['confidence']:.2f})
- 분석: {response['reasoning']}
- 결과: {response['response']}
"""
        
        if sources:
            prompt += f"\n참조 의약품: {', '.join(sources[:5])}"
        
        prompt += """

답변 지침:
1. 전문 에이전트들의 분석을 종합하여 정확한 정보를 제공하세요
2. 의학적 조언이 아닌 일반적인 의약품 정보임을 명시하세요
3. 구체적인 의약품명과 주요 정보를 포함하세요
4. 전문의 상담을 권하는 문구를 포함하세요
5. 명확하고 이해하기 쉽게 설명하세요

종합 답변:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """gemma3:4b 모델로 LLM 호출 (Ollama 전용)"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "gemma3:4b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 800
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '답변을 생성할 수 없습니다.')
            else:
                return f"LLM 서버 오류: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Docker Ollama 컨테이너에 연결할 수 없습니다. 'docker-compose up -d' 명령으로 서비스를 시작해주세요."
        except Exception as e:
            return f"❌ LLM 호출 오류: {e}"
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n💬 Multi-Agent RAG 대화형 모드 시작!")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n질문: ").strip()
                
                if query.lower() in ['quit', 'exit', '종료', 'q']:
                    print("👋 대화를 종료합니다.")
                    break
                
                if not query:
                    continue
                
                print("\n🔄 분석 중...")
                result = self.query_with_llm_synthesis(query)
                
                if "error" in result:
                    print(f"❌ 오류: {result['error']}")
                    continue
                
                print(f"\n🤖 답변:")
                print(result["final_answer"])
                
                # 에이전트 분석 정보 (선택적 표시)
                show_details = input("\n상세 분석 정보를 보시겠습니까? (y/N): ").lower() == 'y'
                if show_details:
                    self._show_agent_details(result["agent_analysis"])
                
            except KeyboardInterrupt:
                print("\n\n👋 대화를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    
    def _show_agent_details(self, agent_analysis: dict):
        """에이전트 분석 상세 정보 표시"""
        print("\n📊 에이전트 분석 상세:")
        print("-" * 30)
        
        for response in agent_analysis.get("detailed_responses", []):
            print(f"\n🤖 {response['agent']} (신뢰도: {response['confidence']:.2f})")
            print(f"   분석: {response['reasoning']}")
            print(f"   결과: {response['response']}")
        
        sources = agent_analysis.get("all_sources", [])
        if sources:
            print(f"\n📚 참조 의약품: {', '.join(sources)}")

def check_system_status():
    """시스템 상태 확인"""
    print("🔍 시스템 상태 확인 중...")
    
    # 1. 데이터베이스 확인
    try:
        from src.database import VectorDB
        db = VectorDB()
        cursor = db.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cursor.fetchone()[0]
        db.close()
        
        print(f"✅ 데이터베이스: {doc_count}개 문서 저장됨")
        
        if doc_count < 100:
            print("⚠️ 문서 수가 적습니다. 더 나은 결과를 위해 데이터 로딩을 완료해주세요.")
            
    except Exception as e:
        print(f"❌ 데이터베이스 오류: {e}")
        return False
    
    # 2. Ollama 로컬 서버 및 gemma3:4b 모델 확인
    try:
        print("🔗 Ollama 로컬 서버 연결 중...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            print(f"📋 로컬에서 사용 가능한 모델: {len(models)}개")
            
            if 'gemma3:4b' in model_names:
                print("✅ Ollama 로컬 서버: gemma3:4b 모델 사용 가능")
                
                # 모델 정보 표시
                for model in models:
                    if model['name'] == 'gemma3:4b':
                        size = model.get('size', 'Unknown')
                        print(f"📊 gemma3:4b 모델 크기: {size}")
                        break
            else:
                print("⚠️ gemma3:4b 모델이 로컬에 설치되지 않았습니다.")
                print("💡 다음 명령으로 모델을 설치해주세요:")
                print("   ollama pull gemma3:4b")
                return False
                
        else:
            print(f"❌ Ollama 서버 응답 오류: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Docker Ollama 컨테이너에 연결할 수 없습니다.")
        print("💡 다음 단계를 실행해주세요:")
        print("   1. docker-compose up -d (Docker 서비스 시작)")
        print("   2. python setup_docker_ollama.py (자동 설정)")
        return False
    except Exception as e:
        print(f"❌ Ollama 연결 오류: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Enhanced Multi-Agent RAG 시스템 (gemma3:4b)")
    print("=" * 50)
    
    # 시스템 상태 확인
    if not check_system_status():
        print("\n❌ 시스템 준비가 완료되지 않았습니다.")
        print("💡 다음 명령을 실행해주세요:")
        print("   1. docker-compose up -d")
        print("   2. python setup_docker_ollama.py")
        exit(1)
    
    # Multi-Agent RAG 시스템 초기화 (gemma3:4b 고정)
    enhanced_rag = EnhancedMultiAgentRAG(use_local_llm=False, llm_model="gemma3:4b")
    
    # 실행 모드 선택
    print("\n실행 모드를 선택하세요:")
    print("1. 대화형 모드")
    print("2. 테스트 모드")
    
    choice = input("선택 (1 또는 2): ").strip()
    
    if choice == "1":
        enhanced_rag.interactive_mode()
    else:
        # 테스트 모드
        test_queries = [
            "두통에 좋은 약이 있나요?",
            "게보린정의 용법과 용량을 알려주세요",
            "임신 중에 피해야 할 약물은 무엇인가요?"
        ]
        
        print("\n🧪 테스트 모드 실행")
        for i, query in enumerate(test_queries, 1):
            print(f"\n[테스트 {i}] {query}")
            result = enhanced_rag.query_with_llm_synthesis(query)
            print(f"답변: {result.get('final_answer', 'No response')}")
            print("-" * 50)