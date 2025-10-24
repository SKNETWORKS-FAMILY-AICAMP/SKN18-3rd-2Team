"""
RAG 기반 약품 정보 시스템
벡터 검색 + Self-RAG 에이전트 통합
"""
# 로컬 모듈 임포트
from RAG.langgraph import run_medicine_rag
from RAG.model import llm


def main():
    """메인 함수 - RAG 기반 약품 정보 시스템 대화형 모드"""
    print("=== 💊 RAG 기반 약품 정보 시스템 ===\n")
    print("약품 또는 아픈 증상에 대해 말씀해주시면 관련된 약품 정보를 제공해드립니다.")

    try:
        # 1. 모델 로드 확인
        print("모델 로드 확인 중...")
        print(f"LLM 모델: {type(llm).__name__}")
        print("모델 로드 완료 ✅\n")

        print("이제 질문을 입력해보세요! (종료하려면 'exit' 또는 'quit' 입력)\n")

        # 2. 대화형 루프 시작
        while True:
            question = input("질문 💬 > ").strip()
            if question.lower() in {"exit", "quit"}:
                print("시스템 종료 중...")
                break

            if not question:
                continue

            try:
                result = run_medicine_rag(question)
                print(f"\n🧠 답변: {result.get('final_answer', '답변을 생성할 수 없습니다.')}\n")
            except Exception as e:
                print(f"❌ 질문 처리 중 오류 발생: {e}\n")

    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        print("\n다음을 확인해주세요:")
        print("1. PostgreSQL 데이터베이스가 실행 중인지")
        print("2. 필요한 Python 패키지들이 설치되어 있는지")
        print("3. 데이터가 로드되어 있는지")
    
    finally:
        print("\n=== 시스템 종료 ===")


if __name__ == "__main__":
    main()
