# drug_info_agent.py
# 워크플로우 2번의, 사용자가 증상을 이야기 하면 약을 추천해주고 그 약의 관련정보를 주는 에이전트 함수(노드)

# db_utils.py에서 DB와 통신하는 데 필요한 클래스와 설정값들을 가져옵니다.
from db_utils import CustomPGVector, CONNECTION_STRING, TABLE_NAME, MODEL_NAME
from langchain_community.embeddings import HuggingFaceEmbeddings

def search_drug_info(user_query: str) -> str:
    """
    사용자의 질문을 바탕으로 PGVector DB에서 약품 정보를 검색합니다.
    """
    print(f">> 'drug_info_agent' 실행: '{user_query}'에 대한 정보 검색 중...")

    try:
        # 1. 임베딩 모델 준비 (db_utils.py와 동일한 모델 사용)
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

        # 2. VectorStore 객체 생성 (DB 전문가)
        vectorstore = CustomPGVector(
            conn_str=CONNECTION_STRING,
            embedding_fn=embeddings,
            table=TABLE_NAME
        )
        
        # 3. 유사도 검색 실행
        # k=5는 가장 유사한 문서 5개를 찾아오라는 의미입니다.
        results = vectorstore.similarity_search(user_query, k=5)

        if not results:
            print("   - 검색 결과 없음.")
            return "관련 정보를 찾을 수 없습니다."

        # 4. 검색된 문서 내용들을 하나의 '참고 정보(context)' 텍스트로 결합
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        
        print(f"   - {len(results)}개의 관련 문서를 찾았습니다.")
        return context
    
    except Exception as e:
        print(f"❌ 'drug_info_agent' 실행 중 오류 발생: {e}")
        return "정보를 검색하는 중 오류가 발생했습니다."

# --- 테스트를 위한 실행 코드 ---
if __name__ == "__main__":
    # 테스트 질문
    # drug_info 테이블에 '아스피린' 관련 데이터가 있다고 가정합니다.
    query = "아스피린의 효능과 사용법에 대해 알려줘."
    
    print("\n--- 'drug_info_agent' 단독 테스트 ---")
    retrieved_context = search_drug_info(query)
    
    print("\n[검색된 최종 정보(Context)]")
    print("---------------------------------")
    print(retrieved_context)