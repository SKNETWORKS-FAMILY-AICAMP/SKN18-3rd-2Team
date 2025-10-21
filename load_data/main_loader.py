"""
메인 데이터 로더 - 약품 정보 JSON → VectorDB
"""

from tqdm import tqdm
from config import (
    connect_db, 
    load_json_documents, 
    get_text_splitter,
    get_embedding_model, 
    create_embedding
)

def insert_data(json_path=None, clear_mode=False):
    """JSON을 VectorDB에 삽입"""
    
    # 경로 자동 설정
    if json_path is None:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "..", "data", "drug_info_preprocessed.json")
    
    # 연결 및 준비
    conn = connect_db()
    print("✅ DB 연결 완료")
    
    # JSON 문서 로드
    documents = load_json_documents(json_path)
    print(f"✅ JSON 로드: {len(documents)}개 약품")
    
    # 텍스트 스플리터 준비
    splitter = get_text_splitter()
    print("✅ 텍스트 스플리터 준비")
    
    # 문서 청킹 먼저 수행
    print("\n📄 문서 청킹 중...")
    all_chunks = []
    
    for doc in documents:
        # 각 약품 정보를 청킹
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    print(f"✅ 청킹 완료: {len(all_chunks)}개 청크")
    
    # 임베딩 플랫폼 이름 가져오기 (예: HuggingFace, OpenAI 등)
    model, platform_name = get_embedding_model(return_platform_name=True) if 'return_platform_name' in get_embedding_model.__code__.co_varnames else (get_embedding_model(), "Unknown Platform")
    print(f"\n🤖 {platform_name} 임베딩 모델 로드 중...")
    print("✅ 임베딩 모델 로드 완료")
    
    cursor = conn.cursor()
    
    try:
        # 기존 데이터 삭제 (clear_mode가 True일 때)
        if clear_mode:
            print("🗑️  기존 데이터 삭제 중...")
            cursor.execute("DELETE FROM qa_embedding")
            conn.commit()
            print("✅ 기존 데이터 삭제 완료")
        
        # VectorDB에 임베딩 삽입
        print("\n🔮 VectorDB 임베딩 삽입 중...")
        inserted_count = 0
        
        for chunk in tqdm(all_chunks):
            # 청크 텍스트로 임베딩 생성
            embedding = create_embedding(chunk.page_content, model)
            
            # 제품명을 메타데이터에서 추출
            product_name = chunk.metadata.get('제품명', 'Unknown')
            
            # qa_embedding 테이블에 삽입 (RDB 테이블 없이 VectorDB만 사용)
            import psycopg2.extras
            metadata_dict = {'제품명': product_name, 'content': chunk.page_content}
            cursor.execute("""
                INSERT INTO qa_embedding (embedding, metadata)
                VALUES (%s, %s)
            """, (embedding, psycopg2.extras.Json(metadata_dict)))
            
            inserted_count += 1
        
        conn.commit()
        print(f"✅ 임베딩 삽입 완료: {inserted_count}개")
        
        # 검증
        cursor.execute("SELECT COUNT(*) FROM qa_embedding")
        embedding_count = cursor.fetchone()[0]
        
        print(f"\n📊 결과:")
        print(f"- 임베딩: {embedding_count}개")
        print("🎉 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    insert_data()