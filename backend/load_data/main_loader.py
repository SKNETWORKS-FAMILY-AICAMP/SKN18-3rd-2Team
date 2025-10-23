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
    
    # 경로 자동 설정 - backend/data/drug_info_preprocessed.json 사용
    if json_path is None:
        import os
        # 현재 파일의 절대 경로에서 backend/data로 이동
        current_file = os.path.abspath(__file__)
        # backend/load_data/main_loader.py -> backend/data/drug_info_preprocessed.json
        backend_dir = os.path.dirname(os.path.dirname(current_file))  # backend 폴더
        json_path = os.path.join(backend_dir, "data", "drug_info_preprocessed.json")
        print(f"현재 파일: {current_file}")
        print(f"Backend 디렉토리: {backend_dir}")
        print(f"JSON 파일 경로: {json_path}")
        print(f"파일 존재 여부: {os.path.exists(json_path)}")
    
    # 연결 및 준비
    conn = connect_db()
    print("DB 연결 완료")
    
    # JSON 문서 로드
    documents = load_json_documents(json_path)
    print(f"JSON 로드: {len(documents)}개 약품")
    
    # 텍스트 스플리터 준비
    splitter = get_text_splitter()
    print("텍스트 스플리터 준비")
    
    # 문서 청킹 먼저 수행
    print("\n문서 청킹 중...")
    all_chunks = []
    
    for doc in documents:
        # 각 약품 정보를 청킹
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    print(f"청킹 완료: {len(all_chunks)}개 청크")
    
    # 임베딩 모델 로드 부분 주석처리 (데이터만 삽입)
    model, platform_name = get_embedding_model(return_platform_name=True) if 'return_platform_name' in get_embedding_model.__code__.co_varnames else (get_embedding_model(), "Unknown Platform")
    print(f"\n{platform_name} 임베딩 모델 로드 중...")
    print("임베딩 모델 로드 완료")
    
    cursor = conn.cursor()
    
    try:
        # 기존 데이터 삭제 (clear_mode가 True일 때)
        if clear_mode:
            print("기존 데이터 삭제 중...")
            cursor.execute("DELETE FROM qa_embedding")
            conn.commit()
            print("기존 데이터 삭제 완료")
        
        # VectorDB에 임베딩 삽입 (배치 처리)
        print("\nVectorDB 임베딩 삽입 중...")
        inserted_count = 0
        
        # 배치 크기 설정 (OpenAI는 100개씩, HuggingFace는 10개씩)
        batch_size = 1
        
        # 배치별로 처리 
        for i in tqdm(range(0, len(all_chunks), batch_size)):
            batch_chunks = all_chunks[i:i + batch_size]
            
            batch_texts = [chunk.page_content for chunk in batch_chunks]
            if hasattr(model, 'embed_documents'):
                embeddings = model.embed_documents(batch_texts)
            elif hasattr(model, 'embed_query'):
                embeddings = [model.embed_query(text) for text in batch_texts]
            else:
                embeddings = model.encode(batch_texts)
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
            
            # 데이터만 삽입
            import psycopg2.extras
            for chunk in batch_chunks:
                product_name = chunk.metadata.get('제품명', 'Unknown')
                content = chunk.page_content 
                metadata_dict = {'제품명': product_name}
                cursor.execute("""
                    INSERT INTO qa_embedding (embedding, metadata, content)
                    VALUES (NULL, %s, %s)
                """, (psycopg2.extras.Json(metadata_dict), content))
                inserted_count += 1
        
        conn.commit()
        print(f"임베딩 삽입 완료: {inserted_count}개")
        
        # 검증
        cursor.execute("SELECT COUNT(*) FROM qa_embedding")
        embedding_count = cursor.fetchone()[0]
        
        print(f"\n결과:")
        print(f"- 임베딩: {embedding_count}개")
        print("완료!")
        
    except Exception as e:
        print(f"오류: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    insert_data()