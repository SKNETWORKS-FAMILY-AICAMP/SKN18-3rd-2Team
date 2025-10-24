-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 제품 문서 테이블 생성
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,        -- 제품 설명/문서 내용
    embedding VECTOR(1024),       -- E5-large-instruct 모델 임베딩 크기 (1024차원)
    product_name VARCHAR(255) NOT NULL,  -- 제품명
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 생성 시간
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 수정 시간
);

-- 제품명으로 검색하기 위한 인덱스
CREATE INDEX idx_documents_product_name ON documents(product_name);

-- 다른 차원의 임베딩을 사용할 경우 테이블 수정:
-- ALTER TABLE documents ALTER COLUMN embedding TYPE VECTOR(384);   -- 384차원 모델용
-- ALTER TABLE documents ALTER COLUMN embedding TYPE VECTOR(768);   -- 768차원 모델용


-- updated_at 자동 업데이트 함수 생성
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- updated_at 자동 업데이트 트리거 생성
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

/*
-- 벡터 인덱스 생성 (성능 향상)
-- 데이터가 충분히 쌓인 후에 아래 명령어로 인덱스를 생성하세요.
-- 
-- 권장사항:
-- - 1,000개 미만: 인덱스 없이 사용 (brute force가 더 빠름)
-- - 1,000~10,000개: lists = 100
-- - 10,000개 이상: lists = 1000
-- 
-- 코사인 유사도 검색용 인덱스:
-- CREATE INDEX documents_embedding_cosine_idx ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- 
-- L2 거리 검색용 인덱스:
-- CREATE INDEX documents_embedding_l2_idx ON documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
-- 
-- 내적 검색용 인덱스:
-- CREATE INDEX documents_embedding_ip_idx ON documents USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);
*/
