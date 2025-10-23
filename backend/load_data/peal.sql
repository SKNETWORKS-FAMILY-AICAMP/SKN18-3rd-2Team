-- PostgreSQL + pgvector 확장 기능을 위한 초기화 스크립트
-- 약품 정보 벡터 DB 구축을 위한 SQL

-- 1. pgvector 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. qa_embedding 테이블 생성 (VectorDB만 사용)
CREATE TABLE IF NOT EXISTS qa_embedding (
    id BIGSERIAL PRIMARY KEY,
    embedding VECTOR(768) NOT NULL,  -- 768차원 임베딩 (ko-sroberta-multitask)
    content TEXT NOT NULL,  -- 데이터
    metadata JSONB,  -- 제품명
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 인덱스 생성 (벡터 유사도 검색 최적화)
CREATE INDEX IF NOT EXISTS qa_embedding_vector_idx 
ON qa_embedding USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- 4. 메타데이터 검색을 위한 인덱스
CREATE INDEX IF NOT EXISTS qa_embedding_metadata_idx 
ON qa_embedding USING gin(metadata);

-- 5. 유용한 쿼리들

-- 벡터 유사도 검색 쿼리 템플릿
-- 사용법: '[0.1, 0.2, ...]' 부분을 실제 검색 벡터로 치환
/*
SELECT 
    id,
    metadata->>'제품명' as product_name,
    metadata->>'content' as content,
    1 - (embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
FROM qa_embedding
ORDER BY similarity DESC
LIMIT 10;
*/

-- 특정 제품 검색
/*
SELECT 
    id,
    metadata->>'제품명' as product_name,
    metadata->>'content' as content,
    1 - (embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
FROM qa_embedding
WHERE metadata->>'제품명' LIKE '%게보린%'
ORDER BY similarity DESC
LIMIT 5;
*/

-- 통계 정보 조회
SELECT 
    '총 임베딩 개수' as metric,
    COUNT(*) as value
FROM qa_embedding
UNION ALL
SELECT 
    '벡터 차원' as metric,
    768 as value
UNION ALL
SELECT 
    '고유 제품 수' as metric,
    COUNT(DISTINCT metadata->>'제품명') as value
FROM qa_embedding;

-- 제품별 청크 개수
SELECT 
    metadata->>'제품명' as product_name,
    COUNT(*) as chunk_count
FROM qa_embedding
GROUP BY metadata->>'제품명'
ORDER BY chunk_count DESC
LIMIT 10;

-- 테이블 상태 확인
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE tablename = 'qa_embedding';