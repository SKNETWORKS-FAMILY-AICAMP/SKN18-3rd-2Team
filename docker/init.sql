-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE drug_info (
    id SERIAL PRIMARY KEY,
    content TEXT,                 -- 문서 내용
    embedding VECTOR(1024),       -- OpenAI 등 임베딩 크기에 맞춤
    metadata JSONB                -- 메타데이터
);

