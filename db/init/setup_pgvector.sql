-- pgvector 확장 설치 
CREATE EXTENSION IF NOT EXISTS vector;

-- 대분류, 중분류, 질문, 답변, 데이터 수정시각을 저장하는 테이블
CREATE TABLE IF NOT EXISTS qa_text (
  id BIGSERIAL PRIMARY KEY,
  big_category TEXT NOT NULL,
  mid_category TEXT NOT NULL,
  question     TEXT NOT NULL,
  answer       TEXT NOT NULL,
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- pgvector: 임베딩 저장 테이블
-- qa_embedding: qa_text의 각 질문에 대한 문장 임베딩 벡터를 저장
-- qa_id: qa_text.id를 참조하는 외래 키
-- embedding: 벡터 타입으로 임베딩 모델의 차원의 수를 지정해줘야 함!
-- ON DELETE CASCADE: qa_text 데이블에서 삭제된 데이터는 임베딩도 자동으로 삭제
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.tables WHERE table_name='qa_embedding'
  ) THEN
    CREATE TABLE qa_embedding (
      qa_id BIGINT PRIMARY KEY REFERENCES qa_text(id) ON DELETE CASCADE,
      embedding VECTOR(384)
    );
  END IF;
END $$;

-- 인덱스(검색 성능을 향상)
-- idx_qa_embedding: embedding필드에 대해서 HNSW 인덱스를 생성, 코사인 유사도 기반 검색
DROP INDEX IF EXISTS idx_qa_embedding;
CREATE INDEX idx_qa_embedding ON qa_embedding USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_qa_mid ON qa_text(mid_category);
CREATE INDEX IF NOT EXISTS idx_qa_big ON qa_text(big_category);
