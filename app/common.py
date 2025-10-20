"""
공용 유틸리티 모듈
- Postgres 연결/SQL 실행 헬퍼
- LangChain Embeddings 래퍼 (로컬/OPENAI 지원)
- pgvector 차원 조회 및 공통 검색 SQL
"""
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine


def make_conn_str() -> str:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    pwd = os.getenv("PGPASSWORD")
    db = os.getenv("PGDATABASE")
    driver = os.getenv("PG_DB_DRIVER")
    return f"postgresql+{driver}://{user}:{pwd}@{host}:{port}/{db}"


def get_engine(**kwargs) -> Engine:
    return create_engine(make_conn_str(), **kwargs)


@dataclass
class Embedder:
    backend: str
    model_name: str
    dim: int
    embedder: Embeddings

    def embed_one(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        inputs = list(texts)
        if not inputs:
            return []
        vectors = self.embedder.embed_documents(inputs)
        return [list(vec) for vec in vectors]


def get_embedder() -> Embedder:
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()

    model = os.getenv("LOCAL_EMBEDDING_MODEL")
    normalize = os.getenv("LOCAL_EMBEDDING_NORMALIZE").lower()
    embed_model = HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": normalize},
    )
    dim_env = os.getenv("LOCAL_EMBEDDING_DIM")
    dim = int(dim_env) if dim_env else len(embed_model.embed_query("test"))
    return Embedder(backend=backend, model_name=model, dim=dim, embedder=embed_model)


def fetch_vector_dim(
    conn: Connection,
    table: str = "qa_embedding",
    column: str = "embedding",
) -> int | None:
    sql = text(
        """
        SELECT atttypmod AS dim
        FROM pg_attribute
        WHERE attrelid = to_regclass(:table_name)
          AND attname = :column_name
        """
    )
    return conn.execute(sql, {"table_name": table, "column_name": column}).scalar()


SEARCH_SQL = """
SELECT
  1 - (e.embedding <=> CAST(:qvec AS vector)) AS score,
  t.id,
  t.big_category,
  t.mid_category,
  t.question,
  t.answer
FROM qa_embedding e
JOIN qa_text t ON t.id = e.qa_id
WHERE e.embedding IS NOT NULL
  AND (:big IS NULL OR t.big_category = :big)
  AND (:mid IS NULL OR t.mid_category = :mid)
ORDER BY e.embedding <=> CAST(:qvec AS vector)
LIMIT :k;
"""


def search_similar(
    query: str,
    *,
    k: int = 5,
    big: str | None = None,
    mid: str | None = None,
    engine: Engine | None = None,
    embedder: Embedder | None = None,
) -> Sequence:
    embedder = embedder or get_embedder()
    engine = engine or get_engine()
    qvec = embedder.embed_one(query)
    with engine.begin() as conn:
        rows = conn.execute(
            text(SEARCH_SQL),
            {"qvec": qvec, "k": k, "big": big, "mid": mid},
        ).fetchall()
    return rows
