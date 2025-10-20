import argparse

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

from common import Embedder, fetch_vector_dim, get_embedder, get_engine


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        on_bad_lines="skip",
    )
    required = ["대분류", "중분류", "질문", "답변"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV에 '{col}' 컬럼이 필요합니다. 현재 컬럼: {df.columns.tolist()}")
    df = df[required].dropna(subset=["질문", "답변"]).reset_index(drop=True)
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    return df


def prepare_texts(df: pd.DataFrame) -> list[str]:
    return (
        "대분류: " + df["대분류"]
        + " | 중분류: " + df["중분류"]
        + " | 질문: " + df["질문"]
    ).tolist()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="CSV -> Postgres(pgvector) 적재")
    parser.add_argument("--csv", default="../data/AI_Backend_interview_QA.csv", help="입력 CSV 경로")
    args = parser.parse_args()

    df = load_dataframe(args.csv)
    embedder: Embedder = get_embedder()
    vectors = embedder.embed_batch(prepare_texts(df))

    engine = get_engine()
    ins_text_sql = text(
        """
        INSERT INTO qa_text (big_category, mid_category, question, answer)
        VALUES (:big, :mid, :q, :a)
        RETURNING id;
        """
    )
    upsert_emb_sql = text(
        """
        INSERT INTO qa_embedding (qa_id, embedding)
        VALUES (:qid, :emb)
        ON CONFLICT (qa_id) DO UPDATE SET embedding = EXCLUDED.embedding;
        """
    )

    with engine.begin() as conn:
        dim_in_db = fetch_vector_dim(conn)
        if dim_in_db and int(dim_in_db) != int(embedder.dim):
            raise RuntimeError(f"DB vector dim({dim_in_db}) != embedder dim({embedder.dim}). setup_pgvector.sql 확인 필요")

        for idx in tqdm(range(len(df)), desc="Ingesting"):
            record = df.loc[idx]
            qa_id = conn.execute(
                ins_text_sql,
                {
                    "big": record["대분류"],
                    "mid": record["중분류"],
                    "q": record["질문"],
                    "a": record["답변"],
                },
            ).scalar_one()
            conn.execute(
                upsert_emb_sql,
                {"qid": qa_id, "emb": vectors[idx]},
            )

    print(f"✅ Done. Inserted {len(df)} rows.")

if __name__ == "__main__":
    main()
