#DB 기반설정/RAG관련 탐색 함수 정의/csv읽기&가공/빈 테이블 생성/읽은 데이터 테이블에 적재 하는 코드.
#최초 한번만 실행하면 되고, 이미 데이터가 적재되어 있으면 실행 x
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain.vectorstores.base import VectorStore
from typing import List, Dict, Any, Optional
from psycopg2.extras import Json
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm # ✅ tqdm 라이브러리 import
import json

# --- 1. 설정 변수 ---
# ... (이전과 동일) ...
CONNECTION_STRING = "postgresql://admin:admin123@localhost:5432/vectordb"
TABLE_NAME = "drug_info"
CSV_PATH = "./data/drug_info_preprocessed.csv" 
MODEL_NAME = "jhgan/ko-sroberta-multitask"

# --- 2. 클래스 및 함수 정의 ---

def setup_database():
    # ... (이전과 동일) ...
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        content TEXT,
        metadata JSONB,
        embedding VECTOR(768)
    );
    """
    conn = None
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        conn.autocommit = True
        cursor = conn.cursor()
        print("1. pgvector 확장 활성화를 시도합니다...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        print(f"2. '{TABLE_NAME}' 테이블 생성을 시도합니다...")
        cursor.execute(create_table_query)
        print("✅ 데이터베이스 초기 설정이 완료되었습니다.")
        cursor.close()
    except Exception as e:
        print(f"❌ 데이터베이스 설정 중 오류 발생: {e}")
    finally:
        if conn is not None:
            conn.close()

class CustomPGVector(VectorStore):
    # ... (이전과 동일, 필수 메서드 포함) ...
    def __init__(self, conn_str: str, embedding_fn, table: str):
        self.conn = psycopg2.connect(conn_str)
        self.embedding_fn = embedding_fn
        self.table = table

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        metadatas = metadatas or [{} for _ in texts]
        
        # ✅ tqdm을 사용하여 임베딩 진행 상태를 표시합니다.
        print("   - 텍스트를 벡터로 변환하는 중...")
        embeddings = self.embedding_fn.embed_documents(texts)
        
        print("   - 변환된 벡터를 DB에 저장하는 중...")
        with self.conn.cursor() as cur:
            # ✅ tqdm을 사용하여 DB 저장 진행 상태를 표시합니다.
            for text, emb, meta in tqdm(zip(texts, embeddings, metadatas), total=len(texts)):
                cur.execute(
                    f"INSERT INTO {self.table} (content, embedding, metadata) VALUES (%s, %s, %s)",
                    (text, emb, Json(meta)),
                )
        self.conn.commit()
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.add_texts(texts=texts, metadatas=metadatas)
        return []

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        사용자 질문과 가장 유사한 문서를 DB에서 검색하는 필수 메서드
        """
        query_emb = self.embedding_fn.embed_query(query)
        
        params = []
        sql_query_template = f"SELECT content, metadata FROM {self.table}"
        
        where_clauses = []
        if filter:
            filter_json = json.dumps(filter)
            where_clauses.append("metadata @> %s::jsonb")
            params.append(filter_json)

        if where_clauses:
            sql_query_template += " WHERE 1=1 AND" + " AND ".join(where_clauses)
        
        sql_query_template += " ORDER BY embedding <-> %s::vector LIMIT %s"
        params.append(query_emb)
        params.append(k)
        
        with self.conn.cursor() as cur:
            cur.execute(sql_query_template, tuple(params))
            rows = cur.fetchall()

        return [Document(page_content=row[0], metadata=row[1]) for row in rows]

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Any, metadatas: Optional[List[dict]] = None, **kwargs):
        conn_str = kwargs.get("connection_string")
        table_name = kwargs.get("collection_name")
        store = cls(conn_str, embedding, table_name)
        store.add_texts(texts, metadatas)
        return store

def ingest_data():
    """CSV 데이터를 읽어 DB에 저장합니다."""
    print("3. 데이터 주입(Ingestion) 프로세스를 시작합니다...")
    try:
        df = pd.read_csv(CSV_PATH)
        # ✅ NaN 값을 빈 문자열로 변환하는 코드 추가!
        df = df.fillna('')
        print(f"   - '{CSV_PATH}' 파일에서 {len(df)}개의 데이터를 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{CSV_PATH}' 파일을 찾을 수 없습니다.")
        return

    documents = []
    print("   - 데이터를 의미 단위(Chunk)로 분할 및 변환하는 중...")
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        pn = (row.get('제품명', '') or '').strip()

        # 1) content 문서: 모든 핵심 섹션을 개별 문서로 임베딩
        field_map = {
            "효능": row.get('효능', ''),
            "사용법": row.get('사용법', ''),
            "사용 전 주의": row.get('사용 전 주의', ''),
            "사용상 주의사항": row.get('사용상 주의사항', ''),
            "약/음식 주의": row.get('약/음식 주의', ''),
            "이상반응": row.get('이상반응', ''),
            "보관법": row.get('보관법', ''),
        }

        for section, text in field_map.items():
            text = (text or '').strip()
            if not text:
                continue  # ✅ 비어있으면 추가 안 함 (기존 정책 유지)
            documents.append(Document(
                page_content=f"[제품명] {pn}\n[{section}] {text}",
                metadata={"product_name": pn, "doc_type": "content", "data_type": section}
            ))

        # 2) meta 문서: 주의/상호작용/보관/이상반응 묶음(있는 것만)
        meta_sections = ["사용 전 주의", "사용상 주의사항", "약/음식 주의", "이상반응", "보관법"]
        meta_parts = []
        for sec in meta_sections:
            val = (row.get(sec, '') or '').strip()
            if val:
                meta_parts.append(f"{sec}: {val}")
        if meta_parts:
            meta_text = " / ".join(meta_parts)
            documents.append(Document(
                page_content=f"[제품명] {pn}\n[메타] {meta_text}",
                metadata={"product_name": pn, "doc_type": "meta", "data_type": "메타"}
            ))

        # 3) summary 문서: 전 섹션을 한 줄 요약(있는 것만)
        summary_parts = []
        for sec, val in field_map.items():
            val = (val or '').strip()
            if val:
                summary_parts.append(f"{sec}: {val}")
        if summary_parts:
            summary_text = " / ".join(summary_parts)
            documents.append(Document(
                page_content=f"[제품명] {pn}\n[요약] {summary_text}",
                metadata={"product_name": pn, "doc_type": "summary", "data_type": "요약"}
            ))

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    vectorstore = CustomPGVector(
        conn_str=CONNECTION_STRING,
        embedding_fn=embeddings,
        table=TABLE_NAME
    )
    
    print("   - 임베딩 및 DB 저장을 시작합니다...")
    vectorstore.add_documents(documents)
    print(f"✅ 4. {len(documents)}개의 문서를 데이터베이스에 성공적으로 저장했습니다.")

# --- 3. 메인 실행 블록 ---
if __name__ == "__main__":
    print("--- 전체 데이터베이스 파이프라인을 시작합니다 ---")
    setup_database()
    ingest_data()
    print("--- 모든 파이프라인 작업이 완료되었습니다 ---")