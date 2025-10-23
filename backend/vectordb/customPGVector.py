from langchain_core.vectorstores import VectorStore
from typing import List, Dict, Any, Optional, Tuple
import json
from langchain_core.documents import Document
from psycopg2.extras import Json
import psycopg2

# from .embedding_models import get_embedding_model_openai  # 사용하지 않음


class Singleton(type(VectorStore)):
	_instances = {}

	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls)\
				.__call__(*args, **kwargs)
		return cls._instances[cls]


class CustomPGVector(VectorStore, metaclass=Singleton):
    def __init__(self, conn_str: str, embedding_fn, table: str = "documents"):
        self.conn = psycopg2.connect(conn_str)
        self.embedding_fn = embedding_fn
        self.table = table

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_fn,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        conn_str: str = None,
        table: str = "my_vectors",
        **kwargs,
    ):
        store = cls(conn_str=conn_str, embedding_fn=embedding_fn, table=table)
        store.add_texts(texts, metadatas=metadatas)
        return store

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        metadatas = metadatas or [{} for _ in texts]
        embeddings = self.embedding_fn.embed_documents(texts)

        with self.conn.cursor() as cur:
            for text, emb, meta in zip(texts, embeddings, metadatas):
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                    """,
                    (text, emb, Json(meta)),
                )
        self.conn.commit()

    def similarity_search(self, query: str, k: int = 4,
                          filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        
        # LangChain 호환 임베딩 모델 사용
        if hasattr(self.embedding_fn, 'embed_query'):
            query_emb = self.embedding_fn.embed_query(query)
        else:
            # SentenceTransformer 직접 사용
            query_emb = self.embedding_fn.encode(query).tolist()
        
        # 쿼리 매개변수 리스트 초기화. 필터 매개변수가 있다면 여기에 먼저 추가됩니다.
        params = []
        
        # SQL 쿼리 기본 구조 설정
        sql_query_template = f"""
            SELECT content, metadata
            FROM {self.table}
        """
        
        # WHERE 절을 위한 리스트
        where_clauses = []
        
        if filter:
            # 1. 필터 딕셔너리를 JSON 문자열로 변환합니다.
            filter_json = json.dumps(filter)
            
            # 2. WHERE 절에 'metadata @> %s::jsonb' 조건을 추가합니다.
            where_clauses.append("metadata @> %s::jsonb")
            
            # 3. 필터 JSON 문자열을 params 리스트에 먼저 추가합니다.
            #    이것이 SQL 쿼리에서 가장 먼저 나오는 %s에 바인딩됩니다.
            params.append(filter_json)

        if where_clauses:
            sql_query_template += " WHERE 1=1 AND " + " AND ".join(where_clauses)
        
        # ORDER BY 및 LIMIT 절 추가
        # ORDER BY에는 임베딩 비교가 들어가며, 이는 필터가 있든 없든 항상 두 번째 (혹은 첫 번째) %s가 됩니다.
        sql_query_template += """
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        
        # 4. 임베딩 벡터를 params에 추가합니다.
        #    이는 ORDER BY의 %s에 바인딩됩니다.
        params.append(query_emb)
        
        # 5. LIMIT 값 (k)을 params에 마지막으로 추가합니다.
        #    이는 LIMIT의 %s에 바인딩됩니다.
        params.append(k)
        
        # 최종 SQL 쿼리: (필터가 있을 경우) WHERE [조건] ORDER BY [임베딩] LIMIT [k]
        
        with self.conn.cursor() as cur:
            # 쿼리와 매개변수를 실행
            # 매개변수의 순서는 SQL 쿼리에 나타나는 %s의 순서와 정확히 일치해야 합니다.
            cur.execute(sql_query_template, tuple(params))
            rows = self.__get_unique_documents(cur.fetchall())

        return [Document(page_content=row[0], metadata=row[1]) for row in rows]


    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """쿼리와 유사도 점수를 함께 반환"""
        query_emb = self.embedding_fn.embed_query(query)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT content, metadata, (embedding <-> %s::vector) AS score
                FROM {self.table}
                ORDER BY score
                LIMIT %s
                """,
                (query_emb, k),
            )
            rows = self.__get_unique_documents(cur.fetchall())
            

        return [
            (Document(page_content=row[0], metadata=row[1]), float(row[2]))
            for row in rows
        ]
    
    def __get_unique_documents(self, rows):
        # 중복 제거를 위한 후처리
        unique_contents = set()
        unique_documents = []
        
        for row in rows:
            content = row[0]
            
            if content not in unique_contents:
                unique_contents.add(content)
                unique_documents.append(row) # 중복이 아닐 때 원본 튜플을 저장

        return unique_documents # 중복 제거된 리스트 반환

if __name__ == "__main__":
    str_connection_info = "postgresql://admin:admin123@localhost:5432/vectordb"
    vectorstore = CustomPGVector(
        conn_str=str_connection_info,
        embedding_fn=get_embedding_model_openai()
    )

    print("vectorstore created")