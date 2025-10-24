"""
PostgreSQL + pgvector 데이터베이스 연결 및 벡터 검색 기능
"""
import os
import psycopg2
from typing import List, Tuple, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """PostgreSQL 데이터베이스 연결"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', 5432),
                database=os.getenv('DB_NAME', 'vectordb'),
                user=os.getenv('DB_USER', 'admin'),
                password=os.getenv('DB_PASSWORD', 'admin123')
            )
            print("✅ 데이터베이스 연결 성공")
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            raise
    
    def insert_document(self, content: str, embedding: List[float], product_name: str) -> int:
        """제품 문서와 임베딩을 데이터베이스에 저장"""
        with self.connection.cursor() as cursor:
            # 벡터를 PostgreSQL VECTOR 타입으로 변환
            vector_str = '[' + ','.join(map(str, embedding)) + ']'
            
            cursor.execute("""
                INSERT INTO documents (content, embedding, product_name)
                VALUES (%s, %s::vector, %s)
                RETURNING id
            """, (content, vector_str, product_name))
            
            doc_id = cursor.fetchone()[0]
            self.connection.commit()
            return doc_id
    
    def similarity_search(self, query_embedding: List[float], limit: int = 5, product_name: str = None) -> List[Tuple]:
        """벡터 유사도 검색"""
        with self.connection.cursor() as cursor:
            # 벡터를 PostgreSQL VECTOR 타입으로 변환
            vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            if product_name:
                # 특정 제품에서만 검색
                cursor.execute("""
                    SELECT id, content, product_name, 
                           embedding <=> %s::vector as distance
                    FROM documents
                    WHERE product_name = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vector_str, product_name, vector_str, limit))
            else:
                # 전체 제품에서 검색
                cursor.execute("""
                    SELECT id, content, product_name, 
                           embedding <=> %s::vector as distance
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vector_str, vector_str, limit))
            
            return cursor.fetchall()
    
    def get_products(self) -> List[str]:
        """등록된 모든 제품명 조회"""
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT product_name FROM documents ORDER BY product_name")
            return [row[0] for row in cursor.fetchall()]
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.connection:
            self.connection.close()