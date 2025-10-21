"""
벡터 임베딩 검색을 위한 Retriever 클래스
PostgreSQL + pgvector를 사용한 유사도 검색
"""
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
from load_data.config import connect_db, get_embedding_model


class VectorRetriever:
    """벡터 임베딩 기반 유사도 검색 클래스"""
    
    def __init__(self, model_name: str = 'dragonkue/snowflake-arctic-embed-l-v2.0-ko'):
        """
        Args:
            model_name: 사용할 임베딩 모델명
        """
        self.embedding_model = get_embedding_model()
        self.conn = connect_db()
        self.cursor = self.conn.cursor()
        
    def create_embedding(self, text: str) -> List[float]:
        """텍스트를 벡터 임베딩으로 변환"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 문서들을 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            유사한 문서들의 리스트
        """
        # 쿼리 임베딩 생성
        query_embedding = self.create_embedding(query)
        
        # PostgreSQL에서 코사인 유사도 검색
        search_query = """
        SELECT 
            id,
            content,
            metadata,
            1 - (embedding <=> %s) as similarity
        FROM drug_embeddings 
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        
        self.cursor.execute(search_query, (query_embedding, query_embedding, top_k))
        results = self.cursor.fetchall()
        
        # 결과를 딕셔너리 형태로 변환
        documents = []
        for row in results:
            doc = {
                'id': row[0],
                'content': row[1],
                'metadata': row[2],
                'similarity': float(row[3])
            }
            documents.append(doc)
            
        return documents
    
    def search_by_product_name(self, product_name: str) -> List[Dict[str, Any]]:
        """
        제품명으로 정확한 검색
        
        Args:
            product_name: 검색할 제품명
            
        Returns:
            매칭되는 문서들
        """
        search_query = """
        SELECT 
            id,
            content,
            metadata,
            1.0 as similarity
        FROM drug_embeddings 
        WHERE metadata->>'제품명' ILIKE %s
        ORDER BY similarity DESC;
        """
        
        self.cursor.execute(search_query, (f'%{product_name}%',))
        results = self.cursor.fetchall()
        
        documents = []
        for row in results:
            doc = {
                'id': row[0],
                'content': row[1],
                'metadata': row[2],
                'similarity': float(row[3])
            }
            documents.append(doc)
            
        return documents
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (벡터 검색 + 키워드 검색)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            
        Returns:
            검색 결과 문서들
        """
        # 1. 벡터 유사도 검색
        vector_results = self.search_similar_documents(query, top_k)
        
        # 2. 키워드 검색 (제품명, 효능, 사용법 등에서 검색)
        keyword_query = """
        SELECT 
            id,
            content,
            metadata,
            CASE 
                WHEN content ILIKE %s THEN 0.8
                WHEN content ILIKE %s THEN 0.6
                ELSE 0.4
            END as similarity
        FROM drug_embeddings 
        WHERE content ILIKE %s OR content ILIKE %s
        ORDER BY similarity DESC
        LIMIT %s;
        """
        
        search_term = f'%{query}%'
        self.cursor.execute(keyword_query, (search_term, search_term, search_term, search_term, top_k))
        keyword_results = self.cursor.fetchall()
        
        # 키워드 검색 결과를 딕셔너리로 변환
        keyword_docs = []
        for row in keyword_results:
            doc = {
                'id': row[0],
                'content': row[1],
                'metadata': row[2],
                'similarity': float(row[3])
            }
            keyword_docs.append(doc)
        
        # 결과 합치기 및 중복 제거
        all_results = vector_results + keyword_docs
        seen_ids = set()
        unique_results = []
        
        for doc in all_results:
            if doc['id'] not in seen_ids:
                unique_results.append(doc)
                seen_ids.add(doc['id'])
        
        # 유사도 순으로 정렬
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return unique_results[:top_k]
    
    def get_context_for_llm(self, query: str, top_k: int = 3) -> str:
        """
        LLM을 위한 컨텍스트 생성
        
        Args:
            query: 검색 쿼리
            top_k: 사용할 문서 수
            
        Returns:
            LLM이 사용할 컨텍스트 문자열
        """
        results = self.hybrid_search(query, top_k)
        
        if not results:
            return "관련 정보를 찾을 수 없습니다."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            product_name = doc['metadata'].get('제품명', 'Unknown')
            content = doc['content']
            similarity = doc['similarity']
            
            context_parts.append(f"""
문서 {i} (제품명: {product_name}, 유사도: {similarity:.3f}):
{content}
""")
        
        return "\n".join(context_parts)
    
    def close(self):
        """연결 종료"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def test_retriever():
    """Retriever 테스트 함수"""
    retriever = VectorRetriever()
    
    # 테스트 쿼리
    test_queries = [
        "두통에 좋은 약",
        "소화불량 치료제",
        "감기약",
        "알레르기 약"
    ]
    
    for query in test_queries:
        print(f"\n=== 쿼리: {query} ===")
        results = retriever.hybrid_search(query, top_k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n결과 {i}:")
            print(f"제품명: {doc['metadata'].get('제품명', 'Unknown')}")
            print(f"유사도: {doc['similarity']:.3f}")
            print(f"내용: {doc['content'][:200]}...")
    
    retriever.close()


if __name__ == "__main__":
    test_retriever()
