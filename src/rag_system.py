"""
RAG (Retrieval-Augmented Generation) 시스템
Hugging Face 모델 사용
"""
import os
from typing import List, Dict
from dotenv import load_dotenv
from .database import VectorDB
from .embeddings import EmbeddingGenerator, RECOMMENDED_MODELS

load_dotenv()

class RAGSystem:
    def __init__(self, embedding_model: str = None):
        # 임베딩 모델 설정
        model_key = os.getenv('EMBEDDING_MODEL', 'multilingual_small')
        if model_key in RECOMMENDED_MODELS:
            model_name = RECOMMENDED_MODELS[model_key]
        else:
            model_name = model_key  # 직접 모델명이 입력된 경우
        
        self.embedding_generator = EmbeddingGenerator(embedding_model or model_name)
        self.db = VectorDB()
        
        print(f"🚀 RAG 시스템 초기화 완료!")
        print(f"📊 임베딩 차원: {self.embedding_generator.get_embedding_dimension()}")
    
    def add_document(self, content: str, product_name: str) -> int:
        """제품 문서를 시스템에 추가"""
        print(f"📄 제품 '{product_name}' 문서 추가 중: {content[:50]}...")
        
        # 문서용 임베딩 생성 (is_query=False)
        embedding = self.embedding_generator.generate_embedding(content, is_query=False)
        
        # 데이터베이스에 저장
        doc_id = self.db.insert_document(content, embedding, product_name)
        
        print(f"✅ 문서 추가 완료 (ID: {doc_id})")
        return doc_id
    
    def query(self, question: str, product_name: str = None, top_k: int = 3) -> str:
        """질문에 대한 답변 생성"""
        if product_name:
            print(f"🔍 '{product_name}' 제품 관련 질문: {question}")
        else:
            print(f"🔍 전체 제품 질문: {question}")
        
        # 질문을 임베딩으로 변환 (is_query=True)
        query_embedding = self.embedding_generator.generate_embedding(question, is_query=True)
        
        # 유사한 문서 검색
        similar_docs = self.db.similarity_search(query_embedding, limit=top_k, product_name=product_name)
        
        if not similar_docs:
            if product_name:
                return f"죄송합니다. '{product_name}' 제품과 관련된 정보를 찾을 수 없습니다."
            else:
                return "죄송합니다. 관련된 정보를 찾을 수 없습니다."
        
        # 검색된 문서들을 컨텍스트로 구성
        context = "\n\n".join([doc[1] for doc in similar_docs])  # doc[1]은 content
        
        # 간단한 규칙 기반 답변 생성
        answer = self._generate_simple_answer(question, context, similar_docs)
        print(f"💡 답변: {answer}")
        return answer
    
    def get_products(self) -> List[str]:
        """등록된 모든 제품 목록 조회"""
        return self.db.get_products()
    
    def _generate_simple_answer(self, question: str, context: str, similar_docs: List) -> str:
        """간단한 규칙 기반 답변 생성"""
        # 가장 유사한 문서의 거리 확인
        best_distance = similar_docs[0][3]  # distance
        best_product = similar_docs[0][2]   # product_name
        
        if best_distance > 0.8:  # 유사도가 너무 낮으면
            return "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다."
        
        # 컨텍스트를 요약해서 답변 생성
        answer_parts = []
        answer_parts.append(f"📋 '{best_product}' 제품 정보:")
        answer_parts.append("")
        
        # 가장 관련성 높은 문서 내용 포함
        best_content = similar_docs[0][1]  # content
        if len(best_content) > 300:
            best_content = best_content[:300] + "..."
        
        answer_parts.append(best_content)
        
        # 다른 관련 제품들도 표시
        other_products = set([doc[2] for doc in similar_docs[1:]])
        if other_products:
            answer_parts.append("")
            answer_parts.append(f"🔗 관련 제품: {', '.join(other_products)}")
        
        answer_parts.append("")
        answer_parts.append(f"📊 유사도: {1-best_distance:.2f} | 참조 문서: {len(similar_docs)}개")
        
        return "\n".join(answer_parts)
    
    def close(self):
        """시스템 종료"""
        self.db.close()