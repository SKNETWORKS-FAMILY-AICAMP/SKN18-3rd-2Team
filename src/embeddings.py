"""
텍스트 임베딩 생성 기능
Hugging Face 모델 사용 (로컬 실행)
E5 모델 최적화 포함
"""
import os
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, model_name: str = None):
        # 모델 이름 결정
        model_key = model_name or os.getenv('EMBEDDING_MODEL', 'e5_large_instruct')
        
        # 추천 모델 키워드인지 확인
        if model_key in RECOMMENDED_MODELS:
            self.model_name = RECOMMENDED_MODELS[model_key]
        else:
            self.model_name = model_key  # 직접 모델명이 입력된 경우
        
        # E5 모델인지 확인 (더 정확한 감지)
        self.is_e5_model = ('e5' in self.model_name.lower() or 
                           'multilingual-e5' in self.model_name.lower())
        
        print(f"🤖 임베딩 모델 로딩 중: {self.model_name}")
        if self.is_e5_model:
            print("📋 E5 모델 감지 - 특별한 프롬프트 형식 사용")
            print("   - 문서: 'passage: <text>'")
            print("   - 쿼리: 'query: <text>'")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ 모델 로딩 완료! 임베딩 차원: {embedding_dim}")
            
            # E5 모델의 경우 1024차원인지 확인
            if self.is_e5_model and embedding_dim != 1024:
                print(f"⚠️ 주의: E5 모델이지만 임베딩 차원이 {embedding_dim}입니다. 1024차원이 예상됩니다.")
                
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        return self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """텍스트를 벡터로 변환"""
        try:
            # 텍스트 전처리 (너무 긴 텍스트 자르기)
            max_length = 512  # 대부분의 모델이 지원하는 최대 길이
            if len(text) > max_length * 4:  # 대략적인 토큰 수 추정
                text = text[:max_length * 4]
            
            # E5 모델의 경우 특별한 프롬프트 형식 사용
            if self.is_e5_model:
                if is_query:
                    # 질문/검색 쿼리용 프롬프트
                    formatted_text = f"query: {text.strip()}"
                else:
                    # 문서/패시지용 프롬프트
                    formatted_text = f"passage: {text.strip()}"
            else:
                formatted_text = text.strip()
            
            # 임베딩 생성
            embedding = self.model.encode(
                formatted_text, 
                convert_to_tensor=False,
                normalize_embeddings=True  # 정규화로 성능 향상
            )
            
            # numpy array를 list로 변환
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            else:
                return embedding
                
        except Exception as e:
            print(f"❌ 임베딩 생성 실패: {e}")
            print(f"   텍스트 길이: {len(text)}")
            print(f"   텍스트 미리보기: {text[:100]}...")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], is_query: bool = False, batch_size: int = 4) -> List[List[float]]:
        """여러 텍스트를 한번에 벡터로 변환"""
        try:
            # 텍스트 전처리
            max_length = 512 * 4
            processed_texts = []
            
            for text in texts:
                if len(text) > max_length:
                    text = text[:max_length]
                processed_texts.append(text.strip())
            
            # E5 모델의 경우 특별한 프롬프트 형식 사용
            if self.is_e5_model:
                if is_query:
                    formatted_texts = [f"query: {text}" for text in processed_texts]
                else:
                    formatted_texts = [f"passage: {text}" for text in processed_texts]
            else:
                formatted_texts = processed_texts
            
            # 배치 처리로 메모리 효율성 향상
            all_embeddings = []
            for i in range(0, len(formatted_texts), batch_size):
                batch = formatted_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    batch_size=min(batch_size, len(batch))
                )
                
                # numpy array를 list로 변환
                if isinstance(batch_embeddings, np.ndarray):
                    all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
                else:
                    all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            print(f"❌ 배치 임베딩 생성 실패: {e}")
            print(f"   배치 크기: {len(texts)}")
            raise

# 추천 모델들 (속도 vs 성능)
RECOMMENDED_MODELS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",  # 384차원, 매우 빠름
    "multilingual_small": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384차원, 빠름
    "multilingual_medium": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 768차원, 균형
    "korean_optimized": "jhgan/ko-sroberta-multitask",  # 768차원, 한국어 특화
    "e5_base": "intfloat/multilingual-e5-base",  # 768차원, 빠른 E5
    "e5_large": "intfloat/multilingual-e5-large",  # 1024차원, 성능 좋음
    "e5_large_instruct": "intfloat/multilingual-e5-large-instruct",  # 1024차원, 최고 성능 (느림)
}