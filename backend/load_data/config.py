"""
데이터베이스 및 설정 관리 - 약품 정보 JSON용
"""
import psycopg2  # PostgreSQL 데이터베이스 연결용 파이썬 드라이버
import json
from pgvector.psycopg2 import register_vector
# sentence_transformers는 문장 임베딩 생성용 라이브러리 (HuggingFace 기반, 다양한 프리트레인 모델 지원)
from sentence_transformers import SentenceTransformer
import os
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# DB 설정
DB_CONFIG = {
    'host': 'localhost',
    'port': 55432,
    'database': 'vectordb', 
    'user': 'admin',
    'password': 'admin123'
}

def connect_db():
    """DB 연결"""
    conn = psycopg2.connect(**DB_CONFIG)
    # pgvector 등록 (확장이 설치되지 않은 경우 건너뛰기)
    try:
        register_vector(conn)
    except Exception as e:
        print(f"pgvector 확장 등록 실패 (무시): {e}")
        # 확장이 없어도 연결은 유지
    return conn

def load_json_documents(json_path):
    """JSON 문서 로드 (간단한 방식)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        # 제품명을 제외한 모든 필드를 텍스트로 결합
        content_parts = []
        for key, value in item.items():
            if key != '제품명':
                content_parts.append(f"{key}: {value}")
        
        content = "\n".join(content_parts)
        
        doc = Document(
            page_content=content,
            metadata={"제품명": item.get("제품명", "Unknown")}
        )
        documents.append(doc)
    
    return documents

def get_text_splitter():
    """텍스트 스플리터 (JSON 대신 텍스트 기반)"""
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,  # 오버랩 없음
        length_function=len
    )

def get_embedding_model(platform='huggingface', return_platform_name=False):
    """
    플랫폼별 임베딩 모델 로더

    Args:
        platform (str): 'huggingface' 또는 'openai' 등 사용 가능 (기본값: 'huggingface')
        return_platform_name (bool): True일 경우 (model, platform_name) 반환

    Returns:
        model 또는 (model, platform_name)
    """
    if platform == 'huggingface':
        # 최고 성능의 다국어 모델 (한국어 포함, 768차원)
        model = SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')
        platform_name = 'HuggingFace'
    elif platform == 'openai':
        # 예시: OpenAI 임베딩 (1536차원), 실제로 쓰려면 openai 라이브러리 및 API 키 필요
        from langchain.embeddings import OpenAIEmbeddings
        model = OpenAIEmbeddings(model="text-embedding-small-3")
        platform_name = 'OpenAI'
    else:
        raise ValueError(f"지원하지 않는 플랫폼: {platform}")

    if return_platform_name:
        return platform_name

def get_embedding_model():
    """임베딩 모델 선택 (한국어 최적화 모델 사용)"""
    # LangChain 호환 HuggingFaceEmbeddings 사용
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-multitask',
        model_kwargs={'device': 'cpu'}
    )

def create_embedding(text, model):
    """임베딩 생성 (OpenAI 또는 HuggingFace)"""
    try:
        # OpenAI 임베딩인 경우
        if hasattr(model, 'embed_query'):
            embedding = model.embed_query(text)
            return embedding
        
        # HuggingFace 임베딩인 경우
        else:
            embedding = model.encode(text)
            return embedding.tolist()
    
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        return None
