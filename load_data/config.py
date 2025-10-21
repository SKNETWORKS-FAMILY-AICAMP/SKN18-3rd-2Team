"""
데이터베이스 및 설정 관리 - 약품 정보 JSON용
"""
import psycopg2
import json
from pgvector.psycopg2 import register_vector
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
    register_vector(conn)
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

def get_embedding_model():
    """HuggingFace 최고 성능 다국어 임베딩 모델 (768차원)"""
    # 최고 성능의 다국어 모델 (한국어 포함)
    return SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')

def create_embedding(text, model):
    """HuggingFace 임베딩 생성"""
    # SentenceTransformer 임베딩 생성
    return model.encode(text).tolist()
