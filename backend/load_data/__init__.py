"""
backend.load_data 패키지 - JSON 파일을 VectorDB로 로딩
"""
from .config import connect_db, get_embedding_model, create_embedding, load_json_documents, get_text_splitter
from .main_loader import insert_data

__all__ = [
    'connect_db', 
    'get_embedding_model', 
    'create_embedding', 
    'load_json_documents',
    'get_text_splitter',
    'insert_data'
]
