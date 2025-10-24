"""
의약품 정보 RAG 시스템 패키지

주요 모듈:
- database: PostgreSQL + pgvector 데이터베이스 연동
- embeddings: E5 모델 임베딩 생성
- rag_system: 기본 RAG 시스템
- multi_agent_system: Multi-Agent RAG 시스템
"""

__version__ = "1.0.0"
__author__ = "Drug RAG Team"

# 주요 클래스들을 패키지 레벨에서 임포트 가능하게 함
try:
    from .database import VectorDB
    from .embeddings import EmbeddingGenerator
    from .rag_system import RAGSystem
    from .multi_agent_system import MultiAgentCoordinator
    
    __all__ = [
        'VectorDB',
        'EmbeddingGenerator', 
        'RAGSystem',
        'MultiAgentCoordinator'
    ]
except ImportError:
    # 의존성이 설치되지 않은 경우 무시
    pass