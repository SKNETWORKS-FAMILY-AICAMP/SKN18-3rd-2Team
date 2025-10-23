'''
LLM 및 임베딩 모델을 정의하는 코드
OpenAI, HuggingFace, Groq, Ollama 모델 지원
'''

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer


# =============================================================================
# OpenAI 모델들
# =============================================================================

# def get_openai_llm():
#     """OpenAI LLM 모델"""
#     return ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.1,
#         max_tokens=1000
#     )

# def get_openai_embeddings():
#     """OpenAI 임베딩 모델"""
#     return OpenAIEmbeddings(
#         model="text-embedding-3-small"
#     )


# =============================================================================
# Ollama 모델들 (현재 사용 중)
# =============================================================================

def get_ollama_llm():
    """Ollama LLM 모델 (현재 사용)"""
    return ChatOllama(
        model="gemma3:1b",
        temperature=0.1,
        num_ctx=1024
    )

def get_ollama_embeddings():
    """Ollama 임베딩 모델 (HuggingFace 사용)"""
    return SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')

# =============================================================================
# HuggingFace 모델들 (현재 사용 중)
# =============================================================================

def get_huggingface_embeddings():
    """HuggingFace 임베딩 모델 (현재 사용)"""
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

# =============================================================================
# 현재 사용 중인 모델들
# =============================================================================

# LLM: Ollama의 gemma3:1b 사용
llm = get_ollama_llm()

# 임베딩: HuggingFace의 ko-sroberta-multitask 사용
embeddings_model = get_huggingface_embeddings()

