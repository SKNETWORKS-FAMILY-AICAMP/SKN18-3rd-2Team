import os
from functools import lru_cache
from typing import Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1) # 함수 결과를 메모리에 저장해 두는 파이썬 표준 라이브러리
def _load_embeddings() -> Tuple[HuggingFaceEmbeddings, int]:
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL")
    normalize = os.getenv("LOCAL_EMBEDDING_NORMALIZE", "false").lower() == "true"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": normalize},
    )

    dim_env = os.getenv("LOCAL_EMBEDDING_DIM")
    if dim_env:
        dimension = int(dim_env)
    else:
        dimension = len(embeddings.embed_query("dimension probe"))
    return embeddings, dimension


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the cached embedding model instance."""
    return _load_embeddings()[0]


def get_embedding_dim() -> int:
    """Return the embedding dimension for the current model."""
    return _load_embeddings()[1]
