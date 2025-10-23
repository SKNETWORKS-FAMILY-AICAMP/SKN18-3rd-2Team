import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from custom_pgvector import CustomPGVector
from db_utils import make_conn_str
from custom_loader import DrugCSVLoader
from embedding_utils import get_embedding_model


@dataclass
class IngestConfig:
    """데이터를 vectordb에 적재하기 위해서 필요한 변수들"""
    csv_path: str
    table_name: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    reset: bool


class CustomVectorIngestor:
    """DrugCSVLoader 데이터를 CustomPGVector 테이블로 저장한다."""
    def __init__(self, config: IngestConfig) -> None:
        self.config = config
        self.connection_str = make_conn_str()
        self.embedding_model = None
        self.vectorstore = None
        self.splitter = None

    def run(self) -> dict:
        """LangChain Runnable 파이프라인으로 전체 적재 과정을 실행한다."""
        pipeline = (
            RunnableLambda(lambda _: self._prepare_storage())
            | RunnableLambda(lambda _: self._load_documents())
            | RunnableLambda(self._split_documents)
            | RunnableLambda(self._persist_documents)
        )
        return pipeline.invoke(None)

    def _prepare_storage(self) -> None:
        """init.sql로 생성된 테이블을 사용할 준비를 하고 VectorStore를 초기화한다."""
        if self.config.reset:
            self._truncate_table()
        self.embedding_model = get_embedding_model()
        self.vectorstore = CustomPGVector(
            conn_str=self.connection_str,
            embedding_fn=self.embedding_model,
            table=self.config.table_name,
        )

    def _truncate_table(self) -> None:
        """재적재를 위해 테이블 내용을 비운다."""
        with psycopg2.connect(self.connection_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("TRUNCATE TABLE {}").format(
                        sql.Identifier(self.config.table_name)
                    )
                )
            conn.commit()

    def _load_documents(self) -> List[Document]:
        """CSV 파일을 Document 리스트로 변환한다."""
        loader = DrugCSVLoader(self.config.csv_path)
        documents = loader.load()
        if not documents:
            raise RuntimeError("적재할 Document가 없습니다. CSV 내용을 확인하세요.")
        return documents

    def batched(self, items: Sequence[Document], batch_size: int) -> Iterable[Sequence[Document]]:
        """Sequence를 batch_size 단위로 분할한다."""
        total = len(items)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield items[start:end]

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Document를 RecursiveCharacterTextSplitter로 청크 단위로 나눈다."""
        chunk_docs: List[Document] = []
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_clean = chunk_text.strip()
                if not chunk_clean:
                    continue
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = chunk_idx
                chunk_docs.append(Document(page_content=chunk_clean, metadata=metadata))
        return chunk_docs

    def _persist_documents(self, documents: List[Document]) -> dict:
        """청크 Document를 CustomPGVector 테이블에 저장한다."""
        if self.vectorstore is None:
            raise RuntimeError("VectorStore가 초기화되지 않았습니다.")

        total_chunks = len(documents)
        total_batches = math.ceil(total_chunks / self.config.batch_size) if total_chunks else 0
        products = {
            doc.metadata.get("product_name") or doc.metadata.get("제품명")
            for doc in documents
        }
        products.discard(None)

        if not total_chunks:
            return {"chunks": 0, "products": len(products)}

        with tqdm(total=total_batches, desc="Uploading chunks") as progress:
            for batch in self.batched(documents, self.config.batch_size):
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                self.vectorstore.add_texts(texts, metadatas=metadatas)
                progress.update(1)

        return {"chunks": total_chunks, "products": len(products)}


def parse_args() -> IngestConfig:
    parser = argparse.ArgumentParser(description="CSV 문서를 CustomPGVector 테이블에 적재합니다.")
    parser.add_argument(
        "--csv",
        default="../data/drug_info_preprocessed.csv",
        help="적재할 CSV 파일 경로",
    )
    parser.add_argument(
        "--table",
        default="drug_info",
        help="pgvector 테이블명",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="텍스트 청크 크기",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=300,
        help="청크 간 겹치는 토큰 수",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="DB 적재 시 배치 크기",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터를 제거하고 다시 적재",
    )
    args = parser.parse_args()
    return IngestConfig(
        csv_path=args.csv,
        table_name=args.table,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        reset=args.reset,
    )


def main() -> None:
    load_dotenv()
    config = parse_args()
    ingestor = CustomVectorIngestor(config)
    stats = ingestor.run()
    print(
        f"✅ Done. Inserted {stats['chunks']} chunks "
        f"from {stats['products']} products into table '{config.table_name}'."
    )


if __name__ == "__main__":
    main()

