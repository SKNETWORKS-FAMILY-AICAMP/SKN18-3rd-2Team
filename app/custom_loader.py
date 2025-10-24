import os
from typing import Iterable, List

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


class DrugCSVLoader(BaseLoader):

    DEFAULT_CONTENT_COLUMNS: List[str] = [
        "제품명",
        "효능",
        "사용법",
        "사용 전 주의",
        "사용상 주의사항",
        "약/음식 주의",
        "이상반응",
        "보관법",
    ]
    DEFAULT_METADATA_COLUMNS: List[str] = ["제품명"]

    def __init__(
        self,
        file_path: str,
        *,
        content_columns: Iterable[str] | None = None,
        metadata_columns: Iterable[str] | None = None,
        sep: str = ",",
        encoding: str = "utf-8",
        na_fill: str = "",
        dataframe: pd.DataFrame | None = None,
        read_kwargs: dict | None = None,
    ) -> None:
        # csv 파일경로
        self.file_path = file_path
        # page_content를 만들때 사용할 컬럼목록
        self.content_columns = list(content_columns or self.DEFAULT_CONTENT_COLUMNS)
        # metadata에 포함할 컬럼목록
        self.metadata_columns = list(metadata_columns or self.DEFAULT_METADATA_COLUMNS)
        # csv 구분자
        self.sep = sep
        # 파일 인코딩
        self.encoding = encoding
        # 결측치를 어떻게 처리할지
        self.na_fill = na_fill
        # Dataframe
        self.dataframe = dataframe
        # pd.read_csv에 그대로 전달할 추가 인자 딕셔너리
        self.read_kwargs = read_kwargs or {}

    def load(self) -> List[Document]:
        """
            csv를 읽거나 전달된 dataframe 읽고 결측치를 처리하고 
            "컬럼명": "값" 문자열을 만들어서 page_content로
            metadata컬럼은 metadata로 담아 Document 객체로 리턴
        """
        df = (
            self.dataframe.copy()
            if self.dataframe is not None
            else pd.read_csv(
                self.file_path,
                sep=self.sep,
                encoding=self.encoding,
                **self.read_kwargs,
            )
        )
        df = df.fillna(self.na_fill)

        # Document 객체를 담을 리스트
        documents = [] 
        for idx, row in df.iterrows():
            content_parts = []
            used_fields = []
            for col in self.content_columns:
                if col not in df.columns:
                    continue
                value = str(row[col]).strip()
                if not value:
                    continue
                content_parts.append(f"{col}: {value}")
                used_fields.append(col)

            if not content_parts:
                continue

            metadata = {
                col: str(row[col])
                for col in self.metadata_columns
                if col in df.columns
            }
            metadata["row_index"] = idx
            metadata["source"] = os.path.basename(self.file_path)
            metadata["source_fields"] = used_fields
            if "제품명" in metadata and "product_name" not in metadata:
                metadata["product_name"] = metadata["제품명"]

            documents.append(
                Document(page_content=" | ".join(content_parts), metadata=metadata)
            )

        return documents
