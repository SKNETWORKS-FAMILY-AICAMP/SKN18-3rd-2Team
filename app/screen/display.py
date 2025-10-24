import streamlit as st
from collections.abc import Iterable

def _stream_to_placeholder(generator: Iterable) -> str:
    placeholder = st.empty()
    chunks = []
    for part in generator:
        chunks.append(str(part))
        placeholder.markdown("".join(chunks))
    return "".join(chunks)

def print_message(role: str, content_or_gen):
    """
    role: "user" | "assistant"
    content_or_gen: str 또는 generator(스트리밍)
    반환값: 최종 출력 문자열(assistant일 때 히스토리 저장용)
    """
    with st.chat_message(role):
        if isinstance(content_or_gen, str):
            st.markdown(content_or_gen)
            return content_or_gen
        # generator/iterator면 스트리밍 표시
        return _stream_to_placeholder(content_or_gen)
