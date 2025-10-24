# MINIPROJ3/app/screen/utils.py
import streamlit as st
from dotenv import load_dotenv
from graph_drug_rag import run_once  # RAG 핵심 실행 함수 가져오기


def init_page():
    load_dotenv()
    st.set_page_config(
        page_title="의약품 AI 약사 챗봇",
        page_icon="💊",
        layout="wide",
    )


@st.cache_resource(show_spinner=False)
def _get_chain():
    """
    그래프 기반 RAG 파이프라인을 초기화합니다.
    (graph_drug_rag의 run_once를 호출할 준비)
    """
    return run_once


def init_display():
    """
    Streamlit app.py에서 호출되는 provider 규약:
      provider(prompt: str) -> generator[str]
    """
    rag_runner = _get_chain()

    def _provider(prompt: str):
        """
        run_once(question, collection_name="drug_info")의 결과를
        Streamlit 스트리밍 형식으로 전달
        """
        try:
            result = rag_runner(prompt, collection_name="drug_info")
            answer = result.get("answer", "")
            yield answer
        except Exception as e:
            yield f"❗ 오류 발생: {e}"

    return _provider
