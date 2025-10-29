# MINIPROJ3/app/screen/utils.py
import streamlit as st
from dotenv import load_dotenv
from graph_drug_rag import (
    get_compiled_graph,
    warm_up_pipeline,
)


def init_page():
    load_dotenv()
    st.set_page_config(
        page_title="의약품 AI 약사 챗봇",
        page_icon="💊",
        layout="wide",
    )


@st.cache_resource(show_spinner=False)
def _get_runner():
    """
    LangGraph를 한 번만 컴파일해 재사용할 실행 함수를 반환합니다.
    """
    warm_up_pipeline()
    graph = get_compiled_graph()

    def _run(question: str, collection_name: str = "drug_info", k: int = 4):
        initial_state = {"question": question, "collection_name": collection_name, "k": k}
        final_state = graph.invoke(initial_state)
        return {
            "question": final_state["question"],
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "in_domain": final_state.get("in_domain", False),
        }

    return _run


def init_display():
    """
    Streamlit app.py에서 호출되는 provider 규약:
      provider(prompt: str) -> generator[str]
    """
    rag_runner = _get_runner()

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
