# MINIPROJ3/app/screen/history.py
import html
import streamlit as st
from streamlit.components.v1 import html as html_component
from .constant import ROLE_TYPE


def _ensure():
    if "history" not in st.session_state:
        st.session_state.history = []


def ensure_initial_greeting(message: str):
    _ensure()
    if not st.session_state.history:
        st.session_state.history.append(
            {"role": ROLE_TYPE.assistant.value, "content": message}
        )


def clear_history():
    st.session_state.history = []


def add_history(role: ROLE_TYPE, content: str):
    _ensure()
    st.session_state.history.append({"role": role.value, "content": content})


def _build_chat_box_html(height: str, extra_html: str = "") -> str:
    """히스토리 + (옵션)추가 말풍선(extra_html)을 포함한 채팅박스 HTML 생성"""
    _ensure()

    style = f"""
    <style>
    .chat-box {{
        height: {height};
        overflow-y: auto;
        padding: 8px 12px;
        border: 1px solid #eee;
        border-radius: 12px;
        background: #f8fafc;
        scroll-behavior: smooth;
    }}
    .msg {{
        display: flex; gap: 10px;
        margin: 10px 0;
        padding: 10px 12px;
        border-radius: 10px;
        background: #fff;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.03);
    }}
    .msg.user::before {{ content: "🧑‍💬"; }}
    .msg.assistant::before {{ content: "🧪"; }}
    .msg .content {{ white-space: pre-wrap; line-height: 1.5; }}
    </style>
    """

    items = []
    for m in st.session_state.history:
        role = "user" if m["role"] == "user" else "assistant"
        content = html.escape(m["content"] or "").replace("\n", "<br>")
        items.append(f'<div class="msg {role}"><div class="content">{content}</div></div>')

    # 자동 스크롤 (렌더 직후 & 약간 지연 두 번 보장)
    autoscroll_js = """
    <script>
        function scrollBottom(){
            const box = window.parent.document.querySelector('.chat-box');
            if (box) box.scrollTop = box.scrollHeight;
        }
        window.requestAnimationFrame(scrollBottom);
        setTimeout(scrollBottom, 120);
    </script>
    """

    body = "".join(items) + (extra_html or "")
    return style + f'<div class="chat-box">{body}</div>' + autoscroll_js


def render_chat_box(height: str = "60vh", typing_html: str | None = None):
    """
    채팅박스를 하나의 placeholder에 렌더링한다.
    - typing_html이 주어지면 히스토리 아래에 임시 말풍선을 함께 표시
    - 반환값: update 함수 (typing/최종 답변 후 재렌더링용)
    """
    placeholder = st.empty()
    html_block = _build_chat_box_html(height, extra_html=(typing_html or ""))
    placeholder.markdown(html_block, unsafe_allow_html=True)

    def update(new_typing_html: str | None = None):
        placeholder.markdown(
            _build_chat_box_html(height, extra_html=(new_typing_html or "")),
            unsafe_allow_html=True,
        )

    return update


def render_scroll_to_bottom_button(label: str = "⬇️ 최근 메시지 보기"):
    """
    채팅박스(.chat-box) 맨 아래로 스크롤시키는 작은 버튼을 렌더링.
    components.html 로 JS 실행.
    """
    btn_html = f"""
    <div style="text-align:center; margin-top:7px;">
      <button id="scroll_bottom_btn"
              style="padding:6px 12px; border-radius:10px; border:1px solid #e5e7eb;
                     background:#fff; cursor:pointer;">
        {label}
      </button>
    </div>
    <script>
      const btn = window.document.getElementById('scroll_bottom_btn');
      if (btn) {{
        btn.onclick = function() {{
          const box = window.parent.document.querySelector('.chat-box');
          if (box) {{
            box.scrollTo({{ top: box.scrollHeight, behavior: 'smooth' }});
          }}
        }}
      }}
    </script>
    """
    html_component(btn_html, height=60, scrolling=False)
