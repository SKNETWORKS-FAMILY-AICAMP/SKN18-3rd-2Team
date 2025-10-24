# MINIPROJ3/app/app.py
import time
import streamlit as st

from screen.constant import ROLE_TYPE
from screen.history import add_history, clear_history, render_chat_box, render_scroll_to_bottom_button
from screen.input import get_prompt
from screen.utils import init_page, init_display
from screen.top10 import render_top10
from screen.pill_wallet import render_pill_wallet, render_pending_suggestions



def main():
    init_page()

    col_left, col_right = st.columns([1.0, 2.2], gap="large")

    with col_left:
        render_top10()
        st.markdown("")
        render_pill_wallet()

    with col_right:
        st.title("의약품 정보 제공 챗봇")
        st.caption("AI 약사에게 궁금한 점을 질문해보세요")

        _left, _sp, _btn = st.columns([10, 0.2, 1])
        with _btn:
            if st.button("🧹", help="대화 지우기", use_container_width=True):
                clear_history()
                st.rerun()

        CHAT_BOX_HEIGHT = "60vh"
        update_chat_box = render_chat_box(height=CHAT_BOX_HEIGHT)

        # ⬇️ 버튼(채팅박스 바로 아래)
        render_scroll_to_bottom_button("⬇️ 최근 메시지 보기")

        prompt = get_prompt()
        provider = init_display()

        if prompt:
            add_history(ROLE_TYPE.user, prompt)
            update_chat_box()  # 사용자의 질문까지 반영

            # 👨‍⚕️ 약사 이모지 blink 로딩 말풍선
            typing_html = """
            <div class="msg assistant"><div class="content">
              <span class="pharm" title="답변 생성 중">🧑‍⚕️</span>
              <span style='color:#666; margin-left:6px;'>답변 생성 중입니다</span>
            </div></div>
            <style>
              @keyframes pharmblink { 0%,60%{opacity:1;} 60.01%,100%{opacity:0.3;} }
              .pharm { display:inline-block; animation: pharmblink 1s infinite; font-size:1.2rem; }
            </style>
            """
            update_chat_box(typing_html)

            chunks = []
            for part in provider(prompt):
                chunks.append(str(part))
                time.sleep(0.02)
            final_answer = "".join(chunks)

            add_history(ROLE_TYPE.assistant, final_answer)
            update_chat_box()  # 로딩 말풍선 제거 + 최종 답변 반영

            process_user_message(prompt)
            st.rerun()

        render_pending_suggestions()

        st.markdown(
            """
            <div style='text-align:center; font-size:0.85rem; color:#777; margin-top:16px;'>
                ⚠️ <b>주의사항</b><br>
                본 서비스는 AI를 활용한 의약품 정보 제공을 목적으로 하며, 실제 전문의 또는 약사의 상담을 대체하지 않습니다.<br>
                제공되는 정보의 정확성과 최신성은 보장되지 않으며, 복약 및 치료 결정에 대한 책임은 사용자 본인에게 있습니다.<br>
                정확한 복약 상담은 반드시 <b>전문 약사</b>에게 문의하시기 바랍니다.
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
