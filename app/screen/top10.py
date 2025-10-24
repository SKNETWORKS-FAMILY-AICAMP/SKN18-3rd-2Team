# MINIPROJ3/app/screen/top10.py
import streamlit as st

# ---- 목업 데이터 (원하면 주기적으로 바꿔도 됨) ----
# change: +n(상승), -n(하락), 0(변동없음)
MOCK_TOP10 = [
    {"rank": 1,  "name": "아세트아미노펜", "change": +2},
    {"rank": 2,  "name": "이부프로펜",     "change": -1},
    {"rank": 3,  "name": "훼스탈",         "change":  0},
    {"rank": 4,  "name": "베아제",         "change": +3},
    {"rank": 5,  "name": "로페라마이드",   "change": -2},
    {"rank": 6,  "name": "지르텍",         "change":  0},
    {"rank": 7,  "name": "겔포스",         "change": +1},
    {"rank": 8,  "name": "알마겔",         "change": -1},
    {"rank": 9,  "name": "판콜에이",       "change": +1},
    {"rank": 10, "name": "콜대원",         "change":  0},
]


def _arrow_html(change: int) -> str:
    """상승/하락/유지 아이콘 HTML"""
    if change > 0:
        return f'<span class="delta up">▲ {abs(change)}</span>'
    if change < 0:
        return f'<span class="delta down">▼ {abs(change)}</span>'
    return '<span class="delta same">–</span>'


def render_top10():
    # 스타일 주입
    st.markdown(
        """
        <style>
        .topbox {
            border: 1px solid #e6e6e6;
            border-radius: 14px;
            padding: 14px 16px;
            background: #fafafa;
        }
        .topbox h3 {
            margin: 0 0 10px 0;
            font-size: 1.05rem;
        }
        .toplist {list-style: none; margin: 0; padding: 0;}
        .topitem {
            display: flex; align-items: center; justify-content: space-between;
            padding: 6px 6px; border-radius: 10px;
        }
        .topitem:hover { background: #ffffff; }
        .rank {
            font-weight: 600; width: 20px; color: #7a7a7a; flex: 0 0 auto;
        }
        .drugname {
            flex: 1 1 auto; margin: 0 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .delta {
            font-weight: 600; font-size: 0.85rem;
            padding: 2px 6px; border-radius: 8px;
        }
        .delta.up   { color: #0c7a43; background: #e9f7ef; }
        .delta.down { color: #a1191b; background: #fdecea; }
        .delta.same { color: #6b7280; background: #f3f4f6; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 컨테이너
    st.markdown('<div class="topbox"><h3>자주 검색되는 약 Top 10</h3><ul class="toplist">', unsafe_allow_html=True)

    # 항목 렌더링
    for row in MOCK_TOP10:
        html = (
            f'<li class="topitem">'
            f'<span class="rank">{row["rank"]}</span>'
            f'<span class="drugname">{row["name"]}</span>'
            f'{_arrow_html(row["change"])}'
            f'</li>'
        )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('</ul></div>', unsafe_allow_html=True)
