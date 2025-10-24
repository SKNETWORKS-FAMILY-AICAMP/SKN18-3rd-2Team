import os
import time
import json
import pandas as pd
import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta

# =========================================================
# 🔧 데이터 소스 설정 (있으면 DB→CSV→목업 순으로 로드)
DB_TABLE = os.getenv("DRUG_TABLE", "drug_info")
DB_NAME_COL = os.getenv("DRUG_NAME_COL", "product_name")
DB_INGR_COL = os.getenv("DRUG_INGR_COL", "ingredient")
DB_SYNONYM_COLS = json.loads(os.getenv("DRUG_SYNONYM_COLS", '["brand_name","generic_name","korean_name","english_name"]'))

CSV_PATH = os.getenv("DRUG_CSV", "./data/drug_info_preprocessed.csv")
CSV_NAME_COL = os.getenv("CSV_NAME_COL", "제품명")
CSV_INGR_COL = os.getenv("CSV_INGR_COL", "성분명")
CSV_SYNONYM_COLS = json.loads(os.getenv("CSV_SYNONYM_COLS", '["제품명영문","브랜드명","일반명"]'))
# =========================================================

POS_TRIGGERS = ["복용 중", "먹고 있어", "먹고있어", "처방받", "처방 받", "지어줬", "지어 줬", "먹는 중", "먹습니다", "먹어요"]
NEG_TRIGGERS = ["중단", "끊었", "안 먹", "안먹", "먹지 않", "안먹을", "그만 먹"]

RECENT_DAYS = 30
MENTION_THRESHOLD = 2


def _ensure_states():
    if "pill_wallet" not in st.session_state:
        st.session_state.pill_wallet = []  # [{name, ingredient, added_at}]
    if "pill_candidates" not in st.session_state:
        st.session_state.pill_candidates = defaultdict(lambda: {
            "count": 0,
            "last_mentions": [],
            "ingredient": "",
            "display": "",
        })
    if "pill_pending_suggestions" not in st.session_state:
        st.session_state.pill_pending_suggestions = []
    if "wallet_manual_name" not in st.session_state:
        st.session_state.wallet_manual_name = ""
    if "wallet_manual_ingr" not in st.session_state:
        st.session_state.wallet_manual_ingr = ""


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _split_multi(value: str):
    if value is None:
        return []
    txt = str(value)
    for sep in [";", "/", "|"]:
        txt = txt.replace(sep, ",")
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    return parts


@st.cache_resource(show_spinner=False)
def load_drug_synonyms() -> dict:
    """
    alias(lower) -> (display_name, main_ingredient)
    1) Postgres 테이블 → 2) CSV → 3) 목업
    """
    # 1) DB
    try:
        from sqlalchemy import text
        from db_utils import make_conn_str
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401 (ensure LC deps ready)
        import psycopg2  # ensure driver available

        import sqlalchemy
        engine = sqlalchemy.create_engine(make_conn_str())
        with engine.begin() as conn:
            exists = conn.execute(text("SELECT to_regclass(:t) IS NOT NULL"), {"t": DB_TABLE}).scalar()
            if exists:
                cols = conn.execute(
                    text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
                    {"t": DB_TABLE},
                ).fetchall()
                colset = {c[0].lower() for c in cols}
                name_col = DB_NAME_COL if DB_NAME_COL.lower() in colset else None
                ingr_col = DB_INGR_COL if DB_INGR_COL.lower() in colset else None
                syn_cols = [c for c in DB_SYNONYM_COLS if c.lower() in colset]

                if name_col and ingr_col:
                    select_cols = [name_col, ingr_col] + syn_cols
                    rows = conn.execute(text(f'SELECT {", ".join(select_cols)} FROM {DB_TABLE}')).fetchall()
                    mapping = {}
                    for row in rows:
                        rec = dict(zip(select_cols, row))
                        display = str(rec[name_col]).strip()
                        main_ingr = ", ".join(_split_multi(rec.get(ingr_col, ""))) or str(rec.get(ingr_col, "")).strip()
                        aliases = set([display])
                        for ing in _split_multi(rec.get(ingr_col, "")):
                            aliases.add(ing)
                        for sc in syn_cols:
                            vals = rec.get(sc, "")
                            cand = _split_multi(vals) or [vals]
                            for a in cand:
                                if a:
                                    aliases.add(str(a).strip())
                        for a in aliases:
                            a_norm = _normalize(a)
                            if a_norm:
                                mapping[a_norm] = (display, main_ingr)
                    if mapping:
                        return mapping
    except Exception:
        pass

    # 2) CSV
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH, engine="python")
            cols = {c.lower(): c for c in df.columns}
            name_col = cols.get(CSV_NAME_COL.lower())
            ingr_col = cols.get(CSV_INGR_COL.lower())
            syn_cols = [cols[c.lower()] for c in CSV_SYNONYM_COLS if c.lower() in cols]
            if name_col and ingr_col:
                mapping = {}
                for _, rec in df.iterrows():
                    display = str(rec[name_col]).strip()
                    main_ingr = ", ".join(_split_multi(rec.get(ingr_col, ""))) or str(rec.get(ingr_col, "")).strip()
                    aliases = set([display])
                    for ing in _split_multi(rec.get(ingr_col, "")):
                        aliases.add(ing)
                    for sc in syn_cols:
                        vals = rec.get(sc, "")
                        cand = _split_multi(vals) or [vals]
                        for a in cand:
                            if a:
                                aliases.add(str(a).strip())
                    for a in aliases:
                        a_norm = _normalize(a)
                        if a_norm:
                            mapping[a_norm] = (display, main_ingr)
                if mapping:
                    return mapping
    except Exception:
        pass

    # 3) 목업
    return {
        "아세트아미노펜": ("타이레놀", "아세트아미노펜"),
        "타이레놀": ("타이레놀", "아세트아미노펜"),
        "파나돌": ("파나돌", "아세트아미노펜"),
        "이부프로펜": ("이부프로펜", "이부프로펜"),
        "애드빌": ("애드빌", "이부프로펜"),
        "훼스탈": ("훼스탈", "소화효소"),
        "베아제": ("베아제", "소화효소"),
        "겔포스": ("겔포스", "알긴산/제산제"),
        "알마겔": ("알마겔", "알루미늄/마그네슘 제산제"),
        "지르텍": ("지르텍", "세티리진"),
        "로페라마이드": ("로페라마이드", "로페라마이드"),
        "파몰에이": ("파몰에이", "복합감기약"),
        "콜대원": ("콜대원", "복합감기약"),
    }


def _already_in_wallet(display: str) -> bool:
    for item in st.session_state.pill_wallet:
        if item["name"] == display:
            return True
    return False


def process_user_message(user_text: str):
    """사용자 입력 → 약 엔터티 추출/의도 판단 → 후보 카운트 → 임계치 도달 시 제안"""
    _ensure_states()
    now = time.time()
    drugs = _extract_drugs(user_text)
    if not drugs:
        return
    positive = any(k in _normalize(user_text) for k in [k.lower() for k in POS_TRIGGERS])
    negative = any(k in _normalize(user_text) for k in [k.lower() for k in NEG_TRIGGERS])

    if negative:
        for display, _ in drugs:
            if display in st.session_state.pill_candidates:
                del st.session_state.pill_candidates[display]
        return

    if positive:
        for display, ingr in drugs:
            if _already_in_wallet(display):
                continue
            cand = st.session_state.pill_candidates[display]
            cand["display"] = display
            cand["ingredient"] = ingr
            cand["last_mentions"].append(now)
            cutoff = datetime.now() - timedelta(days=RECENT_DAYS)
            cand["last_mentions"] = [ts for ts in cand["last_mentions"] if datetime.fromtimestamp(ts) >= cutoff]
            cand["count"] = len(cand["last_mentions"])
            if cand["count"] >= MENTION_THRESHOLD and display not in st.session_state.pill_pending_suggestions:
                st.session_state.pill_pending_suggestions.append(display)


def _extract_drugs(text: str):
    """사전 기반 간단 매칭"""
    lex = load_drug_synonyms()
    t = _normalize(text)
    found = []
    for alias in sorted(lex.keys(), key=len, reverse=True):
        if alias and alias in t:
            found.append(lex[alias])
    uniq, seen = [], set()
    for d, i in found:
        if (d, i) not in seen:
            uniq.append((d, i))
            seen.add((d, i))
    return uniq


def _add_to_wallet(display: str, ingredient: str):
    if _already_in_wallet(display):
        st.toast("이미 약 지갑에 있습니다.", icon="ℹ️")
        return
    st.session_state.pill_wallet.append({
        "name": display,
        "ingredient": ingredient,
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    # 후보/제안 목록에서도 제거
    if display in st.session_state.pill_candidates:
        del st.session_state.pill_candidates[display]
    if display in st.session_state.pill_pending_suggestions:
        st.session_state.pill_pending_suggestions.remove(display)


def _reject_suggestion(display: str):
    if display in st.session_state.pill_pending_suggestions:
        st.session_state.pill_pending_suggestions.remove(display)


def render_pending_suggestions():
    """이번 턴 제안 UI: 약 지갑에 추가하시겠어요?"""
    _ensure_states()
    if not st.session_state.pill_pending_suggestions:
        return
    st.markdown("---")
    for display in list(st.session_state.pill_pending_suggestions):
        cand = st.session_state.pill_candidates.get(display)
        if not cand:
            continue
        with st.container():
            st.info(f"💊 **{cand['display']}** (*{cand['ingredient']}*) 을(를) ‘내 약 지갑’에 추가할까요?", icon="💡")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("추가", key=f"add_{display}"):
                    _add_to_wallet(cand["display"], cand["ingredient"])
                    st.success(f"‘{display}’이(가) 약 지갑에 추가되었습니다.")
            with c2:
                if st.button("아니요", key=f"reject_{display}"):
                    _reject_suggestion(display)
                    st.toast("제안을 숨겼습니다.", icon="❎")


def render_pill_wallet():
    _ensure_states()

    # 🔲 박스 컨테이너 시작(여기 안에 모든 UI 배치)
    with st.container(border=True):
        st.markdown("### 💊 내 약 지갑")

        # ── 1) 직접 추가 (박스 내부)
        with st.expander("➕ 직접 추가", expanded=False):
            st.caption("제품명(또는 복용 중인 약 이름)과 성분명을 입력해주세요.")

            # 세션 초기값 보장
            if "wallet_manual_name" not in st.session_state:
                st.session_state.wallet_manual_name = ""
            if "wallet_manual_ingr" not in st.session_state:
                st.session_state.wallet_manual_ingr = ""
            if "__pill_wallet_msg" not in st.session_state:
                st.session_state.__pill_wallet_msg = ""
            if "__pill_wallet_msg_type" not in st.session_state:
                st.session_state.__pill_wallet_msg_type = "info"

            c1, c2 = st.columns([2, 3])
            with c1:
                st.text_input("제품명", key="wallet_manual_name", placeholder="예) 타이레놀")
            with c2:
                st.text_input("성분명", key="wallet_manual_ingr", placeholder="예) 아세트아미노펜")

            # 콜백들
            def _on_add():
                n = (st.session_state.wallet_manual_name or "").strip()
                g = (st.session_state.wallet_manual_ingr or "").strip()
                if not n:
                    st.session_state.__pill_wallet_msg = "제품명을 입력해주세요."
                    st.session_state.__pill_wallet_msg_type = "warning"
                    return
                # 실제 추가
                _add_to_wallet(n, g or "성분 미상")
                # 입력칸 비우기
                st.session_state.wallet_manual_name = ""
                st.session_state.wallet_manual_ingr = ""
                st.session_state.__pill_wallet_msg = f"‘{n}’이(가) 약 지갑에 추가되었습니다."
                st.session_state.__pill_wallet_msg_type = "success"

            def _on_clear():
                st.session_state.wallet_manual_name = ""
                st.session_state.wallet_manual_ingr = ""
                st.session_state.__pill_wallet_msg = ""
                st.session_state.__pill_wallet_msg_type = "info"

            add_col1, add_col2, _ = st.columns([1, 1, 3])
            with add_col1:
                st.button("추가하기", type="primary", on_click=_on_add, use_container_width=True)
            with add_col2:
                st.button("입력 지우기", on_click=_on_clear, use_container_width=True)

            # 메시지 표시
            if st.session_state.__pill_wallet_msg:
                if st.session_state.__pill_wallet_msg_type == "success":
                    st.success(st.session_state.__pill_wallet_msg)
                elif st.session_state.__pill_wallet_msg_type == "warning":
                    st.warning(st.session_state.__pill_wallet_msg)
                else:
                    st.info(st.session_state.__pill_wallet_msg)

        # ── 2) 지갑 목록 (✅ 같은 박스 내부에 렌더링)
        if not st.session_state.pill_wallet:
            st.caption("지갑이 비어 있어요. 대화 중 ‘복용 중’, ‘처방받음’과 함께 약 이름을 말하면 제안해 드리거나, 위의 ‘➕ 직접 추가’를 이용하세요.")
            return

        # 항목 렌더
        for idx, item in enumerate(st.session_state.pill_wallet):
            name = item["name"]
            ingr = item["ingredient"]
            added = item["added_at"]

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"**{name}**  <span style='color:#6b7280;'>({ingr})</span><br>"
                    f"<span style='color:#9ca3af; font-size:0.85rem;'>추가: {added}</span>",
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("삭제", key=f"del_{idx}_{name}", use_container_width=True):
                    st.session_state.pill_wallet = [
                        x for x in st.session_state.pill_wallet
                        if not (x['name'] == name and x['added_at'] == added)
                    ]
                    st.rerun()
