"""
D3 에이전트 실습 — 공공데이터로 보고서 만들기 (Streamlit)

교육생이 직접 눌러가며 익히는 실습 도구.
1. 탐색: 관심 주제를 입력하면 KOSIS 통합검색으로 관련 통계표를 찾아줌
2. 수집: 선택한 통계표의 실제 데이터를 KOSIS Open API로 가져옴
3. 전처리: tools/d3_preprocess.py의 범용 전처리 엔진을 그대로 실행
4. 그래프: 전처리 결과를 간단한 꺾은선 그래프로 확인

중요: KOSIS API 키는 각 교육생이 본인 것을 입력합니다. 이 키는 화면에도, 파일에도
저장되지 않고 그 실행 세션(브라우저 탭) 안에서만 메모리에 있다가 사라집니다.
"""

import io
import os
import sys

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from d3_preprocess import load_csv, profile, run_preprocessing  # noqa: E402

KOSIS_SEARCH_URL = "https://kosis.kr/openapi/statisticsSearch.do"
KOSIS_DATA_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
PRD_SE_CANDIDATES = ["Y", "M", "Q", "H"]  # 연간/월간/분기/반기 — 통계표마다 다르므로 순서대로 시도

st.set_page_config(page_title="D3 에이전트 실습", layout="wide")

st.title("D3 에이전트 실습 — 공공데이터로 보고서 만들기")
st.caption("탐색 → 수집 → 전처리 → 그래프. 각 단계를 직접 눌러보면서 과정을 익혀보세요.")

for key, default in [
    ("search_results", None),
    ("selected_table", None),
    ("raw_csv_bytes", None),
    ("clean_df", None),
    ("log", None),
    ("dict_notes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("설정")
    api_key = st.text_input(
        "KOSIS Open API 키",
        type="password",
        help="kosis.kr → 공유서비스(OpenAPI) → 인증키 신청에서 본인 명의로 무료 발급받은 키를 입력하세요.",
    )
    st.caption("이 키는 저장되지 않고 이 실습 세션에서만 사용됩니다.")
    st.divider()
    st.caption("D3 에이전트 실습 도구 · assaaza/d3-agent")

st.divider()

# ============================================================
# 1단계 — 탐색
# ============================================================
st.subheader("1단계 · 탐색 — 관심 주제를 입력하세요")
st.caption("KOSIS 통합검색 API로 입력한 키워드와 관련된 통계표를 찾습니다. 실제 결과를 보고 어떤 통계표를 쓸지는 직접 골라야 합니다 — 이게 '탐색' 단계에서 사람의 판단이 필요한 이유입니다.")

topic = st.text_input("주제 키워드", placeholder="예: 1인 가구, 최저임금, 반려동물 등록")
search_clicked = st.button("탐색 실행", disabled=not (api_key and topic))

if search_clicked:
    with st.spinner("KOSIS 통합검색으로 관련 통계표를 찾는 중..."):
        try:
            resp = requests.get(
                KOSIS_SEARCH_URL,
                params={
                    "method": "getList",
                    "apiKey": api_key,
                    "format": "json",
                    "jsonVD": "Y",
                    "jsonMVD": "Y",
                    "searchNm": topic,
                },
                timeout=15,
            )
            data = resp.json()
            if isinstance(data, dict) and (data.get("errMsg") or data.get("err")):
                st.error(f"검색 오류: {data.get('errMsg', data.get('err'))}")
                st.session_state["search_results"] = None
            elif isinstance(data, list) and data:
                st.session_state["search_results"] = pd.DataFrame(data)
            else:
                st.warning("검색 결과가 없습니다. 다른 키워드로 다시 시도해보세요.")
                st.session_state["search_results"] = None
        except Exception as e:
            st.error(f"검색 요청이 실패했습니다: {e}")
            st.session_state["search_results"] = None

if st.session_state["search_results"] is not None:
    df = st.session_state["search_results"]
    show_cols = [c for c in ["TBL_NM", "ORG_NM", "STRT_PRD_DE", "END_PRD_DE", "ORG_ID", "TBL_ID"] if c in df.columns]
    st.write(f"검색 결과 {len(df)}건 — 사용할 통계표를 하나 선택하세요.")
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    def _label(i):
        row = df.iloc[i]
        return f"{row.get('TBL_NM', '(이름 없음)')} · {row.get('ORG_NM', '')} ({row.get('STRT_PRD_DE', '?')}~{row.get('END_PRD_DE', '?')})"

    idx = st.selectbox("통계표 선택", options=range(len(df)), format_func=_label, key="table_select")
    st.session_state["selected_table"] = df.iloc[idx].to_dict()

st.divider()

# ============================================================
# 2단계 — 수집
# ============================================================
st.subheader("2단계 · 데이터 수집 실행")
sel = st.session_state["selected_table"]
if sel:
    st.write(f"선택된 통계표: **{sel.get('TBL_NM')}** ({sel.get('ORG_NM')}, 통계표ID: {sel.get('TBL_ID')})")
    st.caption("전국 합계·전체 분류 기준으로 자동 조회를 시도합니다. 통계표 구조가 복잡하면 자동 조회가 안 될 수 있는데, 그 경우 원인을 그대로 보여드립니다 — 추측해서 임의로 채우지 않습니다.")
else:
    st.caption("먼저 1단계에서 통계표를 선택하세요.")

collect_clicked = st.button("데이터 수집 실행", disabled=not (api_key and sel))

if collect_clicked:
    with st.spinner("KOSIS에서 실제 데이터를 가져오는 중..."):
        last_err = None
        result_data = None
        used_prd_se = None
        for prd_se in PRD_SE_CANDIDATES:
            params = {
                "method": "getList",
                "apiKey": api_key,
                "format": "json",
                "jsonVD": "Y",
                "orgId": sel["ORG_ID"],
                "tblId": sel["TBL_ID"],
                "objL1": "ALL",
                "objL2": "ALL",
                "objL3": "ALL",
                "itmId": "ALL",
                "prdSe": prd_se,
            }
            try:
                resp = requests.get(KOSIS_DATA_URL, params=params, timeout=20)
                data = resp.json()
            except Exception as e:
                last_err = str(e)
                continue
            if isinstance(data, list) and data:
                result_data = data
                used_prd_se = prd_se
                break
            if isinstance(data, dict):
                last_err = data.get("errMsg") or data.get("err") or str(data)

        if result_data is None:
            st.error(
                f"자동 수집에 실패했습니다 (마지막 오류: {last_err}). "
                "이 통계표는 분류 구조가 더 복잡해서 기본 설정(전체/ALL)으로는 안 되는 것 같습니다 — "
                "다른 주제를 골라보거나 강사에게 문의해보세요."
            )
            st.session_state["raw_csv_bytes"] = None
        else:
            raw_df = pd.DataFrame(result_data)
            note = ""
            if len(raw_df) > 3000:
                note = f" (원래 {len(raw_df)}행이었으나 분류 조합이 많아 실습용으로 앞 3000행만 사용)"
                raw_df = raw_df.head(3000)
            st.session_state["raw_csv_bytes"] = raw_df.to_csv(index=False).encode("utf-8-sig")
            st.session_state["clean_df"] = None  # 새로 수집했으면 이전 전처리 결과는 무효화
            st.success(f"수집 완료 — {len(raw_df)}행 (수록주기: {used_prd_se}){note}")
            st.dataframe(raw_df.head(20), use_container_width=True)

if st.session_state["raw_csv_bytes"] is not None:
    st.download_button(
        "원본 데이터 CSV 다운로드",
        data=st.session_state["raw_csv_bytes"],
        file_name="raw_data.csv",
        mime="text/csv",
    )

st.divider()

# ============================================================
# 3단계 — 전처리 + 그래프
# ============================================================
st.subheader("3단계 · 전처리 실행 및 그래프 확인")
st.caption("주제별로 스크립트를 새로 짜지 않고, 검증된 범용 전처리 엔진(d3_preprocess.py)을 그대로 사용합니다 — 컬럼 이름을 모르는 데이터에도 똑같이 작동합니다.")

preprocess_clicked = st.button("전처리 실행", disabled=st.session_state["raw_csv_bytes"] is None)

if preprocess_clicked:
    raw_io = io.BytesIO(st.session_state["raw_csv_bytes"])
    df_in = load_csv(raw_io)
    prof = profile(df_in)
    clean_df, log, dict_notes = run_preprocessing(df_in, prof)
    st.session_state["clean_df"] = clean_df
    st.session_state["log"] = log
    st.session_state["dict_notes"] = dict_notes

if st.session_state["clean_df"] is not None:
    st.markdown("**전처리 체크리스트 — 자동 적용된 항목**")
    log = st.session_state["log"]
    if log:
        for line in log:
            st.write(f"- {line}")
    else:
        st.write("- 자동 적용된 항목 없음 (원본이 이미 깨끗합니다)")

    manual_notes = st.session_state["dict_notes"]["manual_review_needed"]
    if manual_notes:
        st.warning("**수동 확인이 필요한 항목** (애매해서 자동으로 처리하지 않았습니다)\n\n" + "\n".join(f"- {n}" for n in manual_notes))

    st.markdown("**전처리 결과**")
    clean_df = st.session_state["clean_df"]
    st.dataframe(clean_df, use_container_width=True)
    st.download_button(
        "전처리 결과 CSV 다운로드",
        data=clean_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="clean_data.csv",
        mime="text/csv",
    )

    st.markdown("**간단한 그래프**")
    numeric_cols = clean_df.select_dtypes(include="number").columns.tolist()
    time_candidates = [c for c in clean_df.columns if str(c).strip().lower() in ("year", "연도", "date", "날짜", "prd_de")]
    if numeric_cols:
        chart_df = clean_df.copy()
        if time_candidates:
            chart_df = chart_df.set_index(time_candidates[0])
            numeric_cols = [c for c in numeric_cols if c != time_candidates[0]]
        if numeric_cols:
            st.line_chart(chart_df[numeric_cols])
        else:
            st.info("연도/날짜 컬럼 외에 그래프로 그릴 숫자 컬럼이 없습니다.")
    else:
        st.info("그래프로 그릴 숫자형 컬럼이 없습니다.")
else:
    st.caption("먼저 2단계에서 데이터를 수집하세요.")
