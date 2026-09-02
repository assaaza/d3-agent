"""
D3 에이전트 — 범용 전처리 엔진

원칙: 애매하면 자동으로 처리하지 않는다. 확실한 것만 자동 적용하고,
판단이 애매한 항목은 "수동 확인 필요"로 표시만 하고 그대로 둔다.
컬럼 이름을 코드에 하드코딩하지 않는다 — 어떤 CSV가 와도 데이터를 보고 스스로 판단한다.

사용법:
    from d3_preprocess import load_csv, profile, run_preprocessing, to_dictionary_md

    df = load_csv("raw.csv")
    prof = profile(df)              # 체크리스트 탐지 결과 (사람이 먼저 봄)
    clean_df, log, dict_notes = run_preprocessing(df, prof)  # 안전한 것만 자동 적용
"""

import pandas as pd
import numpy as np

MISSING_MARKERS = {"", "na", "n/a", "null", "none", "-", "nan", "결측", "미상"}


def load_csv(path, encoding="utf-8-sig"):
    return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)


def _is_missing(v):
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in MISSING_MARKERS


def _is_numeric_like(v):
    if _is_missing(v):
        return False
    s = str(v).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    if s == "":
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _to_number(v):
    s = str(v).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    return float(s)


def _looks_like_index(name):
    """연도/ID/번호처럼 '측정값'이 아니라 '식별자' 역할인 컬럼 이름 — 이상치·재구성 판단에서 제외."""
    n = str(name).strip().lower()
    return n in {"year", "연도", "id", "번호", "no", "date", "날짜"}


def profile(df: pd.DataFrame) -> dict:
    """데이터를 보고 체크리스트 항목별 탐지 결과를 만든다. 아직 아무것도 바꾸지 않는다."""
    p = {"row_count": len(df), "col_count": len(df.columns)}

    # 1. 컬럼명 정리
    p["trimmable_headers"] = [c for c in df.columns if c != c.strip() or "  " in c]

    # 2. 완전히 빈 행/열
    def _row_all_missing(row):
        return all(_is_missing(v) for v in row)
    p["empty_rows"] = int(df.apply(_row_all_missing, axis=1).sum())
    p["empty_columns"] = [c for c in df.columns if all(_is_missing(v) for v in df[c])]

    # 3. 중복 행
    p["duplicate_rows"] = int(df.duplicated().sum())

    # 4. 상수 컬럼 (모든 값이 동일 — 행이 2개 이상일 때만 의미 있음)
    p["constant_columns"] = {}
    if len(df) > 1:
        for c in df.columns:
            uniq = df[c].unique()
            if len(uniq) == 1:
                p["constant_columns"][c] = uniq[0]

    # 5. 숫자형으로 변환 가능한 컬럼 (결측 제외 전부 숫자로 보일 때만 — 섞여 있으면 자동 변환 대상 아님)
    p["numeric_columns"] = []
    p["mixed_columns"] = []  # 숫자와 텍스트가 섞여 있어 자동 변환하지 않는 컬럼
    for c in df.columns:
        non_missing = [v for v in df[c] if not _is_missing(v)]
        if not non_missing:
            continue
        numeric_flags = [_is_numeric_like(v) for v in non_missing]
        if all(numeric_flags):
            needs_cleanup = any(("," in str(v)) or str(v).strip().endswith("%") for v in non_missing)
            p["numeric_columns"].append({"col": c, "needs_cleanup": needs_cleanup})
        elif any(numeric_flags):
            p["mixed_columns"].append(c)

    numeric_col_names = [nc["col"] for nc in p["numeric_columns"]]

    # 6. 결측치 (일반적인 결측 표기 인식: "", NA, N/A, null, -, 등)
    p["missing_cells"] = {}
    for c in df.columns:
        n = int(df[c].apply(_is_missing).sum())
        if n > 0:
            p["missing_cells"][c] = n

    # 7. 이상치 (IQR, 숫자형 컬럼만, 결측 제외, year/id 같은 식별자성 컬럼은 제외)
    p["outliers"] = {}
    for nc in p["numeric_columns"]:
        c = nc["col"]
        if _looks_like_index(c):
            continue
        vals = [_to_number(v) for v in df[c] if not _is_missing(v)]
        if len(vals) < 4:
            continue
        s = pd.Series(vals)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        idx_outliers = [i for i, v in enumerate(df[c]) if not _is_missing(v) and not (lo <= _to_number(v) <= hi)]
        if idx_outliers:
            p["outliers"][c] = {"lo": lo, "hi": hi, "rows": idx_outliers}

    # 8. 재구성(long -> wide) 후보 — 애매하면 후보 없음으로 처리 (자동으로 고르지 않음)
    p["reshape"] = None
    p["reshape_ambiguous"] = False
    non_numeric_cols = [c for c in df.columns if c not in numeric_col_names]
    pivot_candidates = [c for c in non_numeric_cols
                         if c not in p["constant_columns"]
                         and 2 <= df[c].nunique(dropna=False) <= 8]

    value_candidates = [nc["col"] for nc in p["numeric_columns"]
                         if nc["col"] not in p["constant_columns"] and not _looks_like_index(nc["col"])]
    if not value_candidates:
        value_candidates = [nc["col"] for nc in p["numeric_columns"] if nc["col"] not in p["constant_columns"]]

    # pivot 후보(그룹을 나눌 컬럼) 자체가 없으면 애초에 "재구성"이라는 개념이 적용되지 않음 —
    # 이건 "애매함"이 아니라 "이미 tidy해서 불필요"한 경우이므로 reshape_ambiguous를 켜지 않는다.
    if len(pivot_candidates) == 0:
        pass
    elif len(pivot_candidates) == 1 and len(value_candidates) == 1:
        index_cols = [c for c in df.columns if c not in (pivot_candidates[0], value_candidates[0])]
        # index_cols + pivot 조합이 유일한지 확인 (아니면 pivot 시 값이 겹쳐서 덮어써짐 -> 위험하니 보류)
        key_cols = index_cols + [pivot_candidates[0]]
        if not df.duplicated(subset=key_cols).any():
            p["reshape"] = {"pivot_col": pivot_candidates[0], "value_col": value_candidates[0], "index_cols": index_cols}
        else:
            p["reshape_ambiguous"] = True
    else:
        # pivot 후보가 2개 이상이거나(어느 컬럼 기준으로 나눌지 불명확),
        # pivot 후보는 1개지만 값 컬럼이 여러 개라 어느 걸 펼칠지 불명확한 경우
        p["reshape_ambiguous"] = True

    return p


def run_preprocessing(df: pd.DataFrame, prof: dict, options: dict = None):
    """
    안전하게 자동 적용 가능한 항목만 실행한다.
    options로 끄고 켤 수 있지만, 기본값(None)은 "안전한 것 전부 적용, 애매한 것은 보류".
    반환: (clean_df, log 리스트, dictionary_notes 딕셔너리 — 데이터 사전에 들어갈 내용)
    """
    opts = {
        "trim_headers": True,
        "remove_empty": True,
        "remove_duplicates": True,
        "separate_constants": True,
        "convert_numeric": True,
        "missing_strategy": "flag_only",  # "flag_only" | "drop" | "mean"
        "outlier_strategy": "flag",       # "flag" | "remove" | "none"
        "reshape": True,                  # profile에서 애매하지 않게 찾았을 때만 실제로 적용됨
    }
    if options:
        opts.update(options)

    d = df.copy()
    log = []
    dictionary_notes = {"constant_metadata": {}, "manual_review_needed": []}

    if opts["trim_headers"] and prof["trimmable_headers"]:
        d.columns = [c.strip() for c in d.columns]
        log.append(f"컬럼명 정리: {len(prof['trimmable_headers'])}개 컬럼명의 공백 정리")

    if opts["remove_empty"]:
        before = len(d)
        d = d[~d.apply(lambda row: all(_is_missing(v) for v in row), axis=1)]
        empty_cols = [c for c in d.columns if all(_is_missing(v) for v in d[c])]
        if empty_cols:
            d = d.drop(columns=empty_cols)
        removed_rows = before - len(d)
        if removed_rows or empty_cols:
            log.append(f"빈 행/열 제거: 빈 행 {removed_rows}개, 빈 열 {len(empty_cols)}개")

    if opts["remove_duplicates"] and prof["duplicate_rows"] > 0:
        before = len(d)
        d = d.drop_duplicates()
        log.append(f"중복 행 제거: {before - len(d)}개")

    if opts["separate_constants"] and prof["constant_columns"]:
        cols = [c for c in prof["constant_columns"] if c in d.columns]
        for c in cols:
            dictionary_notes["constant_metadata"][c] = prof["constant_columns"][c]
        d = d.drop(columns=cols)
        log.append(f"상수 컬럼 {len(cols)}개를 데이터 사전으로 분리: {', '.join(cols)}")

    if prof["mixed_columns"]:
        dictionary_notes["manual_review_needed"].append(
            f"숫자/텍스트가 섞인 컬럼이라 자동 변환하지 않음: {', '.join(prof['mixed_columns'])} — 수동 확인 필요"
        )

    if opts["convert_numeric"] and prof["numeric_columns"]:
        cols = [nc["col"] for nc in prof["numeric_columns"] if nc["col"] in d.columns]
        for c in cols:
            d[c] = d[c].apply(lambda v: _to_number(v) if not _is_missing(v) else np.nan)
        if cols:
            log.append(f"숫자형 변환: {', '.join(cols)} ({len(cols)}개 컬럼)")

    if opts["missing_strategy"] != "flag_only" and prof["missing_cells"]:
        numeric_names = [nc["col"] for nc in prof["numeric_columns"]]
        cols = [c for c in prof["missing_cells"] if c in d.columns]
        if opts["missing_strategy"] == "drop":
            before = len(d)
            d = d.dropna(subset=[c for c in cols if c in numeric_names]) if any(c in numeric_names for c in cols) else d
            # 숫자 아닌 컬럼의 결측(빈 문자열 등)도 같이 처리
            non_numeric_missing = [c for c in cols if c not in numeric_names]
            if non_numeric_missing:
                d = d[~d[non_numeric_missing].apply(lambda row: any(_is_missing(v) for v in row), axis=1)]
            log.append(f"결측치 처리(행 제거): {before - len(d)}개 행 제거")
        elif opts["missing_strategy"] == "mean":
            for c in cols:
                if c in numeric_names:
                    mean_val = d[c].mean()
                    filled = int(d[c].isna().sum())
                    d[c] = d[c].fillna(round(mean_val, 2))
                    log.append(f"결측치 처리(평균값 채움): {c} 컬럼 {filled}개")
                else:
                    dictionary_notes["manual_review_needed"].append(f"{c}: 텍스트 컬럼이라 평균 채우기 불가 — 그대로 둠")
    elif prof["missing_cells"]:
        dictionary_notes["manual_review_needed"].append(
            f"결측치가 있지만 처리하지 않고 표시만 함: {prof['missing_cells']}"
        )

    if prof["outliers"]:
        for c, info in prof["outliers"].items():
            if c not in d.columns:
                continue
            if opts["outlier_strategy"] == "flag":
                flag_col = f"{c}_이상치여부"
                lo, hi = info["lo"], info["hi"]
                d[flag_col] = d[c].apply(lambda v: "Y" if pd.notna(v) and not (lo <= v <= hi) else "N")
                log.append(f"이상치 표시: {c} 컬럼에 {len(info['rows'])}개 표시 ({flag_col} 컬럼 추가, 삭제하지 않음)")
            elif opts["outlier_strategy"] == "remove":
                before = len(d)
                lo, hi = info["lo"], info["hi"]
                d = d[(d[c].isna()) | ((d[c] >= lo) & (d[c] <= hi))]
                log.append(f"이상치 제거: {c} 컬럼에서 {before - len(d)}개 행 제거")

    if opts["reshape"] and prof["reshape"]:
        pivot_col, value_col, index_cols = prof["reshape"]["pivot_col"], prof["reshape"]["value_col"], prof["reshape"]["index_cols"]
        index_cols = [c for c in index_cols if c in d.columns]  # 이미 제거된(상수) 컬럼은 제외
        if pivot_col in d.columns and value_col in d.columns:
            wide = d.pivot_table(index=index_cols, columns=pivot_col, values=value_col, aggfunc="first").reset_index()
            wide.columns.name = None
            d = wide
            log.append(f"구조 변환: '{pivot_col}' 기준으로 넓은 형식 변환 (값: {value_col})")
    elif prof["reshape_ambiguous"]:
        dictionary_notes["manual_review_needed"].append(
            "재구성(long→wide) 후보가 여러 개이거나 기준이 애매해서 자동으로 적용하지 않음 — 필요하면 수동으로 결정할 것"
        )

    return d, log, dictionary_notes


def to_dictionary_md(title, prof, log, dictionary_notes, source_note=""):
    lines = [f"# 데이터 사전 — {title}", ""]
    if source_note:
        lines += [source_note, ""]
    lines += ["## 처리 로그", ""]
    for l in log:
        lines.append(f"- {l}")
    if not log:
        lines.append("- (자동 적용된 항목 없음 — 원본이 이미 깨끗함)")
    lines.append("")
    if dictionary_notes["constant_metadata"]:
        lines += ["## 데이터셋 메타데이터 (상수 컬럼에서 분리됨)", ""]
        for k, v in dictionary_notes["constant_metadata"].items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    if dictionary_notes["manual_review_needed"]:
        lines += ["## 수동 확인이 필요한 항목", ""]
        for n in dictionary_notes["manual_review_needed"]:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)
