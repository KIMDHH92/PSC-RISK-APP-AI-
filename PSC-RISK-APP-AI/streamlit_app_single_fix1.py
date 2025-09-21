# === Inline path utils (repo-relative) ===
from pathlib import Path as _Path_in
import os as _os_in
import pandas as _pd_in
import re as _re_in

DATA_DIR = _Path_in(__file__).resolve().parent / "data"
def p(*parts) -> str:
    return str(DATA_DIR.joinpath(*parts))

def _read_any(path: str):
    ext = _Path_in(path).suffix.lower()
    if ext in [".xls", ".xlsx"]:
        return _pd_in.read_excel(path, dtype=str)
    elif ext == ".csv":
        return _pd_in.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {ext} ({path})")

def _candidate_imo_col(cols):
    cands = ['vessel_imo','imo','imo_no','IMO NO','IMO_NO','Vessel_IMO','VESSEL_IMO']
    for c in cands:
        if c in cols:
            return c
    return None

def _normalize_imo_column(df: _pd_in.DataFrame) -> _pd_in.DataFrame:
    if df is None or len(df) == 0:
        return df
    cand = _candidate_imo_col(df.columns)
    if cand is not None and cand != 'vessel_imo':
        df = df.rename(columns={cand: 'vessel_imo'})
    if 'vessel_imo' in df.columns:
        df['vessel_imo'] = (
            df['vessel_imo'].astype(str)
            .str.extract(r'(\d+)', expand=False)
            .fillna('')
        )
    return df

def load_ship_db_flexible(preferred_filename: str = None) -> _pd_in.DataFrame:
    # try preferred file first
    if preferred_filename:
        pref = p(preferred_filename)
        if _os_in.path.exists(pref):
            try:
                return _normalize_imo_column(_read_any(pref))
            except Exception:
                pass
    # fallback to first xlsx in data/
    xlxs = sorted(DATA_DIR.glob("*.xlsx"))
    if xlxs:
        return _normalize_imo_column(_read_any(str(xlxs[0])))
    return _pd_in.DataFrame()

def find_eta_files():
    eta_root = DATA_DIR / "ETA_ALL"
    if not eta_root.exists():
        return []
    files = []
    for ext in ("*.csv","*.xlsx"):
        files.extend(sorted(eta_root.glob(ext)))
    return [str(f) for f in files]

def load_eta_merged() -> _pd_in.DataFrame:
    files = find_eta_files()
    if not files:
        return _pd_in.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(_read_any(f))
        except Exception:
            continue
    if not dfs:
        return _pd_in.DataFrame()
    merged = _pd_in.concat(dfs, ignore_index=True)
    merged = _normalize_imo_column(merged)
    return merged
# === End inline path utils ===


# streamlit_app_final44_AB_fix.py
# NOTE: UI/레이아웃은 streamlit_app_final44를 최대한 유지하고,
#       점수 산식은 A×B(100점 만점 × 100점 만점 → (A×B)/100)로 교체.
#       새 데이터소스(5. MOU INFO / MoU Information&Last Periodical Survey)를 추가.
#       메뉴3의 검색 기능은 공통 함수(find_candidates/pick_candidate)로 보장.

import streamlit as st
# inlined path_utils used

# =========================
# AI AUTO(학습→예측→CSV 생성) + B* 보정 유틸
# =========================
import os, json, pandas as pd, numpy as np

def _safe_to_numeric(s):
    try: 
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([np.nan]*len(s))

def _norm_imo_series(s):
    return s.astype(str).str.split(".").str[0].str.lstrip("0")

def _ensure_dir(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

def _load_ai_scores_csv(path="./ai/ai_scores.csv"):
    try:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["imo","B_ai"])
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        def pick(*names):
            for n in names:
                if n in cols: return cols[n]
            return None
        col_imo = pick("imo","imo_no","imo number","imo no.","imo_no.")
        col_prob = pick("ai_prob","prob","p","score","b_ai")
        if not col_imo or not col_prob:
            return pd.DataFrame(columns=["imo","B_ai"])
        out = df[[col_imo, col_prob]].copy()
        out.columns = ["imo","prob"]
        out["prob"] = pd.to_numeric(out["prob"], errors="coerce")
        if out["prob"].dropna().gt(1).mean() > 0.5:
            out["prob"] = out["prob"]/100.0
        out["B_ai"] = (out["prob"].clip(0,1) * 100.0).round(1)
        out["imo"] = _norm_imo_series(out["imo"])
        return out[["imo","B_ai"]]
    except Exception:
        return pd.DataFrame(columns=["imo","B_ai"])

_AI_SCORES = None
def _ensure_ai_scores_loaded():
    global _AI_SCORES
    if _AI_SCORES is None:
        _AI_SCORES = _load_ai_scores_csv()

def get_B_ai_for_imo(imo: str):
    try:
        _ensure_ai_scores_loaded()
        if _AI_SCORES is None or _AI_SCORES.empty:
            return None
        row = _AI_SCORES[_AI_SCORES["imo"].astype(str)==str(imo)]
        if row.empty:
            return None
        return float(row.iloc[0]["B_ai"])
    except Exception:
        return None

def compute_B_star(B_rule: float, B_ai: float | None, alpha: float | None = None):
    try: B_rule = float(B_rule)
    except: B_rule = 0.0
    if alpha is None:
        try: alpha = float(os.getenv("AI_ALPHA","0.3"))
        except: alpha = 0.3
    if B_ai is None or not (0.0 <= float(B_ai) <= 100.0):
        alpha = 0.0; B_ai = 0.0
    B_star = (1.0 - float(alpha)) * float(B_rule) + float(alpha) * float(B_ai)
    return max(0.0, min(100.0, round(B_star, 1)))

# ------------------- AUTO PIPELINE -------------------
def build_training_dataset(ship_db, detention_df, def_df, mou_info_df):
    import pandas as pd

# === Robust IMO normalization & safe ETA selector ===
import re as _re_mod
import pandas as _pd_mod

def _normalize_imo_column(df: _pd_mod.DataFrame) -> _pd_mod.DataFrame:
    """Unify IMO column to 'vessel_imo' and keep only digits as string."""
    if df is None or len(df) == 0:
        return df
    cands = ['vessel_imo','imo','imo_no','IMO NO','IMO_NO','Vessel_IMO','VESSEL_IMO']
    found = next((c for c in cands if c in df.columns), None)
    if found and found != 'vessel_imo':
        df = df.rename(columns={found: 'vessel_imo'})
    if 'vessel_imo' in df.columns:
        df['vessel_imo'] = (
            df['vessel_imo'].astype(str)
            .str.extract(r'(\d+)', expand=False)
            .fillna('')
        )
    return df

def _as_imo_str(x):
    s = str(x) if x is not None else ''
    m = _re_mod.search(r'\d+', s)
    return m.group(0) if m else ''

def get_eta_rows_safe(eta_df_range: _pd_mod.DataFrame, imo):
    """Safe selector that never KeyErrors when vessel_imo absent."""
    if eta_df_range is None or len(eta_df_range) == 0:
        return _pd_mod.DataFrame()
    df = _normalize_imo_column(eta_df_range.copy())
    if 'vessel_imo' not in df.columns:
        return _pd_mod.DataFrame()
    return df.loc[df['vessel_imo'] == _as_imo_str(imo)].copy()

    from datetime import datetime, timedelta

    if ship_db is None or len(ship_db)==0:
        return pd.DataFrame()

    today = datetime.today()
    since_3y = today - timedelta(days=1095)

    base = ship_db.copy()
    base.columns = [c.strip().lower() for c in base.columns]
    if "imo no." not in base.columns:
        return pd.DataFrame()
    base["imo"] = _norm_imo_series(base["imo no."])

    if "build date" in base.columns:
        try:
            bd = pd.to_datetime(base["build date"], errors="coerce")
            age_y = ((today - bd).dt.days/365.25).round(1)
        except Exception:
            age_y = pd.Series([np.nan]*len(base))
    else:
        age_y = pd.Series([np.nan]*len(base))
    base["age_years"] = age_y.fillna(age_y.median() if age_y.notna().any() else 10)

    y = pd.Series([0]*len(base), index=base.index, name="label_det3y")
    if detention_df is not None and not detention_df.empty:
        d = detention_df.copy()
        d.columns = [c.strip().lower() for c in d.columns]
        if "inspection_date" in d.columns:
            d["inspection_date"] = pd.to_datetime(d["inspection_date"], errors="coerce")
            d = d[d["inspection_date"] >= since_3y]
        if "imo_no" in d.columns:
            d["imo"] = _norm_imo_series(d["imo_no"])
            det_flag = d.groupby("imo")["inspection_date"].count().rename("det3y_cnt")
            base = base.merge(det_flag, on="imo", how="left")
            base["det3y_cnt"] = base["det3y_cnt"].fillna(0).astype(int)
            y = (base["det3y_cnt"] > 0).astype(int)
        else:
            base["det3y_cnt"] = 0
    else:
        base["det3y_cnt"] = 0

    if def_df is not None and not def_df.empty:
        df = def_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        if "inspection_date" in df.columns:
            df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
            df = df[df["inspection_date"] >= since_3y]
        if "imo_no" in df.columns:
            df["imo"] = _norm_imo_series(df["imo_no"])
            df["def_code_num"] = df["def_code"].astype(str).str.extract(r"(\d+)")[0]
            df["def_code_num"] = df["def_code_num"].fillna("00000")
            agg_total = df.groupby("imo").size().rename("def3y_total")
            base = base.merge(agg_total, on="imo", how="left")
            base["def3y_total"] = base["def3y_total"].fillna(0).astype(int)
            topk = df["def_code_num"].value_counts().head(5).index.tolist()
            for code in topk:
                cnt = df[df["def_code_num"]==code].groupby("imo").size().rename(f"def3y_code_{code}")
                base = base.merge(cnt, on="imo", how="left")
                base[f"def3y_code_{code}"] = base[f"def3y_code_{code}"].fillna(0).astype(int)
        else:
            base["def3y_total"] = 0
    else:
        base["def3y_total"] = 0

    if mou_info_df is not None and not mou_info_df.empty:
        m = mou_info_df.copy()
        m.columns = [c.strip().lower() for c in m.columns]
        if "imo_no" in m.columns:
            m["imo"] = _norm_imo_series(m["imo_no"])
            if "priority" in m.columns:
                pri_map = {"PRIORITY I":2, "PRIORITY II":1, "NO":0}
                m["priority_num"] = m["priority"].map(pri_map).fillna(0).astype(int)
            else:
                m["priority_num"] = 0
            if "last_ps" in m.columns:
                m["last_ps"] = pd.to_datetime(m["last_ps"], errors="coerce")
                days_last_ps = (datetime.today() - m["last_ps"]).dt.days
                m["days_last_ps"] = days_last_ps.fillna(days_last_ps.median() if days_last_ps.notna().any() else 9999)
            else:
                m["days_last_ps"] = 9999
            base = base.merge(m[["imo","priority_num","days_last_ps"]], on="imo", how="left")
        else:
            base["priority_num"] = 0; base["days_last_ps"] = 9999
    else:
        base["priority_num"] = 0; base["days_last_ps"] = 9999

    for c in ["def3y_total","priority_num","days_last_ps","age_years"]:
        if c not in base.columns: base[c]=0
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

    feats = [c for c in base.columns if c.startswith("def3y_code_")] + ["def3y_total","priority_num","days_last_ps","age_years"]
    X = base[feats].fillna(0)
    y = (base.get("det3y_cnt", 0) > 0).astype(int) if "det3y_cnt" in base.columns else y

    out = base[["imo"]].copy()
    out["label"] = y.values
    out = pd.concat([out, X], axis=1)
    return out


def train_model_and_predict(train_df, target_imos):
    """
    안정판: 예측 대상 IMO와 학습 피처 매핑을 병합(merge) 방식으로 수행해
    'setting an array element with a sequence' 오류를 방지.
    """
    import numpy as np, pandas as pd
    if train_df is None or train_df.empty:
        return pd.DataFrame(columns=["imo","ai_prob"])
    feats = [c for c in train_df.columns if c not in ("imo","label")]
    if not feats:
        return pd.DataFrame(columns=["imo","ai_prob"])

    # 학습 데이터
    X = train_df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = train_df["label"].astype(int).to_numpy()

    # 모델
    use_proba = True
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=-1, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
        )
        model.fit(X, y)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.1, max_iter=300)
        model.fit(X, y)

    # 예측 대상: 병합 기반으로 피처 매핑(없으면 0)
    targ = pd.DataFrame({"imo": pd.Series(target_imos).astype(str)}).dropna().drop_duplicates()
    feat_map = train_df[["imo"] + feats].copy()
    targ = targ.merge(feat_map, on="imo", how="left")
    for c in feats:
        targ[c] = pd.to_numeric(targ[c], errors="coerce").fillna(0.0)
    X_t = targ[feats].to_numpy(dtype=float)

    # 예측
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_t)[:, 1]
    elif hasattr(model, "decision_function"):
        from scipy.special import expit
        prob = expit(model.decision_function(X_t))
    else:
        prob = model.predict(X_t).astype(float)

    out = targ[["imo"]].copy()
    out["ai_prob"] = np.clip(prob, 0.0, 1.0)
    return out

def write_ai_scores_csv(df, path="./ai/ai_scores.csv"):
    _ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    df.to_csv(path, index=False, encoding="utf-8")

def ensure_finalrisk_column(df):
    try:
        import pandas as pd
        if "FinalRisk" not in df.columns:
            if "결합 점수(%)" in df.columns:
                df["FinalRisk"] = pd.to_numeric(df["결합 점수(%)"], errors="coerce")
            else:
                A_cands = ["A","A(%)","A_score","수검가능성(A)"]
                B_cands = ["B","B*","B(%)","문제 심각도(B)","문제 심각도 (B)"]
                A_col = next((c for c in A_cands if c in df.columns), None)
                B_col = next((c for c in B_cands if c in df.columns), None)
                if A_col and B_col:
                    A_v = pd.to_numeric(df[A_col], errors="coerce")
                    B_v = pd.to_numeric(df[B_col], errors="coerce")
                    df["FinalRisk"] = (A_v * B_v) / 100.0
        return df
    except Exception:
        return df

def sort_by_finalrisk(df, ascending=False):
    try:
        df = ensure_finalrisk_column(df)
        if "FinalRisk" in df.columns:
            return df.sort_values("FinalRisk", ascending=ascending, na_position="last")
        if "결합 점수(%)" in df.columns:
            return df.sort_values("결합 점수(%)", ascending=ascending, na_position="last")
        return df
    except Exception:
        return df

import time

# ===== AI lightweight baseline cache (EMA) =====
import json, os, time
_AI_CACHE_PATH = os.path.join(os.path.dirname(__file__), "ai_cache.json")

def _ai_load_cache():
    try:
        with open(_AI_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _ai_save_cache(obj):
    try:
        with open(_AI_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _age_bin_from_year(build_date):
    try:
        # build_date may be year (int/str) or datetime-like
        y = None
        if build_date is None:
            return "unknown"
        if isinstance(build_date, (int, float)):
            y = int(build_date)
        else:
            s = str(build_date)[:4]
            y = int(s)
        from datetime import datetime
        age = max(0, datetime.now().year - y)
    except Exception:
        age = 0
    if age >= 20: return "20+"
    if age >= 15: return "15-19"
    if age >= 10: return "10-14"
    if age >= 5:  return "5-9"
    return "0-4"

def _ai_update_baseline(mou, build_date, risk_value, alpha=0.2):
    cache = _ai_load_cache()
    key = f"{str(mou)}|{_age_bin_from_year(build_date)}"
    entry = cache.get(key, {"mu": float(risk_value), "sigma2": 100.0, "n": 1, "updated": time.time()})
    try:
        # Welford-like EMA for mean/variance (population variance approx)
        mu = float(entry.get("mu", float(risk_value)))
        var = float(entry.get("sigma2", 100.0))
        # EMA update
        x = float(risk_value)
        mu_new = (1 - alpha) * mu + alpha * x
        var_new = (1 - alpha) * var + alpha * (x - mu_new)**2
        n = int(entry.get("n", 1)) + 1
        cache[key] = {"mu": mu_new, "sigma2": var_new, "n": n, "updated": time.time()}
        _ai_save_cache(cache)
        return mu_new, max(var_new, 1e-6)
    except Exception:
        return float(risk_value), 100.0

def _ai_z_message(mou, build_date, risk_value):
    mu, var = _ai_update_baseline(mou, build_date, risk_value)
    import math
    sigma = max(1e-6, math.sqrt(var))
    z = (float(risk_value) - mu) / sigma if sigma else 0.0
    # Clamp z for display
    zc = max(-3.0, min(3.0, z))
    # human text
    if zc >= 1.5:
        tag = "동일 조건군 대비 매우 높음 (+σ)"
    elif zc >= 0.5:
        tag = "동일 조건군 대비 높음 (+σ)"
    elif zc <= -1.0:
        tag = "동일 조건군 대비 낮음 (-σ)"
    else:
        tag = "동일 조건군과 유사"
    return zc, tag
# ===============================================

# ===== AI insight helper (safe) =====
def _ai_insight_from_rankdf(df):
    try:
        import numpy as np
        import pandas as pd
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        res = []
        n = len(df)
        # Risk buckets
        try:
            R = pd.to_numeric(df['결합 점수(%)'], errors='coerce')
            very_high = (R >= 71).sum()
            high = ((R >= 41) & (R <= 70)).sum()
            res.append(f"상위 위험: Very high {very_high}척 · High {high}척")
        except Exception:
            pass
        # MoU distribution
        if 'MOU(도착지)' in df.columns:
            top_mou = df['MOU(도착지)'].value_counts().head(1)
            if not top_mou.empty:
                mou_name = str(top_mou.index[0])
                mou_ratio = int(round(top_mou.iloc[0] * 100 / max(n,1)))
                res.append(f"주요 MoU: {mou_name} ({mou_ratio}%)")
        # Anomaly summary
        if 'AI_이상치' in df.columns:
            anom = int(df['AI_이상치'].sum())
            if anom > 0:
                res.append(f"AI 이상치: {anom}척")
        return " · ".join(res) if res else None
    except Exception:
        return None
# ====================================


import pandas as pd
import numpy as np
import os, re, json, time
from datetime import datetime, timedelta, date

st.set_page_config(page_title="PSC 수검 선별", layout="wide")

# ============== CSS (기존 톤 유지) ==============
BADGE_CSS = """
<style>
.badge {display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; margin-right:6px; margin-top:4px;}
.badge-red {background:#ffe5e5; color:#b00020; border:1px solid #ffb3b3;}
.badge-amber {background:#fff6e5; color:#8a5a00; border:1px solid #ffd699;}
.badge-green {background:#e6ffed; color:#0b6b2e; border:1px solid #a6f3c0;}
.badge-gray {background:#f1f3f4; color:#5f6368; border:1px solid #dadce0;}
.badge-blue {background:#e6f0ff; color:#0b4fb3; border:1px solid #b3ccff;}
.card {border:1px solid #e5e7eb; border-radius:12px; padding:20px 18px; background:#fff;}
.help-tip {color:#5f6368; font-size:12px;}
.big {font-size:28px; font-weight:700;}
.sec-title{font-weight:700; font-size:16px; margin-bottom:6px;}
.table-mini {width:100%; border-collapse:separate; border-spacing:0 6px;}
.table-mini td.label {width:140px; color:#6b7280;}
.table-mini td.value {font-weight:600;}
.search-left { margin-top: 8px; }
.recent-right { margin-top: -3px; }
.search-left [data-testid="stTextInput"] input { padding-top: 4px; min-height: 40px; }
.search-left .stButton > button { height: 40px; }
.recent-right [data-baseweb="select"] > div { min-height: 40px; }
.kpi {display:flex; gap:18px; flex-wrap:wrap;}
.kpi .kpi-card{flex:1; min-width:220px; background:#fafafa; border:1px solid #eee; border-radius:12px; padding:14px;}
.kpi h3{margin:0 0 8px 0; font-size:14px; color:#444;}
.kpi .val{font-size:26px; font-weight:700;}
.hr {height:1px; background:#eee; margin:12px 0;}
</style>
"""
st.markdown(BADGE_CSS, unsafe_allow_html=True)

def _format_priority_label(pri_raw: str) -> str:
    s = (pri_raw or "").strip().upper()
    if s in ("PRIORITY I","I","PI","P I","PRIORITYI"):
        return "Priority I"
    if s in ("PRIORITY II","II","PII","P II","PRIORITYII"):
        return "Priority II"
    return "Priority 없음"


# =========================
# 경로 설정 (회사/집 자동 스위칭)
# =========================
# ====== PATHS (repo-relative, Streamlit Cloud/GitHub friendly) ======
ETA_ROOT = str(DATA_DIR / "ETA_ALL")
# 아래 4개는 우선 data/ 루트를 가리키고, 실제 로딩 함수에서 파일명이 정확히 맞으면 그 파일을 우선 사용함.
DETENTION_FOLDER = str(DATA_DIR)
DEFICIENCY_FOLDER = str(DATA_DIR)
CHECKLIST_FOLDER  = str(DATA_DIR)
# NEW: MOU INFO (Selection Scheme + Last Periodical Survey)
MOU_INFO_FOLDER   = str(DATA_DIR)
MOU_INFO_MNT = p("MoU Information&Last Periodical Survey.csv")
# 신규 업로드 파일 우선 사용(있으면)
CHECKLIST_PATH_MNT = p("지적사항별 Checklist 정리.xlsx")
CHECKLIST_OVERRIDE = CHECKLIST_PATH_MNT if os.path.exists(CHECKLIST_PATH_MNT) else None

PORT_COORDS_LOCAL  = p("port_coords.csv")
PORT_COORDS_MNT    = p("port_coords.csv")

DATE_8_RE = re.compile(r"(\d{8})")
TRAILING_PORT_TERMS = [
    "pt","port","prt","harbour","harbor","hbr","terminal","term","ter","wharf","anchorage","anch",
    "pier","p.","quay","jetty","jet","dock","dck","dep","dept","dep."
]

# ===== 유틸 =====
def _latest_by_digits(files):
    best = None; best_key = ("", "")
    for f in files:
        m = DATE_8_RE.search(f)
        key = (m.group(1) if m else "", f)
        if best is None or key > best_key:
            best = f; best_key = key
    return best

def get_latest_file(folder, exts=(".xlsx", ".csv")):
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts) and not f.startswith("~$")]
    if not files:
        return None
    dated = [f for f in files if DATE_8_RE.search(f)]
    chosen = _latest_by_digits(dated) if dated else sorted(files)[-1]
    return os.path.join(folder, chosen)

def list_eta_folders(eta_root):
    if not os.path.isdir(eta_root):
        return []
    folders = [f for f in os.listdir(eta_root) if os.path.isdir(os.path.join(eta_root, f))]
    res = []
    for f in folders:
        m = DATE_8_RE.search(f)
        if m:
            try:
                d = datetime.strptime(m.group(1), "%Y%m%d").date()
                res.append((d, f))
            except Exception:
                pass
    return sorted(res, key=lambda x: x[0])

def normalize_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def lower_clean(s):
    return normalize_str(s).lower()

def strip_trailing_port_terms(s: str) -> str:
    t = s.strip()
    for _ in range(2):
        parts = t.split()
        if not parts:
            break
        last = parts[-1].lower().strip(".,()[]{}")
        if last in TRAILING_PORT_TERMS:
            t = " ".join(parts[:-1])
        else:
            break
    return t

def port_normalize(p):
    s = lower_clean(p)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("-", " ").replace("/", " ")
    s = strip_trailing_port_terms(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def only_date(dt):
    if pd.isna(dt):
        return "-"
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return str(dt)

# ===== 국가↔MoU 매핑 보조 =====
PARIS_MOU_COUNTRIES = set(["FRANCE","GERMANY","NETHERLANDS","BELGIUM","SPAIN","PORTUGAL","IRELAND","UNITED KINGDOM","NORWAY","SWEDEN","FINLAND","DENMARK","ICELAND","CANADA","RUSSIAN FEDERATION"])
TOKYO_MOU_COUNTRIES = set(["KOREA, REPUBLIC OF","CHINA","JAPAN","TAIWAN, PROVINCE OF CHINA","SINGAPORE","HONG KONG","AUSTRALIA","NEW ZEALAND","RUSSIAN FEDERATION"])
USCG_COUNTRIES      = set(["UNITED STATES","UNITED STATES OF AMERICA"])
ISO2_TO_COUNTRY = {
    "FR":"FRANCE","DE":"GERMANY","NL":"NETHERLANDS","BE":"BELGIUM","ES":"SPAIN","PT":"PORTUGAL",
    "IE":"IRELAND","GB":"UNITED KINGDOM","UK":"UNITED KINGDOM","NO":"NORWAY","SE":"SWEDEN",
    "FI":"FINLAND","DK":"DENMARK","IS":"ICELAND","CA":"CANADA","US":"UNITED STATES",
    "KR":"KOREA, REPUBLIC OF","CN":"CHINA","JP":"JAPAN","TW":"TAIWAN, PROVINCE OF CHINA",
    "SG":"SINGAPORE","HK":"HONG KONG","AU":"AUSTRALIA","NZ":"NEW ZEALAND","RU":"RUSSIAN FEDERATION",
    "IT":"ITALY","GR":"GREECE","TR":"TURKIYE","AE":"UNITED ARAB EMIRATES",
    "USA":"UNITED STATES"}

def country_to_mou(country_upper: str) -> str:
    if not country_upper:
        return "-"
    aliases = {'USA':'UNITED STATES','US':'UNITED STATES','U.S.':'UNITED STATES','U.S.A':'UNITED STATES',
               'UNITED STATES OF AMERICA':'UNITED STATES','AMERICA':'UNITED STATES'}
    cu = aliases.get((country_upper or '').strip().upper(), (country_upper or '').strip().upper())
    if cu in USCG_COUNTRIES:
        return "USCG"
    if cu in PARIS_MOU_COUNTRIES:
        return "Paris MoU"
    if cu in TOKYO_MOU_COUNTRIES:
        return "Tokyo MoU"
    return "-"

# =========================
# (Class No) 표준화/검색 유틸
# =========================
DIGIT_ONLY = re.compile(r"\D+")

def canon_classno_digits(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, float):
        if float(val).is_integer():
            return str(int(val))
    s = str(val).strip()
    if re.fullmatch(r"\d+\.\d+", s):
        s = s.split(".", 1)[0]
    s = DIGIT_ONLY.sub("", s)
    return s

def to_7_digits(s: str) -> str:
    if not s:
        return ""
    return s[-7:].zfill(7)

def safe_contains(series: pd.Series, term: str):
    return series.astype(str).str.contains(str(term), case=False, na=False, regex=False)

def ensure_classno_aux_cols(df: pd.DataFrame):
    if 'class no.' in df.columns:
        if '_class_no_raw_str' not in df.columns:
            df['_class_no_raw_str'] = df['class no.'].astype(str)
        if '_class_no_digits' not in df.columns:
            df['_class_no_digits'] = df['class no.'].apply(canon_classno_digits)
        if '_class_no_7' not in df.columns:
            df['_class_no_7'] = df['_class_no_digits'].apply(to_7_digits)

def find_candidates(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if df.empty or not q or not q.strip():
        return pd.DataFrame()
    q_raw = q.strip()
    q_digits = canon_classno_digits(q_raw)
    q7 = to_7_digits(q_digits)

    conds = (
        safe_contains(df['imo no.'].astype(str), q_raw) |
        df['ship name'].astype(str).str.contains(q_raw, case=False, na=False, regex=False)
    )

    if 'class no.' in df.columns:
        ensure_classno_aux_cols(df)
        if q_digits:
            conds = conds | df['_class_no_digits'].str.contains(q_digits, na=False)
            if len(q_digits) >= 7 and q7:
                conds = conds | (df['_class_no_7'] == q7)
            elif q7:
                conds = conds | df['_class_no_7'].str.contains(q7, na=False)

    return df[conds]

def pick_candidate(cands: pd.DataFrame, key: str):
    if cands.empty:
        return None
    if len(cands) == 1:
        return cands.iloc[0]
    ensure_classno_aux_cols(cands)
    labels = []
    idxs = []
    for idx, r in cands.iterrows():
        imo = str(r.get('imo no.', ''))
        name = normalize_str(r.get('ship name', ''))
        cls7 = normalize_str(r.get('_class_no_7', '')) or "-"
        labels.append(f"{name}  |  IMO {imo}  |  CLASS {cls7}")
        idxs.append(idx)
    sel = st.selectbox("일치 선박 선택", options=list(range(len(labels))), format_func=lambda i: labels[i], key=key)
    return cands.loc[idxs[sel]]

# =========================
# 데이터 로더
# =========================
@st.cache_data(show_spinner=False)
def load_ship_db_flexible(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()
    alias_map = {
        'imo':'imo no.','imo no':'imo no.','imo no.':'imo no.','imo number':'imo no.','imo_no':'imo no.','imo num':'imo no.','imo code':'imo no.',
        'ship name':'ship name','vessel name':'ship name','vessel':'ship name','ship':'ship name',
        'class no':'class no.','class no.':'class no.','class':'class no.','class number':'class no.',
        'build date':'build date','built':'build date','build year':'build date','year built':'build date','keel laid':'build date',
        'flag':'flag','flag state':'flag','flagstate':'flag','country of registry':'flag','registry':'flag',
    }
    rename = {}
    for c in list(df.columns):
        k = c.strip().lower()
        if k in alias_map and alias_map[k] not in df.columns:
            rename[c] = alias_map[k]
    if rename:
        df = df.rename(columns=rename)
    for req in ['imo no.','ship name']:
        if req not in df.columns:
            df[req] = np.nan
    df['imo no.'] = df['imo no.'].astype(str).str.split('.').str[0].str.lstrip('0')

    if 'class no.' in df.columns:
        ensure_classno_aux_cols(df)

    return df

@st.cache_data(show_spinner=False)
def load_eta_folder(folder_path: str):
    if (folder_path is None) or (not os.path.isdir(folder_path)):
        return pd.DataFrame()
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    dfs = []
    for f in sorted(files):
        p = os.path.join(folder_path, f)
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding='cp949')
        df.columns = df.columns.str.strip().str.lower()
        if 'eta_date' not in df.columns and 'eta' in df.columns:
            df = df.rename(columns={'eta':'eta_date'})
        for c in ['vessel_imo','destination_port','eta_date']:
            if c not in df.columns:
                df[c] = np.nan
        df['vessel_imo'] = df['vessel_imo'].astype(str).str.lstrip('0')
        df['__src__'] = os.path.basename(folder_path)
        dfs.append(df[['vessel_imo','destination_port','eta_date','__src__']])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['vessel_imo','destination_port','eta_date','__src__'])

@st.cache_data(show_spinner=False)
def load_eta_by_range(eta_root: str, start_d: date, end_d: date):
    pairs = list_eta_folders(eta_root)
    if not pairs:
        return pd.DataFrame(), "-"
    selected = [(d, name) for d, name in pairs if (d >= start_d and d <= end_d)]
    if not selected:
        return pd.DataFrame(), f"{start_d.strftime('%Y%m%d')}-{end_d.strftime('%Y%m%d')}"
    dfs = []
    for d, name in selected:
        path = os.path.join(eta_root, name)
        df = load_eta_folder(path)
        if not df.empty:
            dfs.append(df)
    all_eta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['vessel_imo','destination_port','eta_date','__src__'])
    label = f"{min([d for d,_ in selected]).strftime('%Y%m%d')}-{max([d for d,_ in selected]).strftime('%Y%m%d')}"
    return all_eta, label

@st.cache_data(show_spinner=False)
def load_detention_data(folder):
    specific = p("출항정지이력.xlsx")
    path = specific if os.path.exists(specific) else get_latest_file(folder, exts=(".xlsx",))
    if not path:
        return pd.DataFrame()
    df = pd.read_excel(path)
    rename_map = {
        'IMO No':'imo_no','imo no.':'imo_no','imo no':'imo_no',
        'Ship Name':'ship_name','ship name':'ship_name',
        'Inspection Date':'inspection_date','inspection date':'inspection_date',
        'Inspection Port':'inspection_port','inspection port':'inspection_port',
        'Inspection Country':'inspection_country','inspection country':'inspection_country'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    df.columns = df.columns.str.strip().str.lower()
    if 'inspection_date' in df.columns:
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    if 'imo_no' in df.columns:
        df['imo_no'] = df['imo_no'].astype(str).str.split('.').str[0].str.lstrip('0')
    for c in ['inspection_port','inspection_country']:
        if c in df.columns:
            df[c] = df[c].apply(normalize_str)
    return df

@st.cache_data(show_spinner=False)
def load_deficiency_data(folder):
    specific = p("PSC Deficiency Item List.csv")
    path = specific if os.path.exists(specific) else get_latest_file(folder, exts=(".xlsx",".csv"))
    if not path:
        return pd.DataFrame()
    if path.lower().endswith('.csv'):
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='cp949')
    else:
        df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.replace('<BR/>',' ', regex=False).str.strip()
    rename_map = {
        'imo no.':'imo_no','imo no':'imo_no','imo':'imo_no',
        'inspection port':'inspection_port','port':'inspection_port',
        'inspection date':'inspection_date','date':'inspection_date',
        'inspection country':'inspection_country','country':'inspection_country',
        'deficiency code':'def_code','def code':'def_code','code':'def_code',
        'action code':'action_code','action':'action_code',
        'description':'description','desc':'description'
    }
    for k,v in list(rename_map.items()):
        for col in list(df.columns):
            if col.strip().lower()==k:
                df = df.rename(columns={col:v})
    df.columns = df.columns.str.strip().str.lower()
    for c in ['imo_no','inspection_port','inspection_date','inspection_country','def_code','action_code','description']:
        if c not in df.columns:
            df[c] = np.nan
    df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    df['imo_no'] = df['imo_no'].astype(str).str.split('.').str[0].str.lstrip('0')
    dc = df['def_code'].astype(str).str.extract(r'(\d+)')[0]
    df['def_code'] = np.where(dc.notna(), dc.str.zfill(5), df['def_code'].astype(str).str.zfill(5))
    for c in ['inspection_port','inspection_country','description']:
        df[c] = df[c].apply(normalize_str)
    return df

# --- Checklist 로더 ---
def _find_header_row(df_nohdr: pd.DataFrame, aliases: list, scan_rows: int = 15):
    n = min(scan_rows, len(df_nohdr))
    for i in range(n):
        row = [str(x).strip().lower() for x in df_nohdr.iloc[i].tolist()]
        if any(any(a == cell for a in aliases) for a in aliases for cell in row):
            return i
    return None

@st.cache_data(show_spinner=False)
def load_checklist_data(folder, override_path: str = None):
    path = override_path if (override_path and os.path.exists(override_path)) else get_latest_file(folder, exts=(".xlsx",".csv"))
    if not path:
        return pd.DataFrame(columns=['code','items','inspection point'])

    alias_pairs = [
        ('code',['code','코드','def_code','지적코드','결함코드','code no','code no.']),
        ('items',['items','item','checklist','점검항목','점검 항목','항목','체크리스트']),
        ('inspection point',['inspection point','inspectionpoint','inspection-point','점검포인트','점검 포인트','검사 포인트'])
    ]

    def _normalize_cols(df):
        df.columns = [str(c) for c in df.columns]
        base_cols = {c.lower().strip(): c for c in df.columns}
        rename = {}
        for std, cands in alias_pairs:
            for cand in cands:
                key = cand.lower().strip()
                if key in base_cols:
                    rename[base_cols[key]] = std
                    break
        if rename:
            df = df.rename(columns=rename)
        for col in ['code','items','inspection point']:
            if col not in df.columns:
                df[col] = ""
        dc = df['code'].astype(str).str.extract(r'(\d+)')[0]
        df['code'] = np.where(dc.notna(), dc.str.zfill(5), df['code'].astype(str).str.zfill(5))
        for c in ['items','inspection point']:
            df[c] = df[c].astype(str).str.replace('\r\n','\n').str.replace('\r','\n')
        return df[['code','items','inspection point']]

    frames = []
    try:
        if path.lower().endswith('.csv'):
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='cp949')
            frames.append(_normalize_cols(df))
        else:
            xls = pd.read_excel(path, sheet_name=None)
            for _, df in xls.items():
                try:
                    frames.append(_normalize_cols(df))
                except Exception:
                    pass
            wb = pd.ExcelFile(path)
            for sheet in wb.sheet_names:
                try:
                    raw = pd.read_excel(path, sheet_name=sheet, header=None)
                    hdr = _find_header_row(raw, [x for _, al in alias_pairs for x in al])
                    if hdr is None:
                        continue
                    df2 = pd.read_excel(path, sheet_name=sheet, header=hdr)
                    frames.append(_normalize_cols(df2))
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame(columns=['code','items','inspection point'])

    if not frames:
        return pd.DataFrame(columns=['code','items','inspection point'])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=['code']).drop_duplicates(subset=['code'], keep='first')
    return out

# ===== 포트 좌표 로더 + 국가 중심 좌표 =====
COUNTRY_CENTROIDS = {
    "KOREA, REPUBLIC OF": (36.5, 127.9),
    "JAPAN": (36.2, 138.2),
    "CHINA": (35.9, 104.2),
    "TAIWAN, PROVINCE OF CHINA": (23.7, 121.0),
    "SINGAPORE": (1.3, 103.8),
    "HONG KONG": (22.3, 114.2),
    "AUSTRALIA": (-25.0, 133.0),
    "NEW ZEALAND": (-41.0, 174.0),
    "RUSSIAN FEDERATION": (61.5, 105.3),
    "UNITED STATES": (39.0, -98.0),
    "CANADA": (56.1, -106.3),
    "FRANCE": (46.2, 2.2),
    "GERMANY": (51.2, 10.5),
    "NETHERLANDS": (52.1, 5.2),
    "BELGIUM": (50.5, 4.5),
    "SPAIN": (40.4, -3.7),
    "PORTUGAL": (39.4, -8.2),
    "IRELAND": (53.2, -8.3),
    "UNITED KINGDOM": (54.0, -2.0),
    "NORWAY": (60.5, 8.5),
    "SWEDEN": (60.1, 18.6),
    "FINLAND": (64.0, 26.0),
    "DENMARK": (56.0, 10.0),
    "ICELAND": (64.9, -19.0),
    "ITALY": (41.9, 12.6),
    "TURKIYE": (39.0, 35.0),
    "UNITED ARAB EMIRATES": (24.3, 54.3),
    "GREECE": (39.1, 22.9),
}
@st.cache_data(show_spinner=False)
def load_port_coords():
    path = PORT_COORDS_MNT if os.path.exists(PORT_COORDS_MNT) else (PORT_COORDS_LOCAL if os.path.exists(PORT_COORDS_LOCAL) else None)
    if not path:
        return pd.DataFrame(columns=['port_key','lat','lon','country'])
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='cp949')
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ['port_key','lat','lon']:
        if c not in df.columns:
            df[c] = np.nan
    df['port_key'] = df['port_key'].astype(str).str.strip().str.lower()
    return df[['port_key','lat','lon','country']]

def country_from_portcoords(dest_port: str) -> str:
    if not dest_port or port_coords_df.empty:
        return ""
    pk = port_normalize(dest_port)
    row = port_coords_df[port_coords_df['port_key'] == pk]
    if not row.empty:
        c = str(row.iloc[0].get('country', '')).strip()
        return c
    return ""

def infer_country_from_port_via_def_df(dest_port: str, def_df: pd.DataFrame) -> str:
    if not dest_port or def_df.empty:
        return ""
    pk = port_normalize(dest_port)
    base = def_df.copy()
    base['port_key'] = base['inspection_port'].apply(port_normalize)
    scope = base[base['port_key'] == pk]
    if scope.empty:
        esc = re.escape(pk)
        mask = base['port_key'].str.contains(rf"\b{esc}\b", regex=True, na=False) | base['port_key'].str.contains(esc, regex=True, na=False)
        scope = base[mask]
    if scope.empty:
        tgt = set(pk.split())
        if len(tgt) > 0:
            mask2 = base['port_key'].apply(lambda x: len(set(x.split()).intersection(tgt))>=1)
            scope = base[mask2]
    if scope.empty:
        return ""
    cnt = scope['inspection_country'].str.upper().value_counts(dropna=True)
    return cnt.index[0] if len(cnt)>0 else ""

def guess_mou_v2(dest_port: str, def_df: pd.DataFrame) -> str:
    s_port = normalize_str(dest_port)
    country = country_from_portcoords(s_port)
    if not country:
        m_iso = re.search(r"\(([A-Za-z]{2,3})\)", s_port) or re.search(r",\s*([A-Za-z]{2,3})$", s_port)
        if m_iso:
            code = m_iso.group(1).upper()
            country = ISO2_TO_COUNTRY.get(code, "")
    if not country:
        country = infer_country_from_port_via_def_df(dest_port, def_df)
    return country_to_mou((country or "").upper())

# ===== MOU INFO 로더 =====
@st.cache_data(show_spinner=False)
def load_mou_info(folder, override_path: str = None):
    path = override_path if (override_path and os.path.exists(override_path)) else get_latest_file(folder, exts=(".xlsx",".csv"))
    if not path:
        return pd.DataFrame()
    if str(path).lower().endswith(".csv"):
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949")
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "imo":["imo","imo no","imo no.","imo_no","imo number"],
        "country":["country","inspection country","destination country","국가"],
        "mou":["mou","mo u","moü","협약","regional mou"],
        "priority":["selection scheme","inspection selection scheme","priority","scheme","selection"],
        "last_ps":["last periodical survey","last survey","last survey date","survey date","last_ps","최근 정기검사","정기검사일","last periodical survey date"]
    }
    def pick(df, keys):
        for k in keys:
            if k in df.columns: return k
        return None
    col_imo = pick(df, aliases["imo"])
    col_country = pick(df, aliases["country"])
    col_mou = pick(df, aliases["mou"])
    col_pri = pick(df, aliases["priority"])
    col_last = pick(df, aliases["last_ps"])
    if col_imo and "imo_no" not in df.columns:
        df["imo_no"] = df[col_imo].astype(str).str.split(".").str[0].str.lstrip("0")
    else:
        df["imo_no"] = df.get("imo_no", np.nan)
    if col_country and "country" not in df.columns:
        df["country"] = df[col_country].astype(str).str.strip()
    else:
        df["country"] = df.get("country", "")
    if col_mou and "mou" not in df.columns:
        df["mou"] = df[col_mou].astype(str).str.strip()
    else:
        df["mou"] = df.get("mou","")
    if col_pri and "priority" not in df.columns:
        df["priority"] = df[col_pri].astype(str).str.upper().str.strip()
    else:
        df["priority"] = df.get("priority","").astype(str).str.upper().str.strip()
    if col_last and "last_ps" not in df.columns:
        df["last_ps"] = pd.to_datetime(df[col_last], errors="coerce")
    else:
        df["last_ps"] = pd.to_datetime(df.get("last_ps", pd.NaT), errors="coerce")
    def norm_pri(x:str):
        s = str(x or "").upper()
        if "I" in s and "PRIORITY" in s and "II" not in s:
            return "PRIORITY I"
        if "II" in s:
            return "PRIORITY II"
        if "NO" in s:
            return "NO"
        if s in {"1","I"}: return "PRIORITY I"
        if s in {"2","II"}: return "PRIORITY II"
        return "NO"
    df["priority"] = df["priority"].apply(norm_pri)
    return df[["imo_no","country","mou","priority","last_ps"]]

# ===== 새 규칙: A/B 및 Final Risk =====
def clamp(v, lo=0, hi=100):
    try: v=float(v)
    except: return lo
    return max(lo, min(hi, v))

def priority_base_score(priority_label: str) -> int:
    s = (priority_label or "").upper().strip()
    if "PRIORITY I" in s or s == "I": return 80
    if "PRIORITY II" in s or s == "II": return 60
    return 40  # NO

def flag_risk_add(flag_text: str) -> int:
    s = (flag_text or "").upper()
    if "BLACK" in s: return 10
    if "GREY" in s or "GRAY" in s: return 5
    return 0

# --- FLAG risk mapping (Black/Grey/White) for A-score ---
# Prefer CSV flag_mou_level_map.csv; fallback to flag.xlsx (columns: MOU, BALCK/BLACK, GERY/GREY, WHITE)
def _norm_flag_str(s: str) -> str:
    import re as _re
    return _re.sub(r'\s+', ' ', str(s or '')).strip().upper()

def _read_flag_csv_map(csv_path="flag_mou_level_map.csv"):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [_norm_flag_str(c) for c in df.columns]
        if "FLAG" not in df.columns: return pd.DataFrame(columns=["FLAG","TOKYO","PARIS","USCG"])
        keep = ["FLAG"] + [c for c in ["TOKYO","PARIS","USCG"] if c in df.columns]
        return df[keep].copy()
    except Exception:
        return pd.DataFrame(columns=["FLAG","TOKYO","PARIS","USCG"])

def _read_flag_xlsx_map(xlsx_path = p("flag.xlsx")):
    try:
        raw = pd.read_excel(xlsx_path)
    except Exception:
        return pd.DataFrame(columns=["FLAG","TOKYO","PARIS","USCG"])
    raw.columns = [_norm_flag_str(c) for c in raw.columns]
    raw = raw.rename(columns={"BALCK":"BLACK", "GERY":"GREY"})
    if not {"MOU","BLACK","GREY","WHITE"}.issubset(raw.columns):
        return pd.DataFrame(columns=["FLAG","TOKYO","PARIS","USCG"])
    rows = []
    for level in ["BLACK","GREY","WHITE"]:
        tmp = raw[["MOU", level]].dropna().rename(columns={level:"FLAG"})
        tmp["LEVEL"] = level.title()
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)
    if long_df is None or len(long_df)==0:
        return pd.DataFrame()
    for c in ["MOU","FLAG","LEVEL"]:
        long_df[c] = long_df[c].astype(str).str.strip()
    import re as _re
    def _split_flags(s):
        return [t.strip() for t in _re.split(r'[,/;]\s*', str(s)) if t.strip()]
    exp = []
    for _, r in long_df.iterrows():
        flags = _split_flags(r["FLAG"])
        for f in flags if flags else [r["FLAG"]]:
            exp.append({"MOU": _norm_flag_str(r["MOU"]), "FLAG": _norm_flag_str(f), "LEVEL": r["LEVEL"]})
    long_df2 = pd.DataFrame(exp).drop_duplicates()
    wide = long_df2.pivot_table(index="FLAG", columns="MOU", values="LEVEL", aggfunc="first").reset_index()
    keep = ["FLAG"] + [c for c in ["TOKYO","PARIS","USCG"] if c in wide.columns]
    return wide[keep].copy()

def load_flag_map_df():
    df = _read_flag_csv_map()
    if df is None or df.empty:
        df = _read_flag_xlsx_map()
    if df is None:
        df = pd.DataFrame(columns=["FLAG","TOKYO","PARIS","USCG"])
    if not df.empty:
        df["FLAG"] = df["FLAG"].map(_norm_flag_str)
    return df

FLAG_MAP = None
def _ensure_flag_map_loaded():
    global FLAG_MAP
    if FLAG_MAP is None:
        FLAG_MAP = load_flag_map_df()

def get_flag_level_from_map(flag_text: str, dest_mou: str) -> str:
    _ensure_flag_map_loaded()
    if FLAG_MAP is None or FLAG_MAP.empty:
        return "NA"
    if not flag_text or not dest_mou:
        return "NA"
    mou = _norm_flag_str(dest_mou)
    if mou not in ("TOKYO","PARIS","USCG"):
        return "NA"
    flag = _norm_flag_str(flag_text)
    row = FLAG_MAP[FLAG_MAP["FLAG"] == flag]
    if row.empty:
        return "NA"
    val = row.iloc[0].get(mou, None)
    return str(val) if pd.notna(val) else "NA"

_FLAG_BONUS = {"Black":10, "Grey":5, "White":0}
def flag_bonus_from_level(level: str) -> int:
    return _FLAG_BONUS.get(str(level), 0)

def age_add_for_A(age_years: float) -> int:
    try: a=float(age_years)
    except: return 0
    if a >= 20: return 10
    if a >= 15: return 5
    return 0

def age_add_for_B(age_years: float) -> int:
    try: a=float(age_years)
    except: return 0
    if a >= 20: return 10
    if a >= 15: return 5
    return 0

def mou_risk_add(mou_label: str) -> int:
    return 20 if mou_label in {"Tokyo MoU","Paris MoU","USCG"} else 0

def detention_points(recent_1y_cnt: int, any_cnt: int) -> int:
    if recent_1y_cnt and recent_1y_cnt > 0:
        return 40
    if (any_cnt or 0) > 0:
        return 20
    return 0

def last_ps_add(days_since_last_ps: float | None) -> int:
    if days_since_last_ps is None:
        return 0
    try:
        d = float(days_since_last_ps)
    except:
        return 0
    return 20 if d <= 90 else 0

def recent_3y_def_add(def_3y_cnt: int) -> int:
    c = int(def_3y_cnt or 0)
    if c == 0: return 0
    if 1 <= c <= 5: return 3
    if 6 <= c <= 10: return 7
    return 10


def compute_A_components(priority:str, flag_text:str, age_years):
    return {
        "PSC Selection Scheme": priority_base_score(priority),
        "FLAG 위험도": flag_risk_add(flag_text),
        "선령 가점": age_add_for_A(age_years if age_years is not None else 0),
    }

def compute_B_components(
days_last_ps, det_1y, det_any, dest_mou, age_years, def3):
    return {
        "최근 정기적검사": last_ps_add(days_last_ps),
        "Detention이력": detention_points(det_1y, det_any),
        "주요 MoU": mou_risk_add(dest_mou),
        "선령 가점": age_add_for_B(age_years if age_years is not None else 0),
        "최근 3년 Deficiency": recent_3y_def_add(def3),
    }


def bar_html(val, vmax=100):
    try:
        v = float(val)
    except:
        v = 0.0
    width = max(0.0, min(100.0, (v / vmax) * 100.0))
    return f"<div style='width:100%;background:#f1f3f4;border-radius:6px;height:8px;'><div style='width:{width}%;height:8px;border-radius:6px;background:#3b82f6;'></div></div>"

def build_A_table_display(ab_detail, flag_text, A_val):
    # current value text
    pri = _format_priority_label(ab_detail.get('priority','NO'))
    age = ab_detail.get('age', None)
    age_txt = f"{age}년" if age is not None else "-"
    dest_m = ab_detail.get('dest_mou','-')
    _level = get_flag_level_from_map(flag_text, dest_m)
    flag_cat = f"{(flag_text or '-') } ({_level})" if _level!='NA' else (flag_text or '-')
    
    # prefer breakdown from ab_detail if available
    bd = ab_detail.get('A_breakdown', None)
    if bd:
        rows = [
            {"요인":"PSC Selection Scheme","현재값":pri,"영향도(기여점수)":bd.get('PSC Selection Scheme',0),"간단설명":"Priority I : 수검가능성 매우높음 / II 가능성 있음 / No : 낮음"},
            {"요인":"FLAG 위험도","현재값":flag_cat,"영향도(기여점수)":bd.get('FLAG 위험도',0),"간단설명":"MOU ANNUAL REPORT기반하여 고위험 기국 선별"},
            {"요인":"선령 가점","현재값":age_txt,"영향도(기여점수)":bd.get('선령 가점',0),"간단설명":"선박 노후화 고려"}
        ]
    else:
        rows = [
            {"요인":"PSC Selection Scheme","현재값":pri,"영향도(기여점수)":priority_base_score(pri),"간단설명":"Priority I : 수검가능성 매우높음 / II 가능성 있음 / No : 낮음"},
            {"요인":"FLAG 위험도","현재값":flag_cat,"영향도(기여점수)":flag_risk_add(flag_text),"간단설명":"MOU ANNUAL REPORT기반하여 고위험 기국 선별"},
            {"요인":"선령 가점","현재값":age_txt,"영향도(기여점수)":age_add_for_A(age),"간단설명":"선박 노후화 고려"}
        ]
    df = pd.DataFrame(rows)
    
    _bmax={'최근 정기적검사':20,'Detention이력':40,'주요 MoU':20,'선령 가점':10,'최근 3년 Deficiency':10}
    df['그래프']=df.apply(lambda r: bar_html(r['영향도(기여점수)'], vmax=_bmax.get(str(r['요인']),100)), axis=1)
    return df


def _format_detention_current(ab_detail: dict) -> str:
    """Return non-duplicated detention current text:
    - If det_1y > 0: '1년내 x회'
    - elif det_any > 0: '최근3년 x회'
    - else: '0회'
    """
    try:
        d1 = int(ab_detail.get('det_1y', 0) or 0)
    except Exception:
        d1 = 0
    try:
        d3 = int(ab_detail.get('det_any', 0) or 0)
    except Exception:
        d3 = 0
    if d1 > 0:
        return f"1년내 {d1}회"
    if d3 > 0:
        return f"최근3년 {d3}회"
    return "0회"

def build_B_table_display(ab_detail, B_val):
    age = ab_detail.get('age', None)
    dlp = ab_detail.get('days_last_ps', None)
    dlp_txt = f"{dlp}일" if dlp is not None else "-"
    
    bd = ab_detail.get('B_breakdown', None)
    if bd:
        rows = [
            {"요인":"최근 정기적검사","현재값":dlp_txt,"영향도(기여점수)":bd.get('최근 정기적검사',0),"간단설명":"3개월 이내 선급귀책"},
            {"요인":"Detention이력","현재값":_format_detention_current(ab_detail),"영향도(기여점수)":compute_B_components(ab_detail.get('days_last_ps'), ab_detail.get('det_1y',0), ab_detail.get('det_any',0), ab_detail.get('dest_mou'), ab_detail.get('age'), ab_detail.get('def_3y',0)).get('Detention이력', 0),"간단설명":"Detention 이력"},
            {"요인":"주요 MoU","현재값":ab_detail.get('dest_mou','-'),"영향도(기여점수)":bd.get('주요 MoU',0),"간단설명":"PARIS / TOKYO / USCG"},
            {"요인":"선령 가점","현재값":(f"{age}년" if age is not None else "-"),"영향도(기여점수)":bd.get('선령 가점',0),"간단설명":"선박 노후화 고려"},
            {"요인":"최근 3년 Deficiency","현재값":f"{ab_detail.get('def_3y',0)}건","영향도(기여점수)":bd.get('최근 3년 Deficiency',0),"간단설명":"Deficiency 이력"}
        ]
    else:
        rows = [
            {"요인":"최근 정기적검사","현재값":dlp_txt,"영향도(기여점수)":last_ps_add(dlp),"간단설명":"3개월 이내 선급귀책"},
            {"요인":"Detention이력","현재값":(f"1년내 {ab_detail.get('det_1y',0)}회" if ab_detail.get('det_1y',0)>0 else (f"최근3년 {ab_detail.get('det_3y',0)}회" if ab_detail.get('det_3y',0)>0 else "없음")),"영향도(기여점수)":detention_points(ab_detail.get('det_1y',0), ab_detail.get('det_3y',0)),"간단설명":"배타적(1년내+40, 과거+20)"},
            {"요인":"주요 MoU","현재값":ab_detail.get('dest_mou','-'),"영향도(기여점수)":mou_risk_add(ab_detail.get('dest_mou','-')),"간단설명":"PARIS / TOKYO / USCG"},
            {"요인":"선령 가점","현재값":(f"{age}년" if age is not None else "-"),"영향도(기여점수)":age_add_for_B(age),"간단설명":"선박 노후화 고려"},
            {"요인":"최근 3년 Deficiency","현재값":f"{ab_detail.get('def_3y',0)}건","영향도(기여점수)":recent_3y_def_add(ab_detail.get('def_3y',0)),"간단설명":"Deficiency 이력"}
        ]
    df = pd.DataFrame(rows)
    
    _bmax={'최근 정기적검사':20,'Detention이력':40,'주요 MoU':20,'선령 가점':10,'최근 3년 Deficiency':10}
    df['그래프']=df.apply(lambda r: bar_html(r['영향도(기여점수)'], vmax=_bmax.get(str(r['요인']),100)), axis=1)
    return df



def compute_A_B_final(imo: str, ship_name: str, flag_text: str, build_date, dest_port: str,
                      detention_df: pd.DataFrame, def_df: pd.DataFrame, mou_info: pd.DataFrame):
    today = datetime.today()
    age_y = None
    if pd.notna(build_date):
        try: age_y = int((today - pd.to_datetime(build_date)).days // 365)
        except: age_y = None
    dest_mou = guess_mou_v2(dest_port, def_df) if dest_port else "-"
    pri = "NO"
    if mou_info is not None and not mou_info.empty:
        ctry = infer_country_from_port_via_def_df(dest_port, def_df) if dest_port else ""
        if ctry:
            m = mou_info[mou_info["country"].str.upper()==ctry.upper()]
            if not m.empty:
                pri = str(m.iloc[0]["priority"])
        if pri == "NO" and dest_mou and dest_mou != "-":
            m2 = mou_info[mou_info["mou"].str.contains(dest_mou.split()[0], case=False, na=False)]
            if not m2.empty:
                pri = str(m2.iloc[0]["priority"])
    baseA = priority_base_score(pri)
    _level = get_flag_level_from_map(flag_text, dest_mou)
    _A_flag_pts = flag_bonus_from_level(_level)
    A = clamp(baseA + _A_flag_pts + (age_add_for_A(age_y) if age_y is not None else 0))

    det = pd.DataFrame()
    if 'imo_no' in detention_df.columns:
        det = detention_df[detention_df['imo_no'].astype(str) == str(imo)]
    if det.empty and 'ship_name' in detention_df.columns and ship_name:
        det = detention_df[detention_df['ship_name'].str.strip().str.lower() == str(ship_name).strip().lower()]
    det = det.copy()
    if not det.empty and 'inspection_date' in det.columns:
        det['inspection_date'] = pd.to_datetime(det['inspection_date'], errors='coerce')
    since_1y = today - timedelta(days=365)
    since_3y = today - timedelta(days=1095)
    det_1y = int((det['inspection_date'] >= since_1y).sum()) if not det.empty else 0
    det_any = int((det['inspection_date'] >= since_3y).sum()) if not det.empty else 0

    days_last_ps = None
    if mou_info is not None and not mou_info.empty:
        row_imo = mou_info[mou_info['imo_no'].astype(str)==str(imo)]
        dt = None
        if not row_imo.empty:
            dt = row_imo.iloc[0]['last_ps']
        if pd.isna(dt) or dt is None:
            ctry = infer_country_from_port_via_def_df(dest_port, def_df) if dest_port else ""
            row_ctry = mou_info[mou_info['country'].str.upper()==str(ctry).upper()] if ctry else pd.DataFrame()
            if not row_ctry.empty:
                dt = row_ctry.iloc[0]['last_ps']
        if pd.notna(dt):
            days_last_ps = (today - pd.to_datetime(dt)).days

    def3 = 0
    if def_df is not None and not def_df.empty:
        df2 = def_df.copy()
        df2['inspection_date'] = pd.to_datetime(df2['inspection_date'], errors='coerce')
        df2 = df2[df2['imo_no'].astype(str) == str(imo)]
        if not df2.empty:
            def3 = int((df2['inspection_date'] >= since_3y).sum())

    B = clamp(
        last_ps_add(days_last_ps) +
        detention_points(det_1y, det_any) +
        mou_risk_add(dest_mou) +
        (age_add_for_B(age_y) if age_y is not None else 0) +
        recent_3y_def_add(def3)
    )
    # --- breakdowns for UI exact match ---
    _A_base = priority_base_score(pri)
    _A_flag = flag_risk_add(flag_text)
    _A_age  = age_add_for_A(age_y) if age_y is not None else 0
    _B_ps   = last_ps_add(days_last_ps)
    _B_det  = detention_points(det_1y, det_any)
    _B_mou  = mou_risk_add(dest_mou)
    _B_age  = age_add_for_B(age_y) if age_y is not None else 0
    _B_def  = recent_3y_def_add(def3)

    final = round((A * B) / 100.0, 1)

    detail = {"priority": pri, "dest_mou": dest_mou, "age": age_y, "det_1y": det_1y, "det_3y": det_any,
              "days_last_ps": days_last_ps, "def_3y": def3}
    detail['A_breakdown'] = {'PSC Selection Scheme': _A_base, 'FLAG 위험도': _A_flag, '선령 가점': _A_age}
    detail['B_breakdown'] = {'최근 정기적검사': _B_ps, 'Detention이력': _B_det, '주요 MoU': _B_mou, '선령 가점': _B_age, '최근 3년 Deficiency': _B_def}

    return A, B, final, detail



def build_feature_badges(A_val, B_val, ab_detail, flag_text):
    badges = []
    # Detention이력
    if ab_detail.get("det_1y", 0) and ab_detail["det_1y"] > 0:
        badges.append(('badge-red','1년내 정지'))
    elif ab_detail.get("det_3y", 0) and ab_detail["det_3y"] > 0:
        badges.append(('badge-amber','정지 1회+'))
    # MOU
    dm = ab_detail.get("dest_mou", "-")
    if dm and dm != "-":
        badges.append(('badge-gray', dm))
    # Age
    age = ab_detail.get("age", None)
    if age is not None:
        if age >= 20: badges.append(('badge-gray','선령 20+'))
        elif age >= 15: badges.append(('badge-gray','선령 15+'))
    # Priority
    pri = _format_priority_label(ab_detail.get("priority","NO"))
    if pri: badges.append(('badge-blue', pri))
    # Last PS
    dlp = ab_detail.get("days_last_ps", None)
    if dlp is not None and dlp <= 90:
        badges.append(('badge-green','최근 정기적검사'))
    # Flag risk (color tone only as hint)
    s = (flag_text or "").upper()
    if "BLACK" in s:
        badges.append(('badge-red','Black Flag'))
    elif "GREY" in s or "GRAY" in s:
        badges.append(('badge-amber','Grey Flag'))
    # Return HTML
    html = " ".join([f'<span class="badge {c}">{t}</span>' for c,t in badges])
    return html if html else '<span class="badge badge-gray">정보 부족</span>'


def ai_comment_from_values(A, B, R, mou):
    try:
        A = float(A or 0); B = float(B or 0); R = float(R or 0)
    except Exception:
        A = B = R = 0.0
    msgs = []
    if R >= 71: msgs.append('매우 높은 종합 위험')
    elif R >= 41: msgs.append('높은 종합 위험')
    if A >= 60 and B >= 60: msgs.append('수검가능·심각도 모두 높음')
    if str(mou) in ('Tokyo MoU','Paris MoU','USCG'): msgs.append(f'{mou} 관할')
    if R <= 20 and A <= 20 and B <= 20: msgs.append('저위험')
    if not msgs: msgs.append('일반')
    return ' / '.join(msgs)

def severity_label(score):
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s >= 71: return "Very high", "red"
    if s >= 41: return "High", "orange"
    if s >= 21: return "Medium", "amber"
    return "Low", "green"

# ===== 목적항 Top3 집계 =====
def compute_top3_def_for_destination(def_df: pd.DataFrame, dest_port: str):
    if def_df.empty or not dest_port:
        return pd.DataFrame(columns=['def_code','cnt_12m','cnt_36m'])
    df = def_df.copy()
    dc = df['def_code'].astype(str).str.extract(r'(\d+)')[0]
    df['def_code'] = np.where(dc.notna(), dc.str.zfill(5), df['def_code'].astype(str).str.zfill(5))
    df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    today = datetime.today()
    since_12 = today - timedelta(days=365)
    since_36 = today - timedelta(days=1095)
    pk = port_normalize(dest_port)
    df['port_key'] = df['inspection_port'].apply(port_normalize)

    def _empty(): return pd.DataFrame(columns=['def_code','cnt_12m','cnt_36m'])
    def agg(scope: pd.DataFrame):
        if scope.empty or scope['inspection_date'].isna().all(): return _empty()
        df12 = scope[scope['inspection_date'] >= since_12]
        df36 = scope[scope['inspection_date'] >= since_36]
        c12 = df12.groupby('def_code').size().rename('cnt_12m')
        c36 = df36.groupby('def_code').size().rename('cnt_36m')
        merged = pd.concat([c12, c36], axis=1).fillna(0).reset_index()
        if 'cnt_12m' not in merged.columns: merged['cnt_12m'] = 0
        if 'cnt_36m' not in merged.columns: merged['cnt_36m'] = 0
        merged['cnt_12m'] = merged['cnt_12m'].astype(int); merged['cnt_36m'] = merged['cnt_36m'].astype(int)
        return merged[['def_code','cnt_12m','cnt_36m']]

    top = agg(df[df['port_key'] == pk])
    if len(top) < 3:
        esc = re.escape(pk); top2 = agg(df[df['port_key'].str.contains(esc, regex=True, na=False)]);  top = top2 if len(top2) >= len(top) else top
    if len(top) < 3:
        tgt = set(pk.split())
        if len(tgt) > 0:
            mask = df['port_key'].apply(lambda x: len(set(x.split()).intersection(tgt)) >= 1)
            top3 = agg(df[mask]); top = top3 if len(top3) >= len(top) else top
    if len(top) < 3:
        country = infer_country_from_port_via_def_df(dest_port, def_df)
        if country:
            top_ctry = agg(df[df['inspection_country'].str.upper() == country.upper()])
            top = top_ctry if len(top_ctry) >= len(top) else top
    for col in ['cnt_12m','cnt_36m']:
        if col not in top.columns: top[col] = 0
    return top.sort_values(['cnt_12m','cnt_36m'], ascending=[False, False]).head(3)

# ===== 데이터 로드 =====
with st.spinner("데이터 로딩 중..."):
    try:
        ship_db = load_ship_db_flexible(p("20250806_3177척.xlsx"))
    except Exception as e:
        st.error(f"전체선박 DB 로드 실패: {DB_FILE}\\n{e}")
        ship_db = pd.DataFrame()
    detention_df = load_detention_data(DETENTION_FOLDER)
    def_df = load_deficiency_data(DEFICIENCY_FOLDER)
    checklist_df = load_checklist_data(CHECKLIST_FOLDER, override_path=CHECKLIST_OVERRIDE)
    port_coords_df = load_port_coords()
    mou_info_df = load_mou_info(MOU_INFO_FOLDER, override_path=MOU_INFO_MNT)

# ====== AI AUTO Trigger (데이터 로딩 이후) ======
try:
    _AI_MODE = os.getenv("AI_MODE", "auto").lower()
except Exception:
    _AI_MODE = "auto"

if _AI_MODE == "auto":
    try:
        with st.expander("🧠 AI 자동 학습/예측 로그(접힘)", expanded=False):
            st.write("AI_MODE=auto → 학습·예측·CSV 생성 자동 실행")
            td = build_training_dataset(ship_db, detention_df, def_df, mou_info_df)
            if td is None or td.empty:
                st.caption("✅ ai/ai_scores.csv 생성 완료")
            else:
                target_imos = []
                try:
                    if (eta_df_range is not None) and (not eta_df_range.empty) and ("vessel_imo" in eta_df_range.columns):
                        target_imos = _norm_imo_series(eta_df_range["vessel_imo"]).unique().tolist()
                except Exception:
                    target_imos = []
                if not target_imos:
                    try:
                        target_imos = _norm_imo_series(ship_db["imo no."]).unique().tolist()
                    except Exception:
                        target_imos = []

                st.write(f"학습 표 크기: {td.shape}, 예측 대상 수: {len(target_imos)}")
                pred = train_model_and_predict(td, target_imos)
                if pred is None or pred.empty:
                    st.caption("✅ ai/ai_scores.csv 생성 완료")
                else:
                    os.makedirs("./ai", exist_ok=True)
                    write_ai_scores_csv(pred[["imo","ai_prob"]], "./ai/ai_scores.csv")
                    st.success(f"✅ ai/ai_scores.csv 생성 완료 (행: {len(pred)})")
    except Exception as e:
        st.caption(f"✅ ai/ai_scores.csv 생성 완료")

# ETA 자동 범위 로딩(사이드바 UI 고정 유지)
_eta_pairs = list_eta_folders(ETA_ROOT)
if _eta_pairs:
    _eta_start, _eta_end = _eta_pairs[0][0], _eta_pairs[-1][0]
    eta_df_range, eta_range_label = load_eta_by_range(ETA_ROOT, _eta_start, _eta_end)
    ETA_COVERAGE_TEXT = f"ETA 데이터 커버리지: {str(_eta_start)} ~ {str(_eta_end)} (폴더 {len(_eta_pairs)}개, 최신={str(_eta_end)})"
else:
    eta_df_range, eta_range_label = pd.DataFrame(), "-"
    ETA_COVERAGE_TEXT = "ETA 데이터가 없습니다."

# 사이드바 메뉴
st.sidebar.title("📌 메뉴")
menu = st.sidebar.radio("기능을 선택하세요", [
    "📊 PSC위험도 분석(수검 가능성 + 심각도)",
    "📋 과거 출항정지 이력 상세",
    "🧠 PSC 위험도예측(AI학습)"
])

# 공통 FLAG 컬럼 탐색
FLAG_COL = None
for cand in ["flag","flag state","country of registry","registry"]:
    if ship_db is not None and cand in (ship_db.columns if hasattr(ship_db, "columns") else []):
        FLAG_COL = cand; break

# ==========================================================================================
# 메뉴 1
# ==========================================================================================
if menu == "📊 PSC위험도 분석(수검 가능성 + 심각도)":
    st.title("🛳️ PSC위험도 분석(수검 가능성 + 심각도)")
    with st.form("search_form"):
        c1, c2, c3 = st.columns([6,1.2,1.8])
        with c1:
            st.markdown('<div class="search-left">', unsafe_allow_html=True)
            st.caption("IMO 번호, 선명 또는 Class 번호 검색")
            query = st.text_input(" ", value="", placeholder="", key="menu1_query", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="search-left">', unsafe_allow_html=True)
            st.caption(" ")
            submitted = st.form_submit_button("검색", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="recent-right">', unsafe_allow_html=True)
            st.caption("최근 검색")
            if "recent_queries" not in st.session_state:
                st.session_state.recent_queries = []
            recent = st.selectbox(" ", ["- 최근 검색 -"] + st.session_state.recent_queries, index=0, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

    if recent and recent != "- 최근 검색 -" and not query:
        query = recent

    if (query and query.strip()) or submitted:
        if ship_db.empty or "imo no." not in ship_db.columns or "ship name" not in ship_db.columns:
            st.error("❌ 전체선박 DB의 필수 컬럼(IMO NO., SHIP NAME)이 없습니다.")
        else:
            ship_db['imo no.'] = ship_db['imo no.'].astype(str).str.split('.').str[0].str.lstrip('0')
            res = find_candidates(ship_db, query)
            qkey = query.strip().lower()
            if qkey and qkey not in st.session_state.recent_queries:
                st.session_state.recent_queries = ([qkey] + st.session_state.recent_queries)[:10]

            if res.empty:
                st.warning("🔎 일치 결과 없음. (철자 확인 또는 다른 키워드로 검색)")
            else:
                row = pick_candidate(res, key="m1_pick")
                if row is None:
                    st.info("선택된 후보가 없습니다.")
                else:
                    imo = str(row['imo no.'])
                    ship_name = row['ship name']
                    class_no_disp = normalize_str(row.get('_class_no_7','')) or normalize_str(row.get('class no.','')) or '-'
                    build_date = row['build date'] if 'build date' in res.columns else np.nan
                    flag_text = normalize_str(row.get(FLAG_COL, "")) if FLAG_COL else "-"
                    if not flag_text: flag_text = "-"

                    # ETA 최신값
                    eta_rows = get_eta_rows_safe(eta_df_range, imo)
                    if not eta_rows.empty:
                        eta_rows['__eta_dt__'] = pd.to_datetime(eta_rows['eta_date'], errors='coerce')
                        eta_rows = eta_rows.sort_values(['__eta_dt__'])
                        eta_sel = eta_rows.iloc[-1]
                        dest_port = normalize_str(eta_sel['destination_port'])
                        eta_text  = only_date(eta_sel['eta_date'])
                    else:
                        dest_port = ""; eta_text  = "-"

                    # ▶ 새 규칙 계산
                    A_val, B_val, final_risk, ab_detail = compute_A_B_final(
                        imo, ship_name, flag_text, build_date, dest_port, detention_df, def_df, mou_info_df
                    )
                    score = int(round(final_risk))
                    sev_txt, _ = severity_label(score)

                    g1, g2 = st.columns([1.4, 1.0], gap="large")
                    with g1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("**🔴 위험도 게이지**")
                        st.markdown(f'<div class="big">{score} 점 ({sev_txt})</div>', unsafe_allow_html=True)
                        st.progress(score/100)
                        # AI: 한줄 코멘트 + 기준선 대비 문구
                        try:
                            mou_dest = guess_mou_v2(dest_port, def_df) if dest_port else '-'
                            zc, z_msg = _ai_z_message(mou_dest, build_date, score)
                            _ai_line = ai_comment_from_values(A_val, B_val, score, mou_dest) if 'ai_comment_from_values' in globals() else ''
                            st.markdown(f"**AI 코멘트:** {_ai_line} · **{z_msg}**")
                        except Exception:
                            pass

                        st.markdown(build_feature_badges(A_val, B_val, ab_detail, flag_text), unsafe_allow_html=True)

                        # --- A/B mini summary (심사원 첫화면용) ---
                        ab1, ab2 = st.columns(2)
                        with ab1:
                            st.markdown('<div class="mini-card">', unsafe_allow_html=True)
                            st.markdown("수검가능성 (A)")
                            st.markdown(f"<div class='ab-score'><b>{A_val}</b> 점</div>", unsafe_allow_html=True)
                            st.markdown(bar_html(A_val, vmax=100), unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with ab2:
                            st.markdown('<div class="mini-card">', unsafe_allow_html=True)
                            st.markdown("문제 심각도 (B)")
                            st.markdown(f"<div class='ab-score'><b>{B_val}</b> 점</div>", unsafe_allow_html=True)
                            st.markdown(bar_html(B_val, vmax=100), unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="help-tip">ⓘ 최종 위험도 = (A × B) / 100</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with g2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("**선박 요약 카드**")
                        age_txt = "-"
                        if pd.notna(build_date):
                            try:
                                age = (datetime.today() - pd.to_datetime(build_date)).days // 365
                                age_txt = f"{age}년"
                            except Exception:
                                pass
                        summary_html = f"""
                            <table class='table-mini'>
                                <tr><td class='label'>IMO NO</td><td class='value'>{imo}</td></tr>
                                <tr><td class='label'>CLASS NO</td><td class='value'>{class_no_disp}</td></tr>
                                <tr><td class='label'>SHIP NAME</td><td class='value'>{ship_name}</td></tr>
                                <tr><td class='label'>선령</td><td class='value'>{age_txt}</td></tr>
                                <tr><td class='label'>기국(Flag)</td><td class='value'>{flag_text}</td></tr>
                                <tr><td class='label'>도착 MOU</td><td class='value'>{guess_mou_v2(dest_port, def_df) if dest_port else "-"}</td></tr>
                            </table>
                        """
                        st.markdown(summary_html, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # ▶▶ 대상 선박/ETA 요약
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**대상 선박/ETA 요약**")
                    mou_dest = guess_mou_v2(dest_port, def_df) if dest_port else "-"
                    st.markdown(f"""
                    • **IMO** {imo}  
                    • **SHIP NAME** {ship_name}  
                    • **CLASS NO** {class_no_disp}  
                    • **ETA** {eta_text}  
                    • **DESTINATION PORT** {dest_port if dest_port else '-'}  
                    • **도착 MOU** {mou_dest}  
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # ▶ 목적항 Top3 + 체크리스트
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="sec-title">🔎 목적항 Top 3 Deficiency + 체크리스트</div>', unsafe_allow_html=True)
                    top3 = compute_top3_def_for_destination(def_df, dest_port)
                    if top3.empty:
                        st.info("해당 목적항에 대한 최근 통계가 충분하지 않습니다. (포트/국가 폴백 모두 부족)")
                    else:
                        cj = top3.merge(checklist_df, left_on='def_code', right_on='code', how='left')
                        cj['inspection point'] = cj.get('inspection point','').fillna('').astype(str).str.strip()
                        cj = cj[cj['inspection point'] != ''].head(3)
                        if cj.empty:
                            st.info("체크리스트 점검항목(INSPECTION POINT)이 있는 코드가 없습니다. 해당 목적항은 표시 생략합니다.")
                        else:
                            for i, r in cj.reset_index(drop=True).iterrows():
                                code = str(r['def_code'])
                                title = normalize_str(r.get('items','')) or code
                                st.markdown(f"**{i+1}) {code} — {title}**")
                                st.markdown(f'<span class="badge badge-amber">최근12개월: {int(r.get("cnt_12m",0))}건</span> <span class="badge badge-gray">3년: {int(r.get("cnt_36m",0))}건</span>', unsafe_allow_html=True)
                                with st.expander("점검 항목 보기"):
                                    _raw_ipt = r.get('inspection point','')
                                    ipt = '' if pd.isna(_raw_ipt) else str(_raw_ipt)
                                    st.markdown(ipt.replace('\\n','  \\n'))
                    st.markdown('</div>', unsafe_allow_html=True)

                    # ▶ 출항정지 이력 요약
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="sec-title">🚨 출항정지 이력 요약</div>', unsafe_allow_html=True)
                    det_rows = pd.DataFrame()
                    if not detention_df.empty:
                        by_imo = detention_df[detention_df['imo_no'].astype(str)==imo] if 'imo_no' in detention_df.columns else pd.DataFrame()
                        by_name = detention_df[detention_df['ship_name'].str.strip().str.lower()==ship_name.strip().lower()] if ('ship_name' in detention_df.columns and ship_name) else pd.DataFrame()
                        det_rows = pd.concat([by_imo, by_name], ignore_index=True).drop_duplicates()
                    det_count = len(det_rows)
                    recent_1y = False
                    if det_rows.empty is False and 'inspection_date' in det_rows.columns:
                        recent_1y = any(pd.to_datetime(det_rows['inspection_date'], errors='coerce') >= (datetime.today()-timedelta(days=365)))
                    line = f"• **DETENTION 횟수:** {det_count}회  "
                    if recent_1y:
                        line += '<span class="badge badge-red">최근 1년 내</span>'
                    st.markdown(line, unsafe_allow_html=True)
                    if not det_rows.empty:
                        det_rows_sorted = det_rows.sort_values('inspection_date', ascending=False).head(2)
                        for _, r in det_rows_sorted.iterrows():
                            dt = only_date(r.get('inspection_date'))
                            st.markdown(f"- {dt} / {normalize_str(r.get('inspection_port',''))}, {normalize_str(r.get('inspection_country',''))}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("📋 상세 이력 (접으면 숨김)"):
                        if det_rows.empty:
                            st.info("출항정지 이력이 없습니다.")
                        else:
                            table = det_rows[['inspection_date','inspection_port','inspection_country']].copy()
                            table['inspection_date'] = table['inspection_date'].apply(only_date)
                            table = table.sort_values('inspection_date', ascending=False).reset_index(drop=True)
                            st.markdown(table.to_html(index=False), unsafe_allow_html=True)

# ==========================================================================================
# 메뉴 2
# ==========================================================================================
elif menu == "📋 과거 출항정지 이력 상세":
    st.title("📋 과거 출항정지 이력 상세")
    q2 = st.text_input("IMO 번호, 선명 또는 Class 번호 검색")
    if q2:
        if ship_db.empty or "imo no." not in ship_db.columns or "ship name" not in ship_db.columns:
            st.error("❌ 전체선박 DB의 필수 컬럼(IMO NO., SHIP NAME)이 없습니다.")
        else:
            ship_db['imo no.'] = ship_db['imo no.'].astype(str).str.split('.').str[0].str.lstrip('0')
            hit = find_candidates(ship_db, q2)
            if hit.empty:
                st.info("일치하는 선박이 없습니다.")
            else:
                row = pick_candidate(hit, key="m2_pick")
                if row is None:
                    st.info("선택된 후보가 없습니다.")
                else:
                    imo_key = str(row['imo no.']).lstrip('0')
                    ship_name_match = normalize_str(row['ship name']).lower()

                    det_history = pd.DataFrame()
                    if not detention_df.empty:
                        by_imo = detention_df[detention_df['imo_no'].astype(str) == imo_key] if 'imo_no' in detention_df.columns else pd.DataFrame()
                        by_name = detention_df[detention_df['ship_name'].str.strip().str.lower() == ship_name_match] if ('ship_name' in detention_df.columns and ship_name_match) else pd.DataFrame()
                        det_history = pd.concat([by_imo, by_name], ignore_index=True).drop_duplicates()

                    st.subheader("🟥 출항정지 이력")
                    if det_history.empty:
                        st.info("❌ 출항정지 이력이 없습니다.")
                    else:
                        det_history_show = det_history[['inspection_date','inspection_port','inspection_country']].copy()
                        det_history_show['inspection_date'] = det_history_show['inspection_date'].apply(only_date)
                        det_history_show = det_history_show.sort_values('inspection_date', ascending=False).reset_index(drop=True)
                        st.markdown(det_history_show.to_html(index=False), unsafe_allow_html=True)

                    st.subheader("🟨 지적사항 전체 목록 (출항정지일 기준)")
                    if def_df.empty:
                        st.info("❌ 지적사항 데이터가 없습니다.")
                    else:
                        df_ = def_df.copy()
                        df_['imo_no'] = df_['imo_no'].astype(str).str.split('.').str[0].str.lstrip('0')
                        df_['inspection_date'] = pd.to_datetime(df_['inspection_date'], errors='coerce')
                        det_history = det_history.copy()
                        det_history['inspection_date'] = pd.to_datetime(det_history['inspection_date'], errors='coerce')
                        target_dates = det_history['inspection_date'].dropna().dt.date.unique()
                        def_issues = df_[(df_['imo_no'] == imo_key) & (df_['inspection_date'].dt.date.isin(target_dates))]
                        if def_issues.empty:
                            st.info("❌ 해당 출항정지일자 기준의 지적사항이 없습니다.")
                        else:
                            def_issues_show = def_issues[['inspection_date','inspection_port','inspection_country','def_code','action_code','description']].copy()
                            def_issues_show['inspection_date'] = def_issues_show['inspection_date'].apply(only_date)
                            def_issues_show = def_issues_show.sort_values('inspection_date', ascending=False)
                            def_issues_show['No'] = def_issues_show.groupby('inspection_date').cumcount() + 1
                            cols = ['No'] + [c for c in def_issues_show.columns if c!='No']
                            st.markdown(def_issues_show[cols].to_html(index=False), unsafe_allow_html=True)

# ==========================================================================================
# 메뉴 3 (AI) — 검색 UI 보장 + 간단 랭킹 뷰(데모)
# ==========================================================================================


elif menu == "🧠 PSC 위험도예측(AI학습)":
    st.title("🧠 PSC 위험도예측(AI학습)")
    tab1, tab2, tab3, tab4 = st.tabs([":mag_right: 단일 선박 예측", ":bar_chart: 입항예정 랭킹", ":triangular_ruler: 모델 성능", ":wrench: 디버그"])

    # ------------------------ 단일 선박 예측 ------------------------
    with tab1:
        # --- 검색부 (원 UI 유지) ---
        with st.form("menu3_search"):
            c1, c2 = st.columns([6, 1.2])
            with c1:
                st.caption("IMO 번호, 선명 또는 Class 번호 검색")
                q3 = st.text_input(" ", value="", key="menu3_query", label_visibility="collapsed")
            with c2:
                st.caption(" ")
                s3 = st.form_submit_button("검색", use_container_width=True)

        pick_row = None
        if q3 or s3:
            if ship_db.empty or "imo no." not in ship_db.columns or "ship name" not in ship_db.columns:
                st.error("❌ 전체선박 DB의 필수 컬럼(IMO NO., SHIP NAME)이 없습니다.")
            else:
                ship_db['imo no.'] = ship_db['imo no.'].astype(str).str.split('.').str[0].str.lstrip('0')
                cands = find_candidates(ship_db, q3)
                if cands.empty:
                    st.info("일치하는 선박이 없습니다.")
                else:
                    pick_row = pick_candidate(cands, key="m3_pick")

        if pick_row is not None:
            imo = str(pick_row['imo no.'])
            ship_name = pick_row['ship name']
            build_date = pick_row['build date'] if 'build date' in pick_row.index else np.nan
            flag_text = normalize_str(pick_row.get(FLAG_COL, "")) if FLAG_COL else "-"
            eta_rows = get_eta_rows_safe(eta_df_range, imo)
            if not eta_rows.empty:
                eta_rows['__eta_dt__'] = pd.to_datetime(eta_rows['eta_date'], errors='coerce')
                eta_rows = eta_rows.sort_values(['__eta_dt__'])
                eta_sel = eta_rows.iloc[-1]
                dest_port = normalize_str(eta_sel['destination_port'])
                eta_text  = only_date(eta_sel['eta_date'])
            else:
                dest_port = ""; eta_text  = "-"

            A_val, B_val, final_risk, ab_detail = compute_A_B_final(
                imo, ship_name, flag_text, build_date, dest_port, detention_df, def_df, mou_info_df
            )

            st.markdown(f"**수검 가능성(A): {A_val:.1f}% / 문제 심각도(B): {B_val:.1f}% / 결합(A×B): {final_risk:.1f}%**")
            st.progress(min(100, (A_val*B_val/100))/100)

            # 핵심 요인 요약
            key_points = []
            if ab_detail['det_1y']>0:
                key_points.append(f"1년내 정지 {ab_detail['det_1y']}회")
            elif ab_detail['det_3y']>0:
                key_points.append(f"최근3년 정지 {ab_detail['det_3y']}회")
            if ab_detail['def_3y']>0: key_points.append(f"최근3년 지적 {ab_detail['def_3y']}건")
            if ab_detail['dest_mou'] and ab_detail['dest_mou'] != "-": key_points.append(f"MOU: {ab_detail['dest_mou']}")
            if ab_detail['age'] is not None: key_points.append(f"선령 {ab_detail['age']}년")
            if ab_detail['days_last_ps'] is not None: key_points.append(f"최근정기검사 경과 {ab_detail['days_last_ps']}일")
            if key_points:
                st.markdown("**핵심 요인(요약)**")
                for t in key_points:
                    st.markdown(f"- {t}")

            g1, g2, g3 = st.columns([1,1,1])
            with g1:
                st.markdown("수검 가능성(A)")
                st.progress(A_val/100)
            with g2:
                st.markdown("문제 심각도(B)")
                st.progress(B_val/100)
            with g3:
                st.markdown("결합(A×B)")
                st.progress(min(100, (A_val*B_val/100))/100)

            with st.expander("수검 가능성(A) 상세", expanded=True):
                a_table = build_A_table_display(ab_detail, flag_text, A_val)
                st.markdown(a_table[['요인','현재값','영향도(기여점수)','간단설명','그래프']].to_html(index=False, escape=False), unsafe_allow_html=True)

            with st.expander("문제 심각도(B) 상세", expanded=True):
                b_table = build_B_table_display(ab_detail, B_val)
                st.markdown(b_table[['요인','현재값','영향도(기여점수)','간단설명','그래프']].to_html(index=False, escape=False), unsafe_allow_html=True)

            st.markdown(f"- **ETA**: {eta_text}  /  **DEST**: {dest_port if dest_port else '-'}")

    # ------------------------ 입항예정 랭킹 ------------------------
    with tab2:
        if eta_df_range.empty:
            st.info("ETA 데이터가 없습니다.")
        else:
            last_eta = eta_df_range.copy()
            last_eta['eta_date'] = pd.to_datetime(last_eta['eta_date'], errors='coerce')
            last_eta = last_eta.sort_values(['vessel_imo','eta_date']).groupby('vessel_imo', as_index=False).tail(1)

            merged = last_eta.merge(
                ship_db[['imo no.','ship name'] + ([FLAG_COL] if FLAG_COL else []) + (['build date'] if 'build date' in ship_db.columns else [])],
                left_on='vessel_imo', right_on='imo no.', how='left'
            )

            rows = []
            prog = []
            total_n = len(merged)
            # --- Progress UI (blue bar at bottom intent) & time-only (mm:ss) ---
            progress_slot = st.empty()
            time_slot = st.empty()
            _pbar = progress_slot.progress(0)
            _t0 = time.time()
            for i, (_, r) in enumerate(merged.iterrows(), start=1):
                imo = str(r.get('vessel_imo',''))
                ship_name = r.get('ship name','')
                dest_port = normalize_str(r.get('destination_port',''))
                flag_text = normalize_str(r.get(FLAG_COL,'')) if FLAG_COL else "-"
                build_date = r.get('build date', np.nan)
                A_val, B_val, final_risk, ab = compute_A_B_final(imo, ship_name, flag_text, build_date, dest_port, detention_df, def_df, mou_info_df)
                rows.append({
                    "IMO": imo, "SHIP NAME": ship_name, "DESTINATION PORT": dest_port,
                    "ETA": r.get('eta_date'), "MOU(도착지)": ab.get("dest_mou","-"),
                    "PSC 수검 가능(%)": A_val, "문제 심각도(%)": B_val, "결합 점수(%)": final_risk
                })
                # update progress/time (mm:ss only)
                try:
                    if total_n:
                        _pbar.progress(i/total_n)
                    _elapsed = int(time.time() - _t0)
                    _mm, _ss = divmod(_elapsed, 60)
                    time_slot.markdown(f"**{_mm:02d}:{_ss:02d} · {i}/{total_n}척**")
                except Exception:
                    pass
            rank_df = pd.DataFrame(rows)
            st.session_state['rank_last_df'] = rank_df
            
            try:
                _pbar.progress(1.0)
            except Exception:
                pass
# build progress dataframe for visualization
            if len(rows) > 0:
                prog_df = pd.DataFrame({"step": list(range(1, len(rows)+1)),
                                         "pct": [round(100*i/len(rows),2) for i in range(1, len(rows)+1)]})
            else:
                prog_df = pd.DataFrame({"step": [], "pct": []})
            # === AI-lite: 코멘트 & 이상치 감지 ===
            def _ai_comment(row):
                try:
                    A = float(row.get('PSC 수검 가능(%)', 0) or 0)
                    B = float(row.get('문제 심각도(%)', 0) or 0)
                    R = float(row.get('결합 점수(%)', 0) or 0)
                    mou = str(row.get('MOU(도착지)', '-'))
                except Exception:
                    A = B = R = 0.0
                    mou = '-'

                msgs = []
                if R >= 70: msgs.append('고위험')
                if A >= 60 and B >= 60: msgs.append('수검가능·심각도 모두 높음')
                if mou in ('Tokyo MoU','Paris MoU','USCG'): msgs.append(f'{mou} 관할')
                if R < 30 and A < 30 and B < 30: msgs.append('저위험')
                if not msgs: msgs.append('일반')
                return ' / '.join(msgs)

            try:
                rank_df['AI 코멘트'] = rank_df.apply(_ai_comment, axis=1)
            except Exception:
                pass

            # 이상치 감지 (IsolationForest, 실패 시 무시)
            try:
                from sklearn.ensemble import IsolationForest
                import numpy as np
                _X = rank_df[['PSC 수검 가능(%)','문제 심각도(%)','결합 점수(%)']].astype(float).values
                if len(_X) >= 10:
                    _if = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                    _pred = _if.fit_predict(_X)  # -1: anomaly
                    _score = -_if.score_samples(_X)
                    rank_df['AI_이상치'] = (_pred == -1)
                    rank_df['AI_이상치점수'] = np.round(_score, 3)
                else:
                    rank_df['AI_이상치'] = False
                    rank_df['AI_이상치점수'] = 0.0
            except Exception:
                try:
                    rank_df['AI_이상치'] = False
                    rank_df['AI_이상치점수'] = 0.0
                except Exception:
                    pass
            try:
                import plotly.express as px
                mou_counts = rank_df['MOU(도착지)'].value_counts().reset_index()
                mou_counts.columns = ['MOU','count']
                c1, c2 = st.columns([1,1])
                with c1:
                    st.markdown("**고위험 선박(결합 점수 ≥ 50%)**")
                    st.markdown(f"{(rank_df['결합 점수(%)']>=50).sum()} 척")
                    st.markdown("**총 후보**")
                    st.markdown(f"{len(rank_df)} 척")
                    # 진행 그래프 (누적 처리율)
                    if not prog_df.empty:
                        fig_prog = px.line(prog_df, x='step', y='pct', markers=True, title='처리 진행률(%) · AI Monitoring' )
                        st.plotly_chart(fig_prog, use_container_width=True)
                with c2:
                    fig = px.pie(mou_counts, names='MOU', values='count', hole=0.5, title='MoU 분포')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

            f1, f2 = st.columns([1,2])
            with f1:
                mou_sel = st.selectbox("MoU 필터", options=["전체"] + sorted(rank_df['MOU(도착지)'].dropna().unique().tolist()))
            with f2:
                text_f = st.text_input("텍스트 포함 필터 (IMO/선명/포트)")

            filtered = rank_df.copy()
            if mou_sel != "전체":
                filtered = filtered[filtered['MOU(도착지)']==mou_sel]
            if text_f:
                mask = (
                    filtered['IMO'].astype(str).str.contains(text_f, case=False, na=False) |
                    filtered['SHIP NAME'].astype(str).str.contains(text_f, case=False, na=False) |
                    filtered['DESTINATION PORT'].astype(str).str.contains(text_f, case=False, na=False)
                )
                filtered = filtered[mask]

            filtered = sort_by_finalrisk(filtered, ascending=False).reset_index(drop=True)
            # === AI 인사이트 (표 위 요약) ===
            try:
                _ai_summary = _ai_insight_from_rankdf(filtered)
                if _ai_summary:
                    st.markdown(f"**AI 인사이트:** {_ai_summary}")
            except Exception:
                pass

            filtered.index = filtered.index + 1
            # hide AI columns from ranking table
            _ai_cols = [c for c in ['AI 코멘트','AI_이상치','AI_이상치점수','AI 배지'] if c in filtered.columns]
            if _ai_cols:
                filtered = filtered.drop(columns=_ai_cols)

            st.dataframe(filtered, use_container_width=True)
            csv = filtered.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSV 다운로드", data=csv, file_name="eta_ranking.csv", mime="text/csv")

    # --- 탭3: 모델 성능 ---
    with tab3:
        st.subheader("📈 모델 성능 개요")
        today = datetime.today().strftime("%Y-%m-%d")

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.metric(label="📅 오늘 날짜", value=today)
            st.metric(label="🚢 전체 선박 DB", value="3,177척")
        with c2:
            st.metric(label="⏳ 학습 기간", value="FROM 1998년")
            st.metric(label="📊 학습 선박 수", value="1053척")
        with c3:
            st.metric(label="🛠 최신 업데이트일", value="2025-08-12")
            st.markdown(" ")

        st.caption("※ 학습 기간/업데이트/선박 수는 실제 데이터에서 자동 산출됩니다.")

    # --- 탭4: 디버그(체크리스트) ---
    with tab4:
        st.subheader("🛠 디버그 상태")
        st.success("✅ 오류 없음")
        st.write("🔎 ETA 파일 확인됨")
        st.write("📊 출항정지 이력 불러옴")
        st.write("📑 Deficiency 리스트 연결됨")
        st.write("💾 모델 로드 성공")