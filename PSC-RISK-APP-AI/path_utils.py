from pathlib import Path
import os
import pandas as pd
import re

# Repo-relative data dir (works on GitHub/Streamlit Cloud)
DATA_DIR = Path(__file__).resolve().parent / "data"

def p(*parts) -> str:
    """Build a path under data/ as string."""
    return str(DATA_DIR.joinpath(*parts))

def _read_any(path: str):
    ext = Path(path).suffix.lower()
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(path, dtype=str)
    elif ext == ".csv":
        # allow large files
        return pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {ext} ({path})")

def _candidate_imo_col(cols):
    cands = ['vessel_imo','imo','imo_no','IMO NO','IMO_NO','Vessel_IMO','VESSEL_IMO']
    for c in cands:
        if c in cols:
            return c
    return None

def _normalize_imo_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    cand = _candidate_imo_col(df.columns)
    if cand is not None and cand != 'vessel_imo':
        df = df.rename(columns={cand: 'vessel_imo'})
    if 'vessel_imo' in df.columns:
        df['vessel_imo'] = (
            df['vessel_imo']
            .astype(str)
            .str.extract(r'(\d+)', expand=False)
            .fillna('')
        )
    return df

def load_ship_db(path: str) -> pd.DataFrame:
    """Load overall ship DB; fallback to any xlsx in data/ if path missing."""
    if not os.path.exists(path):
        # fallback: first xlsx in data/
        xlxs = sorted(DATA_DIR.glob("*.xlsx"))
        if xlxs:
            path = str(xlxs[0])
    df = _read_any(path)
    # Standardize a few common columns
    # Not strictly necessary, but helps downstream
    return _normalize_imo_column(df)

def find_eta_files():
    eta_root = DATA_DIR / "ETA_ALL"
    if not eta_root.exists():
        return []
    files = []
    for ext in ("*.csv", "*.xlsx"):
        files.extend(sorted(eta_root.glob(ext)))
    return [str(f) for f in files]

def load_eta_merged() -> pd.DataFrame:
    files = find_eta_files()
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(_read_any(f))
        except Exception:
            # ignore broken file
            continue
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs, ignore_index=True)
    merged = _normalize_imo_column(merged)
    return merged