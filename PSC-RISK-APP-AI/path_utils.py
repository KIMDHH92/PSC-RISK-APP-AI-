
# path_utils.py
# 공용 경로/로더 유틸 (깃허브/Streamlit Cloud에서 동작하도록 상대경로 사용)
from __future__ import annotations
from pathlib import Path
import pandas as pd
import glob

# 리포 루트 기준 data 폴더
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"

def p(*names) -> str:
    """data/ 하위 파일 경로를 문자열로 반환"""
    return str(DATA_DIR.joinpath(*names))

def load_ship_db() -> pd.DataFrame:
    """
    data/0.전체선박/ 폴더에서 최신 xlsx 하나를 자동 선택해서 읽는다.
    파일이 없으면 빈 DataFrame 반환.
    """
    shipdb_dir = DATA_DIR / "0.전체선박"
    try:
        xlsx_list = list(shipdb_dir.glob("*.xlsx"))
        if not xlsx_list:
            raise FileNotFoundError("data/0.전체선박 폴더에 *.xlsx 없음")
        shipdb_path = max(xlsx_list, key=lambda p: p.stat().st_mtime)
        return pd.read_excel(shipdb_path)
    except Exception as e:
        import streamlit as st
        st.error(f"전체선박 DB 로드 실패: {e}")
        return pd.DataFrame()

def find_eta_files() -> list[str]:
    """
    data/ETA_ALL/ 폴더의 모든 CSV를 찾아서 정렬된 리스트로 반환
    """
    eta_glob = str((DATA_DIR / "ETA_ALL" / "*.csv"))
    return sorted(glob.glob(eta_glob))
