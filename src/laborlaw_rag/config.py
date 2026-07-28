from pathlib import Path


# Project root directory.
# config.py is located at:
# src/laborlaw_rag/config.py
#
# Therefore:
# parent     -> laborlaw_rag
# parent     -> src
# parent     -> project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==============================
# Data Directories
# ==============================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

VECTOR_STORE_DIR = DATA_DIR / "vector_store"


# ==============================
# Legal Source
# ==============================

LABOR_LAW_URL = (
    "https://www.solh.ir/regulation/1/66"
)


# ==============================
# Dataset Files
# ==============================

RAW_HTML_PATH = (
    RAW_DATA_DIR / "labor_law_raw.html"
)

PROCESSED_JSON_PATH = (
    PROCESSED_DATA_DIR / "labor_law_records.json"
)