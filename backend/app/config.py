import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
RAW_CSV = DATA_DIR / "raw" / "mit_stanford_early50.csv"
UPLOAD_DIR = DATA_DIR / "uploads"
ART_DIR = BASE / "artifacts"
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", BASE.parent / "frontend"))

EARLY_CYCLES = 50
SEED = 42
CV_FOLDS, CV_REPEATS = 5, 3
# Second-life tier thresholds on predicted RUL (cycles remaining after the early window)
TIER_GOOD = float(os.getenv("TIER_GOOD", 600))
TIER_POOR = float(os.getenv("TIER_POOR", 250))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
MAX_UPLOAD_MB = 20


def tier(rul: float) -> str:
    if rul > TIER_GOOD:
        return "GOOD"
    if rul > TIER_POOR:
        return "NEEDS TESTING"
    return "POOR"

NASA_DIR = DATA_DIR / "nasa" / "cleaned_dataset"
SP1_XLSX = next((DATA_DIR / "sp1").glob("*.xlsx"), None)
