import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", SRC_DIR.parent.parent))

RAW_DIR = Path(os.getenv("RAW_DIR", PROJECT_ROOT / "data" / "raw"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "data" / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
