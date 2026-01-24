from pathlib import Path
import os

DATA_DIR = Path(os.getenv("PAPERASSIST_DATA_DIR", "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = int(os.getenv("PAPERASSIST_MIN_CHARS", "20"))
TOPK_EVIDENCE = int(os.getenv("PAPERASSIST_TOPK_EVIDENCE", "8"))
NEIGHBOR_WINDOW = int(os.getenv("PAPERASSIST_NEIGHBOR_WINDOW", "1"))

