from pathlib import Path
import os

DATA_DIR = Path(os.getenv("PAPERASSIST_DATA_DIR", "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = int(os.getenv("PAPERASSIST_MIN_CHARS", "20"))
TOPK_EVIDENCE = int(os.getenv("PAPERASSIST_TOPK_EVIDENCE", "8"))
NEIGHBOR_WINDOW = int(os.getenv("PAPERASSIST_NEIGHBOR_WINDOW", "1"))

EMBED_MODEL = os.getenv("PAPERASSIST_EMBED_MODEL", "all-MiniLM-L6-v2")

# --- Agent / LLM settings ---
LLM_PROVIDER = os.getenv("PAPERASSIST_LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("PAPERASSIST_LLM_MODEL", "claude-sonnet-4-20250514")
LLM_API_KEY = os.getenv("PAPERASSIST_LLM_API_KEY", "")
TAVILY_API_KEY = os.getenv("PAPERASSIST_TAVILY_API_KEY", "")
AGENT_MAX_ITER = int(os.getenv("PAPERASSIST_AGENT_MAX_ITER", "10"))
AGENT_TEMPERATURE = float(os.getenv("PAPERASSIST_AGENT_TEMPERATURE", "0.0"))

