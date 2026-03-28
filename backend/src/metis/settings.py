from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("METIS_DATA_DIR", "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = int(os.getenv("METIS_MIN_CHARS", "20"))
TOPK_EVIDENCE = int(os.getenv("METIS_TOPK_EVIDENCE", "8"))
NEIGHBOR_WINDOW = int(os.getenv("METIS_NEIGHBOR_WINDOW", "1"))

EMBED_MODEL = os.getenv("METIS_EMBED_MODEL", "all-MiniLM-L6-v2")

# --- Agent / LLM settings ---
LLM_PROVIDER = os.getenv("METIS_LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("METIS_LLM_MODEL", "claude-sonnet-4-20250514")
LLM_API_KEY = os.getenv("METIS_LLM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("METIS_TAVILY_API_KEY", "") or os.getenv("TAVILY_API_KEY", "")
AGENT_MAX_ITER = int(os.getenv("METIS_AGENT_MAX_ITER", "10"))
AGENT_TEMPERATURE = float(os.getenv("METIS_AGENT_TEMPERATURE", "0.0"))
MMR_LAMBDA = float(os.getenv("METIS_MMR_LAMBDA", "0.7"))
CITATION_MIN_SCORE = float(os.getenv("METIS_CITATION_MIN_SCORE", "0.0"))

# --- Multimodal enrichment ---
ENABLE_ENRICHMENT = os.getenv("METIS_ENABLE_ENRICHMENT", "true").lower() in ("true", "1", "yes")

