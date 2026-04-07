from pathlib import Path
import os
import tomllib
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("METIS_DATA_DIR", "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Config directory — Tauri passes this in as METIS_CONFIG_DIR; falls back to
# the XDG default so the backend also works outside the packaged app.
CONFIG_DIR = Path(os.getenv("METIS_CONFIG_DIR", Path.home() / ".config" / "metis"))

# Load config.toml if present. Environment variables take priority over it.
_toml: dict = {}
_toml_path = CONFIG_DIR / "config.toml"
if _toml_path.exists():
    with open(_toml_path, "rb") as _f:
        _toml = tomllib.load(_f)


def _cfg(env_var: str, toml_key: str, default: str = "") -> str:
    """Return env var if set, then config.toml value, then default."""
    return os.getenv(env_var) or _toml.get(toml_key, default)


MIN_CHARS = int(os.getenv("METIS_MIN_CHARS", "20"))
TOPK_EVIDENCE = int(os.getenv("METIS_TOPK_EVIDENCE", "8"))
NEIGHBOR_WINDOW = int(os.getenv("METIS_NEIGHBOR_WINDOW", "1"))

EMBED_MODEL = os.getenv("METIS_EMBED_MODEL", "all-MiniLM-L6-v2")

# --- Agent / LLM settings ---
LLM_PROVIDER = _cfg("METIS_LLM_PROVIDER", "provider", "anthropic")
LLM_MODEL = _cfg("METIS_LLM_MODEL", "model", "claude-sonnet-4-20250514")
LLM_API_KEY = os.getenv("METIS_LLM_API_KEY", "")

# API keys: env var → config.toml → empty
ANTHROPIC_API_KEY = _cfg("ANTHROPIC_API_KEY", "anthropic_api_key")
OPENAI_API_KEY = _cfg("OPENAI_API_KEY", "openai_api_key")
OPENROUTER_API_KEY = _cfg("OPENROUTER_API_KEY", "openrouter_api_key")
TAVILY_API_KEY = _cfg("METIS_TAVILY_API_KEY", "tavily_api_key") or os.getenv("TAVILY_API_KEY", "")

AGENT_MAX_ITER = int(os.getenv("METIS_AGENT_MAX_ITER", "10"))
AGENT_TEMPERATURE = float(os.getenv("METIS_AGENT_TEMPERATURE", "0.0"))
MMR_LAMBDA = float(os.getenv("METIS_MMR_LAMBDA", "0.7"))
CITATION_MIN_SCORE = float(os.getenv("METIS_CITATION_MIN_SCORE", "0.0"))

