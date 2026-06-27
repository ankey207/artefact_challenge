from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_local_env() -> None:
    """Load simple KEY=VALUE pairs from .env without an extra dependency."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_local_env()


def _resolve_project_path(value: str | None, default_name: str) -> Path:
    """Resolve relative configuration paths from the project root, not the CWD."""
    configured = Path(value) if value else Path(default_name)
    if not configured.is_absolute():
        configured = ROOT_DIR / configured
    return configured.resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


DEFAULT_DB_PATH = _resolve_project_path(
    os.getenv("EDAN_DUCKDB_PATH"),
    "edan_2025_resultat_national_details.duckdb",
)
CHATBOT_VERSION = os.getenv("EDAN_CHATBOT_VERSION", "edan-chat-v1").strip()
PROMPT_VERSION = os.getenv("EDAN_PROMPT_VERSION", "2026-06-24.v1").strip()
LANGFUSE_ENABLED = _env_bool("EDAN_LANGFUSE_ENABLED", True)
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").strip()
LANGFUSE_CAPTURE_CONTENT = _env_bool(
    "EDAN_LANGFUSE_CAPTURE_CONTENT",
    False,
)
LANGFUSE_PROMPTS_ENABLED = _env_bool("EDAN_LANGFUSE_PROMPTS_ENABLED", True)
LANGFUSE_PROMPT_LABEL = os.getenv("EDAN_LANGFUSE_PROMPT_LABEL", "production").strip() or "production"
LANGFUSE_PROMPT_CACHE_TTL_SECONDS = int(os.getenv("EDAN_LANGFUSE_PROMPT_CACHE_TTL_SECONDS", "60"))
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_INPUT_COST_PER_MILLION = float(os.getenv("DEEPSEEK_INPUT_COST_PER_MILLION", "0"))
DEEPSEEK_OUTPUT_COST_PER_MILLION = float(os.getenv("DEEPSEEK_OUTPUT_COST_PER_MILLION", "0"))
DEFAULT_LIMIT = int(os.getenv("SQL_DEFAULT_LIMIT", "100"))
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "20"))
QUERY_INTERRUPT_GRACE_SECONDS = float(os.getenv("QUERY_INTERRUPT_GRACE_SECONDS", "2"))
DUCKDB_MEMORY_LIMIT = os.getenv("DUCKDB_MEMORY_LIMIT", "256MB")
DUCKDB_THREADS = max(1, int(os.getenv("DUCKDB_THREADS", "1")))


RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
ENTITY_SIMILARITY_THRESHOLD = float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.80"))
HISTORY_MAX_EXCHANGES = int(os.getenv("HISTORY_MAX_EXCHANGES", "10"))

# ── Phase 9: Query result cache ───────────────────────────────────────────────
CACHE_ENABLED = _env_bool("EDAN_CACHE_ENABLED", True)
CACHE_DB_PATH = _resolve_project_path(
    os.getenv("EDAN_CACHE_DB_PATH"),
    "data/query_cache.sqlite3",
)
# Time-to-live per cache entry (seconds). Default 24 h.
CACHE_TTL_SECONDS = int(os.getenv("EDAN_CACHE_TTL_SECONDS", str(60 * 60 * 24)))
# Max entries per namespace before LRU eviction kicks in.
CACHE_MAX_ENTRIES_PER_NS = int(os.getenv("EDAN_CACHE_MAX_ENTRIES_PER_NS", "500"))

# ── Phase 10: DeepSeek retry / timeout configuration ─────────────────────────
# Maximum number of *retries* (not total attempts). 0 = no retries.
DEEPSEEK_MAX_RETRIES = int(os.getenv("DEEPSEEK_MAX_RETRIES", "3"))
DEEPSEEK_RETRY_BASE_DELAY = float(os.getenv("DEEPSEEK_RETRY_BASE_DELAY", "2.0"))
DEEPSEEK_RETRY_MAX_DELAY = float(os.getenv("DEEPSEEK_RETRY_MAX_DELAY", "30.0"))
# Separate connection and read timeouts.
DEEPSEEK_CONNECT_TIMEOUT = float(os.getenv("DEEPSEEK_CONNECT_TIMEOUT", "10.0"))
DEEPSEEK_READ_TIMEOUT_TEXT = float(os.getenv("DEEPSEEK_READ_TIMEOUT_TEXT", "30.0"))
DEEPSEEK_READ_TIMEOUT_JSON = float(os.getenv("DEEPSEEK_READ_TIMEOUT_JSON", "45.0"))
# Optional circuit breaker (disabled by default).
DEEPSEEK_CB_ENABLED = _env_bool("DEEPSEEK_CB_ENABLED", False)
DEEPSEEK_CB_FAILURE_THRESHOLD = int(os.getenv("DEEPSEEK_CB_FAILURE_THRESHOLD", "5"))
DEEPSEEK_CB_WINDOW_SECONDS = float(os.getenv("DEEPSEEK_CB_WINDOW_SECONDS", "60.0"))
DEEPSEEK_CB_COOLDOWN_SECONDS = float(os.getenv("DEEPSEEK_CB_COOLDOWN_SECONDS", "30.0"))


def get_api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY", "").strip()
