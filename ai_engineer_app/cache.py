"""
SQLite-backed query result cache with automatic invalidation.

Namespaces
----------
  sql_result        -- DuckDB query results serialized as list[dict]
  rag_result        -- RAG chunk retrieval results (list[dict])
  contextualization -- LLM contextualizer output (non-sensitive only)

Cache key
---------
SHA-256 of:   namespace | dataset_version_id | model | normalized_question
              | memory_hash | retrieval_params

Global invalidation signature
------------------------------
A single hash covering every factor that should bust all entries:
  pdf_sha256, DuckDB schema version, chunking version, embedding model,
  RAG vector/keyword weights, top_k, LLM model, normalization code version,
  prompt version.

When the signature changes the entire cache is flushed atomically.
Individual entries also carry a per-entry TTL and the namespace is bounded
by a configurable LRU limit.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import unicodedata
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version sentinels — bump manually when the corresponding logic changes
# ---------------------------------------------------------------------------

# Bump when _normalize_text() logic changes.
_NORMALIZE_VERSION = "v1"

# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------

NS_SQL = "sql_result"
NS_RAG = "rag_result"
NS_CTX = "contextualization"

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key    TEXT PRIMARY KEY,
    namespace    TEXT NOT NULL,
    value_json   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    accessed_at  REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 1,
    signature    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cache_ns  ON cache_entries (namespace);
CREATE INDEX IF NOT EXISTS idx_cache_acc ON cache_entries (accessed_at);
"""


# ---------------------------------------------------------------------------
# Cache store
# ---------------------------------------------------------------------------


class _CacheStore:
    """Thread-safe SQLite cache store."""

    def __init__(
        self,
        db_path: Path,
        *,
        ttl_seconds: int = 86_400,
        max_entries_per_ns: int = 500,
    ) -> None:
        self._db_path = db_path
        self._ttl = ttl_seconds
        self._max = max_entries_per_ns
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── Signature management ────────────────────────────────────────────────

    def get_stored_signature(self) -> str | None:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT value FROM cache_metadata WHERE key='signature'").fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def set_stored_signature(self, sig: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES ('signature', ?)",
                    (sig,),
                )
        except Exception as exc:
            logger.debug("Cache: failed to store signature: %s", exc)

    # ── Data operations ─────────────────────────────────────────────────────

    def flush_all(self) -> int:
        """Delete every cache entry. Returns number of rows deleted."""
        try:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM cache_entries")
                return cur.rowcount
        except Exception:
            return 0

    def get(self, key: str, *, expected_sig: str) -> Any | None:
        """Return cached value or None on miss / signature mismatch / TTL expiry."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value_json, created_at, signature FROM cache_entries WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                value_json, created_at, sig = row
                if sig != expected_sig:
                    return None
                if self._ttl > 0 and (time.time() - created_at) > self._ttl:
                    return None
                # Update LRU stats (best-effort, non-blocking)
                try:
                    conn.execute(
                        "UPDATE cache_entries SET accessed_at=?, access_count=access_count+1 WHERE cache_key=?",
                        (time.time(), key),
                    )
                except Exception:
                    pass
                return json.loads(value_json)
        except Exception:
            return None

    def set(self, key: str, value: Any, *, namespace: str, signature: str) -> None:
        """Upsert a cache entry and evict oldest entries if the namespace is full."""
        try:
            now = time.time()
            value_json = json.dumps(value, ensure_ascii=False, default=str)
            with self._lock:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO cache_entries
                           (cache_key, namespace, value_json,
                            created_at, accessed_at, access_count, signature)
                           VALUES (?, ?, ?, ?, ?, 1, ?)""",
                        (key, namespace, value_json, now, now, signature),
                    )
                    # LRU eviction: keep only _max entries per namespace
                    count = conn.execute(
                        "SELECT COUNT(*) FROM cache_entries WHERE namespace=?",
                        (namespace,),
                    ).fetchone()[0]
                    if count > self._max:
                        evict = count - self._max
                        conn.execute(
                            """DELETE FROM cache_entries WHERE cache_key IN (
                               SELECT cache_key FROM cache_entries
                               WHERE namespace=?
                               ORDER BY accessed_at ASC
                               LIMIT ?
                            )""",
                            (namespace, evict),
                        )
        except Exception as exc:
            logger.debug("Cache set error: %s", exc)

    def stats(self) -> dict[str, int]:
        """Return per-namespace entry counts."""
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT namespace, COUNT(*) FROM cache_entries GROUP BY namespace").fetchall()
                return {ns: cnt for ns, cnt in rows}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_store: _CacheStore | None = None
_current_signature: str | None = None
_signature_checked: bool = False


def _get_store() -> _CacheStore | None:
    """Return the singleton cache store, or None if cache is disabled / broken."""
    global _store
    if _store is None:
        try:
            from .config import (
                CACHE_DB_PATH,
                CACHE_ENABLED,
                CACHE_MAX_ENTRIES_PER_NS,
                CACHE_TTL_SECONDS,
            )

            if not CACHE_ENABLED:
                return None
            _store = _CacheStore(
                CACHE_DB_PATH,
                ttl_seconds=CACHE_TTL_SECONDS,
                max_entries_per_ns=CACHE_MAX_ENTRIES_PER_NS,
            )
        except Exception as exc:
            logger.warning("Cache: initialization failed: %s", exc)
    return _store


def _compute_signature() -> str:
    """Compute and cache the current invalidation signature."""
    global _current_signature
    if _current_signature is not None:
        return _current_signature
    try:
        from .config import DEEPSEEK_MODEL, DEFAULT_DB_PATH, PROMPT_VERSION, RAG_TOP_K
        from .dataset_version import EMBEDDING_MODEL, get_current_dataset_version
        from .rag_retriever import _KEYWORD_WEIGHT, _VECTOR_WEIGHT

        dv = get_current_dataset_version(DEFAULT_DB_PATH) or {}
        payload = {
            "pdf_sha256": dv.get("pdf_sha256", ""),
            "schema_version": dv.get("schema_version", ""),
            "chunking_version": dv.get("chunking_version", ""),
            "embedding_model": EMBEDDING_MODEL,
            "vector_weight": _VECTOR_WEIGHT,
            "keyword_weight": _KEYWORD_WEIGHT,
            "top_k": RAG_TOP_K,
            "model": DEEPSEEK_MODEL,
            "normalize_version": _NORMALIZE_VERSION,
            "prompt_version": PROMPT_VERSION,
        }
        sig = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    except Exception as exc:
        logger.debug("Cache: could not compute signature: %s", exc)
        sig = "unknown"
    _current_signature = sig
    return sig


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """
    Deterministic, accent-stripped, lower-case normalization for cache keys.
    Version: _NORMALIZE_VERSION — bump that constant when this function changes.
    """
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode().lower()
    return re.sub(r"\s+", " ", text).strip()


def make_cache_key(
    namespace: str,
    question: str,
    *,
    dataset_version_id: str = "",
    model: str = "",
    memory_hash: str = "",
    retrieval_params: str = "",
) -> str:
    """
    Build a deterministic cache key.

    Parameters correspond to the five components specified for Phase 9:
      dataset_version_id, model, normalized_question, memory_hash,
      retrieval_params (top_k / source_type / embedding_model).
    """
    normalized = _normalize_text(question)
    raw = f"{namespace}|{dataset_version_id}|{model}|{normalized}|{memory_hash}|{retrieval_params}"
    return sha256(raw.encode("utf-8")).hexdigest()


def make_sql_cache_key(safe_sql: str, dataset_version_id: str) -> str:
    """Cache key for SQL results: keyed on the exact query + dataset version."""
    raw = f"{NS_SQL}|{dataset_version_id}|{safe_sql}"
    return sha256(raw.encode("utf-8")).hexdigest()


def make_memory_hash(memory: dict) -> str:
    """
    Stable 16-char hash of the structured memory fields used for cache keying.
    Only hashes active_entities and active_metric (the stable, structured part).
    """
    stable = {
        "active_entities": memory.get("active_entities"),
        "active_metric": memory.get("active_metric"),
    }
    return sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public get / set
# ---------------------------------------------------------------------------


def get_cached(key: str) -> Any | None:
    """
    Retrieve a value from the cache.
    Returns None on miss, signature mismatch, TTL expiry, or if cache is disabled.
    """
    store = _get_store()
    if store is None:
        return None
    sig = _compute_signature()
    return store.get(key, expected_sig=sig)


def set_cached(key: str, value: Any, *, namespace: str) -> None:
    """
    Store a value in the cache under the given namespace.
    No-op if cache is disabled or an error occurs.
    """
    store = _get_store()
    if store is None:
        return
    sig = _compute_signature()
    store.set(key, value, namespace=namespace, signature=sig)


def ensure_valid_cache() -> None:
    """
    Called once per request: compare the current signature against the stored
    one and flush all entries if they differ (automatic invalidation).
    """
    global _signature_checked
    if _signature_checked:
        return
    store = _get_store()
    if store is None:
        _signature_checked = True
        return
    current_sig = _compute_signature()
    stored_sig = store.get_stored_signature()
    if stored_sig != current_sig:
        n = store.flush_all()
        store.set_stored_signature(current_sig)
        if stored_sig is not None:
            logger.info(
                "Cache invalidated (signature changed): %d entries removed. Old=%s New=%s",
                n,
                stored_sig[:8],
                current_sig[:8],
            )
        else:
            logger.info("Cache initialized with signature %s", current_sig[:8])
    _signature_checked = True


def reset_signature_check() -> None:
    """
    Reset the per-process signature-checked flag.
    Called at the start of each answer_question() call so that a hot-reload
    or dataset rebuild is detected on the next request.
    """
    global _signature_checked, _current_signature
    _signature_checked = False
    _current_signature = None


def get_cache_stats() -> dict[str, int]:
    """Return per-namespace entry counts (for monitoring / Streamlit dashboard)."""
    store = _get_store()
    return store.stats() if store is not None else {}
