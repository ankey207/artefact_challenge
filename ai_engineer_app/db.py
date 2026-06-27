from __future__ import annotations

import concurrent.futures
import shutil
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import (
    DEFAULT_DB_PATH,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    QUERY_INTERRUPT_GRACE_SECONDS,
    QUERY_TIMEOUT_SECONDS,
)
from .dataset_version import get_current_dataset_version
from .observability import record_current_event

SCHEMA_FOR_PROMPT = """
DuckDB schema available to answer election-result questions:

Primary analysis view:
vw_results_clean(
  region, region_norm, circonscription_code, circonscription,
  circonscription_norm, nb_bv, inscrits, votants, taux_participation_pct,
  bulletins_nuls, suffrages_exprimes, bulletins_blancs_nombre,
  bulletins_blancs_pct, candidat_id, groupement_parti, groupement_parti_norm,
  candidat_liste, candidat_liste_norm, scores, score_pct, elu, page
)

Useful views:
- vw_winners: same columns as vw_results_clean, filtered to elu = TRUE.
- vw_turnout_by_region(region, region_norm, inscrits, votants, taux_participation_pct).

Base tables:
- circonscriptions: one row per circonscription.
- candidats: one row per candidate/list.

Rules:
- Use only SELECT queries.
- Prefer vw_results_clean, vw_winners and vw_turnout_by_region.
- Use *_norm columns for accent/case-insensitive filters.
- Add a LIMIT for result previews unless the query returns a single aggregate.
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as exc:
        if "File is already open" not in str(exc):
            raise
        source = Path(db_path)
        cached = Path(tempfile.gettempdir()) / f"{source.stem}_readonly.duckdb"
        shutil.copy2(source, cached)
        return duckdb.connect(str(cached), read_only=True)


def _configure_query_limits(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply per-connection resource limits before executing generated SQL."""
    conn.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    conn.execute(f"SET threads={DUCKDB_THREADS}")
    conn.execute("SET max_temp_directory_size='0B'")


def run_query(
    sql: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Run one read-only query and actively interrupt DuckDB on timeout."""
    timeout = QUERY_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    started = time.perf_counter()
    conn: duckdb.DuckDBPyConnection | None = None
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    sql_fingerprint = sha256(sql.encode("utf-8")).hexdigest()[:16]
    try:
        conn = connect(db_path)
        _configure_query_limits(conn)
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="edan-duckdb-query",
        )
        future = executor.submit(lambda: conn.execute(sql).fetchdf())
        dataframe = future.result(timeout=timeout)
        record_current_event(
            "sql_query",
            duration_ms=(time.perf_counter() - started) * 1_000,
            payload={
                "sql_sha256": sql_fingerprint,
                "sql_length": len(sql),
                "timeout_seconds": timeout,
                "row_count": len(dataframe),
                "column_count": len(dataframe.columns),
                "memory_limit": DUCKDB_MEMORY_LIMIT,
                "threads": DUCKDB_THREADS,
            },
        )
        return dataframe
    except concurrent.futures.TimeoutError as exc:
        if conn is not None:
            conn.interrupt()
            try:
                future.result(timeout=QUERY_INTERRUPT_GRACE_SECONDS)
            except Exception:
                pass
        error = RuntimeError(f"Query timed out after {timeout:g} seconds.")
        record_current_event(
            "sql_query",
            duration_ms=(time.perf_counter() - started) * 1_000,
            status="error",
            payload={
                "sql_sha256": sql_fingerprint,
                "sql_length": len(sql),
                "timeout_seconds": timeout,
                "timed_out": True,
                "error_type": type(error).__name__,
            },
        )
        raise error from exc
    except Exception as exc:
        record_current_event(
            "sql_query",
            duration_ms=(time.perf_counter() - started) * 1_000,
            status="error",
            payload={
                "sql_sha256": sql_fingerprint,
                "sql_length": len(sql),
                "timeout_seconds": timeout,
                "timed_out": False,
                "error_type": type(exc).__name__,
            },
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if conn is not None:
            conn.close()


def get_database_stats(db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, Any]:
    with connect(db_path) as conn:
        return {
            "circonscriptions": conn.execute("SELECT COUNT(*) FROM circonscriptions").fetchone()[0],
            "candidats": conn.execute("SELECT COUNT(*) FROM candidats").fetchone()[0],
            "winners": conn.execute("SELECT COUNT(*) FROM vw_winners").fetchone()[0],
            "regions": conn.execute("SELECT COUNT(DISTINCT region) FROM circonscriptions").fetchone()[0],
        }


def get_database_version(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    return get_current_dataset_version(db_path)
