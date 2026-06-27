from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

DATASET_SCHEMA_VERSION = "edan-schema-v1"
CHUNKING_VERSION = "edan-rag-chunks-v1"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_version_id(
    pdf_sha256: str,
    schema_version: str = DATASET_SCHEMA_VERSION,
    chunking_version: str = CHUNKING_VERSION,
    embedding_model: str = EMBEDDING_MODEL,
) -> str:
    payload = json.dumps(
        {
            "pdf_sha256": pdf_sha256,
            "schema_version": schema_version,
            "chunking_version": chunking_version,
            "embedding_model": embedding_model,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "dataset_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_version_record(
    pdf_path: Path | str,
    *,
    circonscription_count: int,
    candidate_count: int,
    chunk_count: int,
    embedding_count: int = 0,
    embedding_dimension: int | None = None,
) -> dict[str, Any]:
    source = Path(pdf_path).resolve()
    pdf_sha256 = sha256_file(source)
    return {
        "version_id": make_version_id(pdf_sha256),
        "pdf_sha256": pdf_sha256,
        "pdf_filename": source.name,
        "pdf_size_bytes": source.stat().st_size,
        "schema_version": DATASET_SCHEMA_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": embedding_dimension,
        "embedding_count": embedding_count,
        "circonscription_count": circonscription_count,
        "candidate_count": candidate_count,
        "chunk_count": chunk_count,
        "build_status": ("ready" if chunk_count > 0 and embedding_count == chunk_count else "pending"),
        "built_at": datetime.now(UTC).replace(tzinfo=None),
    }


def create_dataset_versions_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            version_id TEXT PRIMARY KEY,
            pdf_sha256 TEXT NOT NULL,
            pdf_filename TEXT NOT NULL,
            pdf_size_bytes BIGINT NOT NULL,
            schema_version TEXT NOT NULL,
            chunking_version TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER,
            embedding_count INTEGER NOT NULL,
            circonscription_count INTEGER NOT NULL,
            candidate_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            build_status TEXT NOT NULL,
            built_at TIMESTAMP NOT NULL
        )
        """
    )


def upsert_version_record(
    conn: duckdb.DuckDBPyConnection,
    record: dict[str, Any],
) -> None:
    create_dataset_versions_table(conn)
    columns = list(record)
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        "DELETE FROM dataset_versions WHERE version_id = ?",
        [record["version_id"]],
    )
    conn.execute(
        f"INSERT INTO dataset_versions ({', '.join(columns)}) VALUES ({placeholders})",
        [record[column] for column in columns],
    )


def inspect_database_counts(conn: duckdb.DuckDBPyConnection) -> dict[str, int | None]:
    embedding_column = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'rag_chunks' AND column_name = 'embedding'
        """
    ).fetchone()[0]
    embedding_count = 0
    embedding_dimension = None
    if embedding_column:
        embedding_count = conn.execute("SELECT COUNT(*) FROM rag_chunks WHERE embedding IS NOT NULL").fetchone()[0]
        dimension_row = conn.execute(
            "SELECT array_length(embedding) FROM rag_chunks WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        embedding_dimension = dimension_row[0] if dimension_row else None

    return {
        "circonscription_count": conn.execute("SELECT COUNT(*) FROM circonscriptions").fetchone()[0],
        "candidate_count": conn.execute("SELECT COUNT(*) FROM candidats").fetchone()[0],
        "chunk_count": conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0],
        "embedding_count": embedding_count,
        "embedding_dimension": embedding_dimension,
    }


def refresh_dataset_version(
    db_path: Path | str,
    pdf_path: Path | str,
) -> dict[str, Any]:
    with duckdb.connect(str(db_path), read_only=False) as conn:
        counts = inspect_database_counts(conn)
        record = build_version_record(pdf_path, **counts)
        upsert_version_record(conn, record)
    return record


def get_current_dataset_version(
    db_path: Path | str,
) -> dict[str, Any] | None:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'dataset_versions'
            """
        ).fetchone()[0]
        if not exists:
            return None
        cursor = conn.execute("SELECT * FROM dataset_versions ORDER BY built_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(zip([item[0] for item in cursor.description], row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcule et enregistre la version du dataset EDAN.")
    parser.add_argument(
        "--db",
        default="edan_2025_resultat_national_details.duckdb",
        help="Chemin de la base DuckDB.",
    )
    parser.add_argument(
        "--pdf",
        default="docs/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf",
        help="Chemin du PDF source.",
    )
    args = parser.parse_args()
    record = refresh_dataset_version(args.db, args.pdf)
    print(json.dumps(record, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
