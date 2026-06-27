from pathlib import Path

import duckdb

from ai_engineer_app.dataset_version import (
    build_version_record,
    get_current_dataset_version,
    make_version_id,
    sha256_file,
    upsert_version_record,
)


def test_version_id_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"stable-pdf-content")

    digest = sha256_file(source)
    assert digest == sha256_file(source)
    assert make_version_id(digest) == make_version_id(digest)


def test_version_record_can_be_persisted(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf-content")
    database = tmp_path / "version.duckdb"
    record = build_version_record(
        source,
        circonscription_count=205,
        candidate_count=1125,
        chunk_count=1330,
        embedding_count=1330,
        embedding_dimension=384,
    )

    with duckdb.connect(str(database)) as conn:
        upsert_version_record(conn, record)

    stored = get_current_dataset_version(database)
    assert stored is not None
    assert stored["version_id"] == record["version_id"]
    assert stored["pdf_sha256"] == record["pdf_sha256"]
    assert stored["build_status"] == "ready"
    assert stored["embedding_dimension"] == 384
