#!/usr/bin/env python
"""Extract EDAN 2025 national detailed election results PDF to DuckDB.

The output is a relational DuckDB database built directly from the PDF. The
internal dataframe is validated before insertion and is not written to CSV.

PDF handling covered by this script:
- Repeated table headers are skipped on every page.
- Page footers and page numbers are excluded by table extraction; if they appear
  in a table row, the header/empty-row filter prevents candidate creation.
- Broken lines inside cells are normalized to single spaces.
- Wrapped circonscription names are merged into the active circonscription.
- Page-break layout issues are handled by buffering leading candidate rows and
  by exact, documented corrections for known visual/table mismatches.
- Tables are extracted with pdfplumber, then validated with numeric identities.

Entity normalization policy:
- Accents and apostrophes are preserved in output values.
- Casing from the PDF is preserved for parties, candidates and locations.
- Whitespace and line breaks are normalized to a single space.
- Vertically printed region names are reconstructed, uppercased, and corrected
  only through the REGION_FIXES mapping below.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pdfplumber

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_engineer_app.dataset_version import (
    build_version_record,
    upsert_version_record,
)

DEFAULT_INPUT = "EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf"
DEFAULT_OUTPUT = "edan_2025_resultat_national_details.duckdb"
EXPECTED_CIRCONSCRIPTION_CODES = set(range(1, 206))

COLUMNS = [
    "region",
    "circonscription_code",
    "circonscription",
    "nb_bv",
    "inscrits",
    "votants",
    "taux_participation_pct",
    "bulletins_nuls",
    "suffrages_exprimes",
    "bulletins_blancs_nombre",
    "bulletins_blancs_pct",
    "groupement_parti",
    "candidat_liste",
    "scores",
    "score_pct",
    "elu",
    "page",
]

REGION_FIXES = {
    "DISTRICTAUTONOMED'ABIDJAN": "DISTRICT AUTONOME D'ABIDJAN",
    "DISTRICTAUTONOMEDEYAMOUSSOUKRO": "DISTRICT AUTONOME DE YAMOUSSOUKRO",
    "GRANDSPONTS": "GRANDS-PONTS",
    "INDENIE-DJUABLIN": "INDENIE-DJUABLIN",
    "SAN-PEDRO": "SAN-PEDRO",
    "SUD-COMOE": "SUD-COMOE",
}


def clean_text(value: Any) -> str:
    """Normalize extracted cell text without losing accents or apostrophes."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_entity(value: Any) -> str:
    """Return a matching/search key while preserving source text elsewhere.

    Normalization policy for *_norm columns:
    - uppercase text,
    - strip accents/diacritics,
    - normalize apostrophes and separators to spaces,
    - collapse repeated whitespace.

    The original accented/cased values remain in the non-_norm columns.
    """
    text = clean_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.upper()
    text = re.sub(r"['’`´]", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_region(value: Any) -> str:
    """Recover vertically printed region labels extracted bottom-to-top."""
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ""

    compact = raw.replace("\n", "").replace(" ", "").strip()
    if not compact or compact.upper() == "TOTAL":
        return ""

    # Region labels are vertical in the source PDF and pdfplumber returns them
    # in reverse reading order.
    region = compact[::-1].upper()
    return REGION_FIXES.get(region, region)


def parse_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    digits = re.sub(r"[^\d-]", "", text)
    return int(digits) if digits else None


def parse_pct(value: Any) -> float | None:
    text = clean_text(value).replace("%", "").replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_header_or_total(row: list[Any]) -> bool:
    cells = [clean_text(cell).upper() for cell in row]
    joined = " ".join(cells)
    return (
        not any(cells)
        or cells[0] in {"REGI", "ON", "TOTAL"}
        or "CIRCONSCRIPTION" in joined
        or "GROUPEMENTS / PARTIS" in joined
    )


def has_candidate(row: list[Any]) -> bool:
    return bool(clean_text(row[11]) and clean_text(row[12]))


def extract_rows(pdf_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_region = ""
    current_context: dict[str, Any] = {}
    pending_lead_row: dict[str, Any] | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            table = page.extract_table()
            if not table:
                continue

            for row in table:
                row = (row + [""] * 16)[:16]
                if is_header_or_total(row):
                    continue

                region = clean_region(row[0])
                if region:
                    current_region = region

                circonscription_code = clean_text(row[1])
                circonscription_fragment = clean_text(row[2])
                row_metrics = {
                    "inscrits": parse_int(row[4]),
                    "votants": parse_int(row[5]),
                    "taux_participation_pct": parse_pct(row[6]),
                    "bulletins_nuls": parse_int(row[7]),
                    "suffrages_exprimes": parse_int(row[8]),
                    "bulletins_blancs_nombre": parse_int(row[9]),
                    "bulletins_blancs_pct": parse_pct(row[10]),
                }
                has_row_metrics = any(value is not None for value in row_metrics.values())

                if not circonscription_code and not circonscription_fragment and has_row_metrics and has_candidate(row):
                    # In a few wrapped blocks, pdfplumber extracts the first
                    # candidate and vote metrics before the next row containing
                    # the circonscription code/name. Hold it until that context
                    # is complete.
                    pending_lead_row = {
                        "metrics": row_metrics,
                        "candidates": [],
                    }
                    pending_lead_row["candidates"].append(
                        {
                            "groupement_parti": clean_text(row[11]),
                            "candidat_liste": clean_text(row[12]),
                            "scores": parse_int(row[13]),
                            "score_pct": parse_pct(row[14]),
                            "elu": clean_text(row[15]).upper() == "ELU(E)",
                            "page": page_number,
                        }
                    )
                    continue

                if (
                    pending_lead_row
                    and not circonscription_code
                    and not circonscription_fragment
                    and has_candidate(row)
                ):
                    pending_lead_row["candidates"].append(
                        {
                            "groupement_parti": clean_text(row[11]),
                            "candidat_liste": clean_text(row[12]),
                            "scores": parse_int(row[13]),
                            "score_pct": parse_pct(row[14]),
                            "elu": clean_text(row[15]).upper() == "ELU(E)",
                            "page": page_number,
                        }
                    )
                    continue

                if circonscription_code:
                    metrics = row_metrics
                    if pending_lead_row and not has_row_metrics:
                        metrics = pending_lead_row["metrics"]

                    current_context = {
                        "region": current_region,
                        "circonscription_code": circonscription_code,
                        "circonscription": circonscription_fragment,
                        "nb_bv": parse_int(row[3]),
                        **metrics,
                    }
                    if pending_lead_row:
                        for candidate in pending_lead_row["candidates"]:
                            rows.append({**current_context, **candidate})
                        pending_lead_row = None
                elif circonscription_fragment and current_context:
                    # Some wrapped circonscription names are extracted as a
                    # separate row while candidate columns remain populated.
                    previous = current_context.get("circonscription", "")
                    current_context["circonscription"] = clean_text(f"{previous} {circonscription_fragment}")

                if not has_candidate(row) or not current_context:
                    continue

                rows.append(
                    {
                        **current_context,
                        "groupement_parti": clean_text(row[11]),
                        "candidat_liste": clean_text(row[12]),
                        "scores": parse_int(row[13]),
                        "score_pct": parse_pct(row[14]),
                        "elu": clean_text(row[15]).upper() == "ELU(E)",
                        "page": page_number,
                    }
                )

    return apply_known_pdf_layout_corrections(rows)


def row_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    return (
        row.get("groupement_parti", ""),
        row.get("candidat_liste", ""),
        row.get("scores"),
    )


def context_from(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row[column] for column in COLUMNS[:11]}


def apply_context(row: dict[str, Any], context: dict[str, Any]) -> None:
    for key, value in context.items():
        row[key] = value


def apply_known_pdf_layout_corrections(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fix source-specific page-break issues missed by table extraction.

    In the source PDF, a few rows are visually attached to the next
    circonscription, but pdfplumber keeps them under the previous table block.
    One circonscription header (115) is also visible in the text layer but not
    represented as a table row. These corrections are constrained to exact
    candidate/list names and scores.
    """
    context_by_code = {str(row["circonscription_code"]).zfill(3): context_from(row) for row in rows}

    context_115 = {
        "region": "INDENIE-DJUABLIN",
        "circonscription_code": "115",
        "circonscription": (
            "ABENGOUROU, SOUS-PREFECTURE, AMELEKIA, ANIANSSUE, "
            "EBILASSOKRO, NIABLE, YAKASSE-FEYASSE ET ZARANOU, "
            "COMMUNES ET SOUS-PREFECTURES"
        ),
        "nb_bv": 166,
        "inscrits": 51273,
        "votants": 22283,
        "taux_participation_pct": 43.46,
        "bulletins_nuls": 498,
        "suffrages_exprimes": 21785,
        "bulletins_blancs_nombre": 104,
        "bulletins_blancs_pct": 0.48,
    }
    context_by_code["115"] = context_115

    move_targets = {
        ("CODE", "OSONS LE CHANGEMENT", 692): "061",
        ("GP-PAIX", "PAIX-UNITE-PROSPERITE", 300): "061",
        (
            "RHDP",
            "UNE CÔTE DIVOIRE EN PAIX, PROSPÈRE ET SOLIDAIRE",
            17921,
        ): "115",
        ("INDEPENDANT", "KOUAME KOUASSI JEAN MICHEL", 746): "182",
        ("INDEPENDANT", "YAMOUSSOUN MOSSOUN CLEMENT", 1033): "182",
    }

    for row in rows:
        target_code = move_targets.get(row_key(row))
        if target_code:
            apply_context(row, context_by_code[target_code])

    if not any(
        row_key(row) == ("PDCI-RDA", "TOUS ENSEMBLE POUR LA CÔTE D'IVOIRE", 3760)
        and str(row["circonscription_code"]).zfill(3) == "115"
        for row in rows
    ):
        rows.append(
            {
                **context_115,
                "groupement_parti": "PDCI-RDA",
                "candidat_liste": "TOUS ENSEMBLE POUR LA CÔTE D'IVOIRE",
                "scores": 3760,
                "score_pct": 17.26,
                "elu": False,
                "page": 20,
            }
        )

    return rows


def build_dataframe(pdf_path: Path) -> pd.DataFrame:
    rows = extract_rows(pdf_path)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["_circonscription_sort"] = pd.to_numeric(df["circonscription_code"], errors="coerce")
    df = (
        df.sort_values(
            ["_circonscription_sort", "page"],
            kind="mergesort",
        )
        .drop(columns="_circonscription_sort")
        .reset_index(drop=True)
    )

    int_columns = [
        "nb_bv",
        "inscrits",
        "votants",
        "bulletins_nuls",
        "suffrages_exprimes",
        "bulletins_blancs_nombre",
        "scores",
        "page",
    ]
    for column in int_columns:
        df[column] = df[column].astype("Int64")

    return df


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Return validation errors that would make the CSV analytically unsafe."""
    errors: list[str] = []
    circonscriptions = df.drop_duplicates("circonscription_code").copy()
    codes = {int(code) for code in pd.to_numeric(circonscriptions["circonscription_code"], errors="coerce").dropna()}

    missing_values = int(df.isna().sum().sum())
    if missing_values:
        errors.append(f"{missing_values} missing values found")

    missing_codes = sorted(EXPECTED_CIRCONSCRIPTION_CODES - codes)
    extra_codes = sorted(codes - EXPECTED_CIRCONSCRIPTION_CODES)
    if missing_codes:
        errors.append(f"missing circonscription codes: {missing_codes}")
    if extra_codes:
        errors.append(f"unexpected circonscription codes: {extra_codes}")

    c = circonscriptions.set_index("circonscription_code")
    if (c["votants"] != c["bulletins_nuls"] + c["suffrages_exprimes"]).any():
        errors.append("some rows fail: votants = bulletins_nuls + suffrages_exprimes")

    score_sum = df.groupby("circonscription_code")["scores"].sum()
    aligned = c.join(score_sum.rename("score_sum"))
    if (aligned["score_sum"] + aligned["bulletins_blancs_nombre"] != aligned["suffrages_exprimes"]).any():
        errors.append("some rows fail: sum(scores) + bulletins_blancs = suffrages_exprimes")

    participation_diff = ((c["votants"] / c["inscrits"] * 100).round(2) - c["taux_participation_pct"]).abs()
    if (participation_diff > 0.01).any():
        errors.append("some participation percentages do not match PDF metrics")

    blank_diff = (
        (c["bulletins_blancs_nombre"] / c["suffrages_exprimes"] * 100).round(2) - c["bulletins_blancs_pct"]
    ).abs()
    if (blank_diff > 0.01).any():
        errors.append("some blank-ballot percentages do not match PDF metrics")

    score_diff = ((df["scores"] / df["suffrages_exprimes"] * 100).round(2) - df["score_pct"]).abs()
    if (score_diff > 0.01).any():
        errors.append("some candidate score percentages do not match PDF metrics")

    elected_counts = df.groupby("circonscription_code")["elu"].sum()
    if not elected_counts.eq(1).all():
        errors.append("each circonscription must have exactly one elected row")

    return errors


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(clean_text(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def build_relational_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build normalized relational tables from the validated long dataframe."""
    circonscriptions = (
        df[
            [
                "circonscription_code",
                "region",
                "circonscription",
                "nb_bv",
                "inscrits",
                "votants",
                "taux_participation_pct",
                "bulletins_nuls",
                "suffrages_exprimes",
                "bulletins_blancs_nombre",
                "bulletins_blancs_pct",
            ]
        ]
        .drop_duplicates("circonscription_code")
        .copy()
    )
    circonscriptions["region_norm"] = circonscriptions["region"].map(normalize_entity)
    circonscriptions["circonscription_norm"] = circonscriptions["circonscription"].map(normalize_entity)
    page_bounds = (
        df.groupby("circonscription_code")["page"].agg(source_page_start="min", source_page_end="max").reset_index()
    )
    circonscriptions = circonscriptions.merge(page_bounds, on="circonscription_code")
    circonscriptions = circonscriptions[
        [
            "circonscription_code",
            "region",
            "region_norm",
            "circonscription",
            "circonscription_norm",
            "nb_bv",
            "inscrits",
            "votants",
            "taux_participation_pct",
            "bulletins_nuls",
            "suffrages_exprimes",
            "bulletins_blancs_nombre",
            "bulletins_blancs_pct",
            "source_page_start",
            "source_page_end",
        ]
    ].sort_values("circonscription_code")

    candidats = df[
        [
            "circonscription_code",
            "groupement_parti",
            "candidat_liste",
            "scores",
            "score_pct",
            "elu",
            "page",
        ]
    ].copy()
    candidats.insert(
        0,
        "candidat_id",
        [
            stable_id(
                "cand",
                row.circonscription_code,
                row.groupement_parti,
                row.candidat_liste,
                row.scores,
                row.score_pct,
                row.page,
            )
            for row in candidats.itertuples(index=False)
        ],
    )
    candidats["groupement_parti_norm"] = candidats["groupement_parti"].map(normalize_entity)
    candidats["candidat_liste_norm"] = candidats["candidat_liste"].map(normalize_entity)
    candidats = candidats[
        [
            "candidat_id",
            "circonscription_code",
            "groupement_parti",
            "groupement_parti_norm",
            "candidat_liste",
            "candidat_liste_norm",
            "scores",
            "score_pct",
            "elu",
            "page",
        ]
    ]

    alias_records: list[dict[str, Any]] = []
    for entity_type, source_df, value_column, norm_column in [
        ("region", circonscriptions, "region", "region_norm"),
        ("circonscription", circonscriptions, "circonscription", "circonscription_norm"),
        ("parti", candidats, "groupement_parti", "groupement_parti_norm"),
        ("candidat", candidats, "candidat_liste", "candidat_liste_norm"),
    ]:
        for value, norm_value in source_df[[value_column, norm_column]].drop_duplicates().itertuples(index=False):
            alias_records.append(
                {
                    "alias_id": stable_id("alias", entity_type, value, value),
                    "entity_type": entity_type,
                    "canonical_value": value,
                    "canonical_norm": norm_value,
                    "alias_value": value,
                    "alias_norm": norm_value,
                }
            )
    entity_aliases = pd.DataFrame(alias_records).drop_duplicates("alias_id")

    rag_records: list[dict[str, Any]] = []
    for row in circonscriptions.itertuples(index=False):
        text = (
            f"Circonscription {row.circonscription_code}: {row.circonscription}, "
            f"region {row.region}. Inscrits: {row.inscrits}; votants: {row.votants}; "
            f"participation: {row.taux_participation_pct}%; suffrages exprimes: "
            f"{row.suffrages_exprimes}; bulletins nuls: {row.bulletins_nuls}; "
            f"bulletins blancs: {row.bulletins_blancs_nombre}."
        )
        rag_records.append(
            {
                "chunk_id": stable_id("chunk", "circonscription", row.circonscription_code),
                "source_type": "circonscription",
                "source_id": str(row.circonscription_code),
                "source_page": int(row.source_page_start),
                "chunk_text": text,
                "chunk_text_norm": normalize_entity(text),
            }
        )
    for row in candidats.merge(
        circonscriptions[["circonscription_code", "region", "circonscription"]],
        on="circonscription_code",
    ).itertuples(index=False):
        text = (
            f"{row.candidat_liste} ({row.groupement_parti}) a obtenu {row.scores} voix "
            f"({row.score_pct}%) dans la circonscription {row.circonscription_code} - "
            f"{row.circonscription}, region {row.region}. "
            f"Elu: {'oui' if row.elu else 'non'}."
        )
        rag_records.append(
            {
                "chunk_id": stable_id("chunk", "candidat", row.candidat_id),
                "source_type": "candidat",
                "source_id": row.candidat_id,
                "source_page": int(row.page),
                "chunk_text": text,
                "chunk_text_norm": normalize_entity(text),
            }
        )
    rag_chunks = pd.DataFrame(rag_records)

    return {
        "circonscriptions": circonscriptions,
        "candidats": candidats,
        "entity_aliases": entity_aliases,
        "rag_chunks": rag_chunks,
    }


def write_duckdb(
    output_path: Path,
    tables: dict[str, pd.DataFrame],
    version_record: dict[str, Any] | None = None,
) -> None:
    """Create a DuckDB database and insert validated relational tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(output_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DROP VIEW IF EXISTS vw_turnout_by_region")
            conn.execute("DROP VIEW IF EXISTS vw_winners")
            conn.execute("DROP VIEW IF EXISTS vw_results_clean")
            for table in [
                "dataset_versions",
                "query_traces",
                "eval_cases",
                "rag_chunks",
                "entity_aliases",
                "candidats",
                "circonscriptions",
            ]:
                conn.execute(f"DROP TABLE IF EXISTS {table}")

            conn.execute(
                """
                CREATE TABLE circonscriptions (
                    circonscription_code INTEGER PRIMARY KEY,
                    region TEXT NOT NULL,
                    region_norm TEXT NOT NULL,
                    circonscription TEXT NOT NULL,
                    circonscription_norm TEXT NOT NULL,
                    nb_bv INTEGER,
                    inscrits INTEGER,
                    votants INTEGER,
                    taux_participation_pct DOUBLE,
                    bulletins_nuls INTEGER,
                    suffrages_exprimes INTEGER,
                    bulletins_blancs_nombre INTEGER,
                    bulletins_blancs_pct DOUBLE,
                    source_page_start INTEGER,
                    source_page_end INTEGER
                )
                """
            )
            if version_record:
                upsert_version_record(conn, version_record)
            conn.execute(
                """
                CREATE TABLE candidats (
                    candidat_id TEXT PRIMARY KEY,
                    circonscription_code INTEGER NOT NULL,
                    groupement_parti TEXT NOT NULL,
                    groupement_parti_norm TEXT NOT NULL,
                    candidat_liste TEXT NOT NULL,
                    candidat_liste_norm TEXT NOT NULL,
                    scores INTEGER,
                    score_pct DOUBLE,
                    elu BOOLEAN,
                    page INTEGER,
                    FOREIGN KEY (circonscription_code)
                        REFERENCES circonscriptions(circonscription_code)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE entity_aliases (
                    alias_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    canonical_value TEXT NOT NULL,
                    canonical_norm TEXT NOT NULL,
                    alias_value TEXT NOT NULL,
                    alias_norm TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_page INTEGER,
                    chunk_text TEXT NOT NULL,
                    chunk_text_norm TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE query_traces (
                    trace_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_question TEXT NOT NULL,
                    detected_intent TEXT,
                    route_used TEXT,
                    generated_sql TEXT,
                    sql_valid BOOLEAN,
                    rows_returned INTEGER,
                    latency_ms INTEGER,
                    final_answer TEXT,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE eval_cases (
                    eval_id TEXT PRIMARY KEY,
                    level TEXT,
                    question TEXT NOT NULL,
                    expected_answer TEXT,
                    expected_sql TEXT,
                    expected_route TEXT,
                    tolerance DOUBLE,
                    category TEXT
                )
                """
            )

            for name, frame in tables.items():
                conn.register(f"{name}_df", frame)
                conn.execute(f"INSERT INTO {name} SELECT * FROM {name}_df")
                conn.unregister(f"{name}_df")

            conn.execute(
                """
                CREATE VIEW vw_results_clean AS
                SELECT
                    c.region,
                    c.region_norm,
                    c.circonscription_code,
                    c.circonscription,
                    c.circonscription_norm,
                    c.nb_bv,
                    c.inscrits,
                    c.votants,
                    c.taux_participation_pct,
                    c.bulletins_nuls,
                    c.suffrages_exprimes,
                    c.bulletins_blancs_nombre,
                    c.bulletins_blancs_pct,
                    k.candidat_id,
                    k.groupement_parti,
                    k.groupement_parti_norm,
                    k.candidat_liste,
                    k.candidat_liste_norm,
                    k.scores,
                    k.score_pct,
                    k.elu,
                    k.page
                FROM candidats k
                JOIN circonscriptions c
                ON k.circonscription_code = c.circonscription_code
                """
            )
            conn.execute("CREATE VIEW vw_winners AS SELECT * FROM vw_results_clean WHERE elu")
            conn.execute(
                """
                CREATE VIEW vw_turnout_by_region AS
                SELECT
                    region,
                    SUM(inscrits) AS inscrits,
                    SUM(votants) AS votants,
                    ROUND(SUM(votants) * 100.0 / SUM(inscrits), 2)
                        AS taux_participation_pct
                FROM circonscriptions
                GROUP BY region
                """
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def validate_duckdb(db_path: Path) -> list[str]:
    errors: list[str] = []
    with duckdb.connect(str(db_path), read_only=True) as conn:
        checks = {
            "circonscriptions": conn.execute("SELECT COUNT(*) FROM circonscriptions").fetchone()[0],
            "candidats": conn.execute("SELECT COUNT(*) FROM candidats").fetchone()[0],
            "winners": conn.execute("SELECT COUNT(*) FROM candidats WHERE elu").fetchone()[0],
            "rag_chunks": conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0],
            "dataset_versions": conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0],
            "missing_fk": conn.execute(
                """
                SELECT COUNT(*)
                FROM candidats k
                LEFT JOIN circonscriptions c USING (circonscription_code)
                WHERE c.circonscription_code IS NULL
                """
            ).fetchone()[0],
            "bad_votants": conn.execute(
                """
                SELECT COUNT(*)
                FROM circonscriptions
                WHERE votants != bulletins_nuls + suffrages_exprimes
                """
            ).fetchone()[0],
            "bad_scores": conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        c.circonscription_code,
                        SUM(k.scores) + c.bulletins_blancs_nombre AS recomputed,
                        c.suffrages_exprimes
                    FROM circonscriptions c
                    JOIN candidats k USING (circonscription_code)
                    GROUP BY
                        c.circonscription_code,
                        c.bulletins_blancs_nombre,
                        c.suffrages_exprimes
                )
                WHERE recomputed != suffrages_exprimes
                """
            ).fetchone()[0],
            "bad_winner_count": conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT circonscription_code, SUM(CASE WHEN elu THEN 1 ELSE 0 END) n
                    FROM candidats
                    GROUP BY circonscription_code
                )
                WHERE n != 1
                """
            ).fetchone()[0],
        }

    expected = {
        "circonscriptions": 205,
        "candidats": 1125,
        "winners": 205,
        "rag_chunks": 1330,
        "dataset_versions": 1,
        "missing_fk": 0,
        "bad_votants": 0,
        "bad_scores": 0,
        "bad_winner_count": 0,
    }
    for key, expected_value in expected.items():
        if checks[key] != expected_value:
            errors.append(f"{key}: expected {expected_value}, got {checks[key]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EDAN 2025 PDF election result tables to DuckDB.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input PDF path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output DuckDB path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_path = Path(args.output)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    df = build_dataframe(pdf_path)
    if df.empty:
        raise RuntimeError("No candidate rows were extracted from the PDF.")
    validation_errors = validate_dataframe(df)
    if validation_errors:
        raise RuntimeError("Extracted data validation failed:\n- " + "\n- ".join(validation_errors))

    tables = build_relational_tables(df)
    version_record = build_version_record(
        pdf_path,
        circonscription_count=len(tables["circonscriptions"]),
        candidate_count=len(tables["candidats"]),
        chunk_count=len(tables["rag_chunks"]),
    )
    write_duckdb(output_path, tables, version_record)
    db_errors = validate_duckdb(output_path)
    if db_errors:
        raise RuntimeError("DuckDB validation failed:\n- " + "\n- ".join(db_errors))

    print(f"DuckDB written: {output_path}")
    print(f"Candidates: {len(tables['candidats'])}")
    print(f"Circonscriptions: {df['circonscription_code'].nunique()}")
    print(f"Elected rows: {int(df['elu'].sum())}")
    print(f"RAG chunks: {len(tables['rag_chunks'])}")
    print(f"Dataset version: {version_record['version_id']} (pending embeddings)")


if __name__ == "__main__":
    main()
