"""
Build script — enrichit entity_aliases avec les variantes d'orthographe
(partis, régions, circonscriptions).

Doit être exécuté APRÈS step2_fix_views.py.

Usage:
    python db_builders/step3_build_aliases.py edan_2025_resultat_national_details.duckdb
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9\s]", " ", text).strip()


def _alias_id(canonical: str, alias: str) -> str:
    return "alias_" + hashlib.md5(f"{canonical}|{alias}".encode()).hexdigest()[:16]


def _dot_separated(acronym: str) -> str:
    """RHDP -> R.H.D.P"""
    return ".".join(list(acronym))


def _is_acronym(s: str) -> bool:
    """True if s is 2-6 uppercase letters with no spaces."""
    return bool(re.fullmatch(r"[A-Z]{2,6}", s))


def _extract_cities(chunk_text: str) -> list[str]:
    """
    Extract city/sub-prefecture names from a circumscription chunk.
    Example input: 'Circonscription 001: AGBOVILLE, ABOUDE, ..., region AGNEBY-TIASSA.'
    """
    # Strip everything after ", region "
    text = re.sub(r",?\s*region\s+.*$", "", chunk_text, flags=re.IGNORECASE)
    # Strip the "Circonscription NNN:" prefix
    text = re.sub(r"^[^:]+:\s*", "", text)
    # Strip trailing notes / parentheses
    text = re.sub(r"\([^)]*\)", " ", text)
    # Remove known noise phrases
    noise = [
        "COMMUNES ET SOUS",
        "PREFECTURES",
        "SOUS PREFECTURE",
        "COMMUNE",
        "SOUS-PREFECTURE",
        "ET SOUS-",
        "COMMUNES",
    ]
    for n in noise:
        text = text.replace(n, " ")
    raw = [c.strip().strip("-").strip() for c in text.split(",")]
    # Also split on " ET " to extract individual place names
    cities: list[str] = []
    for piece in raw:
        sub = [s.strip() for s in re.split(r"\s+ET\s+", piece, flags=re.IGNORECASE)]
        cities.extend(sub)
    return [c for c in cities if len(c) >= 4]


# ---------------------------------------------------------------------------
# Alias generators
# ---------------------------------------------------------------------------


def party_aliases(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Return (canonical_value, canonical_norm, alias_value, alias_norm) rows."""
    rows = con.execute("SELECT DISTINCT groupement_parti, groupement_parti_norm FROM vw_results_clean").fetchall()

    aliases: list[tuple] = []

    manual: dict[str, list[str]] = {
        "RHDP": [
            "R.H.D.P",
            "R.H.D.P.",
            "RHDP-CI",
            "RASSEMBLEMENT DES HOUPHOUETISTES",
            "RASSEMBLEMENT HOUPHOUETISTES",
            "HOUPHOUETISTES",
        ],
        "PDCI-RDA": [
            "PDCI",
            "P.D.C.I",
            "P.D.C.I-R.D.A",
            "PDCI RDA",
            "PARTI DEMOCRATIQUE DE COTE D IVOIRE",
            "PARTI DEMOCRATIQUE",
        ],
        "FPI": ["F.P.I", "FRONT POPULAIRE IVOIRIEN", "FRONT POPULAIRE"],
        "INDEPENDANT": [
            "IND",
            "INDEPENDANTE",
            "CANDIDAT INDEPENDANT",
            "LISTE INDEPENDANTE",
            "SANS PARTI",
        ],
        "EDS": ["E.D.S", "ENSEMBLE POUR LA DEMOCRATIE ET LA SOUVERAINETE"],
    }

    for canonical, canon_norm in rows:
        added = set()

        # Auto: dot-separated acronym
        clean_norm = canon_norm.strip()
        if _is_acronym(clean_norm):
            dot = _dot_separated(clean_norm)
            if dot != canonical:
                added.add((canonical, _norm(canonical), dot, _norm(dot)))

        # Auto: hyphen-to-space already in canon_norm; add hyphen-to-nothing
        no_sep = re.sub(r"[-\s]+", "", canonical)
        if no_sep != canonical and len(no_sep) >= 2:
            added.add((canonical, _norm(canonical), no_sep, _norm(no_sep)))

        # Manual aliases
        for src_key, alt_list in manual.items():
            if canonical == src_key or canon_norm == _norm(src_key):
                for alt in alt_list:
                    added.add((canonical, _norm(canonical), alt, _norm(alt)))

        aliases.extend(added)

    return aliases


def region_aliases(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    rows = con.execute("SELECT DISTINCT region, region_norm FROM vw_results_clean").fetchall()

    aliases: list[tuple] = []
    manual: dict[str, list[str]] = {
        "DISTRICT AUTONOME D'ABIDJAN": [
            "ABIDJAN",
            "DISTRICT ABIDJAN",
            "DA ABIDJAN",
            "DISTRICT D ABIDJAN",
        ],
        "DISTRICT AUTONOME DE YAMOUSSOUKRO": [
            "YAMOUSSOUKRO",
            "DA YAMOUSSOUKRO",
            "DISTRICT YAMOUSSOUKRO",
        ],
        "AGNEBY-TIASSA": ["AGNEBI TIASSA", "AGNEBY TIASSA"],
        "BOUNKANI": ["BOUNKANI", "BOUNKANY", "BOUNCANI", "BOUNCANY", "BOUKANI", "BOUKANY", "BOUCANI", "BOUCANY"],
        "CAVALLY": ["CAVALLY", "CAVALLI", "CAVALY", "CAVALI"],
        "GRANDS-PONTS": ["GRAND PONT", "GRAND-PONT", "GRAND-PONTS", "GRAND PONTS", "GRANDS PONTS"],
        "HAMBOL": ["AMBOL", "HAMBOLE", "AMBOLE"],
        "HAUT-SASSANDRA": ["HAUT SASSANDRA", "HAUT SASANDRA", "HAUT-SASANDRA"],
        "LOH-DJIBOUA": ["LOH DJIBOUA", "LOH DJIBOU"],
        "INDENIE-DJUABLIN": ["INDENIE DJUABLIN", "INDENIÉ DJUABLIN"],
        "MARAHOUE": ["MARAOUE", "MARRAHOUE"],
        "LAME": ["LA ME"],
        "SAN-PEDRO": ["SAN PEDRO"],
        "TONKPI": ["TONPKI", "TONPI"],
        "SUD-COMOE": ["SUD COMOE", "SUD-COMOÉ", "SUD COMOÉ"],
        "N'ZI": ["NZI", "N ZI"],
    }

    for canonical, canon_norm in rows:
        for src_key, alt_list in manual.items():
            if canonical == src_key or canon_norm == _norm(src_key):
                for alt in alt_list:
                    aliases.append((canonical, _norm(canonical), alt, _norm(alt)))

    return aliases


def _lookup_circ(con: duckdb.DuckDBPyConnection, source_id: str):
    """Resolve source_id (zero-padded like '047') to circumscription row."""
    # rag_chunks.source_id is zero-padded ('047'); circonscription_code is INTEGER (47)
    # Match both formats to handle rebuild order differences
    for code_str in [source_id, source_id.lstrip("0") or "0"]:
        row = con.execute(
            "SELECT circonscription, circonscription_norm FROM circonscriptions "
            "WHERE LPAD(circonscription_code::VARCHAR, 3, '0') = ?",
            [code_str.zfill(3)],
        ).fetchone()
        if row:
            return row
    return None


def circonscription_aliases(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """Add city-name aliases and short-form (city before comma) aliases from chunk_text."""
    rows = con.execute("SELECT source_id, chunk_text FROM rag_chunks WHERE source_type = 'circonscription'").fetchall()

    aliases: list[tuple] = []
    for source_id, chunk_text in rows:
        name_row = _lookup_circ(con, source_id)
        if not name_row:
            continue
        canonical, canon_norm = name_row

        # Short-form alias: first token before comma (e.g. "YOPOUGON" from "YOPOUGON,COMMUNE")
        short = canonical.split(",")[0].strip()
        short_norm = _norm(short)
        if short_norm and short_norm != canon_norm and len(short_norm) >= 4:
            aliases.append((canonical, canon_norm, short, short_norm))

        # City-name aliases extracted from chunk text
        for city in _extract_cities(chunk_text):
            city_norm = _norm(city)
            if city_norm and city_norm != canon_norm and len(city_norm) >= 4:
                aliases.append((canonical, canon_norm, city, city_norm))

    return aliases


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def upsert_aliases(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> int:
    inserted = 0
    for canonical_value, canonical_norm, alias_value, alias_norm in rows:
        aid = _alias_id(canonical_norm, alias_norm)
        existing = con.execute("SELECT 1 FROM entity_aliases WHERE alias_id = ?", [aid]).fetchone()
        if not existing:
            con.execute(
                "INSERT INTO entity_aliases VALUES (?, ?, ?, ?, ?, ?)",
                [aid, "parti", canonical_value, canonical_norm, alias_value, alias_norm],
            )
            inserted += 1
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrichit entity_aliases avec les variantes d'orthographe.")
    parser.add_argument(
        "db",
        nargs="?",
        default="edan_2025_resultat_national_details.duckdb",
        help="Chemin vers la base DuckDB.",
    )
    args = parser.parse_args()
    DB_PATH = Path(args.db)

    if not DB_PATH.exists():
        print(f"ERREUR: base DuckDB introuvable: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    con = duckdb.connect(str(DB_PATH), read_only=False)
    print(f"Connected to {DB_PATH}")

    # --- Party aliases ---
    p_aliases = party_aliases(con)
    for row in p_aliases:
        aid = _alias_id(row[1], row[3])
        if not con.execute("SELECT 1 FROM entity_aliases WHERE alias_id = ?", [aid]).fetchone():
            con.execute(
                "INSERT INTO entity_aliases VALUES (?, 'parti', ?, ?, ?, ?)",
                [aid, row[0], row[1], row[2], row[3]],
            )
    print(f"  Party aliases processed: {len(p_aliases)}")

    # --- Region aliases ---
    r_aliases = region_aliases(con)
    for row in r_aliases:
        aid = _alias_id(row[1], row[3])
        if not con.execute("SELECT 1 FROM entity_aliases WHERE alias_id = ?", [aid]).fetchone():
            con.execute(
                "INSERT INTO entity_aliases VALUES (?, 'region', ?, ?, ?, ?)",
                [aid, row[0], row[1], row[2], row[3]],
            )
    print(f"  Region aliases processed: {len(r_aliases)}")

    # --- Circumscription aliases (cities) ---
    try:
        c_aliases = circonscription_aliases(con)
        for row in c_aliases:
            aid = _alias_id(row[1], row[3])
            if not con.execute("SELECT 1 FROM entity_aliases WHERE alias_id = ?", [aid]).fetchone():
                con.execute(
                    "INSERT INTO entity_aliases VALUES (?, 'circonscription', ?, ?, ?, ?)",
                    [aid, row[0], row[1], row[2], row[3]],
                )
        print(f"  Circumscription city aliases processed: {len(c_aliases)}")
    except Exception as exc:
        print(f"  Circumscription aliases skipped: {exc}")

    # --- Summary ---
    total = con.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0]
    print(f"\nTotal aliases in DB: {total}")
    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
