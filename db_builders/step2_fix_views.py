"""
Build script — corrige ou crée les vues DuckDB qui nécessitent region_norm.

Doit être exécuté APRÈS step1_extract_pdf.py.

Usage:
    python db_builders/step2_fix_views.py edan_2025_resultat_national_details.duckdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb


def fix_views(db_path: Path) -> None:
    con = duckdb.connect(str(db_path), read_only=False)

    # vw_turnout_by_region : recrée avec region_norm + suffrages_exprimes + bulletins_nuls
    con.execute("""
        CREATE OR REPLACE VIEW vw_turnout_by_region AS
        SELECT
            region,
            upper(regexp_replace(region, '[^A-Z0-9 ]', ' ', 'g')) AS region_norm,
            sum(inscrits)                                            AS inscrits,
            sum(votants)                                             AS votants,
            sum(suffrages_exprimes)                                  AS suffrages_exprimes,
            sum(bulletins_nuls)                                      AS bulletins_nuls,
            round(sum(votants) * 100.0 / NULLIF(sum(inscrits), 0), 2) AS taux_participation_pct
        FROM circonscriptions
        GROUP BY region
    """)
    rows = con.execute("SELECT COUNT(*) FROM vw_turnout_by_region").fetchone()[0]
    print(f"  vw_turnout_by_region : {rows} régions")

    # vw_national_summary : totaux nationaux pré-calculés (évite la multiplication par candidats)
    con.execute("""
        CREATE OR REPLACE VIEW vw_national_summary AS
        SELECT
            count(DISTINCT circonscription_code)                        AS nb_circonscriptions,
            sum(inscrits)                                               AS total_inscrits,
            sum(votants)                                               AS total_votants,
            sum(suffrages_exprimes)                                     AS total_suffrages_exprimes,
            sum(bulletins_nuls)                                         AS total_bulletins_nuls,
            round(sum(votants) * 100.0 / NULLIF(sum(inscrits), 0), 2)  AS taux_participation_pct
        FROM circonscriptions
    """)
    r = con.execute("SELECT taux_participation_pct, total_suffrages_exprimes FROM vw_national_summary").fetchone()
    print(f"  vw_national_summary  : taux={r[0]}%  suffrages={r[1]:,}")

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrige les vues DuckDB après extraction PDF.")
    parser.add_argument(
        "db",
        nargs="?",
        default="edan_2025_resultat_national_details.duckdb",
        help="Chemin vers la base DuckDB. Default: edan_2025_resultat_national_details.duckdb",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERREUR: base DuckDB introuvable: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Connexion à {db_path}")
    fix_views(db_path)
    print("Done.")


if __name__ == "__main__":
    main()
