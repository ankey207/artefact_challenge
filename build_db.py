"""
Orchestrateur principal — reconstruit la base DuckDB EDAN 2025 en 4 étapes.

Usage:
    python build_db.py docs/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf

Étapes:
    1. Extraction PDF → DuckDB (tables, vues, RAG chunks)
    2. Correction des vues (ajout region_norm dans vw_turnout_by_region)
    3. Construction des alias d'entités (partis, régions, circonscriptions)
    4. Calcul des embeddings vectoriels (modèle multilingue 384 dims)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DB_BUILDERS = ROOT / "db_builders"
DB_PATH = ROOT / "edan_2025_resultat_national_details.duckdb"

STEPS = [
    {
        "label": "Étape 1/4 — Extraction PDF → DuckDB",
        "script": DB_BUILDERS / "step1_extract_pdf.py",
        "extra_args": lambda pdf: [str(pdf), "-o", str(DB_PATH)],
    },
    {
        "label": "Étape 2/4 — Correction des vues (region_norm)",
        "script": DB_BUILDERS / "step2_fix_views.py",
        "extra_args": lambda _: [str(DB_PATH)],
    },
    {
        "label": "Étape 3/4 — Construction des alias d'entités",
        "script": DB_BUILDERS / "step3_build_aliases.py",
        "extra_args": lambda _: [str(DB_PATH)],
    },
    {
        "label": "Étape 4/4 — Calcul des embeddings vectoriels",
        "script": DB_BUILDERS / "step4_build_embeddings.py",
        "extra_args": lambda pdf: [str(DB_PATH), "--pdf", str(pdf)],
    },
]

BAR = "=" * 62


def run_step(label: str, script: Path, extra_args: list[str]) -> None:
    print(f"\n{BAR}")
    print(f"  {label}")
    print(BAR)
    result = subprocess.run(
        [sys.executable, str(script)] + extra_args,
        check=False,
    )
    if result.returncode != 0:
        print(f"\n[ERREUR] L'étape a échoué (code {result.returncode}). Arrêt.")
        sys.exit(result.returncode)
    print("\n  OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruit la base DuckDB EDAN 2025 complète.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default="docs/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf",
        help="Chemin vers le PDF source. Default: docs/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf",
    )
    cli = parser.parse_args()

    pdf_path = Path(cli.pdf)
    if not pdf_path.exists():
        print(f"ERREUR: PDF introuvable — {pdf_path}", file=sys.stderr)
        print("Usage: python build_db.py docs/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf", file=sys.stderr)
        sys.exit(1)

    print(f"\n{BAR}")
    print("  Construction de la base DuckDB EDAN 2025")
    print(f"  PDF source : {pdf_path}")
    print(f"  DB cible   : {DB_PATH}")
    print(BAR)

    for step in STEPS:
        run_step(
            label=step["label"],
            script=step["script"],
            extra_args=step["extra_args"](pdf_path),
        )

    print(f"\n{BAR}")
    print(f"  Base prête : {DB_PATH}")
    print("  Lancer l'app : streamlit run app.py")
    print(BAR)


if __name__ == "__main__":
    main()
