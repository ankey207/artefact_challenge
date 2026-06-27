"""
Build script — calcule et stocke les embeddings vectoriels pour tous les rag_chunks.

Doit être exécuté APRÈS step3_build_aliases.py.

Usage:
    python db_builders/step4_build_embeddings.py edan_2025_resultat_national_details.duckdb

Modèle : paraphrase-multilingual-MiniLM-L12-v2 (384 dims, ~470 MB au premier lancement)
         Inférence locale — pas de clé API, pas d'appel réseau au moment des requêtes.
         Supporte le français, l'anglais et 50+ langues.

Après ce script, rag_retriever.py bascule automatiquement en mode vector search.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_engineer_app.dataset_version import EMBEDDING_MODEL, refresh_dataset_version

EMBED_MODEL = EMBEDDING_MODEL
BATCH_SIZE = 128


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcule et stocke les embeddings vectoriels pour rag_chunks.")
    parser.add_argument(
        "db",
        nargs="?",
        default="edan_2025_resultat_national_details.duckdb",
        help="Chemin vers la base DuckDB.",
    )
    parser.add_argument(
        "--pdf",
        default="docs/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf",
        help="PDF source utilisé pour calculer la version du dataset.",
    )
    args = parser.parse_args()
    DB_PATH = Path(args.db)

    if not DB_PATH.exists():
        print(f"ERREUR: base DuckDB introuvable: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    # Load model (downloads ~470 MB on first run, cached afterwards)
    print(f"Loading sentence-transformers model '{EMBED_MODEL}'...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer(EMBED_MODEL)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    con = duckdb.connect(str(DB_PATH), read_only=False)

    # Add embedding column if not present
    existing_cols = [r[0] for r in con.execute("DESCRIBE rag_chunks").fetchall()]
    if "embedding" not in existing_cols:
        con.execute("ALTER TABLE rag_chunks ADD COLUMN embedding FLOAT[]")
        print("Added embedding column to rag_chunks.")
    else:
        print("embedding column already exists.")

    # Load only chunks that still need embedding
    pending = con.execute("SELECT chunk_id, chunk_text FROM rag_chunks WHERE embedding IS NULL").fetchall()

    if not pending:
        total = con.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        print(f"All {total} chunks already have embeddings. Nothing to do.")
        con.close()
        pdf_path = Path(args.pdf)
        if pdf_path.exists():
            refresh_dataset_version(DB_PATH, pdf_path)
        return

    total_pending = len(pending)
    print(f"Computing embeddings for {total_pending} chunks...")

    ids = [r[0] for r in pending]
    texts = [r[1] for r in pending]

    # Encode all chunks in one call (sentence-transformers batches internally)
    vectors: np.ndarray = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize → dot product = cosine similarity
    )

    print("Storing embeddings in DuckDB...")
    for chunk_id, vec in zip(ids, vectors):
        con.execute(
            "UPDATE rag_chunks SET embedding = ? WHERE chunk_id = ?",
            [vec.tolist(), chunk_id],
        )

    ready = con.execute("SELECT COUNT(*) FROM rag_chunks WHERE embedding IS NOT NULL").fetchone()[0]
    total_all = con.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    dim = vectors.shape[1]
    con.close()
    pdf_path = Path(args.pdf)
    if pdf_path.exists():
        version = refresh_dataset_version(DB_PATH, pdf_path)
        print(f"Dataset version: {version['version_id']} ({version['build_status']})")

    print(f"\nDone. {ready}/{total_all} chunks now have embeddings (dim={dim}).")
    print("The app will automatically use vector search on next startup.")


if __name__ == "__main__":
    main()
