"""
RAG retrieval.

Two retrieval strategies, selected automatically:

  1. Vector search (preferred) — cosine similarity between the query embedding
     and pre-computed chunk embeddings stored in rag_chunks.embedding (FLOAT[]).
     Requires build_embeddings.py to have been run first.
     Uses numpy for fast batch cosine similarity (~1330 vectors, <1 ms).

  2. Keyword fallback — fraction of query keywords present in chunk_text_norm.
     Used when embeddings are not yet built (no embedding column or NULLs).

Both strategies share the same cache: chunks are loaded once at first call.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata

import numpy as np

from .dataset_version import EMBEDDING_MODEL
from .observability import record_current_event

# Silence HuggingFace Hub network checks and progress bars.
# The model is already cached locally after build_embeddings.py ran;
# no network access is needed at inference time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_NORMALIZE_RE = re.compile(r"[^A-Z0-9\s]")

STOPWORDS: frozenset[str] = frozenset(
    {
        "LE",
        "LA",
        "LES",
        "DE",
        "DU",
        "DES",
        "ET",
        "EN",
        "AU",
        "AUX",
        "UN",
        "UNE",
        "CE",
        "CET",
        "CETTE",
        "SUR",
        "DANS",
        "PAR",
        "POUR",
        "AVEC",
        "OU",
        "QUI",
        "QUE",
        "QUOI",
        "QUEL",
        "QUELLE",
        "COMMENT",
        "COMBIEN",
        "EST",
        "SONT",
        "ETE",
        "THE",
        "A",
        "IN",
        "OF",
        "AND",
        "OR",
        "BY",
        "FROM",
        "WITH",
        "ARE",
        "WAS",
        "SHOW",
        "LIST",
        "FIND",
        "TELL",
        "GIVE",
        "MONTRE",
        "LISTE",
        "DIS",
        "DONNE",
        "AFFICHE",
        "TOP",
        "MOI",
        "ME",
        "LUI",
        "ILS",
        "ELLES",
        "VOUS",
        "NOUS",
        "OBTENU",
        "VOIX",
        "REGION",
        "CIRCONSCRIPTION",
        "INSCRITS",
        "VOTANTS",
        "PARTICIPATION",
        "SUFFRAGES",
        "EXPRIMES",
        "BULLETINS",
        "NULS",
        "BLANCS",
        "ELU",
        "OUI",
        "NON",
    }
)

EMBED_MODEL = EMBEDDING_MODEL

_CHUNKS_CACHE: list[dict] | None = None
_EMBED_MATRIX: np.ndarray | None = None  # shape (N, dim), float32 — pre-normalized
_ST_MODEL = None  # SentenceTransformer, loaded once on demand


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode().upper()
    return _NORMALIZE_RE.sub(" ", text).strip()


def _load_chunks() -> list[dict]:
    """
    Load all chunks from rag_chunks.  If the embedding column exists and all
    rows have values, also build the numpy embedding matrix for vector search.
    """
    global _CHUNKS_CACHE, _EMBED_MATRIX
    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    from .db import connect

    with connect() as conn:
        cols = [r[0] for r in conn.execute("DESCRIBE rag_chunks").fetchall()]
        has_embed = "embedding" in cols

        sql = (
            "SELECT chunk_id, source_type, source_id, source_page, "
            "       chunk_text, chunk_text_norm" + (", embedding" if has_embed else "") + " FROM rag_chunks"
        )
        rows = conn.execute(sql).fetchall()

    _CHUNKS_CACHE = [
        {
            "chunk_id": r[0],
            "source_type": r[1],
            "source_id": r[2],
            "source_page": r[3],
            "chunk_text": r[4],
            "chunk_text_norm": r[5],
        }
        for r in rows
    ]

    if has_embed:
        vecs = [r[6] for r in rows]
        if all(v is not None for v in vecs):
            _EMBED_MATRIX = np.array(vecs, dtype=np.float32)

    return _CHUNKS_CACHE


def _cosine_scores(query_vec: list[float]) -> np.ndarray:
    """Return cosine similarity of query_vec against every row of _EMBED_MATRIX."""
    q = np.array(query_vec, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    norms = np.linalg.norm(_EMBED_MATRIX, axis=1, keepdims=True) + 1e-9
    return (_EMBED_MATRIX / norms) @ q_norm  # shape (N,)


def extract_keywords(question: str) -> list[str]:
    """Return meaningful tokens (no stopwords, length >= 3) from the question."""
    norm = _normalize(question)
    return [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]


_VECTOR_WEIGHT = 0.60
_KEYWORD_WEIGHT = 0.40


def _retrieve_chunks_impl(
    question: str,
    top_k: int | None = None,
    source_type: str | None = None,
) -> list[dict]:
    """
    Return at most `top_k` chunks most relevant to `question`.

    Scoring strategy (automatic, in priority order):
      - hybrid  : 60 % cosine vector + 40 % keyword overlap  (when embeddings ready)
      - keyword : keyword overlap only                         (fallback, no embeddings)

    Each returned dict contains:
        chunk_id, source_type, source_id, source_page, chunk_text,
        chunk_text_norm, score, retrieval_mode
    """
    from .config import RAG_TOP_K

    if top_k is None:
        top_k = RAG_TOP_K

    chunks = _load_chunks()

    # ---- 1. keyword scores (always computed — cheap, no model needed) -------
    keywords = extract_keywords(question)
    kw_scores: list[float] = []
    if keywords:
        for chunk in chunks:
            hits = sum(1 for kw in keywords if kw in chunk["chunk_text_norm"])
            kw_scores.append(hits / len(keywords))
    else:
        kw_scores = [0.0] * len(chunks)

    # ---- 2. vector scores (only when embeddings are available) --------------
    vec_scores: list[float] | None = None
    if _EMBED_MATRIX is not None:
        try:
            global _ST_MODEL
            if _ST_MODEL is None:
                from sentence_transformers import SentenceTransformer

                _ST_MODEL = SentenceTransformer(EMBED_MODEL)

            query_vec = _ST_MODEL.encode(
                question,
                convert_to_numpy=True,
                normalize_embeddings=True,  # chunks were normalized at build time
            )
            vec_scores = (_EMBED_MATRIX @ query_vec).tolist()
        except Exception:
            vec_scores = None  # model error → keyword-only fallback

    # ---- 3. combine & rank --------------------------------------------------
    results: list[dict] = []
    for i, chunk in enumerate(chunks):
        if source_type and chunk["source_type"] != source_type:
            continue

        kw = kw_scores[i]

        if vec_scores is not None:
            v = vec_scores[i]
            score = _VECTOR_WEIGHT * v + _KEYWORD_WEIGHT * kw
            mode = "hybrid"
        else:
            score = kw
            mode = "keyword"

        if score > 0:
            provenance = {
                "source_page": chunk.get("source_page"),
                "source_type": chunk.get("source_type"),
                "source_id": chunk.get("source_id"),
                "chunk_id": chunk.get("chunk_id"),
            }
            results.append(
                {
                    **chunk,
                    "score": round(score, 4),
                    "retrieval_mode": mode,
                    "provenance": provenance,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def retrieve_chunks(
    question: str,
    top_k: int | None = None,
    source_type: str | None = None,
) -> list[dict]:
    """Run hybrid retrieval and emit bounded operational metrics."""
    started = time.perf_counter()
    cache_hit = _CHUNKS_CACHE is not None
    model_cache_hit = _ST_MODEL is not None
    try:
        results = _retrieve_chunks_impl(question, top_k, source_type)
    except Exception as exc:
        record_current_event(
            "rag_retrieval",
            duration_ms=(time.perf_counter() - started) * 1_000,
            status="error",
            payload={
                "error_type": type(exc).__name__,
                "top_k": top_k,
                "source_type_filter": bool(source_type),
                "cache_hit": cache_hit,
                "embedding_matrix_available": _EMBED_MATRIX is not None,
            },
        )
        raise

    scores = [float(item["score"]) for item in results if item.get("score") is not None]
    modes = sorted({str(item["retrieval_mode"]) for item in results if item.get("retrieval_mode")})
    record_current_event(
        "rag_retrieval",
        duration_ms=(time.perf_counter() - started) * 1_000,
        payload={
            "mode": modes[0] if len(modes) == 1 else modes,
            "result_count": len(results),
            "keyword_count": len(extract_keywords(question)),
            "top_k": top_k,
            "source_type_filter": bool(source_type),
            "cache_hit": cache_hit,
            "candidate_chunk_count": len(_CHUNKS_CACHE or []),
            "embedding_matrix_available": _EMBED_MATRIX is not None,
            "embedding_model": (EMBED_MODEL if _EMBED_MATRIX is not None else None),
            "embedding_model_cache_hit": model_cache_hit,
            "top_score": max(scores) if scores else None,
            "lowest_score": min(scores) if scores else None,
            "source_pages": sorted(
                {int(item["source_page"]) for item in results if item.get("source_page") is not None}
            )[:20],
        },
    )
    return results


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered list for the LLM prompt."""
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        page_ref = f" (page {c['source_page']})" if c.get("source_page") else ""
        row_ref = ""
        if c.get("source_type") and c.get("source_id"):
            row_ref = f" [{c['source_type']}:{c['source_id']}; chunk:{c.get('chunk_id', 'n/a')}]"
        lines.append(f"{i}. {c['chunk_text']}{page_ref}{row_ref}")
    return "\n".join(lines)
