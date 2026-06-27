"""
Entity resolution.

Fuzzy-matches tokens/n-grams from the user question against known entities
(parties, regions, circumscriptions) stored in entity_aliases.

Uses difflib.SequenceMatcher (built-in): no extra dependencies.
Candidates only include entity_type in (parti, region, circonscription) to keep
the candidate set small (~280 entries) and avoid false positives on candidate names.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

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
        "PAR",
        "QUI",
        "QUE",
        "QUOI",
        "QUEL",
        "QUELLE",
        "QUELS",
        "QUELLES",
        "COMBIEN",
        "COMMENT",
        "EST",
        "SONT",
        "THE",
        "A",
        "IN",
        "OF",
        "AND",
        "OR",
        "BY",
        "FROM",
        "SHOW",
        "LIST",
        "GET",
        "FIND",
        "TELL",
        "GIVE",
        "MONTRE",
        "LISTE",
        "DIS",
        "DONNE",
        "AFFICHE",
        "TOP",
        "LES",
        "ME",
        "MOI",
        "GAGNE",
        "GAGNES",
        "GAGNER",
        "GAGNANT",
        "REMPORTE",
        "REMPORTES",
        "OBTENU",
        "SIEGE",
        "SIEGES",
        "CANDIDAT",
        "CANDIDATS",
        "PARTI",
        "PARTIS",
        "REGION",
        "REGIONS",
        "RESULTAT",
        "RESULTATS",
        "ELECTION",
        "ELECTIONS",
        "TOUT",
        "TOUS",
        "TOUTE",
        "TOUTES",
        "CES",
        "ZONE",
        "ZONES",
        "DOMINE",
        "DOMINENT",
        "RESTANT",
        "RESTANTE",
        "RESTANTS",
        "RESTANTES",
    }
)

_ALIASES_CACHE: list[dict] | None = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode().upper()
    # Remove straight apostrophes silently so that "N'ZI" → "NZI" (one token, not
    # "N" + "ZI" which are both too short to survive the len≥3 token filter).
    text = text.replace("'", "")
    return _NORMALIZE_RE.sub(" ", text).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _load_aliases() -> list[dict]:
    global _ALIASES_CACHE
    if _ALIASES_CACHE is None:
        from .db import connect

        with connect() as conn:
            rows = conn.execute(
                "SELECT canonical_value, canonical_norm, alias_norm, entity_type "
                "FROM entity_aliases "
                "WHERE entity_type IN ('parti', 'region', 'circonscription')"
            ).fetchall()
        _ALIASES_CACHE = [
            {
                "canonical_value": r[0],
                "canonical_norm": r[1],
                "alias_norm": r[2],
                "entity_type": r[3],
            }
            for r in rows
        ]
    return _ALIASES_CACHE


def format_option_label(canonical_value: str, entity_type: str) -> str:
    """Human-readable label for a disambiguation option (used in QCM)."""
    prefix = {"region": "Région", "circonscription": "Circonscription", "parti": "Parti"}.get(
        entity_type, entity_type.capitalize()
    )
    # Shorten long canonical names (e.g. full circumscription descriptions)
    if len(canonical_value) > 55:
        parts = canonical_value.split(",")
        short = ", ".join(p.strip() for p in parts[:2])
        if len(short) > 55:
            short = short[:52] + "..."
        elif len(parts) > 2:
            short += "…"
        return f"{prefix} : {short}"
    return f"{prefix} : {canonical_value}"


def resolve_entities(
    question: str,
    threshold: float | None = None,
    entity_memory: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """
    Returns a dict keyed by the matched n-gram (normalized) found in the question.
    Each value is:
        {canonical_value, canonical_norm, entity_type, similarity}

    Example:
        resolve_entities("Qui a gagné à TIAPUM ?")
        → {"TIAPUM": {"canonical_value": "TIAPOUM", "entity_type": "circonscription", ...}}
    """
    from .config import ENTITY_SIMILARITY_THRESHOLD

    if threshold is None:
        threshold = ENTITY_SIMILARITY_THRESHOLD
    if entity_memory is None:
        entity_memory = {}

    aliases = _load_aliases()
    norm_q = _normalize(question)
    # Collapse dot-separated acronyms: "R H D P" → "RHDP", "P D C I" → "PDCI"
    norm_q = re.sub(
        r"\b([A-Z])( [A-Z]){1,5}\b",
        lambda m: m.group(0).replace(" ", ""),
        norm_q,
    )
    tokens = [t for t in norm_q.split() if len(t) >= 3 and t not in STOPWORDS]

    resolved: dict[str, dict] = {}
    used: set[int] = set()

    # Try n-grams longest-first so "GRAND BASSAM" wins over "GRAND" alone
    for n in (4, 3, 2, 1):
        if n > len(tokens):
            continue
        for i in range(len(tokens) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            ngram = " ".join(tokens[i : i + n])

            # Session memory takes priority over fuzzy matching
            if ngram in entity_memory:
                resolved[ngram] = entity_memory[ngram]
                for j in range(i, i + n):
                    used.add(j)
                continue

            best: dict | None = None
            best_score = 0.0
            for alias in aliases:
                score = _similarity(ngram, alias["alias_norm"])
                if score > best_score and score >= threshold:
                    best_score = score
                    best = alias

            if best:
                resolved[ngram] = {
                    "canonical_value": best["canonical_value"],
                    "canonical_norm": best["canonical_norm"],
                    "entity_type": best["entity_type"],
                    "similarity": round(best_score, 3),
                }
                for j in range(i, i + n):
                    used.add(j)

    return resolved


def find_ambiguous_entities(
    question: str,
    entity_memory: dict[str, dict] | None = None,
    threshold: float = 0.72,
) -> dict[str, list[dict]]:
    """
    Returns n-grams that match multiple distinct (canonical_value, entity_type) pairs.

    Only reports genuine ambiguity:
      - top match score >= 0.80 (confident match)
      - at least one other match with different canonical_value or entity_type, score >= 0.65

    Skips n-grams already resolved in entity_memory.
    Returns: {ngram: [sorted list of option dicts]} — at most 4 options per n-gram.
    """
    if entity_memory is None:
        entity_memory = {}

    aliases = _load_aliases()
    norm_q = _normalize(question)
    norm_q = re.sub(
        r"\b([A-Z])( [A-Z]){1,5}\b",
        lambda m: m.group(0).replace(" ", ""),
        norm_q,
    )
    tokens = [t for t in norm_q.split() if len(t) >= 3 and t not in STOPWORDS]

    ambiguous: dict[str, list[dict]] = {}
    used: set[int] = set()

    for n in (4, 3, 2, 1):
        if n > len(tokens):
            continue
        for i in range(len(tokens) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            ngram = " ".join(tokens[i : i + n])

            # Already resolved by the user in this session — skip
            if ngram in entity_memory:
                for j in range(i, i + n):
                    used.add(j)
                continue

            # Collect all matches above threshold, deduped by (canonical_value, entity_type)
            seen: dict[tuple, dict] = {}
            for alias in aliases:
                score = _similarity(ngram, alias["alias_norm"])
                if score >= threshold:
                    key = (alias["canonical_value"], alias["entity_type"])
                    if key not in seen or score > seen[key]["score"]:
                        seen[key] = {
                            "canonical_value": alias["canonical_value"],
                            "canonical_norm": alias["canonical_norm"],
                            "entity_type": alias["entity_type"],
                            "score": round(score, 3),
                            "label": format_option_label(alias["canonical_value"], alias["entity_type"]),
                        }

            matches = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

            top_score = matches[0]["score"] if matches else 0.0

            # For multi-gram n-grams with a strong top match: mark indices as used
            # even if there is only one match or the match is unambiguous.
            # This prevents sub-grams (e.g. "AGBOVILLE" inside "AGBOVILLE COMMUNE")
            # from being re-processed and triggering false ambiguity.
            if n > 1 and top_score >= 0.80:
                for j in range(i, i + n):
                    used.add(j)

            if len(matches) < 2:
                continue

            second_score = matches[1]["score"]

            # Only flag genuine ambiguity:
            # - top match is strong (≥0.80)
            # - runner-up is a near-tie (within 8% of the top)
            #   → avoids false positives like RHDP (1.0) vs RDP (0.857)
            #     while catching exact collisions like KORHOGO→ 2 circonscriptions (both 1.0)
            if top_score >= 0.80 and (second_score / top_score) >= 0.93:
                ambiguous[ngram] = matches[:4]

    return ambiguous


def build_entity_context(resolved: dict[str, dict]) -> str:
    """
    Build a hint string to inject into the LLM prompt.
    Includes canonical_norm so the LLM uses the correct normalized form in LIKE patterns.

    Example:
        "'AGNEBI TIASSA' → region 'AGNEBY-TIASSA' (use norm: 'AGNEBY TIASSA')"
    The norm is what *_norm columns actually contain (no accents, no hyphens, uppercase).
    """
    if not resolved:
        return ""
    parts = []
    for k, v in resolved.items():
        canon_norm = v.get("canonical_norm", "")
        norm_hint = f" (use norm: '{canon_norm}')" if canon_norm else ""
        parts.append(f"'{k}' → {v['entity_type']} '{v['canonical_value']}'{norm_hint}")
    return f"[Entity resolution: {'; '.join(parts)}]"
