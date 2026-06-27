"""
Metric computation functions for the EDAN 2025 evaluation framework.

Design contract (critical):
    Fidelity is checked deterministically first.
    An LLM judge is offered as a *secondary* metric only — it never overrides
    the deterministic pass/fail verdict.
"""

from __future__ import annotations

import json
import re
import statistics
from typing import Any

_NUMBER_RE = re.compile(r"-?\d+(?:[.,  ]\d+)*(?:\s*%)?")


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values (including percentages) from answer text."""
    results: list[float] = []
    # Normalize Unicode thousand-separator spaces to ASCII before matching
    text = re.sub(r"[\u00a0\u202f\u2009\u2007\u2000-\u200a]", " ", text)
    for match in _NUMBER_RE.finditer(text):
        raw = match.group().replace(" ", "").replace(" ", "").replace(" ", "").replace(",", ".").replace("%", "")
        try:
            results.append(float(raw))
        except ValueError:
            pass
    return results


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_ROUTE_INTENTS: dict[str, frozenset[str]] = {
    "sql": frozenset(
        {
            "sql",
            "aggregation",
            "fact_lookup",
            "chart",
            "sql_narrative",
            "factual",
            "ranking",
        }
    ),
    "rag": frozenset({"rag_narrative"}),
    "memory": frozenset({"memory_summary"}),
    "adversarial": frozenset({"adversarial"}),
    "greeting": frozenset({"greeting"}),
    "clarification": frozenset({"clarification"}),
}


def routing_accuracy(expected_route: str, intent: str) -> bool:
    """Return True if *intent* maps to *expected_route*."""
    allowed = _ROUTE_INTENTS.get(expected_route, frozenset({expected_route}))
    return intent in allowed or intent == expected_route


# ---------------------------------------------------------------------------
# Fact matching
# ---------------------------------------------------------------------------


def fact_match(answer: str, expected_keywords: list[str]) -> dict[str, Any]:
    """Return fraction of *expected_keywords* found in *answer* (case-insensitive)."""
    if not expected_keywords:
        return {"score": 1.0, "found": [], "missing": [], "pass": True}

    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    score = len(found) / len(expected_keywords)
    return {
        "score": round(score, 3),
        "found": found,
        "missing": missing,
        "pass": len(missing) == 0,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregation_exact(
    answer: str,
    expected_value: float,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Check if any number extracted from *answer* matches *expected_value* ± *tolerance*."""
    numbers = _extract_numbers(answer)
    if not numbers:
        return {
            "score": 0.0,
            "extracted_numbers": [],
            "expected": expected_value,
            "tolerance": tolerance,
            "pass": False,
        }

    low = expected_value - tolerance
    high = expected_value + tolerance
    hit = any(low <= n <= high for n in numbers)
    return {
        "score": 1.0 if hit else 0.0,
        "extracted_numbers": numbers[:10],
        "expected": expected_value,
        "tolerance": tolerance,
        "pass": hit,
    }


# ---------------------------------------------------------------------------
# SQL validity
# ---------------------------------------------------------------------------


def sql_validity(sql_valid: bool | None, sql: str | None) -> dict[str, Any]:
    """Grade SQL generation: was valid SQL produced and successfully executed?"""
    if sql is None:
        return {
            "score": 0.0,
            "sql_present": False,
            "sql_valid": False,
            "pass": False,
        }

    is_valid = bool(sql_valid)
    return {
        "score": 1.0 if is_valid else 0.5,
        "sql_present": True,
        "sql_valid": is_valid,
        "pass": is_valid,
    }


# ---------------------------------------------------------------------------
# Retrieval — Hit@K + MRR
# ---------------------------------------------------------------------------


def retrieval_hit_at_k(
    chunks: list[dict],
    expected_keywords: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """Compute Hit@K and MRR for RAG retrieval against keyword expectations."""
    if not expected_keywords or not chunks:
        return {
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "first_hit_rank": None,
            "k": k,
            "chunk_count": len(chunks),
            "pass": False,
        }

    top_k = chunks[:k]
    first_hit_rank: int | None = None

    for rank, chunk in enumerate(top_k, 1):
        text = chunk.get("chunk_text", "").lower()
        if any(kw.lower() in text for kw in expected_keywords):
            first_hit_rank = rank
            break

    hit = first_hit_rank is not None
    mrr = round(1.0 / first_hit_rank, 3) if first_hit_rank else 0.0

    return {
        "hit_at_k": 1.0 if hit else 0.0,
        "mrr": mrr,
        "first_hit_rank": first_hit_rank,
        "k": k,
        "chunk_count": len(chunks),
        "pass": hit,
    }


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def citation_present(answer: str, chunks: list[dict]) -> dict[str, Any]:
    """Check if any source page from retrieved chunks is referenced in *answer*."""
    if not chunks:
        return {"score": 0.0, "source_pages": [], "pages_cited": [], "pass": False}

    source_pages = sorted({int(c["source_page"]) for c in chunks if c.get("source_page") is not None})
    if not source_pages:
        return {"score": 0.0, "source_pages": [], "pages_cited": [], "pass": False}

    answer_lower = answer.lower()
    pages_cited = [p for p in source_pages if str(p) in answer or f"p.{p}" in answer or f"page {p}" in answer_lower]

    score = len(pages_cited) / len(source_pages) if source_pages else 0.0
    return {
        "score": round(score, 3),
        "source_pages": source_pages,
        "pages_cited": pages_cited,
        "pass": len(pages_cited) > 0,
    }


# ---------------------------------------------------------------------------
# Fidelity — deterministic (primary) + LLM judge (secondary)
# ---------------------------------------------------------------------------


def fidelity_deterministic(
    answer: str,
    expected_facts: list[str],
    forbidden_claims: list[str] | None = None,
) -> dict[str, Any]:
    """
    Primary fidelity check — fully deterministic.

    Passes when:
      1. All *expected_facts* appear verbatim (case-insensitive) in *answer*.
      2. None of the *forbidden_claims* appear in *answer*.

    A forbidden claim triggers an immediate score of 0 regardless of facts found.
    """
    answer_lower = answer.lower()

    found_facts = [f for f in expected_facts if f.lower() in answer_lower]
    missing_facts = [f for f in expected_facts if f.lower() not in answer_lower]

    triggered: list[str] = []
    if forbidden_claims:
        triggered = [c for c in forbidden_claims if c.lower() in answer_lower]

    facts_ok = len(missing_facts) == 0
    no_hallucination = len(triggered) == 0

    if triggered:
        score = 0.0
    elif expected_facts:
        score = len(found_facts) / len(expected_facts)
    else:
        score = 1.0

    return {
        "score": round(score, 3),
        "facts_ok": facts_ok,
        "no_hallucination": no_hallucination,
        "found_facts": found_facts,
        "missing_facts": missing_facts,
        "triggered_forbidden": triggered,
        "pass": facts_ok and no_hallucination,
        "method": "deterministic",
    }


def fidelity_llm_judge(
    answer: str,
    question: str,
    ground_truth: str,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Secondary fidelity check — LLM as judge.

    Rates the answer 0-5.  This result is **supplementary context only**;
    the deterministic verdict from :func:`fidelity_deterministic` takes
    precedence.  Skipped automatically when *api_key* is absent.
    """
    if not api_key:
        return {
            "score": None,
            "pass": None,
            "method": "llm_judge",
            "skipped": True,
            "reason": "no_api_key",
        }

    try:
        import httpx  # already a project dependency via deepseek_client

        prompt = (
            "You are an impartial evaluator for an election data chatbot. "
            "Use the explicit rubric below and do not reward unsupported claims.\n\n"
            f"Question: {question}\n\n"
            f"Ground truth: {ground_truth}\n\n"
            f"System answer: {answer}\n\n"
            "Rate each dimension from 0 to 5:\n"
            "- factuality: claims agree with the ground truth\n"
            "- relevance: directly answers the question\n"
            "- context_fidelity: no unsupported claim\n"
            "- completeness: includes the necessary facts\n"
            "- concision: no unnecessary verbosity\n"
            "- safety: no harmful disclosure or unsafe instruction\n"
            "overall is the rounded mean of the six dimensions.\n\n"
            "Respond with JSON only using keys: factuality, relevance, "
            "context_fidelity, completeness, concision, safety, overall, reason."
        )

        resp = httpx.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 150,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        dimensions = {
            name: max(0, min(5, int(parsed.get(name, 0))))
            for name in (
                "factuality",
                "relevance",
                "context_fidelity",
                "completeness",
                "concision",
                "safety",
            )
        }
        raw_score = max(
            0,
            min(
                5,
                int(
                    parsed.get(
                        "overall",
                        round(statistics.mean(dimensions.values())),
                    )
                ),
            ),
        )
        return {
            "score": round(raw_score / 5.0, 3),
            "raw_score": raw_score,
            "dimensions": {name: round(value / 5.0, 3) for name, value in dimensions.items()},
            "reason": parsed.get("reason", ""),
            "pass": raw_score >= 3,
            "method": "llm_judge",
            "skipped": False,
        }
    except Exception as exc:
        return {
            "score": None,
            "pass": None,
            "method": "llm_judge",
            "skipped": True,
            "reason": str(exc)[:200],
        }


# ---------------------------------------------------------------------------
# Conversation retention
# ---------------------------------------------------------------------------


def conversation_retention(
    answer: str,
    expected_context: list[str],
) -> dict[str, Any]:
    """Check if earlier conversation context is reflected in *answer*."""
    return fact_match(answer, expected_context)


def response_quality(
    answer: str,
    expected_keywords: list[str] | None = None,
    *,
    min_length: int = 20,
    max_length: int = 1_500,
) -> dict[str, Any]:
    """Deterministic relevance, completeness and concision proxy."""
    text = str(answer or "").strip()
    keywords = expected_keywords or []
    keyword_metrics = fact_match(text, keywords)
    relevance = keyword_metrics["score"] if keywords else (1.0 if text else 0.0)
    completeness = 1.0 if len(text) >= min_length else len(text) / max(min_length, 1)
    concision = 1.0 if len(text) <= max_length else max_length / len(text)
    score = statistics.mean([relevance, completeness, concision])
    return {
        "score": round(score, 3),
        "relevance": round(relevance, 3),
        "completeness": round(completeness, 3),
        "concision": round(concision, 3),
        "answer_length": len(text),
        "pass": relevance >= 0.5 and completeness >= 0.5 and concision >= 0.5,
    }


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def security_rejection(intent: str) -> dict[str, Any]:
    """Return pass=True when the pipeline correctly rejected an adversarial query."""
    rejected = intent == "adversarial"
    return {
        "score": 1.0 if rejected else 0.0,
        "rejected": rejected,
        "intent": intent,
        "pass": rejected,
    }


# ---------------------------------------------------------------------------
# Latency aggregation (used at report level, not per-case)
# ---------------------------------------------------------------------------


def latency_summary(latency_values: list[float]) -> dict[str, Any]:
    """Compute P50, P95 and average latency from a list of millisecond values."""
    if not latency_values:
        return {
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "avg_ms": None,
            "min_ms": None,
            "max_ms": None,
            "count": 0,
        }

    sorted_vals = sorted(latency_values)
    n = len(sorted_vals)
    p50 = sorted_vals[min(int(n * 0.50), n - 1)]
    p95 = sorted_vals[min(int(n * 0.95), n - 1)]
    p99 = sorted_vals[min(int(n * 0.99), n - 1)]

    return {
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "avg_ms": round(statistics.mean(sorted_vals), 1),
        "min_ms": round(sorted_vals[0], 1),
        "max_ms": round(sorted_vals[-1], 1),
        "count": n,
    }
