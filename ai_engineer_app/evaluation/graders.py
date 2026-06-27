"""
Per-dimension graders for the EDAN 2025 evaluation framework.

Each grader receives a *case* dict (from test_cases.json) and a *result*
dict (the dict returned by ``answer_question()``), and returns a grade dict
with at minimum: id, category, score (0.0-1.0), pass (bool).

Fidelity contract: the deterministic check is always the primary verdict.
The LLM judge is injected as a supplementary field only and never flips pass/fail.
"""

from __future__ import annotations

from typing import Any

from .metrics import (
    aggregation_exact,
    citation_present,
    conversation_retention,
    fact_match,
    fidelity_deterministic,
    fidelity_llm_judge,
    response_quality,
    retrieval_hit_at_k,
    routing_accuracy,
    security_rejection,
    sql_validity,
)


def _base(case_id: str, category: str, passed: bool, score: float) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "score": round(float(score), 3),
        "pass": bool(passed),
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def grade_routing(case: dict, result: dict) -> dict[str, Any]:
    # For pre-routed RAG (deterministic), intent is never set; fall back to pre_route
    intent = result.get("intent") or result.get("pre_route", "")
    expected = case.get("expected_route", "")
    passed = routing_accuracy(expected, intent)
    return {
        **_base(case["id"], "routing", passed, 1.0 if passed else 0.0),
        "expected_route": expected,
        "actual_intent": intent,
    }


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def grade_facts(case: dict, result: dict) -> dict[str, Any]:
    answer = result.get("answer", "")
    keywords = case.get("expected_answer_contains", [])
    metrics = fact_match(answer, keywords)
    return {
        **_base(case["id"], "facts", metrics["pass"], metrics["score"]),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def grade_aggregation(case: dict, result: dict) -> dict[str, Any]:
    answer = result.get("answer", "")
    expected = case.get("expected_value")
    if expected is None:
        # No numerical expectation — just check an answer was produced
        has_answer = bool(answer.strip())
        return _base(case["id"], "aggregation", has_answer, 1.0 if has_answer else 0.0)

    tolerance = float(case.get("tolerance", 0.0))
    metrics = aggregation_exact(answer, float(expected), tolerance)
    return {
        **_base(case["id"], "aggregation", metrics["pass"], metrics["score"]),
        **metrics,
    }


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def grade_sql(case: dict, result: dict) -> dict[str, Any]:
    sql = result.get("safe_sql") or result.get("sql")
    valid = result.get("sql_valid")
    metrics = sql_validity(valid, sql)
    return {
        **_base(case["id"], "sql", metrics["pass"], metrics["score"]),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def grade_retrieval(case: dict, result: dict, *, k: int = 5) -> dict[str, Any]:
    chunks = result.get("rag_results") or []
    keywords = case.get("expected_chunk_keywords", [])
    if not keywords:
        # No keyword expectation: validate only that some chunks were retrieved
        has_chunks = len(chunks) > 0
        return {
            **_base(case["id"], "retrieval", has_chunks, 1.0 if has_chunks else 0.0),
            "hit_at_k": 1.0 if has_chunks else 0.0,
            "mrr": 0.0,
            "first_hit_rank": None,
            "k": k,
            "chunk_count": len(chunks),
            "note": "no_keywords_expected",
        }
    metrics = retrieval_hit_at_k(chunks, keywords, k=k)
    return {
        **_base(case["id"], "retrieval", metrics["pass"], metrics["hit_at_k"]),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def grade_citation(case: dict, result: dict) -> dict[str, Any]:
    answer = result.get("answer", "")
    chunks = result.get("rag_results") or []
    metrics = citation_present(
        answer,
        chunks,
        expected_pages=case.get("expected_source_pages") or [],
        expected_source_ids=case.get("expected_source_ids") or [],
    )
    return {
        **_base(case["id"], "citation", metrics["pass"], metrics["score"]),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Fidelity (deterministic primary, LLM judge secondary)
# ---------------------------------------------------------------------------


def grade_fidelity(
    case: dict,
    result: dict,
    *,
    llm_judge: bool = False,
    api_key: str | None = None,
) -> dict[str, Any]:
    answer = result.get("answer", "")
    expected_facts = case.get("expected_facts", [])
    forbidden_claims = case.get("forbidden_claims", [])

    # Primary — deterministic, always executed
    det = fidelity_deterministic(answer, expected_facts, forbidden_claims)

    output: dict[str, Any] = {
        **_base(case["id"], "fidelity", det["pass"], det["score"]),
        "deterministic": det,
    }

    # Secondary — LLM judge, supplementary context only
    if llm_judge and case.get("ground_truth"):
        llm = fidelity_llm_judge(
            answer,
            case.get("question", ""),
            case["ground_truth"],
            api_key=api_key,
        )
        output["llm_judge"] = llm
        # The deterministic verdict is NEVER overridden by the LLM judge

    return output


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


def grade_conversation(case: dict, result: dict) -> dict[str, Any]:
    answer = result.get("answer", "")
    expected_context = case.get("expected_context", [])
    metrics = conversation_retention(answer, expected_context)
    return {
        **_base(case["id"], "conversation", metrics["pass"], metrics["score"]),
        **metrics,
    }


def grade_quality(case: dict, result: dict) -> dict[str, Any]:
    metrics = response_quality(
        result.get("answer", ""),
        case.get("expected_answer_contains", []),
        min_length=int(case.get("min_answer_length", 20)),
        max_length=int(case.get("max_answer_length", 1_500)),
    )
    return {
        **_base(case["id"], "quality", metrics["pass"], metrics["score"]),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def grade_security(case: dict, result: dict) -> dict[str, Any]:
    intent = result.get("intent", "")
    metrics = security_rejection(intent)
    return {
        **_base(case["id"], "security", metrics["pass"], metrics["score"]),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Latency (per-case threshold check; aggregate stats are in report.py)
# ---------------------------------------------------------------------------


def grade_latency(
    case: dict,
    result: dict,
    *,
    threshold_ms: float = 12_000,
) -> dict[str, Any]:
    latency = result.get("_latency_ms")
    if latency is None:
        return {**_base(case["id"], "latency", True, 1.0), "latency_ms": None}

    passed = latency <= threshold_ms
    return {
        **_base(case["id"], "latency", passed, 1.0 if passed else 0.0),
        "latency_ms": round(latency, 1),
        "threshold_ms": threshold_ms,
    }
