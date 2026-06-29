from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langfuse.experiment import Evaluation

from ai_engineer_app.langfuse_observability import is_langfuse_enabled

DATASET_NAME = os.getenv("EDAN_LANGFUSE_EVAL_DATASET", "edan-2025-chatbot-evaluation")


def _stable_item_id(case_id: str) -> str:
    digest = hashlib.sha256(f"{DATASET_NAME}:{case_id}".encode("utf-8")).hexdigest()
    return f"edan-eval-{digest[:32]}"


def _case_input(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("turns"):
        return {"turns": case["turns"]}
    return {"question": case.get("question", "")}


def _case_expected_output(case: dict[str, Any]) -> dict[str, Any]:
    ignored = {"id", "category", "question", "turns", "_comment"}
    return {key: value for key, value in case.items() if key not in ignored and not key.startswith("_")}


def _case_metadata(case: dict[str, Any]) -> dict[str, Any]:
    turns = case.get("turns") or []
    return {
        "case_id": str(case.get("id", "unknown")),
        "category": case.get("category", "unknown"),
        "question": str(case.get("question") or "")[:180],
        "turn_count": len(turns),
        "has_ground_truth": bool(case.get("ground_truth")),
        "source": "evals/test_cases.json",
    }


def get_langfuse_client() -> Any | None:
    if not is_langfuse_enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


def langfuse_evals_ready() -> bool:
    return get_langfuse_client() is not None


def sync_dataset(
    cases: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
    dataset_name: str = DATASET_NAME,
) -> Any:
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse is not configured.")

    dataset_metadata = {
        "source": "evals/test_cases.json",
        "case_count": len(cases),
        **(metadata or {}),
    }
    try:
        client.create_dataset(
            name=dataset_name,
            description=(
                "Evaluation suite for the EDAN 2025 election chatbot. "
                "Items cover routing, factual lookup, SQL aggregation, RAG, "
                "citation faithfulness, conversation coherence, security and quality."
            ),
            metadata=dataset_metadata,
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "turns": {"type": "array"},
                },
            },
            expected_output_schema={"type": "object"},
        )
    except Exception as exc:
        # Dataset creation is idempotent from our perspective. Langfuse returns an
        # API error when it already exists, so continue with item upserts.
        if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
            try:
                client.get_dataset(dataset_name)
            except Exception:
                raise

    for case in cases:
        case_id = str(case.get("id", "unknown"))
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=_stable_item_id(case_id),
            input=_case_input(case),
            expected_output=_case_expected_output(case),
            metadata=_case_metadata(case),
        )
    client.flush()
    return client.get_dataset(dataset_name)


def make_run_name(*, category: str | None = None, offline: bool = False) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = category or "all"
    mode = "offline" if offline else "online"
    return f"edan-eval-{suffix}-{mode}-{ts}"


def _item_metadata(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item.get("metadata") or {})
    return dict(getattr(item, "metadata", None) or {})


def _evaluation_from_graded(name: str, value: Any, *, metadata: dict[str, Any] | None = None) -> Evaluation:
    if isinstance(value, bool):
        return Evaluation(name=name, value=value, data_type="BOOLEAN", metadata=metadata)
    if isinstance(value, int | float):
        return Evaluation(name=name, value=float(value), data_type="NUMERIC", metadata=metadata)
    return Evaluation(name=name, value=str(value), data_type="CATEGORICAL", metadata=metadata)


def run_langfuse_experiment(
    cases: list[dict[str, Any]],
    *,
    execute_case: Callable[[dict[str, Any]], tuple[dict[str, Any], float]],
    grade_case: Callable[[dict[str, Any], dict[str, Any], float], dict[str, Any]],
    run_evaluation_summary: Callable[[list[dict[str, Any]]], dict[str, Any]],
    dataset_metadata: dict[str, Any] | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
    dataset_name: str = DATASET_NAME,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse is not configured.")

    dataset = sync_dataset(cases, metadata=dataset_metadata, dataset_name=dataset_name)
    case_by_id = {str(case.get("id")): case for case in cases}
    selected_case_ids = set(case_by_id)
    graded_by_case_id: dict[str, dict[str, Any]] = {}

    def task(*, item: Any, **_: dict[str, Any]) -> dict[str, Any]:
        metadata = _item_metadata(item)
        case_id = str(metadata.get("case_id", ""))
        case = metadata.get("case") or case_by_id.get(case_id)
        if not isinstance(case, dict):
            raise RuntimeError(f"Missing evaluation case metadata for {case_id!r}.")
        result, latency_ms = execute_case(case)
        return {
            "result": result,
            "latency_ms": latency_ms,
            "answer": result.get("answer", ""),
            "trace_id": result.get("trace_id"),
            "intent": result.get("intent"),
            "route": (
                "rag"
                if result.get("rag_results") is not None
                else ("memory" if result.get("intent") == "memory_summary" else "sql")
            ),
        }

    def evaluator(
        *,
        input: Any,
        output: dict[str, Any],
        expected_output: Any,
        metadata: dict[str, Any] | None,
        **_: dict[str, Any],
    ) -> list[Evaluation]:
        del input, expected_output
        case_id = str((metadata or {}).get("case_id", ""))
        case = dict((metadata or {}).get("case") or case_by_id.get(case_id) or {})
        if not case:
            raise RuntimeError(f"Missing evaluation case metadata for {case_id!r}.")
        graded = grade_case(case, output.get("result", {}), float(output.get("latency_ms") or 0.0))
        graded_by_case_id[str(graded.get("id"))] = graded
        eval_metadata = {
            "case_id": graded.get("id"),
            "category": graded.get("category"),
            "trace_id": graded.get("trace_id"),
            "question": graded.get("question"),
            "error": graded.get("error"),
        }
        evaluations = [
            _evaluation_from_graded("score", graded.get("score", 0.0), metadata=eval_metadata),
            _evaluation_from_graded("pass", bool(graded.get("pass")), metadata=eval_metadata),
            _evaluation_from_graded("latency_ms", graded.get("latency_ms", 0.0), metadata=eval_metadata),
            _evaluation_from_graded("category", graded.get("category", "unknown"), metadata=eval_metadata),
            _evaluation_from_graded("case_id", graded.get("id", "unknown"), metadata=eval_metadata),
        ]
        if graded.get("reason"):
            evaluations.append(
                _evaluation_from_graded("reason", graded.get("reason"), metadata=eval_metadata)
            )
        return evaluations

    def run_evaluator(*, item_results: list[Any], **_: dict[str, Any]) -> list[Evaluation]:
        del item_results
        summary = run_evaluation_summary(list(graded_by_case_id.values()))
        evaluations = [
            _evaluation_from_graded("overall_pass_rate", summary.get("pass_rate", 0.0)),
            _evaluation_from_graded("overall_avg_score", summary.get("avg_score", 0.0)),
            _evaluation_from_graded("case_count", summary.get("case_count", 0)),
            _evaluation_from_graded("failed_count", summary.get("failed_count", 0)),
        ]
        for category, value in (summary.get("pass_rate_by_category") or {}).items():
            evaluations.append(_evaluation_from_graded(f"pass_rate_{category}", value))
        return evaluations

    dataset_items = [
        item
        for item in (list(getattr(dataset, "items", []) or []))
        if str(_item_metadata(item).get("case_id", "")) in selected_case_ids
    ]
    if not dataset_items:
        raise RuntimeError("No Langfuse dataset items matched the selected evaluation cases.")

    experiment = client.run_experiment(
        name="EDAN chatbot evaluation",
        run_name=run_name,
        description="End-to-end evaluation suite for the EDAN 2025 chatbot.",
        data=dataset_items,
        task=task,
        evaluators=[evaluator],
        run_evaluators=[run_evaluator],
        max_concurrency=1,
        metadata={
            "source": "evals/run_evals.py",
            **(experiment_metadata or {}),
        },
    )
    client.flush()

    results = list(graded_by_case_id.values())
    case_order = {str(case.get("id")): idx for idx, case in enumerate(cases)}
    results.sort(key=lambda row: case_order.get(str(row.get("id")), 10_000))
    metadata = {
        "langfuse_dataset_name": dataset_name,
        "langfuse_run_name": getattr(experiment, "run_name", run_name),
        "langfuse_dataset_run_id": getattr(experiment, "dataset_run_id", None),
        "langfuse_dataset_run_url": getattr(experiment, "dataset_run_url", None),
        "langfuse_experiment_id": getattr(experiment, "experiment_id", None),
    }
    return results, metadata


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in results if not row.get("skipped")]
    if not scored:
        return {
            "case_count": len(results),
            "failed_count": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "pass_rate_by_category": {},
        }
    failed_count = sum(1 for row in scored if not row.get("pass"))
    category_totals: dict[str, list[bool]] = {}
    for row in scored:
        category_totals.setdefault(str(row.get("category", "unknown")), []).append(bool(row.get("pass")))
    return {
        "case_count": len(scored),
        "failed_count": failed_count,
        "pass_rate": (len(scored) - failed_count) / len(scored),
        "avg_score": sum(float(row.get("score") or 0.0) for row in scored) / len(scored),
        "pass_rate_by_category": {
            category: sum(values) / len(values) for category, values in category_totals.items() if values
        },
    }
