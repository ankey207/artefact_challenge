#!/usr/bin/env python
"""
Evaluation runner for the Côte d'Ivoire 2025 Legislative Elections chat app.

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --category routing
    python evals/run_evals.py --llm-judge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable regardless of where the script is invoked
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai_engineer_app.config import DEEPSEEK_MODEL, get_api_key
from ai_engineer_app.evaluation.graders import (
    grade_aggregation,
    grade_citation,
    grade_conversation,
    grade_facts,
    grade_fidelity,
    grade_latency,
    grade_quality,
    grade_retrieval,
    grade_routing,
    grade_security,
    grade_sql,
)
from ai_engineer_app.evaluation.langfuse_eval import (
    langfuse_evals_ready,
    make_run_name,
    run_langfuse_experiment,
    summarize_results,
)

_EVALS_DIR = Path(__file__).resolve().parent
_DEFAULT_CASES = _EVALS_DIR / "test_cases.json"

_CATEGORY_GRADER = {
    "routing": grade_routing,
    "facts": grade_facts,
    "aggregation": grade_aggregation,
    "sql": grade_sql,
    "retrieval": grade_retrieval,
    "citation": grade_citation,
    "fidelity": grade_fidelity,
    "conversation": grade_conversation,
    "security": grade_security,
    "latency": grade_latency,
    "quality": grade_quality,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] Test cases file not found: {path}", file=sys.stderr)
        sys.exit(1)
    raw = _load_json(path)
    if not isinstance(raw, list):
        print("[ERROR] test_cases.json must be a JSON array.", file=sys.stderr)
        sys.exit(1)
    # Strip internal comment keys (keys starting with _)
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in raw]


# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------


def _run_pipeline(
    question: str,
    history: list[dict],
    *,
    entity_memory: dict | None = None,
    conversation_memory: dict | None = None,
) -> tuple[dict, float]:
    """Call answer_question() and return (result, latency_ms)."""
    from ai_engineer_app.graph import answer_question

    t0 = time.perf_counter()
    result = answer_question(
        question,
        history=history,
        entity_memory=entity_memory or {},
        conversation_memory=conversation_memory or {},
    )
    latency_ms = (time.perf_counter() - t0) * 1_000
    return result, latency_ms


def _execute_case(case: dict) -> tuple[dict, float]:
    turns: list[dict] = case.get("turns", [])
    if turns:
        history: list[dict] = []
        entity_memory: dict = {}
        conversation_memory: dict = {}
        latency_ms = 0.0
        last_result: dict = {}
        for turn in turns:
            q = turn.get("question", "")
            last_result, last_lat = _run_pipeline(
                q,
                history,
                entity_memory=entity_memory,
                conversation_memory=conversation_memory,
            )
            latency_ms += last_lat
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": last_result.get("answer", "")})
            if isinstance(last_result.get("conversation_memory"), dict):
                conversation_memory = last_result["conversation_memory"]
        return last_result, latency_ms
    return _run_pipeline(case.get("question", ""), [])


def _grade_case_result(
    case: dict,
    result: dict,
    latency_ms: float,
    *,
    llm_judge: bool,
    api_key: str | None,
) -> dict:
    category = case.get("category", "unknown")
    grader = _CATEGORY_GRADER.get(category)
    if grader is None:
        return {
            "id": case["id"],
            "category": category,
            "score": 0.0,
            "pass": False,
            "error": f"unknown_category:{category}",
        }

    result["_latency_ms"] = latency_ms
    if category == "fidelity":
        graded = grade_fidelity(case, result, llm_judge=llm_judge, api_key=api_key)
    else:
        graded = grader(case, result)

    turns: list[dict] = case.get("turns", [])
    graded["latency_ms"] = round(latency_ms, 1)
    graded["trace_id"] = result.get("trace_id")
    graded["question"] = case.get("question") or (turns[-1].get("question") if turns else "")
    graded["answer_preview"] = (result.get("answer") or "")[:150]

    return graded


def run_case(
    case: dict,
    *,
    llm_judge: bool,
    api_key: str | None,
) -> dict:
    """Run one evaluation case and return a graded dict."""
    result, latency_ms = _execute_case(case)
    return _grade_case_result(
        case,
        result,
        latency_ms,
        llm_judge=llm_judge,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python evals/run_evals.py",
        description="Run evaluations for the EDAN 2025 election chat app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evals/run_evals.py
  python evals/run_evals.py --category routing
  python evals/run_evals.py --llm-judge
  python evals/run_evals.py --langfuse-run-name edan-eval-main-v1
""",
    )
    p.add_argument(
        "--category",
        choices=sorted(_CATEGORY_GRADER),
        help="Only run cases in this category.",
        metavar="CATEGORY",
    )
    p.add_argument(
        "--llm-judge",
        action="store_true",
        dest="llm_judge",
        help=(
            "Enable LLM-as-judge for fidelity cases. Adds supplementary scores — never overrides deterministic verdict."
        ),
    )
    p.add_argument(
        "--cases",
        metavar="FILE",
        default=str(_DEFAULT_CASES),
        help="Path to test_cases.json (default: evals/test_cases.json).",
    )
    p.add_argument(
        "--langfuse-run-name",
        metavar="NAME",
        help="Explicit Langfuse dataset run name. Default: generated timestamped name.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    api_key = get_api_key() if args.llm_judge else None

    # Apply filters
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]

    if not cases:
        print("No test cases to run.")
        return

    if not langfuse_evals_ready():
        print("[ERROR] Langfuse is required for evaluation. Configure LANGFUSE_* keys.", file=sys.stderr)
        sys.exit(2)

    run_name = args.langfuse_run_name or make_run_name(category=args.category, offline=False)
    print(f"\nRunning {len(cases)} test case(s) as Langfuse experiment: {run_name}\n", flush=True)
    results, langfuse_metadata = run_langfuse_experiment(
        cases,
        execute_case=_execute_case,
        grade_case=lambda case, result, latency_ms: _grade_case_result(
            case,
            result,
            latency_ms,
            llm_judge=args.llm_judge,
            api_key=api_key,
        ),
        run_evaluation_summary=summarize_results,
        dataset_metadata={
            "model": DEEPSEEK_MODEL,
            "dataset_version": _safe_dataset_version(),
            "cases_file": str(Path(args.cases).resolve()),
        },
        experiment_metadata={
            "category": args.category,
            "llm_judge": args.llm_judge,
            "model": DEEPSEEK_MODEL,
        },
        run_name=run_name,
    )

    for idx, graded in enumerate(results, 1):
        status = "PASS" if graded.get("pass") else "FAIL"
        lat = graded.get("latency_ms") or 0
        print(
            f"  [{idx:02d}/{len(results)}] {graded.get('id', '?'):<12} "
            f"({graded.get('category', '?'):<14})  {status}  ({lat:.0f} ms)"
        )
    if langfuse_metadata.get("langfuse_dataset_run_url"):
        print(f"\nLangfuse run: {langfuse_metadata['langfuse_dataset_run_url']}\n")

    summary = summarize_results(results)
    print(
        "Summary: "
        f"{summary['case_count'] - summary['failed_count']}/{summary['case_count']} passed, "
        f"pass_rate={summary['pass_rate']:.1%}, avg_score={summary['avg_score']:.3f}"
    )


def _safe_dataset_version() -> dict | None:
    try:
        from ai_engineer_app.db import get_database_version

        return get_database_version()
    except Exception:
        return None


if __name__ == "__main__":
    main()
