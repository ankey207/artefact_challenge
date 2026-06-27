from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .config import CHATBOT_VERSION, PROMPT_VERSION
from .langfuse_observability import (
    finish_langfuse_trace,
    observe_langfuse_event,
    record_langfuse_event,
    record_langfuse_feedback,
    record_langfuse_scores,
)

_TRACE_CONTEXT: ContextVar[tuple[ObservabilityStore, str] | None] = ContextVar(
    "edan_langfuse_trace_compat",
    default=None,
)


class ObservabilityStore:
    """
    Langfuse-only compatibility adapter.

    Older application code calls ObservabilityStore to start runs, record events,
    feedback and evaluation scores. This adapter preserves that API but never
    writes local SQLite files. Langfuse is the only persistent observability
    backend.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def initialize(self) -> None:
        return

    def start_run(
        self,
        question: str,
        *,
        dataset_version_id: str | None = None,
        session_id: str | None = None,
        anonymous_user_id: str | None = None,
        chatbot_version: str = CHATBOT_VERSION,
        prompt_version: str = PROMPT_VERSION,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del question, dataset_version_id, session_id, anonymous_user_id, chatbot_version, prompt_version, metadata
        return uuid.uuid4().hex

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        latency_ms: float,
        route: str | None = None,
        intent: str | None = None,
        result_row_count: int | None = None,
        rag_chunk_count: int | None = None,
        chart_type: str | None = None,
        sql_valid: bool | None = None,
        final_response: str | None = None,
        evaluation_scores: dict[str, Any] | None = None,
        error: BaseException | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        finish_langfuse_trace(
            run_id,
            final_response=final_response,
            status=status,
            route=route,
            intent=intent,
            latency_ms=latency_ms,
            evaluation_scores=evaluation_scores,
            metadata={
                **(metadata or {}),
                "result_row_count": result_row_count,
                "rag_chunk_count": rag_chunk_count,
                "chart_type": chart_type,
                "sql_valid": sql_valid,
                "error_type": type(error).__name__ if error else None,
            },
        )

    def record_event(
        self,
        run_id: str,
        event_name: str,
        *,
        node_name: str | None = None,
        duration_ms: float | None = None,
        status: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> int:
        record_langfuse_event(
            run_id,
            event_name,
            node_name=node_name,
            duration_ms=duration_ms,
            status=status,
            payload=payload,
        )
        return 0

    @contextmanager
    def observe_event(
        self,
        run_id: str,
        event_name: str,
        *,
        node_name: str | None = None,
        status: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        with observe_langfuse_event(
            run_id,
            event_name,
            node_name=node_name,
            status=status,
            payload=payload,
        ) as observation:
            yield observation

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        del run_id
        return None

    def record_evaluation(self, run_id: str, scores: dict[str, Any]) -> None:
        record_langfuse_scores(run_id, scores)

    def record_feedback(
        self,
        run_id: str,
        rating: int,
        comment: str | None = None,
    ) -> None:
        record_langfuse_feedback(run_id, rating, comment)


def get_observability_store() -> ObservabilityStore:
    return ObservabilityStore()


@contextmanager
def bind_trace(
    store: ObservabilityStore | None,
    run_id: str | None,
) -> Iterator[None]:
    if store is None or run_id is None:
        yield
        return
    token = _TRACE_CONTEXT.set((store, run_id))
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)


def record_current_event(
    event_name: str,
    *,
    duration_ms: float | None = None,
    status: str = "ok",
    payload: dict[str, Any] | None = None,
) -> None:
    context = _TRACE_CONTEXT.get()
    if context is None:
        return
    store, run_id = context
    try:
        store.record_event(
            run_id,
            event_name,
            duration_ms=duration_ms,
            status=status,
            payload=payload,
        )
    except Exception:
        pass


@contextmanager
def observe_current_event(
    event_name: str,
    *,
    status: str = "ok",
    payload: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    context = _TRACE_CONTEXT.get()
    if context is None:
        yield {}
        return
    store, run_id = context
    try:
        event_cm = store.observe_event(run_id, event_name, status=status, payload=payload)
    except Exception:
        yield {}
        return

    with event_cm as observation:
        yield observation


def record_run_event(
    run_id: str | None,
    event_name: str,
    *,
    duration_ms: float | None = None,
    status: str = "ok",
    payload: dict[str, Any] | None = None,
) -> None:
    if not run_id:
        return
    try:
        record_langfuse_event(
            run_id,
            event_name,
            duration_ms=duration_ms,
            status=status,
            payload=payload,
        )
    except Exception:
        pass
