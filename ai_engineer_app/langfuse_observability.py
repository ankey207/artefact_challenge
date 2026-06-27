from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import Any

from .config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_CAPTURE_CONTENT,
    LANGFUSE_ENABLED,
    PROMPT_VERSION,
)

_LANGFUSE_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "edan_langfuse_trace",
    default=None,
)
_FLUSH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edan-langfuse-flush")


def _configured() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("EDAN_LANGFUSE_ENABLE_IN_TESTS", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    return bool(
        LANGFUSE_ENABLED
        and os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    )


def _client() -> Any | None:
    if not _configured():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


def langfuse_trace_id(run_id: str) -> str | None:
    client = _client()
    if client is None or not run_id:
        return None
    try:
        return client.create_trace_id(seed=run_id)
    except Exception:
        return sha256(run_id.encode("utf-8")).digest()[:16].hex()


def is_langfuse_enabled() -> bool:
    return _client() is not None


def _safe_value(value: Any, *, max_length: int = 2_000) -> Any:
    if isinstance(value, str):
        if LANGFUSE_CAPTURE_CONTENT:
            return value[:max_length]
        return {
            "length": len(value),
            "sha256": sha256(value.encode("utf-8")).hexdigest()[:16] if value else None,
        }
    if isinstance(value, dict):
        return {str(k): _safe_value(v, max_length=max_length) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(item, max_length=max_length) for item in value[:50]]
    return value


def _clean_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        safe_key = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(key))
        cleaned[safe_key[:80]] = _safe_value(value)
    return cleaned


def _observation_type(event_name: str) -> str:
    if event_name == "llm_call":
        return "generation"
    if event_name == "rag_retrieval":
        return "retriever"
    if event_name in {"sql_query", "chart_generation"}:
        return "tool"
    if event_name == "evaluation_recorded":
        return "evaluator"
    if event_name in {"detect_adversarial", "sql_guardrail"}:
        return "guardrail"
    return "span"


def _observation_level(status: str) -> str:
    if status == "error":
        return "ERROR"
    if status == "skipped":
        return "WARNING"
    return "DEFAULT"


def _flush_async(client: Any) -> None:
    """
    Send queued Langfuse data outside the user-facing response path.

    Langfuse SDK calls enqueue telemetry locally. A synchronous flush at the end
    of each Streamlit request makes the user wait for a network round-trip even
    though the answer is already available. This keeps durability best-effort
    without blocking the UI.
    """

    try:
        _FLUSH_EXECUTOR.submit(client.flush)
    except Exception:
        return


@contextmanager
def bind_langfuse_trace(
    run_id: str | None,
    *,
    question: str,
    session_id: str | None,
    anonymous_user_id: str | None,
    chatbot_version: str | None,
    prompt_version: str = PROMPT_VERSION,
    dataset_version_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    client = _client()
    if client is None or not run_id:
        yield
        return

    trace_id = langfuse_trace_id(run_id)
    if not trace_id:
        yield
        return

    try:
        from langfuse import propagate_attributes

        root_metadata = {
            "local_run_id": run_id,
            "dataset_version_id": dataset_version_id,
            "prompt_version": prompt_version,
            "langfuse_base_url": LANGFUSE_BASE_URL,
            **(metadata or {}),
        }
        propagation_metadata = {
            key: value
            for key, value in root_metadata.items()
            if key not in {"tools_available"}
        }
        trace_context = {"trace_id": trace_id}
        with client.start_as_current_observation(
            name="edan_chat_request",
            as_type="agent",
            trace_context=trace_context,
            input={"question": _safe_value(question)},
            metadata=_clean_metadata(root_metadata),
                version=chatbot_version,
        ) as root:
            with propagate_attributes(
                user_id=anonymous_user_id,
                session_id=session_id,
                version=chatbot_version,
                trace_name="edan_chat_request",
                metadata=_clean_metadata(propagation_metadata),
                tags=["streamlit", "edan-2025", "langgraph"],
            ):
                token = _LANGFUSE_CONTEXT.set(
                    {
                        "client": client,
                        "root": root,
                        "trace_id": trace_id,
                        "run_id": run_id,
                        "session_id": session_id,
                    }
                )
                try:
                    yield
                finally:
                    _LANGFUSE_CONTEXT.reset(token)
    except Exception:
        yield


def record_langfuse_event(
    run_id: str | None,
    event_name: str,
    *,
    node_name: str | None = None,
    duration_ms: float | None = None,
    status: str = "ok",
    payload: dict[str, Any] | None = None,
) -> None:
    client = _client()
    if client is None or not run_id:
        return

    active = _LANGFUSE_CONTEXT.get()
    trace_id = (active or {}).get("trace_id") or langfuse_trace_id(run_id)
    if not trace_id:
        return

    data = payload or {}
    metadata = _clean_metadata(
        {
            **data,
            "local_run_id": run_id,
            "node_name": node_name,
            "duration_ms": duration_ms,
            "status": status,
        }
    )
    observation_name = f"{event_name}:{node_name}" if node_name else event_name
    observation_type = _observation_type(event_name)
    trace_context = None if active else {"trace_id": trace_id}

    try:
        if observation_type == "generation":
            usage = {
                "input_tokens": int(data.get("prompt_tokens") or 0),
                "output_tokens": int(data.get("completion_tokens") or 0),
                "total_tokens": int(data.get("total_tokens") or 0),
            }
            cost = {"total": float(data.get("estimated_cost_usd") or 0.0)}
            model_parameters = {
                "temperature": data.get("temperature"),
                "response_format": data.get("response_format"),
                "call_type": data.get("call_type"),
            }
            generation_input = {
                "prompt_name": data.get("prompt_name"),
                "prompt_version": data.get("prompt_version") or PROMPT_VERSION,
                "system_prompt_sha256": data.get("system_prompt_sha256"),
                "system_prompt_length": data.get("system_prompt_length"),
                "user_prompt_length": data.get("user_prompt_length"),
            }
            if LANGFUSE_CAPTURE_CONTENT:
                generation_input["system_prompt"] = data.get("system_prompt")
                generation_input["user_prompt"] = data.get("user_prompt")
            with client.start_as_current_observation(
                name=observation_name,
                as_type="generation",
                trace_context=trace_context,
                input=generation_input,
                output={"response_length": data.get("response_length")},
                metadata=metadata,
                level=_observation_level(status),
                status_message=str(data.get("error_type") or status),
                model=str(data.get("model") or ""),
                model_parameters={k: v for k, v in model_parameters.items() if v is not None},
                usage_details=usage,
                cost_details=cost,
            ):
                pass
            return

        with client.start_as_current_observation(
            name=observation_name,
            as_type=observation_type,
            trace_context=trace_context,
            input=metadata.get("input"),
            output=metadata.get("output"),
            metadata=metadata,
            level=_observation_level(status),
            status_message=str(data.get("error_type") or status),
            version=str(data.get("prompt_version") or PROMPT_VERSION),
        ):
            pass
    except Exception:
        return


@contextmanager
def observe_langfuse_event(
    run_id: str | None,
    event_name: str,
    *,
    node_name: str | None = None,
    status: str = "ok",
    payload: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Measure a real code block as a Langfuse observation.

    `record_langfuse_event` is useful for post-hoc facts, but its Langfuse
    duration is close to zero because the observation is opened after the work
    completed. This context manager wraps the actual work, so Langfuse captures
    precise step latency for graph nodes and other expensive operations.
    """

    client = _client()
    if client is None or not run_id:
        yield {}
        return

    active = _LANGFUSE_CONTEXT.get()
    trace_id = (active or {}).get("trace_id") or langfuse_trace_id(run_id)
    if not trace_id:
        yield {}
        return

    data = payload or {}
    observation_name = f"{event_name}:{node_name}" if node_name else event_name
    observation_type = _observation_type(event_name)
    trace_context = None if active else {"trace_id": trace_id}
    update_payload: dict[str, Any] = {}
    metadata = _clean_metadata(
        {
            **data,
            "local_run_id": run_id,
            "node_name": node_name,
            "status": status,
        }
    )
    observation_kwargs: dict[str, Any] = {
        "name": observation_name,
        "as_type": observation_type,
        "trace_context": trace_context,
        "input": metadata.get("input"),
        "metadata": metadata,
        "level": _observation_level(status),
        "status_message": str(data.get("error_type") or status),
        "version": str(data.get("prompt_version") or PROMPT_VERSION),
    }
    if observation_type == "generation":
        model_parameters = {
            "temperature": data.get("temperature"),
            "response_format": data.get("response_format"),
            "call_type": data.get("call_type"),
        }
        observation_kwargs.update(
            {
                "input": {
                    "prompt_name": data.get("prompt_name"),
                    "prompt_version": data.get("prompt_version") or PROMPT_VERSION,
                    "system_prompt_sha256": data.get("system_prompt_sha256"),
                    "system_prompt_length": data.get("system_prompt_length"),
                    "user_prompt_length": data.get("user_prompt_length"),
                },
                "model": str(data.get("model") or ""),
                "model_parameters": {k: v for k, v in model_parameters.items() if v is not None},
            }
        )
        if LANGFUSE_CAPTURE_CONTENT:
            observation_kwargs["input"]["system_prompt"] = data.get("system_prompt")
            observation_kwargs["input"]["user_prompt"] = data.get("user_prompt")

    try:
        observation_cm = client.start_as_current_observation(**observation_kwargs)
    except Exception:
        yield {}
        return

    with observation_cm:
        try:
            yield update_payload
        except Exception as exc:
            try:
                update = client.update_current_generation if observation_type == "generation" else client.update_current_span
                update(
                    metadata=_clean_metadata(
                        {
                            **data,
                            "local_run_id": run_id,
                            "node_name": node_name,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:500],
                        }
                    ),
                    level="ERROR",
                    status_message=type(exc).__name__,
                )
            except Exception:
                pass
            raise
        else:
            try:
                update = client.update_current_generation if observation_type == "generation" else client.update_current_span
                update_kwargs: dict[str, Any] = {
                    "output": _safe_value(update_payload.get("output")),
                    "metadata": _clean_metadata(
                        {
                            **data,
                            **(update_payload.get("metadata") or {}),
                            "local_run_id": run_id,
                            "node_name": node_name,
                            "status": update_payload.get("status") or status,
                        }
                    ),
                    "level": _observation_level(str(update_payload.get("status") or status)),
                    "status_message": str(update_payload.get("status_message") or status),
                }
                if observation_type == "generation":
                    usage = update_payload.get("usage_details") or {}
                    cost = update_payload.get("cost_details") or {}
                    if usage:
                        update_kwargs["usage_details"] = usage
                    if cost:
                        update_kwargs["cost_details"] = cost
                update(
                    **update_kwargs,
                )
            except Exception:
                pass


def finish_langfuse_trace(
    run_id: str | None,
    *,
    final_response: str | None,
    status: str,
    route: str | None = None,
    intent: str | None = None,
    latency_ms: float | None = None,
    evaluation_scores: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    active = _LANGFUSE_CONTEXT.get()
    client = (active or {}).get("client") or _client()
    if client is None or not run_id:
        return
    trace_id = (active or {}).get("trace_id") or langfuse_trace_id(run_id)
    if not trace_id:
        return
    try:
        root = (active or {}).get("root")
        if root is not None:
            root.update(
                output={"answer": _safe_value(final_response or "")},
                metadata=_clean_metadata(
                    {
                        **(metadata or {}),
                        "local_run_id": run_id,
                        "route": route,
                        "intent": intent,
                        "latency_ms": latency_ms,
                        "status": status,
                    }
                ),
                level=_observation_level("error" if status == "failed" else "ok"),
            )
        for name, value in (evaluation_scores or {}).items():
            if isinstance(value, bool):
                score_value: float | str = 1.0 if value else 0.0
                data_type = "BOOLEAN"
            elif isinstance(value, int | float):
                score_value = float(value)
                data_type = "NUMERIC"
            else:
                score_value = str(value)
                data_type = "CATEGORICAL"
            client.create_score(
                name=str(name),
                value=score_value,
                trace_id=trace_id,
                data_type=data_type,
                metadata={"local_run_id": run_id, "source": "edan_app"},
            )
        _flush_async(client)
    except Exception:
        return


def record_langfuse_feedback(run_id: str, rating: int, comment: str | None = None) -> None:
    client = _client()
    trace_id = langfuse_trace_id(run_id)
    if client is None or not trace_id:
        return
    try:
        client.create_score(
            name="user_feedback",
            value=float(rating),
            trace_id=trace_id,
            data_type="NUMERIC",
            comment=comment,
            metadata={"local_run_id": run_id, "source": "streamlit_feedback"},
        )
        _flush_async(client)
    except Exception:
        return


def record_langfuse_scores(run_id: str, scores: dict[str, Any]) -> None:
    finish_langfuse_trace(
        run_id,
        final_response=None,
        status="succeeded",
        evaluation_scores=scores,
        metadata={"score_source": "evaluation"},
    )
