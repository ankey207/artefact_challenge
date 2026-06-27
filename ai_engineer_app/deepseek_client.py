"""
DeepSeek API client with reliability hardening (Phase 10).

Retry policy
------------
- Retries ONLY on HTTP 429 (rate-limit) and 5xx server errors.
- Network-level errors (ConnectError, ReadTimeout, ConnectTimeout) are also
  retried — they are transient by nature.
- Functional errors (400, 401, 403, 404, …) are NOT retried; they are raised
  immediately.
- Backoff: exponential with 10 % random jitter, capped at DEEPSEEK_RETRY_MAX_DELAY.
- Each attempt is recorded as a separate observability event.

Timeouts
--------
- Connection timeout and read timeout are configured separately via
  DEEPSEEK_CONNECT_TIMEOUT / DEEPSEEK_READ_TIMEOUT_TEXT / DEEPSEEK_READ_TIMEOUT_JSON.

Circuit breaker (optional, disabled by default)
-----------------------------------------------
- Enable with DEEPSEEK_CB_ENABLED=true.
- Opens after DEEPSEEK_CB_FAILURE_THRESHOLD failures inside DEEPSEEK_CB_WINDOW_SECONDS.
- Stays open for DEEPSEEK_CB_COOLDOWN_SECONDS, then moves to HALF-OPEN.
- First successful call in HALF-OPEN closes the breaker.
- While OPEN: calls raise DeepSeekUnavailableError immediately (no HTTP attempt).

User-facing stable error
------------------------
DeepSeekUnavailableError.user_message is a stable, language-neutral string
safe to present directly to users.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx

from .config import (
    DEEPSEEK_API_URL,
    DEEPSEEK_CB_COOLDOWN_SECONDS,
    DEEPSEEK_CB_ENABLED,
    DEEPSEEK_CB_FAILURE_THRESHOLD,
    DEEPSEEK_CB_WINDOW_SECONDS,
    DEEPSEEK_CONNECT_TIMEOUT,
    DEEPSEEK_INPUT_COST_PER_MILLION,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_MODEL,
    DEEPSEEK_OUTPUT_COST_PER_MILLION,
    DEEPSEEK_READ_TIMEOUT_JSON,
    DEEPSEEK_READ_TIMEOUT_TEXT,
    DEEPSEEK_RETRY_BASE_DELAY,
    DEEPSEEK_RETRY_MAX_DELAY,
    PROMPT_VERSION,
    LANGFUSE_CAPTURE_CONTENT,
    get_api_key,
)
from .observability import observe_current_event, record_current_event
from .prompt_registry import describe_prompt

logger = logging.getLogger(__name__)

# Stable message returned to users when the service is unavailable after retries.
_USER_STABLE_ERROR = "The AI service is temporarily unavailable. Please try again in a moment."

# HTTP status codes that are safe to retry.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DeepSeekUnavailableError(RuntimeError):
    """Raised when all retries are exhausted or the circuit breaker is open."""

    user_message: str = _USER_STABLE_ERROR

    def __init__(self, message: str = _USER_STABLE_ERROR) -> None:
        super().__init__(message)
        self.user_message = message


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class _CircuitBreaker:
    """
    Thread-safe circuit breaker with three states:
    CLOSED -> OPEN (on failure burst) -> HALF-OPEN (after cooldown) -> CLOSED.
    """

    _CLOSED = "CLOSED"
    _OPEN = "OPEN"
    _HALF_OPEN = "HALF-OPEN"

    def __init__(
        self,
        *,
        failure_threshold: int,
        window_seconds: float,
        cooldown_seconds: float,
    ) -> None:
        self._threshold = failure_threshold
        self._window = window_seconds
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._state = self._CLOSED
        self._failure_times: list[float] = []
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        """True when calls should be blocked (state is OPEN after cooldown check)."""
        if not DEEPSEEK_CB_ENABLED:
            return False
        with self._lock:
            return self._evaluate_state()

    def _evaluate_state(self) -> bool:
        now = time.time()
        if self._state == self._OPEN:
            if now - (self._opened_at or 0) >= self._cooldown:
                self._state = self._HALF_OPEN
                logger.info("CircuitBreaker: OPEN -> HALF-OPEN (probe call allowed)")
                return False  # allow one probe
            return True
        return False

    def record_failure(self) -> None:
        if not DEEPSEEK_CB_ENABLED:
            return
        with self._lock:
            now = time.time()
            cutoff = now - self._window
            self._failure_times = [t for t in self._failure_times if t >= cutoff]
            self._failure_times.append(now)
            if len(self._failure_times) >= self._threshold and self._state != self._OPEN:
                self._state = self._OPEN
                self._opened_at = now
                logger.warning(
                    "CircuitBreaker: -> OPEN (%d failures in %.0fs window)",
                    len(self._failure_times),
                    self._window,
                )

    def record_success(self) -> None:
        if not DEEPSEEK_CB_ENABLED:
            return
        with self._lock:
            if self._state in (self._HALF_OPEN, self._OPEN):
                logger.info("CircuitBreaker: %s -> CLOSED", self._state)
            self._state = self._CLOSED
            self._failure_times.clear()
            self._opened_at = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


_circuit_breaker = _CircuitBreaker(
    failure_threshold=DEEPSEEK_CB_FAILURE_THRESHOLD,
    window_seconds=DEEPSEEK_CB_WINDOW_SECONDS,
    cooldown_seconds=DEEPSEEK_CB_COOLDOWN_SECONDS,
)

_STREAM_CALLBACK: ContextVar[Callable[[str], None] | None] = ContextVar(
    "edan_deepseek_stream_callback",
    default=None,
)


@contextmanager
def bind_stream_callback(callback: Callable[[str], None] | None) -> Iterator[None]:
    token = _STREAM_CALLBACK.set(callback)
    try:
        yield
    finally:
        _STREAM_CALLBACK.reset(token)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _usage_payload(response_body: dict[str, Any]) -> dict[str, Any]:
    usage = response_body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    estimated_cost = (
        prompt_tokens * DEEPSEEK_INPUT_COST_PER_MILLION + completion_tokens * DEEPSEEK_OUTPUT_COST_PER_MILLION
    ) / 1_000_000
    return {
        "model": response_body.get("model") or DEEPSEEK_MODEL,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or 0),
        "estimated_cost_usd": round(estimated_cost, 8),
    }


def _request_metadata(payload: dict[str, Any], call_type: str) -> dict[str, Any]:
    messages = payload.get("messages") or []
    system_prompt = next(
        (str(message.get("content", "")) for message in messages if message.get("role") == "system"),
        "",
    )
    user_prompt = next(
        (str(message.get("content", "")) for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    prompt = describe_prompt(system_prompt)
    metadata = {
        "provider": "deepseek",
        "model": payload.get("model") or DEEPSEEK_MODEL,
        "call_type": call_type,
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_source": prompt.source,
        "langfuse_prompt_version": prompt.langfuse_version,
        "langfuse_prompt_label": prompt.langfuse_label,
        "system_prompt_sha256": prompt.sha256[:16] if system_prompt else None,
        "system_prompt_length": len(system_prompt),
        "user_prompt_length": len(user_prompt),
        "message_count": len(messages),
        "temperature": payload.get("temperature"),
        "response_format": (payload.get("response_format") or {}).get("type"),
        "tools_count": len(payload.get("tools") or []),
        "streaming": bool(payload.get("stream", False)),
        "ttft_ms": None,
        "tpot_ms": None,
    }
    if LANGFUSE_CAPTURE_CONTENT:
        metadata["system_prompt"] = system_prompt
        metadata["user_prompt"] = user_prompt
    return metadata


def _record_skipped(call_type: str) -> None:
    record_current_event(
        "llm_call",
        status="skipped",
        payload={
            "provider": "deepseek",
            "model": DEEPSEEK_MODEL,
            "call_type": call_type,
            "prompt_version": PROMPT_VERSION,
            "reason": "missing_api_key",
        },
    )


def _should_retry(exc: Exception) -> bool:
    """True for transient errors: 429, 5xx, or network-level failures."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout))


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with 10 % uniform jitter, capped at max delay."""
    delay = min(DEEPSEEK_RETRY_BASE_DELAY * (2**attempt), DEEPSEEK_RETRY_MAX_DELAY)
    return delay + random.uniform(0.0, 0.1 * delay)


# ---------------------------------------------------------------------------
# Core HTTP caller with retry loop
# ---------------------------------------------------------------------------


def _post_deepseek(
    payload: dict[str, Any],
    *,
    connect_timeout: float,
    read_timeout: float,
    call_type: str,
) -> dict[str, Any]:
    """
    POST to the DeepSeek completions endpoint with retry/backoff.

    Each attempt is recorded as an individual observability event so that
    retry chains are fully auditable.

    Raises
    ------
    DeepSeekUnavailableError
        When all retries are exhausted for a retryable error.
    httpx.HTTPStatusError
        Immediately for non-retryable HTTP errors (4xx except 429).
    """
    max_attempts = DEEPSEEK_MAX_RETRIES + 1
    last_exc: Exception | None = None
    request_metadata = _request_metadata(payload, call_type)

    for attempt in range(max_attempts):
        started = time.perf_counter()
        try:
            with observe_current_event(
                "llm_call",
                payload={
                    **request_metadata,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                },
            ) as observation:
                headers = {
                    "Authorization": f"Bearer {get_api_key()}",
                    "Content-Type": "application/json",
                }
                timeout = httpx.Timeout(
                    connect=connect_timeout,
                    read=read_timeout,
                    write=30.0,
                    pool=5.0,
                )
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
                    response.raise_for_status()

                response_body = response.json()
                event_payload = {
                    **request_metadata,
                    "http_status": response.status_code,
                    "attempt": attempt + 1,
                    "response_length": len(
                        str(response_body.get("choices", [{}])[0].get("message", {}).get("content", ""))
                    ),
                    **_usage_payload(response_body),
                }
                observation["metadata"] = event_payload
                observation["output"] = {"response_length": event_payload["response_length"]}
                observation["usage_details"] = {
                    "input_tokens": int(event_payload.get("prompt_tokens") or 0),
                    "output_tokens": int(event_payload.get("completion_tokens") or 0),
                    "total_tokens": int(event_payload.get("total_tokens") or 0),
                }
                observation["cost_details"] = {"total": float(event_payload.get("estimated_cost_usd") or 0.0)}

        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1_000
            http_status: int | None = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            retryable = _should_retry(exc)
            record_current_event(
                "llm_call",
                duration_ms=duration_ms,
                status="error",
                payload={
                    **request_metadata,
                    "error_type": type(exc).__name__,
                    "http_status": http_status,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "retryable": retryable,
                },
            )
            last_exc = exc

            if not retryable:
                # Functional error — propagate immediately, no retry
                logger.error(
                    "DeepSeek non-retryable error (attempt %d/%d): %s HTTP %s",
                    attempt + 1,
                    max_attempts,
                    type(exc).__name__,
                    http_status,
                )
                raise

            if attempt >= max_attempts - 1:
                break  # exhausted

            delay = _backoff_seconds(attempt)
            logger.warning(
                "DeepSeek %s (attempt %d/%d, HTTP %s) — retry in %.1f s",
                type(exc).__name__,
                attempt + 1,
                max_attempts,
                http_status if http_status is not None else "n/a",
                delay,
            )
            time.sleep(delay)
            continue

        # ── Success path ────────────────────────────────────────────────────
        return response_body

    # All attempts exhausted — wrap in a stable error
    raise DeepSeekUnavailableError() from last_exc


def _post_deepseek_streaming_text(
    payload: dict[str, Any],
    *,
    connect_timeout: float,
    read_timeout: float,
    call_type: str,
    on_token: Callable[[str], None] | None,
) -> str:
    max_attempts = DEEPSEEK_MAX_RETRIES + 1
    last_exc: Exception | None = None
    request_metadata = _request_metadata(payload, call_type)

    for attempt in range(max_attempts):
        started = time.perf_counter()
        first_token_at: float | None = None
        token_count = 0
        chunks: list[str] = []
        try:
            with observe_current_event(
                "llm_call",
                payload={
                    **request_metadata,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "streaming": True,
                },
            ) as observation:
                headers = {
                    "Authorization": f"Bearer {get_api_key()}",
                    "Content-Type": "application/json",
                }
                timeout = httpx.Timeout(
                    connect=connect_timeout,
                    read=read_timeout,
                    write=30.0,
                    pool=5.0,
                )
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("POST", DEEPSEEK_API_URL, headers=headers, json=payload) as response:
                        response.raise_for_status()
                        usage: dict[str, Any] = {}
                        response_model = DEEPSEEK_MODEL
                        for line in response.iter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                line = line[5:].strip()
                            if line == "[DONE]":
                                break
                            event = json.loads(line)
                            response_model = event.get("model") or response_model
                            if event.get("usage"):
                                usage = event.get("usage") or {}
                            for choice in event.get("choices") or []:
                                delta = choice.get("delta") or {}
                                token = delta.get("content") or ""
                                if not token:
                                    continue
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                token_count += 1
                                chunks.append(token)
                                if on_token is not None:
                                    try:
                                        on_token(token)
                                    except Exception:
                                        on_token = None

                text = "".join(chunks).strip()
                duration_ms = (time.perf_counter() - started) * 1_000
                ttft_ms = (first_token_at - started) * 1_000 if first_token_at is not None else None
                tpot_ms = (
                    (duration_ms - (ttft_ms or 0.0)) / max(token_count - 1, 1)
                    if token_count
                    else None
                )
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or 0)
                estimated_cost = (
                    prompt_tokens * DEEPSEEK_INPUT_COST_PER_MILLION
                    + completion_tokens * DEEPSEEK_OUTPUT_COST_PER_MILLION
                ) / 1_000_000
                metadata = {
                    **request_metadata,
                    "model": response_model,
                    "http_status": 200,
                    "attempt": attempt + 1,
                    "response_length": len(text),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": round(estimated_cost, 8),
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "stream_token_count": token_count,
                    "streaming": True,
                }
                observation["metadata"] = metadata
                observation["output"] = {"response_length": len(text)}
                observation["usage_details"] = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
                observation["cost_details"] = {"total": round(estimated_cost, 8)}
            return text

        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1_000
            http_status: int | None = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            retryable = _should_retry(exc)
            record_current_event(
                "llm_call",
                duration_ms=duration_ms,
                status="error",
                payload={
                    **request_metadata,
                    "error_type": type(exc).__name__,
                    "http_status": http_status,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "retryable": retryable,
                    "streaming": True,
                },
            )
            last_exc = exc
            if not retryable:
                raise
            if attempt >= max_attempts - 1:
                break
            time.sleep(_backoff_seconds(attempt))

    raise DeepSeekUnavailableError() from last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_deepseek_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0,
) -> str:
    api_key = get_api_key()
    if not api_key:
        _record_skipped("text")
        return ""

    if _circuit_breaker.is_open():
        logger.warning("DeepSeek circuit breaker OPEN — skipping text call")
        raise DeepSeekUnavailableError()

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        stream_callback = _STREAM_CALLBACK.get()
        if stream_callback is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            try:
                text = _post_deepseek_streaming_text(
                    payload,
                    connect_timeout=DEEPSEEK_CONNECT_TIMEOUT,
                    read_timeout=DEEPSEEK_READ_TIMEOUT_TEXT,
                    call_type="text",
                    on_token=stream_callback,
                )
                _circuit_breaker.record_success()
                return text
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "DeepSeek streaming rejected with HTTP %s; falling back to non-streaming text call",
                    exc.response.status_code,
                )
                payload.pop("stream", None)
                payload.pop("stream_options", None)
        response_body = _post_deepseek(
            payload,
            connect_timeout=DEEPSEEK_CONNECT_TIMEOUT,
            read_timeout=DEEPSEEK_READ_TIMEOUT_TEXT,
            call_type="text",
        )
        _circuit_breaker.record_success()
        return response_body["choices"][0]["message"]["content"].strip()
    except Exception:
        _circuit_breaker.record_failure()
        raise


def call_deepseek_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0,
) -> dict[str, Any]:
    api_key = get_api_key()
    if not api_key:
        _record_skipped("json")
        raise RuntimeError("DEEPSEEK_API_KEY is not configured. Set it in the environment or .env.")

    if _circuit_breaker.is_open():
        logger.warning("DeepSeek circuit breaker OPEN — skipping json call")
        raise DeepSeekUnavailableError()

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        response_body = _post_deepseek(
            payload,
            connect_timeout=DEEPSEEK_CONNECT_TIMEOUT,
            read_timeout=DEEPSEEK_READ_TIMEOUT_JSON,
            call_type="json",
        )
        _circuit_breaker.record_success()
        content = response_body["choices"][0]["message"]["content"]
        return extract_json_object(content)
    except Exception:
        _circuit_breaker.record_failure()
        raise
