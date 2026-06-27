from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from threading import Lock

from .config import (
    LANGFUSE_PROMPT_CACHE_TTL_SECONDS,
    LANGFUSE_PROMPT_LABEL,
    LANGFUSE_PROMPTS_ENABLED,
    PROMPT_VERSION,
)


@dataclass(frozen=True)
class PromptDescriptor:
    name: str
    version: str
    sha256: str
    length: int
    source: str = "local"
    langfuse_version: int | None = None
    langfuse_label: str | None = None


_LOCK = Lock()
_BY_HASH: dict[str, PromptDescriptor] = {}
_BY_NAME: dict[str, PromptDescriptor] = {}


def register_prompt(
    name: str,
    text: str,
    *,
    version: str = PROMPT_VERSION,
) -> PromptDescriptor:
    """Register a prompt immutably for the lifetime of the process."""
    digest = sha256(text.encode("utf-8")).hexdigest()
    descriptor = PromptDescriptor(
        name=name,
        version=version,
        sha256=digest,
        length=len(text),
    )
    _remember_descriptor(name, digest, descriptor)
    return descriptor


def _remember_descriptor(name: str, digest: str, descriptor: PromptDescriptor) -> None:
    with _LOCK:
        existing = _BY_NAME.get(name)
        if existing is not None and existing != descriptor:
            raise RuntimeError(f"Prompt {name!r} changed without a new immutable name/version.")
        _BY_NAME[name] = descriptor
        _BY_HASH[digest] = descriptor


def _langfuse_client():
    if not LANGFUSE_PROMPTS_ENABLED:
        return None
    if __import__("os").getenv("PYTEST_CURRENT_TEST"):
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


def _langfuse_prompt_name(name: str) -> str:
    return f"edan-{name}"


def managed_prompt(
    name: str,
    default_text: str,
    *,
    version: str = PROMPT_VERSION,
) -> str:
    """
    Resolve a prompt from Langfuse Prompt Management.

    The code prompt is used as the seed/fallback. On first run, the prompt is
    created in Langfuse with the configured production label. Once present,
    the Langfuse prompt text is the runtime source used by the app.
    """
    langfuse_name = _langfuse_prompt_name(name)
    client = _langfuse_client()
    effective_text = default_text
    descriptor_source = "local"
    langfuse_version: int | None = None

    if client is not None:
        try:
            prompt = client.get_prompt(
                langfuse_name,
                label=LANGFUSE_PROMPT_LABEL,
                type="text",
                fallback=default_text,
                cache_ttl_seconds=LANGFUSE_PROMPT_CACHE_TTL_SECONDS,
            )
            if getattr(prompt, "is_fallback", False):
                try:
                    prompt = client.create_prompt(
                        name=langfuse_name,
                        prompt=default_text,
                        labels=[LANGFUSE_PROMPT_LABEL],
                        tags=["edan-2025", "chatbot"],
                        type="text",
                        config={
                            "app_prompt_name": name,
                            "app_prompt_version": version,
                            "sha256": sha256(default_text.encode("utf-8")).hexdigest(),
                        },
                        commit_message=f"Seed {name} from application version {version}",
                    )
                    client.flush()
                except Exception:
                    # Prompt may already exist or the network may be unavailable.
                    prompt = client.get_prompt(
                        langfuse_name,
                        label=LANGFUSE_PROMPT_LABEL,
                        type="text",
                        fallback=default_text,
                        cache_ttl_seconds=LANGFUSE_PROMPT_CACHE_TTL_SECONDS,
                    )
            effective_text = str(getattr(prompt, "prompt", default_text) or default_text)
            descriptor_source = "local_fallback" if getattr(prompt, "is_fallback", False) else "langfuse"
            langfuse_version = int(getattr(prompt, "version", 0) or 0) or None
        except Exception:
            effective_text = default_text
            descriptor_source = "local_fallback"

    digest = sha256(effective_text.encode("utf-8")).hexdigest()
    descriptor = PromptDescriptor(
        name=langfuse_name,
        version=version,
        sha256=digest,
        length=len(effective_text),
        source=descriptor_source,
        langfuse_version=langfuse_version,
        langfuse_label=LANGFUSE_PROMPT_LABEL if client is not None else None,
    )
    _remember_descriptor(langfuse_name, digest, descriptor)
    return effective_text


def describe_prompt(text: str) -> PromptDescriptor:
    digest = sha256(text.encode("utf-8")).hexdigest()
    with _LOCK:
        registered = _BY_HASH.get(digest)
    if registered is not None:
        return registered
    return PromptDescriptor(
        name="unregistered",
        version=PROMPT_VERSION,
        sha256=digest,
        length=len(text),
    )


def prompt_catalog() -> tuple[PromptDescriptor, ...]:
    with _LOCK:
        return tuple(sorted(_BY_NAME.values(), key=lambda item: item.name))
