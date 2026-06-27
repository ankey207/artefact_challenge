from __future__ import annotations

import pytest

from ai_engineer_app.evaluation.metrics import latency_summary, response_quality
from ai_engineer_app.prompt_registry import describe_prompt, managed_prompt, register_prompt


def test_prompt_registry_exposes_immutable_name_version_and_hash():
    descriptor = register_prompt(
        "test_prompt_immutable",
        "Prompt content",
        version="test-v1",
    )

    assert describe_prompt("Prompt content") == descriptor
    assert descriptor.name == "test_prompt_immutable"
    assert descriptor.version == "test-v1"
    assert len(descriptor.sha256) == 64
    assert descriptor.source == "local"

    with pytest.raises(RuntimeError):
        register_prompt(
            "test_prompt_immutable",
            "Changed without versioning",
            version="test-v1",
        )


def test_managed_prompt_uses_local_seed_when_langfuse_disabled(monkeypatch):
    monkeypatch.setattr("ai_engineer_app.prompt_registry.LANGFUSE_PROMPTS_ENABLED", False)

    text = managed_prompt("test_managed_local", "Managed prompt", version="test-v1")
    descriptor = describe_prompt(text)

    assert text == "Managed prompt"
    assert descriptor.name == "edan-test_managed_local"
    assert descriptor.source == "local"


def test_quality_metrics_and_p99_are_reported():
    quality = response_quality(
        "Le RHDP a remporté le plus de sièges.",
        ["RHDP"],
        min_length=10,
        max_length=100,
    )
    latency = latency_summary([10, 20, 30, 40, 100])

    assert quality["pass"] is True
    assert quality["relevance"] == 1.0
    assert latency["p50_ms"] is not None
    assert latency["p95_ms"] is not None
    assert latency["p99_ms"] is not None
