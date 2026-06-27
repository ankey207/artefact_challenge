from __future__ import annotations

from ai_engineer_app.observability import ObservabilityStore, bind_trace, record_current_event


def test_observability_adapter_does_not_create_local_storage(tmp_path, monkeypatch):
    calls = []

    def fake_event(run_id, event_name, **kwargs):
        calls.append((run_id, event_name, kwargs))

    monkeypatch.setattr("ai_engineer_app.observability.record_langfuse_event", fake_event)

    db_path = tmp_path / "should_not_be_created.sqlite3"
    store = ObservabilityStore(db_path)
    run_id = store.start_run("question")

    with bind_trace(store, run_id):
        record_current_event("node_completed", payload={"ok": True})

    assert not db_path.exists()
    assert calls == [
        (
            run_id,
            "node_completed",
            {
                "node_name": None,
                "duration_ms": None,
                "status": "ok",
                "payload": {"ok": True},
            },
        )
    ]
