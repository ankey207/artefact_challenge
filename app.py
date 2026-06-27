from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from ai_engineer_app.charts import make_chart
from ai_engineer_app.config import DEFAULT_DB_PATH, HISTORY_MAX_EXCHANGES, LANGFUSE_BASE_URL, get_api_key
from ai_engineer_app.db import get_database_version
from ai_engineer_app.deepseek_client import bind_stream_callback
from ai_engineer_app.graph import answer_question
from ai_engineer_app.langfuse_observability import is_langfuse_enabled
from ai_engineer_app.observability import get_observability_store

_ASSETS = Path(__file__).parent / "assets"
_FAVICON = Image.open(_ASSETS / "favicon_cei.png")

st.set_page_config(
    page_title="Côte d'Ivoire 2025 Legislative Elections — Results Chat",
    page_icon=_FAVICON,
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
_css = (_ASSETS / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.image(_FAVICON, width=64)
with col_title:
    st.title("Côte d'Ivoire 2025 Legislative Elections — Results Chat")
    st.caption(
        "Ask questions about the 2025 legislative election results: candidates, parties, regions, vote counts, participation rates."
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dataset")
    if not DEFAULT_DB_PATH.exists():
        st.error(f"Database not found: {DEFAULT_DB_PATH}")
    else:
        dataset_version = get_database_version()
        if dataset_version:
            st.caption(f"Version: `{dataset_version['version_id']}` · {dataset_version['build_status']}")

    st.header("Provider")
    st.write("DeepSeek via `DEEPSEEK_API_KEY`")
    if get_api_key():
        st.success("API key configured")
    else:
        st.warning("API key missing")

    st.header("Observability")
    if is_langfuse_enabled():
        st.success("Langfuse enabled")
        st.link_button("Open Langfuse dashboard", LANGFUSE_BASE_URL)
    else:
        st.error("Langfuse credentials missing or SDK unavailable")

    # Session memory display
    entity_mem = st.session_state.get("entity_memory", {})
    if entity_mem:
        st.header("Session Memory")
        for ngram, v in entity_mem.items():
            st.markdown(f"- **{ngram}** → {v['canonical_value']}")
        if st.button("Clear memory"):
            st.session_state["entity_memory"] = {}
            st.session_state["conversation_memory"] = {}
            st.session_state["llm_history"] = []
            st.rerun()

if not is_langfuse_enabled():
    st.error(
        "Langfuse est requis pour la traçabilité. "
        "Configure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY et LANGFUSE_BASE_URL, puis relance l'application."
    )
    st.stop()

# ── Session state initialisation ─────────────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("llm_history", [])  # clean [{role, content}] for LLM
st.session_state.setdefault("entity_memory", {})  # {ngram: {canonical_value, ...}}
st.session_state.setdefault("conversation_memory", {})  # structured cross-turn context
st.session_state.setdefault("clarification_pending", None)
st.session_state.setdefault("session_id", uuid.uuid4().hex)
st.session_state.setdefault(
    "anonymous_user_id",
    "anon_" + hashlib.sha256(st.session_state["session_id"].encode("utf-8")).hexdigest()[:16],
)


def _add_to_llm_history(role: str, content: str) -> None:
    """Add a message to the LLM history, capped at HISTORY_MAX_EXCHANGES exchanges."""
    st.session_state["llm_history"].append({"role": role, "content": content})
    max_msgs = HISTORY_MAX_EXCHANGES * 2
    if len(st.session_state["llm_history"]) > max_msgs:
        st.session_state["llm_history"] = st.session_state["llm_history"][-max_msgs:]


_TABLE_REQUEST_RE = re.compile(
    r"\b("
    r"tableau|tableaux|table|dataframe|classement|class[eé]ment|classer|liste|lister|"
    r"affiche(?:r|z)?\s+(?:le\s+)?tableau|montre(?:r|z)?\s+(?:le\s+)?tableau|"
    r"top\s*\d+|top|ranking|rank"
    r")\b",
    re.IGNORECASE,
)


def _user_requested_table(question: str, chart_type: str | None) -> bool:
    del chart_type
    return bool(_TABLE_REQUEST_RE.search(question or ""))


# ── Chat history rendering ────────────────────────────────────────────────────
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "dataframe" in message and isinstance(message["dataframe"], pd.DataFrame):
            st.dataframe(message["dataframe"], use_container_width=True)
        if "chart" in message and message["chart"] is not None:
            st.plotly_chart(message["chart"], use_container_width=True)
        # if message.get("sql"):
        #     with st.expander("SQL & routing details"):
        #         st.code(message["sql"], language="sql")
        #         st.write(f"Intent: `{message.get('intent', '')}`")


# ── Chat input — ALWAYS rendered first so the widget is never absent ─────────
# (if st.stop() is called below, the widget was already registered by Streamlit)
_chat_input = st.chat_input("Ask a question about the 2025 legislative election results…")

# ── Clarification UI (blocks normal input while pending) ─────────────────────
cp = st.session_state.get("clarification_pending")
if cp:
    options = cp.get("options", [])
    labels = [o["label"] for o in options]

    if labels:
        # QCM mode — modalities are known
        with st.container(border=True):
            st.markdown("**Please clarify your question:**")
            with st.form("clarif_form"):
                choice_label = st.radio(
                    "Choose an option:",
                    labels + ["Other (specify manually)"],
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Confirm", type="primary")

            if submitted:
                if choice_label == "Other (specify manually)":
                    # Fall back to open text — remove options to trigger open-text mode
                    st.session_state["clarification_pending"]["options"] = []
                    st.rerun()
                else:
                    idx = labels.index(choice_label)
                    chosen = options[idx]
                    ngram = cp["ngram"]
                    # Store in session entity memory
                    st.session_state["entity_memory"][ngram] = {
                        "canonical_value": chosen["canonical_value"],
                        "canonical_norm": chosen["canonical_norm"],
                        "entity_type": chosen["entity_type"],
                    }
                    _add_to_llm_history("user", f"[Clarification: {ngram} = {chosen['canonical_value']}]")
                    st.session_state["messages"].append(
                        {
                            "role": "user",
                            "content": f"→ {choice_label}",
                        }
                    )
                    # Re-run original question with updated memory
                    st.session_state["pending_question"] = cp["original_question"]
                    st.session_state["clarification_pending"] = None
                    st.rerun()
    else:
        # Open-text mode — modalities unknown or user chose "Autre"
        with st.container(border=True):
            st.markdown("**Please clarify your question:**")
            with st.form("clarif_open_form"):
                user_clarif = st.text_input(
                    "Your clarification:",
                    placeholder="e.g. the city of Korhogo, San Pedro region…",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Send", type="primary")

            if submitted and user_clarif.strip():
                refined = f"{cp['original_question']} — {user_clarif.strip()}"
                st.session_state["messages"].append(
                    {
                        "role": "user",
                        "content": user_clarif.strip(),
                    }
                )
                _add_to_llm_history("user", user_clarif.strip())
                st.session_state["pending_question"] = refined
                st.session_state["clarification_pending"] = None
                st.rerun()

    st.stop()  # hide the normal chat input while clarification is pending


# ── Normal question handling ──────────────────────────────────────────────────
question = st.session_state.pop("pending_question", None) or _chat_input

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        streamed_answer_parts: list[str] = []
        stream_placeholder = st.empty()
        stream_state = {"last_update": 0.0}

        def _stream_answer_token(token: str) -> None:
            streamed_answer_parts.append(token)
            now = time.perf_counter()
            if now - stream_state["last_update"] < 0.05 and token not in {".", "!", "?", "\n"}:
                return
            stream_state["last_update"] = now
            stream_placeholder.markdown("".join(streamed_answer_parts) + "▌")

        with st.spinner("Analysing…"):
            with bind_stream_callback(_stream_answer_token):
                result = answer_question(
                    question,
                    history=st.session_state["llm_history"],
                    entity_memory=st.session_state["entity_memory"],
                    conversation_memory=st.session_state["conversation_memory"],
                    session_id=st.session_state["session_id"],
                    anonymous_user_id=st.session_state["anonymous_user_id"],
                )

        answer = result.get("answer", "")
        intent = result.get("intent", "")
        df = result.get("dataframe")
        chart = None
        chart_type = result.get("chart_type", "none")
        safe_sql = result.get("safe_sql") or result.get("sql")
        rag_results = result.get("rag_results") or []
        updated_memory = result.get("conversation_memory")
        show_table = _user_requested_table(question, chart_type)

        # ── Clarification requested ───────────────────────────────────────
        if intent == "clarification":
            stream_placeholder.markdown(answer)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
            st.session_state["clarification_pending"] = {
                "original_question": question,
                "ngram": result.get("clarification_ngram", ""),
                "options": result.get("clarification_options", []),
            }
            # Don't add to llm_history yet — wait for user's choice
            st.rerun()

        # ── Normal answer ─────────────────────────────────────────────────
        stream_placeholder.markdown(answer)
        if show_table and isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True)
        if isinstance(df, pd.DataFrame) and not df.empty and chart_type and chart_type.lower() != "none":
            chart = make_chart(
                df,
                chart_type,
                trace_id=result.get("trace_id"),
            )
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

        # ── Provenance (RAG path) ─────────────────────────────────────────
        if rag_results and intent == "rag_narrative":
            pages = sorted({c["source_page"] for c in rag_results if c.get("source_page")})
            page_label = ", ".join(f"p.{p}" for p in pages) if pages else "—"
            with st.expander(f"Sources ({len(rag_results)} excerpt(s) — {page_label})"):
                for i, chunk in enumerate(rag_results, 1):
                    page = chunk.get("source_page", "?")
                    source_type = chunk.get("source_type") or (chunk.get("provenance") or {}).get("source_type")
                    source_id = chunk.get("source_id") or (chunk.get("provenance") or {}).get("source_id")
                    chunk_id = chunk.get("chunk_id") or (chunk.get("provenance") or {}).get("chunk_id")
                    excerpt = chunk.get("chunk_text", "")[:200]
                    st.markdown(f"**Excerpt {i}** — page {page}")
                    st.caption(
                        "Provenance: "
                        f"`source_type={source_type or 'n/a'}` · "
                        f"`source_id={source_id or 'n/a'}` · "
                        f"`chunk_id={chunk_id or 'n/a'}`"
                    )
                    st.caption(excerpt + ("…" if len(chunk.get("chunk_text", "")) > 200 else ""))

        trace_id = result.get("trace_id")
        if trace_id:
            feedback_left, feedback_right, _ = st.columns([1, 1, 8])
            if feedback_left.button(
                "👍",
                key=f"feedback_up_{trace_id}",
                help="Réponse utile",
            ):
                store = get_observability_store()
                if store is not None:
                    store.record_feedback(trace_id, 1)
                st.toast("Merci pour votre retour.")
            if feedback_right.button(
                "👎",
                key=f"feedback_down_{trace_id}",
                help="Réponse à améliorer",
            ):
                store = get_observability_store()
                if store is not None:
                    store.record_feedback(trace_id, -1)
                st.toast("Retour enregistré.")

        # if safe_sql:
        #     with st.expander("SQL & routing details"):
        #         st.code(safe_sql, language="sql")
        #         st.write(f"Intent: `{intent}`")
        #         st.write(f"SQL valid: `{result.get('sql_valid', False)}`")

    # Update LLM history (capped)
    _add_to_llm_history("user", question)
    _add_to_llm_history("assistant", answer)
    if isinstance(updated_memory, dict):
        st.session_state["conversation_memory"] = updated_memory

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "dataframe": df if show_table and isinstance(df, pd.DataFrame) else None,
            "chart": chart,
            "sql": safe_sql,
            "intent": intent,
        }
    )
