from __future__ import annotations

import time

import pandas as pd
import plotly.express as px

from .observability import record_current_event, record_run_event

_CATEGORY_PRIORITY = (
    "groupement_parti",
    "parti",
    "candidat_liste",
    "vainqueur",
    "region",
    "circonscription",
)
_METRIC_PRIORITY = (
    "winners",
    "seats_won",
    "sieges",
    "count",
    "total",
    "scores",
    "voix",
    "score_pct",
    "taux_participation_pct",
    "region_participation_pct",
    "votants",
    "inscrits",
    "suffrages_exprimes",
)
_IDENTIFIER_COLUMNS = {
    "candidat_id",
    "circonscription_code",
    "page",
    "source_page",
    "source_page_start",
    "source_page_end",
    "nb_bv",
}


def _first_ranked(columns: list[str], priorities: tuple[str, ...]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for preferred in priorities:
        if preferred in by_lower:
            return by_lower[preferred]
    return None


def infer_chart_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col.lower() not in _IDENTIFIER_COLUMNS
        and not pd.api.types.is_bool_dtype(df[col])
    ]
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and col.lower() not in _IDENTIFIER_COLUMNS and col.lower() != "elu"
    ]
    x_col = _first_ranked(categorical_cols, _CATEGORY_PRIORITY)
    if x_col is None:
        x_col = categorical_cols[0] if categorical_cols else None
    y_col = _first_ranked(numeric_cols, _METRIC_PRIORITY)
    if y_col is None:
        y_col = numeric_cols[0] if numeric_cols else None
    if x_col is None and y_col is None:
        return None, None
    if x_col is None:
        x_col = y_col
    return x_col, y_col


def make_chart(
    df: pd.DataFrame,
    chart_type: str,
    trace_id: str | None = None,
):
    started = time.perf_counter()
    requested_type = (chart_type or "bar").lower()
    try:
        x_col, y_col = infer_chart_columns(df)
        if x_col is None:
            duration_ms = (time.perf_counter() - started) * 1_000
            payload = {
                "requested_type": requested_type,
                "generated": False,
                "reason": "no_chartable_column",
                "row_count": len(df),
                "column_count": len(df.columns),
            }
            record_current_event(
                "chart_generation",
                duration_ms=duration_ms,
                payload=payload,
            )
            record_run_event(
                trace_id,
                "chart_generation",
                duration_ms=duration_ms,
                payload=payload,
            )
            return None

        labels = {column: column.replace("_", " ").title() for column in df.columns}
        generated_type = requested_type
        if requested_type == "pie" and y_col and x_col != y_col:
            figure = px.pie(df, names=x_col, values=y_col, labels=labels)
        elif requested_type == "histogram" and y_col:
            figure = px.histogram(df, x=y_col, labels=labels)
        elif y_col and x_col != y_col:
            generated_type = "bar"
            figure = px.bar(df, x=x_col, y=y_col, labels=labels)
            if len(df) > 8:
                figure.update_xaxes(tickangle=-35)
        else:
            generated_type = "histogram"
            figure = px.histogram(df, x=x_col, labels=labels)
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1_000
        payload = {
            "requested_type": requested_type,
            "row_count": len(df),
            "column_count": len(df.columns),
            "error_type": type(exc).__name__,
        }
        record_current_event(
            "chart_generation",
            duration_ms=duration_ms,
            status="error",
            payload=payload,
        )
        record_run_event(
            trace_id,
            "chart_generation",
            duration_ms=duration_ms,
            status="error",
            payload=payload,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1_000
    payload = {
        "requested_type": requested_type,
        "generated_type": generated_type,
        "generated": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "x_column": x_col,
        "y_column": y_col,
        "trace_count": len(figure.data),
    }
    record_current_event(
        "chart_generation",
        duration_ms=duration_ms,
        payload=payload,
    )
    record_run_event(
        trace_id,
        "chart_generation",
        duration_ms=duration_ms,
        payload=payload,
    )
    return figure
