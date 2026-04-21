from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from scripts.utils import (
    SUPABASE_CONNECTION_NAME,
    SUPABASE_IMAGE_TABLE,
    SupabaseConnection,
    build_aggregate_run,
    build_label_distribution_frame,
    configure_page,
    load_dashboard_data,
    render_class_distribution_chart,
    render_page_header,
)


PAGE_LINKS = (
    ("Summary", "pages/1_Summary.py"),
    ("Detail", "pages/2_Detail.py"),  #    ("Fine-tuning", "pages/3_Fine_tuning.py"),
    ("Setting", "pages/4_Setting.py"),
    ("Log", "pages/5_Log.py"),
)

CSV_DEFAULT_QUERY_START_DATE = date(2026, 1, 1)
CSV_DEFAULT_QUERY_END_DATE = date(2026, 5, 31)


def render_home_page(config: dict[str, Any], runs: list[dict[str, Any]], log_entries: list[dict[str, str]]) -> None:
    summary_run = build_aggregate_run(runs)
    label_frame = build_label_distribution_frame(summary_run)
    render_page_header("Dashboard Home", "Use Streamlit page navigation in the sidebar to move between screens.")

    if summary_run:
        cols = st.columns(4)
        cols[0].metric("Total Product", f"{summary_run['total_count']:,}")
        cols[1].metric("OK", f"{summary_run['good_count']:,}")
        cols[2].metric("NG", f"{summary_run['bad_count']:,}")
        cols[3].metric("Avg Inference", f"{summary_run['average_inference_ms']:.2f} ms")
        st.caption(f"Aggregated across all runs: {len(runs)} groups, {summary_run['total_count']:,} image files.")
    else:
        st.info("No inference result has been loaded yet.")

    st.subheader("Pages")
    link_cols = st.columns(len(PAGE_LINKS))
    for idx, (label, target) in enumerate(PAGE_LINKS):
        with link_cols[idx]:
            st.page_link(target, label=label, width="stretch")

    with st.container(border=True):
        st.subheader("All Image Distribution")
        render_class_distribution_chart(label_frame)

    recent_runs = pd.DataFrame(
        [
            {
                "timestamp": run["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "total": run["total_count"],
                "OK": run["good_count"],
                "NG": run["bad_count"],
                "avg_inference_ms": round(run["average_inference_ms"], 2),
            }
            for run in reversed(runs[-5:])
        ]
    )
    key_logs = pd.DataFrame.from_records(
        log_entries[:5],
        columns=["date", "time", "source", "log_type", "content"],
    )

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        with st.container(border=True):
            st.subheader("Recent Runs")
            if recent_runs.empty:
                st.info("No recent runs available.")
            else:
                st.dataframe(recent_runs, width="stretch", hide_index=True)

    with right_col:
        with st.container(border=True):
            st.subheader("Latest Logs")
            if key_logs.empty:
                st.info("No logs available.")
            else:
                st.dataframe(key_logs, width="stretch", hide_index=True)

    with st.expander("Current configuration", expanded=False):
        config_frame = pd.DataFrame(
            [{"Setting": key, "Value": str(value)} for key, value in config.items()]
        )
        st.dataframe(config_frame, width="stretch", hide_index=True)


def _fetch_supabase_table_list() -> list[str]:
    """Return public table names from the connected Supabase project."""
    if SupabaseConnection is None:
        return [SUPABASE_IMAGE_TABLE]
    try:
        connection = st.connection(SUPABASE_CONNECTION_NAME, type=SupabaseConnection)
        client = getattr(connection, "client", None)
        if client is None:
            return [SUPABASE_IMAGE_TABLE]
        result = (
            client.table("information_schema.tables")
            .select("table_name")
            .eq("table_schema", "public")
            .eq("table_type", "BASE TABLE")
            .order("table_name")
            .execute()
        )
        rows = getattr(result, "data", []) or []
        names = [row["table_name"] for row in rows if isinstance(row, dict) and row.get("table_name")]
        return names if names else [SUPABASE_IMAGE_TABLE]
    except Exception:
        return [SUPABASE_IMAGE_TABLE]


def set_query_table() -> None:
    st.subheader("Data Source")
    st.caption("Select the Supabase table to query.")

    table_list = _fetch_supabase_table_list()
    current_table = str(st.session_state.get("dashboard_query_table", SUPABASE_IMAGE_TABLE)).strip()
    default_index = table_list.index(current_table) if current_table in table_list else 0

    selected_table = st.selectbox(
        "Table",
        options=table_list,
        index=default_index,
        key="dashboard_query_table_selector",
    )
    if selected_table != st.session_state.get("dashboard_query_table"):
        st.session_state["dashboard_query_table"] = selected_table
        st.session_state["dashboard_data_loaded"] = False
        load_dashboard_data.clear()


def set_query_dates():
    st.subheader("Data Period")
    st.caption("Select the date range to query.")

    query_cols = st.columns(3, gap="large")
    with query_cols[0]:
        pending_start_raw = str(st.session_state.get("dashboard_query_date_start_pending", "")).strip()
        try:
            default_start_date = (
                date.fromisoformat(pending_start_raw)
                if pending_start_raw
                else CSV_DEFAULT_QUERY_START_DATE
            )
        except ValueError:
            default_start_date = CSV_DEFAULT_QUERY_START_DATE

        selected_start_date = st.date_input(
            "Start date",
            value=default_start_date,
            key="dashboard_query_date_start_input",
        )

    with query_cols[1]:
        pending_end_raw = str(st.session_state.get("dashboard_query_date_end_pending", "")).strip()
        try:
            default_end_date = (
                date.fromisoformat(pending_end_raw)
                if pending_end_raw
                else CSV_DEFAULT_QUERY_END_DATE
            )
        except ValueError:
            default_end_date = CSV_DEFAULT_QUERY_END_DATE

        selected_end_date = st.date_input(
            "End date",
            value=default_end_date,
            key="dashboard_query_date_end_input",
        )

    with query_cols[2]:
        st.write("")  # spacer for alignment
        if st.button("Load Data", key="dashboard_query_button", width="stretch"):
            if selected_start_date > selected_end_date:
                st.error("The start date cannot be later than the end date.")
            else:
                start_iso = selected_start_date.isoformat()
                end_iso = selected_end_date.isoformat()
                st.session_state["dashboard_query_date_start_pending"] = start_iso
                st.session_state["dashboard_query_date_end_pending"] = end_iso
                st.session_state["dashboard_query_date_start"] = start_iso
                st.session_state["dashboard_query_date_end"] = end_iso
                st.session_state["dashboard_data_loaded"] = True
                load_dashboard_data.clear()
                st.rerun()

def main() -> None:
    configure_page("Dashboard Home")

    data_loaded = bool(st.session_state.get("dashboard_data_loaded", False))

    with st.expander("Query Settings", expanded=not data_loaded):
        set_query_table()
        st.divider()
        set_query_dates()

    if not data_loaded:
        return

    query_date_start = str(st.session_state.get("dashboard_query_date_start", "")).strip() or None
    query_date_end = str(st.session_state.get("dashboard_query_date_end", "")).strip() or None
    config, runs, _image_records, log_entries = load_dashboard_data(
        query_date_start=query_date_start,
        query_date_end=query_date_end,
    )
    render_home_page(config, runs, log_entries)


if __name__ == "__main__":
    main()
