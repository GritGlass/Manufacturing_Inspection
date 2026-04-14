from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from scripts.utils import (
    build_aggregate_run,
    build_label_distribution_frame,
    configure_page,
    load_dashboard_data,
    render_class_distribution_chart,
    render_page_header,
)


PAGE_LINKS = (
    ("Summary", "pages/1_Summary.py"),
    ("Detail", "pages/2_Detail.py"),
    ("Fine-tuning", "pages/3_Fine_tuning.py"),
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


def main() -> None:
    configure_page("Dashboard Home")

    st.subheader("Data Period")
    st.caption("조회할 데이터 기간을 선택해 주세요.")

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
            "시작 날짜",
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
            "종료 날짜",
            value=default_end_date,
            key="dashboard_query_date_end_input",
        )

    with query_cols[2]:
        st.write("")  # spacer for alignment
        if st.button("데이터 조회", key="dashboard_query_button", width="stretch"):
            if selected_start_date > selected_end_date:
                st.error("시작 날짜가 종료 날짜보다 뒤에 있습니다.")
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
    
    st.divider()

    if not bool(st.session_state.get("dashboard_data_loaded", False)):
        render_home_page(
            {
                "data_source": "supabase",
                "query_date_start": "not_loaded",
                "query_date_end": "not_loaded",
                "status": "대기",
            },
            [],
            [],
        )
        st.info("위에서 조회 날짜 범위를 선택한 뒤 `데이터 조회`를 눌러주세요.")
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
