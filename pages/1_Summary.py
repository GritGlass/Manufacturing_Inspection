from __future__ import annotations

import math
import os
import sys
import textwrap
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.local_gemma_model import (
    are_runtime_dependencies_available,
    generate_response,
    is_model_downloaded,
)
from scripts.utils import (
    CLASS_VISUALIZATION_ORDER,
    PDF_FONT_CANDIDATES,
    SUMMARY_ANALYSIS_SYSTEM_PROMPT,
    _get_discrete_class_colors,
    _get_llm_runtime_settings,
    build_aggregate_run,
    build_label_distribution_frame,
    configure_page,
    load_dashboard_data,
    render_class_distribution_chart,
    render_page_header,
)


def build_overview_frame(latest_run: dict[str, Any] | None) -> pd.DataFrame:
    if not latest_run:
        return pd.DataFrame(
            [
                {"Metric": "Total Product", "Value": 0},
                {"Metric": "OK", "Value": 0},
                {"Metric": "NG", "Value": 0},
            ]
        )

    return pd.DataFrame(
        [
            {"Metric": "Total Product", "Value": latest_run["total_count"]},
            {"Metric": "OK", "Value": latest_run["good_count"]},
            {"Metric": "NG", "Value": latest_run["bad_count"]},
        ]
    )


def build_trend_frame(labels: list[str], values: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"period": labels, "value": values}).set_index("period")


def demo_series(base_value: int, count: int, volatility: float, slope: float) -> list[int]:
    values: list[int] = []
    floor = max(int(base_value * 0.5), 1)
    for idx in range(count):
        wave = math.sin((idx + 1) * 0.9) * base_value * volatility
        drift = idx * slope
        values.append(max(floor, int(round(base_value + wave + drift))))
    return values


def build_trend_data(latest_run: dict[str, Any] | None) -> dict[str, pd.DataFrame]:
    anchor = latest_run["timestamp"] if latest_run else datetime.now()
    base_total = latest_run["total_count"] if latest_run else 60

    month_labels = [(anchor - timedelta(days=30 * offset)).strftime("%b") for offset in range(5, -1, -1)]
    week_labels = [(anchor - timedelta(weeks=offset)).strftime("Wk %U") for offset in range(7, -1, -1)]
    day_labels = [(anchor - timedelta(days=offset)).strftime("%m/%d") for offset in range(9, -1, -1)]

    return {
        "Monthly profit graph": build_trend_frame(
            month_labels,
            demo_series(max(base_total, 35), 6, volatility=0.10, slope=2.4),
        ),
        "Weekly profit graph": build_trend_frame(
            week_labels,
            demo_series(max(int(base_total * 0.88), 28), 8, volatility=0.12, slope=1.0),
        ),
        "Daily profit graph": build_trend_frame(
            day_labels,
            demo_series(max(int(base_total * 0.8), 18), 10, volatility=0.08, slope=0.6),
        ),
    }


def get_run_image_records(
    image_records: list[dict[str, Any]],
    latest_run: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not latest_run:
        return []
    return [
        record
        for record in image_records
        if record["run_name"] == latest_run["name"] and record["exists"]
    ]


def select_summary_image_records(
    image_records: list[dict[str, Any]],
    latest_run: dict[str, Any] | None,
    max_images: int = 4,
) -> list[dict[str, Any]]:
    run_records = get_run_image_records(image_records, latest_run)
    if not run_records:
        return []

    bad_records = [record for record in run_records if record["label"] != "Normal"]
    good_records = [record for record in run_records if record["label"] == "Normal"]

    selected: list[dict[str, Any]] = []
    selected.extend(bad_records[: min(3, max_images)])
    if len(selected) < max_images:
        selected.extend(good_records[: max_images - len(selected)])
    if len(selected) < max_images:
        seen_paths = {record["path"] for record in selected}
        for record in run_records:
            if record["path"] in seen_paths:
                continue
            selected.append(record)
            seen_paths.add(record["path"])
            if len(selected) >= max_images:
                break
    return selected[:max_images]


def _get_record_recency_key(record: dict[str, Any]) -> tuple[float, float, str]:
    timestamp_value = record.get("timestamp")
    timestamp_score = timestamp_value.timestamp() if isinstance(timestamp_value, datetime) else 0.0

    modified_score = timestamp_score
    if record.get("exists") and record.get("path"):
        try:
            modified_score = Path(str(record["path"])).stat().st_mtime
        except OSError:
            modified_score = timestamp_score

    return (
        float(modified_score),
        float(timestamp_score),
        str(record.get("filename", "")),
    )


def select_summary_report_image_records(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records_by_label: dict[str, list[dict[str, Any]]] = {}
    for record in image_records:
        if not record.get("exists"):
            continue
        label = str(record.get("label", "")).strip()
        if not label:
            continue
        records_by_label.setdefault(label, []).append(record)

    if not records_by_label:
        return []

    ordered_labels = [label for label in CLASS_VISUALIZATION_ORDER if label in records_by_label]
    ordered_labels.extend(sorted(set(records_by_label) - set(ordered_labels)))

    selected_records: list[dict[str, Any]] = []
    for label in ordered_labels:
        latest_record = max(records_by_label[label], key=_get_record_recency_key)
        selected_records.append(latest_record)
    return selected_records


def format_trend_for_prompt(frame: pd.DataFrame) -> str:
    return ", ".join(f"{index}:{int(value)}" for index, value in frame["value"].items())


def build_summary_analysis_prompt(
    latest_run: dict[str, Any] | None,
    trends: dict[str, pd.DataFrame],
    sample_records: list[dict[str, Any]],
) -> str:
    if not latest_run:
        return "현재 검사 데이터가 없습니다. 데이터 부재 상황에 대한 짧은 한국어 코멘트를 작성해 주세요."

    class_distribution = ", ".join(
        f"{label} {count}개" for label, count in latest_run["label_counts"].items()
    ) or "분류 데이터 없음"
    sample_descriptions = "\n".join(
        f"- {record['filename']} | 예측 클래스: {record['label']}"
        for record in sample_records
    ) or "- 첨부 이미지 없음"

    return (
        "다음은 반도체 검사 Summary 데이터입니다.\n"
        f"최신 배치 시각: {latest_run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"총 제품 수: {latest_run['total_count']}\n"
        f"양품 수: {latest_run['good_count']}\n"
        f"불량 수: {latest_run['bad_count']}\n"
        f"평균 추론 시간(ms): {latest_run['average_inference_ms']:.2f}\n"
        f"클래스 분포: {class_distribution}\n"
        f"월별 데이터: {format_trend_for_prompt(trends['Monthly profit graph'])}\n"
        f"주별 데이터: {format_trend_for_prompt(trends['Weekly profit graph'])}\n"
        f"일별 데이터: {format_trend_for_prompt(trends['Daily profit graph'])}\n"
        "첨부 이미지 정보:\n"
        f"{sample_descriptions}\n"
        "위 수치와 첨부된 추론 결과 이미지를 함께 보고, 품질 상태와 주요 결함 양상을 요약한 분석 코멘트를 작성해 주세요."
    )


@st.cache_data(show_spinner=False)
def generate_summary_analysis_comment_cached(
    prompt: str,
    image_paths: tuple[str, ...],
    cache_key: str,
    model_dir: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    return generate_response(
        prompt=prompt,
        system_prompt=SUMMARY_ANALYSIS_SYSTEM_PROMPT,
        image_paths=list(image_paths),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        model_dir=model_dir,
    )


def build_summary_analysis_comment(
    latest_run: dict[str, Any] | None,
    image_records: list[dict[str, Any]],
    trends: dict[str, pd.DataFrame],
) -> tuple[str, list[dict[str, Any]]]:
    sample_records = select_summary_image_records(image_records, latest_run)
    if latest_run is None:
        return "검사 데이터가 없어 분석 코멘트를 생성할 수 없습니다.", sample_records

    dependency_ready, dependency_message = are_runtime_dependencies_available()
    if not dependency_ready:
        return dependency_message, sample_records

    llm_settings = _get_llm_runtime_settings()
    if not is_model_downloaded(llm_settings["model_dir"]):
        return "로컬 Gemma 모델이 준비되지 않아 분석 코멘트를 생성하지 못했습니다.", sample_records

    prompt = build_summary_analysis_prompt(latest_run, trends, sample_records)
    image_paths = tuple(record["path"] for record in sample_records if record["exists"])
    cache_key = f"{latest_run['name']}|{len(image_paths)}|{latest_run['bad_count']}|{latest_run['good_count']}"
    try:
        comment = generate_summary_analysis_comment_cached(
            prompt,
            image_paths,
            cache_key,
            str(llm_settings["model_dir"]),
            min(int(llm_settings["max_new_tokens"]), 768),
            float(llm_settings["temperature"]),
        ).strip()
        return comment or "LLM 응답이 비어 있어 분석 코멘트를 생성하지 못했습니다.", sample_records
    except Exception as exc:
        return f"LLM 분석 코멘트 생성 중 오류가 발생했습니다: {exc}", sample_records


def _wrap_text_lines(text: str, width: int = 42) -> str:
    paragraphs = []
    for chunk in text.splitlines():
        stripped = chunk.strip()
        if not stripped:
            paragraphs.append("")
        else:
            paragraphs.append(textwrap.fill(stripped, width=width))
    return "\n".join(paragraphs)


def _prepare_pdf_runtime() -> tuple[Any, Any, Any, Any]:
    cache_dir = BASE_DIR / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fontconfig_cache_dir = BASE_DIR / ".fontconfig-cache"
    fontconfig_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(fontconfig_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = False

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    from PIL import Image

    return plt, PdfPages, FontProperties, Image


def _get_pdf_font(FontProperties: Any) -> Any | None:
    for path in PDF_FONT_CANDIDATES:
        if path.exists():
            return FontProperties(fname=str(path))
    return None


def _apply_font_to_axis(axis: Any, font_prop: Any | None) -> None:
    if font_prop is None:
        return
    axis.title.set_fontproperties(font_prop)
    axis.xaxis.label.set_fontproperties(font_prop)
    axis.yaxis.label.set_fontproperties(font_prop)
    for tick_label in axis.get_xticklabels():
        tick_label.set_fontproperties(font_prop)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontproperties(font_prop)


@st.cache_data(show_spinner=False)
def build_summary_pdf_bytes(
    period_range: str,
    latest_timestamp: str,
    overview_rows: tuple[tuple[str, int], ...],
    label_rows: tuple[tuple[str, int], ...],
    recent_runs_rows: tuple[tuple[str, str, int, int, int, float, float], ...],
    trend_rows: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
    analysis_comment: str,
    sample_images: tuple[tuple[str, str, str], ...],
) -> bytes:
    plt, PdfPages, FontProperties, Image = _prepare_pdf_runtime()
    font_prop = _get_pdf_font(FontProperties)
    pdf_buffer = BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        trend_map = {title.lower(): points for title, points in trend_rows}

        def _sanitize_analysis_comment(text: str) -> str:
            cleaned_lines = [
                line for line in text.splitlines()
                if not line.lstrip().startswith("#")
            ]
            cleaned = "\n".join(cleaned_lines).strip()
            return cleaned if cleaned else "Analysis comment is not available."

        cleaned_analysis_comment = _sanitize_analysis_comment(analysis_comment)

        def _get_trend_points(keyword: str) -> tuple[tuple[str, int], ...]:
            for title, points in trend_map.items():
                if keyword in title:
                    return points
            return tuple()

        def _draw_line_chart(
            axis: Any,
            points: tuple[tuple[str, int], ...],
            *,
            x_tick_size: float = 6.5,
            show_y_label: bool = True,
            y_label: str = "Value",
            x_tick_rotation: float = 0.0,
        ) -> None:
            x_values = [item[0] for item in points]
            y_values = [item[1] for item in points]
            if x_values and y_values:
                axis.plot(x_values, y_values, marker="o", color="#24A0A8", linewidth=1.3, markersize=2.6)
            axis.set_title("")
            if show_y_label:
                axis.set_ylabel(y_label, fontsize=6.2, fontproperties=font_prop)
            else:
                axis.set_ylabel("")
            axis.grid(alpha=0.22, linewidth=0.5)
            axis.tick_params(axis="x", labelsize=x_tick_size, pad=1)
            axis.tick_params(axis="y", labelsize=6.0, pad=1)
            for tick_label in axis.get_xticklabels():
                tick_label.set_rotation(x_tick_rotation)
                tick_label.set_ha("center")
            _apply_font_to_axis(axis, font_prop)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Report", fontsize=16, fontproperties=font_prop, y=0.965)
        fig.text(0.95, 0.92, f"Period : {period_range}", fontsize=9, fontproperties=font_prop, ha="right")

        fig.text(0.08, 0.865, "• Product Amount", fontsize=11, fontproperties=font_prop, weight="bold")
        fig.text(0.41, 0.865, "• Product Class Distribution", fontsize=11, fontproperties=font_prop, weight="bold")

        overview_ax = fig.add_axes([0.08, 0.70, 0.22, 0.15])
        overview_ax.axis("off")
        overview_table = overview_ax.table(
            cellText=[[metric, value] for metric, value in overview_rows],
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.66, 0.34],
        )
        overview_table.auto_set_font_size(False)
        overview_table.set_fontsize(7.8)
        overview_table.scale(1.0, 1.85)
        for cell in overview_table.get_celld().values():
            cell.set_linewidth(0.6)
            cell.set_edgecolor("#C6C6C6")
        if font_prop is not None:
            for cell in overview_table.get_celld().values():
                cell.get_text().set_fontproperties(font_prop)

        label_ax = fig.add_axes([0.41, 0.72, 0.51, 0.12])
        label_ax.set_title("")
        labels = [row[0] for row in label_rows]
        counts = [row[1] for row in label_rows]
        total_label_count = sum(max(int(count), 0) for count in counts)
        label_colors = [_get_discrete_class_colors(labels).get(label, "#5B8FF9") for label in labels]
        label_ax.bar(labels, counts, color=label_colors)
        label_ax.set_ylabel("Count", fontsize=6.2, fontproperties=font_prop)
        label_ax.tick_params(axis="x", labelsize=4.2, rotation=90, pad=1)
        for tick_label in label_ax.get_xticklabels():
            tick_label.set_ha("center")
        label_ax.tick_params(axis="y", labelsize=6.0)
        y_max = max(counts) if counts else 0
        for idx, count in enumerate(counts):
            ratio = (count / total_label_count * 100.0) if total_label_count > 0 else 0.0
            label_ax.text(
                idx,
                max(float(count) * 0.5, max(0.3, y_max * 0.05)),
                f"{count} ({ratio:.0f}%)",
                ha="center",
                va="center",
                rotation=90,
                fontsize=6.4,
                weight="bold",
                color="black",
                fontproperties=font_prop,
            )
        _apply_font_to_axis(label_ax, font_prop)

        fig.text(0.08, 0.628, "• Monthly Graph", fontsize=11, fontproperties=font_prop, weight="bold")
        fig.text(0.51, 0.628, "• Weekly Graph", fontsize=11, fontproperties=font_prop, weight="bold")
        monthly_ax = fig.add_axes([0.16, 0.503, 0.315, 0.11])
        weekly_ax = fig.add_axes([0.549, 0.503, 0.315, 0.11])
        _draw_line_chart(
            monthly_ax,
            _get_trend_points("monthly"),
            x_tick_size=6.0,
            y_label="Count",
        )
        _draw_line_chart(
            weekly_ax,
            _get_trend_points("weekly"),
            x_tick_size=2.1,
            show_y_label=False,
            x_tick_rotation=90,
        )

        fig.text(0.08, 0.449, "• Daily Graph", fontsize=11, fontproperties=font_prop, weight="bold")
        daily_ax = fig.add_axes([0.16, 0.304, 0.704, 0.13])
        _draw_line_chart(daily_ax, _get_trend_points("daily"), x_tick_size=5.8, y_label="Count")
        daily_ax.set_xlabel("Period", fontsize=6.2, fontproperties=font_prop)
        _apply_font_to_axis(daily_ax, font_prop)

        wrapped_analysis_text = _wrap_text_lines(cleaned_analysis_comment, width=80)
        analysis_line_count = wrapped_analysis_text.count("\n") + 1
        analysis_font_size = 9.0
        if analysis_line_count >= 10:
            analysis_font_size = 8.2
        if analysis_line_count >= 13:
            analysis_font_size = 7.6

        fig.text(0.08, 0.231, "• Analysis", fontsize=11, fontproperties=font_prop, weight="bold")
        analysis_ax = fig.add_axes([0.114, 0.061, 0.806, 0.16])
        analysis_ax.axis("off")
        analysis_ax.text(
            0.0,
            0.9,
            wrapped_analysis_text,
            va="top",
            fontsize=analysis_font_size,
            fontproperties=font_prop,
            linespacing=1.28,
        )

        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Report", fontsize=16, fontproperties=font_prop, y=0.965)
        fig.text(0.08, 0.915, "• Inference Image", fontsize=11.5, fontproperties=font_prop, weight="bold")

        image_count = len(sample_images)
        if image_count > 0:
            if image_count <= 8:
                cols = 2
            elif image_count <= 18:
                cols = 3
            else:
                cols = 4

            rows = max(1, math.ceil(image_count / cols))
            grid = fig.add_gridspec(
                rows,
                cols,
                left=0.06,
                right=0.94,
                top=0.88,
                bottom=0.06,
                hspace=0.22,
                wspace=0.08,
            )

            base_font = 7.0 if cols <= 2 else 6.2 if cols == 3 else 5.5
            wrap_width = 26 if cols <= 2 else 17 if cols == 3 else 12

            for idx, (path, label, filename) in enumerate(sample_images):
                axis = fig.add_subplot(grid[idx // cols, idx % cols])
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor("#BFC2C7")

                display_name = textwrap.fill(
                    str(filename),
                    width=wrap_width,
                    max_lines=2,
                    placeholder="...",
                )
                adaptive_font = max(5.0, base_font - max(0.0, (len(str(filename)) - wrap_width * 2) / 18.0))

                try:
                    image = Image.open(path).convert("RGB")
                    img_width, img_height = image.size

                    side = max(img_width, img_height)
                    square_image = Image.new("RGB", (side, side), (255, 255, 255))
                    square_image.paste(image, ((side - img_width) // 2, (side - img_height) // 2))
                    square_image = square_image.resize((512, 512))

                    box_x0, box_x1 = 0.04, 0.96
                    box_y0, box_y1 = 0.24, 0.94
                    box_width = box_x1 - box_x0
                    box_height = box_y1 - box_y0

                    # Keep width, but reduce drawable image height by additional 15% from current setting.
                    draw_box_height = box_height * 0.95 * 0.85
                    box_y0 = box_y0 + (box_height - draw_box_height) / 2.0
                    box_y1 = box_y0 + draw_box_height
                    box_height = draw_box_height

                    draw_side = min(box_width, box_height)
                    draw_width = draw_side
                    draw_height = draw_side

                    draw_x0 = box_x0 + (box_width - draw_width) / 2.0
                    draw_x1 = draw_x0 + draw_width
                    draw_y0 = box_y0 + (box_height - draw_height) / 2.0
                    draw_y1 = draw_y0 + draw_height

                    axis.imshow(square_image, extent=(draw_x0, draw_x1, draw_y0, draw_y1), aspect="equal")
                    axis.set_aspect("equal", adjustable="box")
                except Exception:
                    pass

                axis.set_title(str(label), fontsize=adaptive_font + 0.6, fontproperties=font_prop, pad=1.5)
                axis.set_xlabel(
                    display_name,
                    fontsize=adaptive_font,
                    fontproperties=font_prop,
                    labelpad=2,
                )

        pdf.savefig(fig)
        plt.close(fig)

    return pdf_buffer.getvalue()


def render_summary_page(runs, image_records) -> None:
    render_page_header("Summary")

    summary_run = build_aggregate_run(runs)
    trends = build_trend_data(summary_run)
    overview_frame = build_overview_frame(summary_run)
    label_frame = build_label_distribution_frame(summary_run)
    report_sample_records = select_summary_report_image_records(image_records)
    report_pdf_error = ""

    with st.spinner("Summary 리포트와 AI 코멘트를 준비하는 중입니다..."):
        analysis_comment, _sample_records = build_summary_analysis_comment(summary_run, image_records, trends)

        if summary_run:
            recent_runs_rows = tuple(
                (
                    run["name"],
                    run["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    int(run["total_count"]),
                    int(run["good_count"]),
                    int(run["bad_count"]),
                    float(run["average_inference_ms"]),
                    float(run["total_process_ms"]),
                )
                for run in reversed(runs[-10:])
            )

            trend_rows = tuple(
                (
                    title,
                    tuple((str(index), int(value)) for index, value in frame["value"].items()),
                )
                for title, frame in trends.items()
            )
            if runs:
                period_start = min(run["timestamp"] for run in runs)
                period_end = max(run["timestamp"] for run in runs)
                period_range = f"{period_start.strftime('%y-%m-%d')} ~ {period_end.strftime('%y-%m-%d')}"
            else:
                period_range = summary_run["timestamp"].strftime("%y-%m-%d")

            try:
                report_pdf = build_summary_pdf_bytes(
                    period_range=period_range,
                    latest_timestamp=summary_run["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    overview_rows=tuple(
                        (str(row["Metric"]), int(row["Value"])) for _, row in overview_frame.iterrows()
                    ),
                    label_rows=tuple((str(index), int(row["count"])) for index, row in label_frame.iterrows()),
                    recent_runs_rows=recent_runs_rows,
                    trend_rows=trend_rows,
                    analysis_comment=analysis_comment,
                    sample_images=tuple(
                        (record["path"], record["label"], record["filename"])
                        for record in report_sample_records
                        if record["exists"]
                    ),
                )
            except Exception as exc:
                report_pdf = b""
                report_pdf_error = f"PDF 리포트 생성 중 오류가 발생했습니다: {exc}"
        else:
            report_pdf = b""

    left_col, right_col = st.columns([1.2, 1.0], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("Overview")
            st.dataframe(overview_frame, width="stretch", hide_index=True)

    with right_col:
        total = int(overview_frame.loc[overview_frame["Metric"] == "Total Product", "Value"].iloc[0])
        good = int(overview_frame.loc[overview_frame["Metric"] == "OK", "Value"].iloc[0])
        bad = int(overview_frame.loc[overview_frame["Metric"] == "NG", "Value"].iloc[0])

        metric_cols = st.columns(3)
        metric_cols[0].metric("Total", f"{total:,}")
        metric_cols[1].metric("OK", f"{good:,}")
        metric_cols[2].metric("NG", f"{bad:,}")

        st.download_button(
            "Download Report",
            data=report_pdf,
            file_name="manufacturing_summary_report.pdf",
            mime="application/pdf",
            disabled=not bool(report_pdf),
            width="stretch",
        )

        if summary_run:
            st.caption(
                f"Summary target: {summary_run['name']} | "
                f"Latest timestamp: {summary_run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Average inference: {summary_run['average_inference_ms']:.2f} ms/image"
            )

        if report_pdf_error:
            st.warning(report_pdf_error)

    with st.container(border=True):
        st.subheader("Class distribution graph")
        render_class_distribution_chart(label_frame)

    for chart_title, frame in trends.items():
        with st.container(border=True):
            st.subheader(chart_title)
            try:
                import plotly.graph_objects as go

                x_values = [str(index) for index in frame.index]
                y_values = [int(value) for value in frame["value"].tolist()]
                is_weekly = "weekly" in chart_title.lower()
                y_axis_label = "Count" if ("monthly" in chart_title.lower() or "daily" in chart_title.lower()) else "Value"

                trend_fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode="lines+markers",
                            line=dict(color="#24A0A8", width=2),
                            marker=dict(size=6),
                            showlegend=False,
                        )
                    ]
                )
                trend_fig.update_layout(
                    xaxis_title="Period",
                    yaxis_title=y_axis_label,
                    xaxis=dict(tickangle=90 if is_weekly else 0),
                    margin=dict(l=24, r=16, t=20, b=24),
                )
                st.plotly_chart(trend_fig, width="stretch")
            except ImportError:
                st.line_chart(frame, width="stretch")

    with st.container(border=True):
        st.subheader("AI analysis comment")
        if summary_run:
            if analysis_comment.startswith("LLM 분석 코멘트 생성 중 오류") or analysis_comment.startswith("로컬 Gemma 모델이"):
                st.warning(analysis_comment)
            elif analysis_comment.startswith("로컬 Gemma 실행에 필요한 패키지"):
                st.warning(analysis_comment)
            else:
                st.write(analysis_comment)
        else:
            st.info("분석할 Summary 데이터가 없습니다.")


configure_page("Summary")
_config, runs, image_records, _log_entries = load_dashboard_data()
render_summary_page(runs, image_records)
