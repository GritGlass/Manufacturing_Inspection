from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.detail_finetune_mcp import resolve_base_model_dir
from scripts.utils import (
    CLASS_VISUALIZATION_ORDER,
    _extract_features_from_images,
    _get_cached_detail_inference_result,
    _get_discrete_class_colors,
    _render_classifier_model_selector,
    _suppress_transformers_path_alias_warning,
    _to_project_relative_path,
    configure_page,
    load_dashboard_data,
    render_page_header,
)


def _render_detail_inference_model_selector(selected_records: list[dict[str, Any]]) -> tuple[Path | None, bool, bool]:
    selected_model_dir, model_changed = _render_classifier_model_selector(
        selected_records=selected_records,
        container=st.sidebar,
        selector_key="detail_inference_model_selector",
        active_key="detail_inference_model_active",
        section_title="Image Inference",
        helper_text="모델을 고른 뒤 Start infer를 눌러 현재 선택 이미지를 다시 예측합니다.",
        label="Inference model",
        add_divider=True,
    )
    start_infer = st.sidebar.button(
        "Start infer",
        key="detail_inference_start_button",
        width="stretch",
        disabled=selected_model_dir is None or not selected_records,
    )
    if selected_records and model_changed:
        st.sidebar.caption("모델이 변경되었습니다. `Start infer`를 눌러 새 모델로 재추론하세요.")
    return selected_model_dir, model_changed, start_infer


def _resolve_inference_output_targets(output_path: Path, inference_mode: str) -> tuple[Path, Path, Path]:
    base_output_dir = output_path.parent if output_path.suffix else output_path
    if base_output_dir.exists() and not base_output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {base_output_dir}")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S_") + f"{time.time_ns() % 1_000_000:06d}"
    timestamp_output_dir = base_output_dir / timestamp
    timestamp_output_dir.mkdir(parents=True, exist_ok=False)
    return (
        timestamp_output_dir / f"{inference_mode}_inference_results.json",
        timestamp_output_dir / f"{inference_mode}_inference_timing.txt",
        timestamp_output_dir,
    )


def _save_inference_results(results: dict[str, str], output_json_path: Path) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_results = {
        _to_project_relative_path(path): label for path, label in results.items()
    }
    output_json_path.write_text(
        json.dumps(normalized_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_inference_timing(
    model_path: Path,
    initial_model_load_time_milliseconds: float,
    total_process_time_milliseconds: float,
    per_image_preprocess_times_milliseconds: dict[str, float],
    per_image_inference_times_milliseconds: dict[str, float],
    output_timing_path: Path,
) -> None:
    output_timing_path.parent.mkdir(parents=True, exist_ok=True)
    total_model_load_time_milliseconds = (
        initial_model_load_time_milliseconds + sum(per_image_preprocess_times_milliseconds.values())
    )
    total_inference_time_milliseconds = sum(per_image_inference_times_milliseconds.values())

    lines = [
        f"model_path: {_to_project_relative_path(model_path)}",
        f"initial_model_load_time_milliseconds: {initial_model_load_time_milliseconds:.3f}",
        f"model_load_time_milliseconds: {total_model_load_time_milliseconds:.3f}",
        f"inference_time_milliseconds: {total_inference_time_milliseconds:.3f}",
        f"total_process_time_milliseconds: {total_process_time_milliseconds:.3f}",
        f"image_count: {len(per_image_inference_times_milliseconds)}",
    ]

    if per_image_preprocess_times_milliseconds:
        average_preprocess_time = (
            sum(per_image_preprocess_times_milliseconds.values())
            / len(per_image_preprocess_times_milliseconds)
        )
        lines.append(f"average_per_image_preprocess_time_milliseconds: {average_preprocess_time:.3f}")

    if per_image_inference_times_milliseconds:
        average_inference_time = (
            sum(per_image_inference_times_milliseconds.values())
            / len(per_image_inference_times_milliseconds)
        )
        lines.append(f"average_per_image_inference_time_milliseconds: {average_inference_time:.3f}")

    if per_image_preprocess_times_milliseconds:
        lines.append("per_image_preprocess_time_milliseconds:")
        for path, value in per_image_preprocess_times_milliseconds.items():
            lines.append(f"{path}: {value:.3f}")

    if per_image_inference_times_milliseconds:
        lines.append("per_image_inference_time_milliseconds:")
        for path, value in per_image_inference_times_milliseconds.items():
            lines.append(f"{path}: {value:.3f}")

    output_timing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_detail_classifier_runtime(model_dir: Path) -> tuple[Any, Any, str, float]:
    try:
        import torch

        _suppress_transformers_path_alias_warning()
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError("이미지 재추론에 필요한 torch/transformers 패키지가 없습니다.") from exc

    resolved_model_dir = resolve_base_model_dir(model_dir)
    cached_model_dir = st.session_state.get("detail_inference_runtime_model_dir")
    initial_model_load_time_milliseconds = 0.0
    if cached_model_dir != str(resolved_model_dir):
        cached_model = st.session_state.get("detail_inference_runtime_model")
        if cached_model is not None:
            try:
                cached_model.to("cpu")
            except Exception:
                pass

        st.session_state.pop("detail_inference_runtime_model", None)
        st.session_state.pop("detail_inference_runtime_processor", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        load_start_ns = time.perf_counter_ns()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_processor = AutoImageProcessor.from_pretrained(str(resolved_model_dir))
        model = AutoModelForImageClassification.from_pretrained(str(resolved_model_dir)).to(device)
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        initial_model_load_time_milliseconds = (time.perf_counter_ns() - load_start_ns) / 1_000_000.0

        st.session_state["detail_inference_runtime_model_dir"] = str(resolved_model_dir)
        st.session_state["detail_inference_runtime_processor"] = image_processor
        st.session_state["detail_inference_runtime_model"] = model
        st.session_state["detail_inference_runtime_device"] = device
    st.session_state["detail_inference_runtime_initial_model_load_ms"] = initial_model_load_time_milliseconds

    return (
        st.session_state["detail_inference_runtime_processor"],
        st.session_state["detail_inference_runtime_model"],
        str(st.session_state["detail_inference_runtime_device"]),
        float(st.session_state.get("detail_inference_runtime_initial_model_load_ms", 0.0)),
    )


def _predict_detail_records_with_model(
    selected_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str] | None]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    signature = (
        str(resolved_model_dir),
        tuple(record["path"] for record in selected_records),
    )

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("이미지 재추론에 필요한 Pillow 패키지가 없습니다.") from exc

    image_processor, model, device_name, initial_model_load_time_milliseconds = _load_detail_classifier_runtime(
        resolved_model_dir
    )
    device = torch.device(device_name)
    predicted_records: list[dict[str, Any]] = []
    prediction_errors: list[str] = []
    per_image_preprocess_times_milliseconds: dict[str, float] = {}
    per_image_inference_times_milliseconds: dict[str, float] = {}
    prediction_results: dict[str, str] = {}
    total_start_ns = time.perf_counter_ns()

    def _sync_if_needed() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with st.spinner(f"{resolved_model_dir.name} 모델로 선택 이미지를 다시 예측하는 중입니다..."):
        with torch.no_grad():
            for record in selected_records:
                updated_record = dict(record)
                updated_record["model_dir"] = str(resolved_model_dir)
                updated_record["model_dir_display"] = _to_project_relative_path(resolved_model_dir)

                if not updated_record.get("exists"):
                    predicted_records.append(updated_record)
                    continue

                try:
                    preprocess_start_ns = time.perf_counter_ns()
                    with Image.open(updated_record["path"]) as image:
                        rgb_image = image.convert("RGB")
                    inputs = image_processor(images=rgb_image, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    preprocess_time_milliseconds = (time.perf_counter_ns() - preprocess_start_ns) / 1_000_000.0
                    _sync_if_needed()
                    inference_start_ns = time.perf_counter_ns()
                    logits = model(**inputs).logits
                    predicted_index = logits.argmax(dim=-1).item()
                    _sync_if_needed()
                    inference_time_milliseconds = (time.perf_counter_ns() - inference_start_ns) / 1_000_000.0
                    updated_record["label"] = str(model.config.id2label[predicted_index])
                    relative_image_path = _to_project_relative_path(updated_record["path"])
                    prediction_results[updated_record["path"]] = updated_record["label"]
                    per_image_preprocess_times_milliseconds[relative_image_path] = preprocess_time_milliseconds
                    per_image_inference_times_milliseconds[relative_image_path] = inference_time_milliseconds
                except Exception as exc:
                    prediction_errors.append(f"{Path(updated_record['path']).name}: {exc}")
                predicted_records.append(updated_record)

    artifact_paths: dict[str, str] | None = None
    if prediction_results:
        total_process_time_milliseconds = (time.perf_counter_ns() - total_start_ns) / 1_000_000.0
        output_json_path, output_timing_path, output_dir = _resolve_inference_output_targets(OUTPUT_DIR, "batch")
        _save_inference_results(prediction_results, output_json_path)
        _save_inference_timing(
            model_path=resolved_model_dir,
            initial_model_load_time_milliseconds=initial_model_load_time_milliseconds,
            total_process_time_milliseconds=total_process_time_milliseconds,
            per_image_preprocess_times_milliseconds=per_image_preprocess_times_milliseconds,
            per_image_inference_times_milliseconds=per_image_inference_times_milliseconds,
            output_timing_path=output_timing_path,
        )
        artifact_paths = {
            "output_dir": str(output_dir),
            "results_path": str(output_json_path),
            "timing_path": str(output_timing_path),
        }

    st.session_state["detail_inference_prediction_signature"] = signature
    st.session_state["detail_inference_prediction_records"] = predicted_records
    st.session_state["detail_inference_prediction_errors"] = prediction_errors
    st.session_state["detail_inference_prediction_output_dir"] = (
        artifact_paths["output_dir"] if artifact_paths else ""
    )
    st.session_state["detail_inference_prediction_results_path"] = (
        artifact_paths["results_path"] if artifact_paths else ""
    )
    st.session_state["detail_inference_prediction_timing_path"] = (
        artifact_paths["timing_path"] if artifact_paths else ""
    )
    return predicted_records, prediction_errors, artifact_paths


def _resolve_selected_model_dirs(selected_records: list[dict[str, Any]]) -> list[Path]:
    resolved_dirs: list[Path] = []
    for record in selected_records:
        model_dir = resolve_base_model_dir(record.get("model_dir"))
        if model_dir not in resolved_dirs:
            resolved_dirs.append(model_dir)
    return resolved_dirs


def _get_detail_base_model_dir(selected_records: list[dict[str, Any]]) -> Path:
    resolved_dirs = _resolve_selected_model_dirs(selected_records)
    if resolved_dirs:
        return resolved_dirs[0]
    return resolve_base_model_dir("model")


def _get_detail_selected_records(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_paths = st.session_state.get("detail_selected_image_paths", [])
    if not selected_paths:
        return []
    selected_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in image_records:
        path = record["path"]
        if path in selected_paths and path not in seen_paths:
            selected_records.append(record)
            seen_paths.add(path)
            if len(seen_paths) == len(selected_paths):
                break
    return selected_records


def _reset_detail_finetune_session(selected_paths: list[str]) -> None:
    prefix = "detail_finetune_"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            st.session_state.pop(key, None)
    st.session_state["detail_selected_image_paths"] = selected_paths
    st.session_state["detail_finetune_chat"] = []
    st.session_state["detail_finetune_plan"] = None
    st.session_state["detail_finetune_execution"] = None


def _render_detail_3d_visualization(selected_records: list[dict[str, Any]]) -> None:
    try:
        import numpy as np
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        st.error("필요한 라이브러리가 없습니다. scikit-learn과 plotly를 설치해주세요.")
        return

    st.subheader("3D Model Prediction Visualization")
    st.caption("TensorFlow Projector 스타일의 3D 특성 공간 시각화")

    image_paths = [record["path"] for record in selected_records if record["exists"]]
    if not image_paths:
        st.warning("표시할 이미지가 없습니다.")
        return

    base_model_dir = _get_detail_base_model_dir(selected_records)
    selected_model_dirs = _resolve_selected_model_dirs(selected_records)
    if len(selected_model_dirs) > 1:
        st.warning("선택한 이미지들이 서로 다른 추론 모델에서 생성되었습니다. 가장 첫 번째 모델 기준으로 3D 특성을 계산합니다.")

    with st.spinner("모델에서 특성을 추출하는 중입니다..."):
        features, processed_paths = _extract_features_from_images(image_paths, base_model_dir)

    if features is None or len(features) == 0:
        st.error("특성 추출에 실패했습니다.")
        return

    record_by_path = {record["path"]: record for record in selected_records if record["exists"]}
    plotted_records = [record_by_path[path] for path in processed_paths if path in record_by_path]
    if not plotted_records:
        st.error("시각화에 사용할 라벨 정보를 찾을 수 없습니다.")
        return

    labels = [record["label"] for record in plotted_records]
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    elif features.ndim == 1:
        features = features.reshape(-1, 1)

    col1, col2 = st.columns(2)
    with col1:
        reduction_method = st.selectbox(
            "차원 축소 방법",
            ["PCA", "t-SNE", "UMAP"],
            help="고차원 특성을 3D로 축소하는 방법을 선택하세요",
        )
    with col2:
        show_labels = st.checkbox("라벨 표시", value=False)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if reduction_method == "PCA":
        reducer = PCA(n_components=3, random_state=42)
        features_3d = reducer.fit_transform(features_scaled)
        reduction_info = f"PCA (설명된 분산: {sum(reducer.explained_variance_ratio_):.2%})"
    elif reduction_method == "t-SNE":
        try:
            from sklearn.manifold import TSNE

            with st.spinner("t-SNE 계산 중... (시간이 걸릴 수 있습니다)"):
                reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features) - 1))
                features_3d = reducer.fit_transform(features_scaled)
            reduction_info = "t-SNE"
        except Exception as exc:
            st.error(f"t-SNE 계산 실패: {exc}")
            return
    else:
        try:
            import umap

            with st.spinner("UMAP 계산 중..."):
                reducer = umap.UMAP(n_components=3, random_state=42)
                features_3d = reducer.fit_transform(features_scaled)
            reduction_info = "UMAP"
        except ImportError:
            st.warning("UMAP을 사용하려면 `pip install umap-learn`을 실행해주세요. PCA를 대신 사용합니다.")
            reducer = PCA(n_components=3, random_state=42)
            features_3d = reducer.fit_transform(features_scaled)
            reduction_info = "PCA (UMAP 설치 필요)"

    features_3d_display = np.asarray(features_3d, dtype=np.float32)
    axis_std = features_3d_display.std(axis=0)
    axis_std[axis_std == 0] = 1.0
    features_3d_display = (features_3d_display - features_3d_display.mean(axis=0)) / axis_std

    unique_labels = [label for label in CLASS_VISUALIZATION_ORDER if label in set(labels)]
    unique_labels.extend(sorted(set(labels) - set(unique_labels)))
    color_map = _get_discrete_class_colors(labels)

    fig = go.Figure()
    for label in unique_labels:
        class_indices = [idx for idx, item in enumerate(labels) if item == label]
        if not class_indices:
            continue

        class_records = [plotted_records[idx] for idx in class_indices]
        hover_text = [
            f"{record['label']}<br>File: {record['filename']}"
            for record in class_records
        ]

        fig.add_trace(
            go.Scatter3d(
                x=features_3d_display[class_indices, 0],
                y=features_3d_display[class_indices, 1],
                z=features_3d_display[class_indices, 2],
                mode="markers+text" if show_labels else "markers",
                name=label,
                marker=dict(
                    size=8,
                    color=color_map[label],
                    line=dict(width=0.5, color="white"),
                ),
                text=[label] * len(class_indices) if show_labels else None,
                textposition="top center",
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"3D Model Feature Space ({reduction_info})",
        scene=dict(
            xaxis=dict(title="Feature 1"),
            yaxis=dict(title="Feature 2"),
            zaxis=dict(title="Feature 3"),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.7, y=1.5, z=1.4),
                projection=dict(type="perspective"),
            ),
        ),
        width=1000,
        height=700,
        hovermode="closest",
        dragmode="orbit",
        legend=dict(title="Class"),
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Class Distribution")
    class_counts = pd.DataFrame(
        {"Class": unique_labels, "Count": [labels.count(label) for label in unique_labels]}
    )
    class_distribution_fig = go.Figure(
        data=[
            go.Bar(
                x=class_counts["Class"],
                y=class_counts["Count"],
                marker_color=[color_map[label] for label in class_counts["Class"]],
                text=class_counts["Count"],
                textposition="outside",
                hovertemplate="Class: %{x}<br>Count: %{y}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    class_distribution_fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(class_distribution_fig, width="stretch")

    with st.expander("이미지 상세 정보", expanded=False):
        details_df = pd.DataFrame(
            {
                "Filename": [record["filename"] for record in plotted_records],
                "Prediction": labels,
                "Path": [record["path"] for record in plotted_records],
            }
        )
        st.dataframe(details_df, width="stretch")


def render_detail_page(image_records) -> None:
    render_page_header("Detail")
    all_dates = ["All dates"] + sorted({record["date"] for record in image_records}, reverse=True)
    all_classes = ["All classes"] + sorted({record["label"] for record in image_records})

    filter_cols = st.columns([1, 1], gap="large")
    with filter_cols[0]:
        selected_date = st.selectbox("Date filter", all_dates)
    with filter_cols[1]:
        selected_class = st.selectbox("Class filter", all_classes)

    filtered = image_records
    if selected_date != "All dates":
        filtered = [record for record in filtered if record["date"] == selected_date]
    if selected_class != "All classes":
        filtered = [record for record in filtered if record["label"] == selected_class]

    if not filtered:
        st.info("No images matched the selected filter.")
        return

    select_options = []
    seen_paths = set()
    for record in filtered:
        if record["path"] not in seen_paths:
            select_options.append(record["path"])
            seen_paths.add(record["path"])

    previous_selected_paths = list(st.session_state.get("detail_selected_image_paths", []))
    selected_paths = [path for path in previous_selected_paths if path in select_options]
    selected_paths = st.multiselect(
        "Select images",
        options=select_options,
        default=selected_paths,
        format_func=lambda path: Path(path).name,
        key="detail_multi_select_paths",
    )
    if previous_selected_paths != selected_paths:
        _reset_detail_finetune_session(selected_paths)
    else:
        st.session_state["detail_selected_image_paths"] = selected_paths

    selected_records = []
    if selected_paths:
        raw_selected_records = _get_detail_selected_records(image_records)
        selected_model_dir, model_changed, start_infer = _render_detail_inference_model_selector(raw_selected_records)
        if model_changed:
            _reset_detail_finetune_session(selected_paths)

        selected_records = raw_selected_records
        prediction_errors = []
        artifact_paths = None
        has_cached_result = False
        if selected_model_dir is not None:
            cached_records, cached_errors, cached_artifact_paths = _get_cached_detail_inference_result(
                raw_selected_records,
                selected_model_dir,
            )
            if cached_records is not None:
                has_cached_result = True
                selected_records = cached_records
                prediction_errors = cached_errors
                artifact_paths = cached_artifact_paths

            if start_infer:
                try:
                    selected_records, prediction_errors, artifact_paths = _predict_detail_records_with_model(
                        raw_selected_records,
                        selected_model_dir,
                    )
                except RuntimeError as exc:
                    st.warning(str(exc))

        if selected_model_dir is not None and not start_infer and not has_cached_result:
            st.info("선택한 모델로 다시 예측하려면 사이드바의 `Start infer` 버튼을 누르세요.")

        if prediction_errors:
            st.warning("일부 이미지 재추론에 실패했습니다: " + "; ".join(prediction_errors[:3]))
        if artifact_paths:
            st.caption(f"Saved inference results: {_to_project_relative_path(artifact_paths['results_path'])}")
            st.caption(f"Saved inference timing: {_to_project_relative_path(artifact_paths['timing_path'])}")

        tab1, tab2 = st.tabs(["Result", "3D Visualization"])

        with tab1:
            with st.expander(f"Selected images ({len(selected_records)})", expanded=True):
                cols = st.columns(min(len(selected_records), 3), gap="large")
                for idx, record in enumerate(selected_records):
                    with cols[idx % len(cols)]:
                        if record["exists"]:
                            st.image(record["path"], width="stretch")
                        else:
                            st.info("Image not found")
                        st.caption(record["filename"])
                        st.caption(f"Prediction: {record['label']}")

        with tab2:
            _render_detail_3d_visualization(selected_records)
    else:
        st.info("여러 이미지를 선택하려면 위에서 항목을 여러 개 선택하세요.")

    if selected_records:
        st.caption("Interactive fine-tuning은 `Fine-tuning` 페이지에서 실행할 수 있습니다.")


configure_page("Detail")
_config, _runs, image_records, _log_entries = load_dashboard_data()
render_detail_page(image_records)
