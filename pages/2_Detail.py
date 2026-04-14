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
OPENXAI_DIR = ROOT_DIR / "third_party" / "OpenXAI"
if OPENXAI_DIR.exists() and str(OPENXAI_DIR) not in sys.path:
    sys.path.insert(0, str(OPENXAI_DIR))

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
        helper_text="Choose a model, then click Start infer to run inference again on the currently selected images.",
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
        st.sidebar.caption("The model has changed. Click `Start infer` to rerun inference with the new model.")
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
        raise RuntimeError("The torch/transformers packages required for image reinference are not installed.") from exc

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
        raise RuntimeError("The Pillow package required for image reinference is not installed.") from exc

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

    with st.spinner(f"Running inference again on the selected images with {resolved_model_dir.name}..."):
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
        st.error("Required libraries are missing. Please install scikit-learn and plotly.")
        return

    st.subheader("3D Model Prediction Visualization")
    st.caption("A TensorFlow Projector-style 3D feature space visualization.")

    if len(selected_records) < 3:
        st.info("Select at least 3 images to use 3D Visualization.")
        return

    image_paths = [record["path"] for record in selected_records if record["exists"]]
    if not image_paths:
        st.warning("There are no images to display.")
        return

    base_model_dir = _get_detail_base_model_dir(selected_records)
    selected_model_dirs = _resolve_selected_model_dirs(selected_records)
    if len(selected_model_dirs) > 1:
        st.warning("The selected images were generated with different inference models. The 3D features will be computed using the first model.")

    with st.spinner("Extracting features from the model..."):
        features, processed_paths = _extract_features_from_images(image_paths, base_model_dir)

    if features is None or len(features) == 0:
        st.error("Failed to extract features.")
        return

    record_by_path = {record["path"]: record for record in selected_records if record["exists"]}
    plotted_records = [record_by_path[path] for path in processed_paths if path in record_by_path]
    if not plotted_records:
        st.error("Could not find label information for visualization.")
        return

    labels = [record["label"] for record in plotted_records]
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    elif features.ndim == 1:
        features = features.reshape(-1, 1)

    col1, col2 = st.columns(2)
    with col1:
        reduction_method = st.selectbox(
            "Dimensionality reduction",
            ["PCA", "t-SNE", "UMAP"],
            help="Choose how high-dimensional features should be reduced into 3D.",
        )
    with col2:
        show_labels = st.checkbox("Show labels", value=False)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if reduction_method == "PCA":
        reducer = PCA(n_components=3, random_state=42)
        features_3d = reducer.fit_transform(features_scaled)
        reduction_info = f"PCA (explained variance: {sum(reducer.explained_variance_ratio_):.2%})"
    elif reduction_method == "t-SNE":
        try:
            from sklearn.manifold import TSNE

            with st.spinner("Computing t-SNE... this may take some time."):
                reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features) - 1))
                features_3d = reducer.fit_transform(features_scaled)
            reduction_info = "t-SNE"
        except Exception as exc:
            st.error(f"t-SNE computation failed: {exc}")
            return
    else:
        try:
            import umap

            with st.spinner("Computing UMAP..."):
                reducer = umap.UMAP(n_components=3, random_state=42)
                features_3d = reducer.fit_transform(features_scaled)
            reduction_info = "UMAP"
        except ImportError:
            st.warning("Run `pip install umap-learn` to use UMAP. PCA will be used instead.")
            reducer = PCA(n_components=3, random_state=42)
            features_3d = reducer.fit_transform(features_scaled)
            reduction_info = "PCA (UMAP not installed)"

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

    with st.expander("Image details", expanded=False):
        details_df = pd.DataFrame(
            {
                "Filename": [record["filename"] for record in plotted_records],
                "Prediction": labels,
                "Path": [record["path"] for record in plotted_records],
            }
        )
        st.dataframe(details_df, width="stretch")


def _resolve_target_label_index(model: Any, label_name: str, input_tensor: Any) -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The torch package required for XAI is not installed.") from exc

    label2id = getattr(model.config, "label2id", {}) or {}
    if label_name in label2id:
        return int(label2id[label_name])

    with torch.no_grad():
        logits = model(pixel_values=input_tensor).logits
        return int(logits.argmax(dim=-1).item())


def _render_bottom_right_pagination_controls(
    *,
    total_items: int,
    page_key: str,
    page_size_key: str,
    default_page_size: int,
    show_total_pages_in_size_field: bool = False,
) -> tuple[int, int]:
    if page_size_key not in st.session_state:
        st.session_state[page_size_key] = int(default_page_size)
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    page_size = int(st.session_state[page_size_key])
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    current_page = int(st.session_state[page_key])
    current_page = max(1, min(current_page, total_pages))
    st.session_state[page_key] = current_page
    page_display_key = f"{page_key}_display"
    page_size_display_key = f"{page_size_key}_display"
    st.session_state[page_display_key] = str(current_page)
    size_field_value = total_pages if show_total_pages_in_size_field else page_size
    st.session_state[page_size_display_key] = str(size_field_value)

    spacer, controls_col = st.columns([6, 4], gap="small")
    with spacer:
        st.empty()
    with controls_col:
        label_cols = st.columns([2, 2], gap="small")
        with label_cols[0]:
            st.caption("Page")
            page_cols = st.columns([3, 1, 1], gap="small")
            with page_cols[0]:
                st.text_input(
                    "Current page",
                    key=page_display_key,
                    label_visibility="collapsed",
                    disabled=True,
                )
            with page_cols[1]:
                if st.button("<", key=f"{page_key}_minus", width="stretch"):
                    st.session_state[page_key] = max(1, current_page - 1)
                    st.rerun()
            with page_cols[2]:
                if st.button(">", key=f"{page_key}_plus", width="stretch"):
                    st.session_state[page_key] = min(total_pages, current_page + 1)
                    st.rerun()

        with label_cols[1]:
            st.caption("Total Pages" if show_total_pages_in_size_field else "Page Size")
            page_size_cols = st.columns([3, 1, 1], gap="small")
            with page_size_cols[0]:
                st.text_input(
                    "Page size",
                    key=page_size_display_key,
                    label_visibility="collapsed",
                    disabled=True,
                )
            with page_size_cols[1]:
                st.empty()
            with page_size_cols[2]:
                st.empty()

    page_size = int(st.session_state[page_size_key])
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    current_page = int(st.session_state[page_key])
    st.session_state[page_key] = max(1, min(current_page, total_pages))
    return st.session_state[page_key], page_size


def _render_detail_xai_visualization(
    selected_records: list[dict[str, Any]],
    selected_model_dir: Path | None,
) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from matplotlib import colormaps
        from openxai import Explainer
    except ImportError as exc:
        st.error(
            "The packages required for XAI visualization could not be loaded. "
            "Please make sure `captum` and the OpenXAI dependencies are installed."
        )
        st.caption(f"Import error: {exc}")
        return

    st.subheader("XAI Visualization (OpenXAI)")
    st.caption("Overlays a heatmap on the original image using OpenXAI attributions.")

    if not selected_records:
        st.info("There are no images to display for XAI.")
        return

    if selected_model_dir is None:
        st.info("Please select an inference model in the sidebar first.")
        return

    openxai_methods = ["grad", "sg", "itg", "ig", "lime", "shap", "control"]
    supported_methods = {"grad", "sg", "itg", "ig"}
    selected_method = st.selectbox(
        "OpenXAI method",
        options=openxai_methods,
        index=0,
        key="detail_xai_method_selector",
        help="For the current image classification model, grad, sg, itg, and ig are recommended.",
    )
    overlay_alpha = st.slider(
        "Heatmap overlay strength",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        key="detail_xai_overlay_alpha",
    )
    colormap_name = st.selectbox(
        "Heatmap colormap",
        options=["turbo", "jet", "magma", "viridis"],
        index=0,
        key="detail_xai_colormap",
    )

    if selected_method not in supported_methods:
        st.warning(
            "The selected method is not directly supported by the current image model pipeline. "
            "Please choose from `grad`, `sg`, `itg`, or `ig`."
        )
        return

    valid_records = [record for record in selected_records if record.get("exists")]
    if not valid_records:
        st.info("XAI cannot be computed because no existing images were found.")
        return

    total_images = len(valid_records)
    current_xai_page = int(st.session_state.get("detail_xai_page", 1))
    xai_page_size = int(st.session_state.get("detail_xai_page_size", 5))
    xai_total_pages = max(1, (total_images + xai_page_size - 1) // xai_page_size)
    current_xai_page = max(1, min(current_xai_page, xai_total_pages))
    st.session_state["detail_xai_page"] = current_xai_page
    st.session_state["detail_xai_page_size"] = xai_page_size
    page_start = (current_xai_page - 1) * xai_page_size
    page_end = page_start + xai_page_size
    page_records = valid_records[page_start:page_end]

    st.caption(f"{total_images} images total | Page {current_xai_page}/{xai_total_pages}")

    try:
        image_processor, base_model, device_name, _ = _load_detail_classifier_runtime(selected_model_dir)
        device = torch.device(device_name)

        class _OpenXAILogitsModel(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output = self.model(pixel_values=x)
                return output.logits if hasattr(output, "logits") else output

        openxai_model = _OpenXAILogitsModel(base_model).to(device)
        explainer = Explainer(method=selected_method, model=openxai_model)
    except Exception as exc:
        st.error(f"Failed to initialize the OpenXAI explainer: {exc}")
        return

    cmap = colormaps.get_cmap(colormap_name)
    xai_errors: list[str] = []
    page_items: list[dict[str, Any]] = []

    for record in page_records:
        image_path = record["path"]
        filename = record.get("filename", Path(image_path).name)
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            original_np = np.asarray(rgb_image).astype(np.float32) / 255.0
            height, width = original_np.shape[:2]

            model_inputs = image_processor(images=rgb_image, return_tensors="pt")
            pixel_values = model_inputs["pixel_values"].to(device)
            pixel_values = pixel_values.requires_grad_(True)

            target_idx = _resolve_target_label_index(base_model, str(record["label"]), pixel_values)
            target_tensor = torch.tensor([target_idx], dtype=torch.long, device=device)

            attribution = explainer.get_explanations(pixel_values, target_tensor)
            attribution = attribution.detach().to("cpu")

            if attribution.ndim == 4:
                attribution_map = attribution[0].abs().mean(dim=0)
            elif attribution.ndim == 3:
                attribution_map = attribution[0].abs()
            else:
                raise ValueError(f"Unexpected attribution shape: {tuple(attribution.shape)}")

            attribution_map = attribution_map.unsqueeze(0).unsqueeze(0)
            attribution_map = F.interpolate(
                attribution_map,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            attribution_min = float(attribution_map.min())
            attribution_max = float(attribution_map.max())
            normalized = (attribution_map - attribution_min) / (attribution_max - attribution_min + 1e-8)

            heatmap_rgb = cmap(normalized)[..., :3]
            overlay = np.clip((1.0 - overlay_alpha) * original_np + overlay_alpha * heatmap_rgb, 0.0, 1.0)

            page_items.append(
                {
                    "filename": filename,
                    "label": record["label"],
                    "target_idx": target_idx,
                    "original": (original_np * 255).astype(np.uint8),
                    "heatmap": (heatmap_rgb * 255).astype(np.uint8),
                    "overlay": (overlay * 255).astype(np.uint8),
                }
                )
        except Exception as exc:
            xai_errors.append(f"{filename}: {exc}")

    for item in page_items:
        st.markdown(f"**{item['filename']}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(item["original"], caption="Original", width="stretch")
        with c2:
            st.image(item["heatmap"], caption=f"XAI ({selected_method})", width="stretch")
        with c3:
            st.image(item["overlay"], caption="Overlay", width="stretch")
        st.caption(f"Prediction: {item['label']} | Target ID: {item['target_idx']}")

    _render_bottom_right_pagination_controls(
        total_items=total_images,
        page_key="detail_xai_page",
        page_size_key="detail_xai_page_size",
        default_page_size=5,
        show_total_pages_in_size_field=True,
    )

    if xai_errors:
        st.warning("Some XAI computations failed: " + "; ".join(xai_errors[:3]))


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
    result_total = 0
    current_result_page = 1
    result_page_size = int(st.session_state.get("detail_result_page_size", 25))
    result_total_pages = 1
    result_start = 0
    result_end = result_page_size
    page_records: list[dict[str, Any]] = []
    selected_model_dir: Path | None = None
    if selected_paths:
        raw_selected_records = _get_detail_selected_records(image_records)
        selected_model_dir = _get_detail_base_model_dir(raw_selected_records) if raw_selected_records else None
        result_total = len(raw_selected_records)
        result_total_pages = max(1, (result_total + result_page_size - 1) // result_page_size)
        current_result_page = int(st.session_state.get("detail_result_page", 1))
        current_result_page = max(1, min(current_result_page, result_total_pages))
        st.session_state["detail_result_page"] = current_result_page
        st.session_state["detail_result_page_size"] = result_page_size

        result_start = (current_result_page - 1) * result_page_size
        result_end = result_start + result_page_size
        page_records = raw_selected_records[result_start:result_end]
        selected_records = raw_selected_records

        tab1, tab2, tab3 = st.tabs(["Result", "3D Visualization", "XAI"])

        result_total = len(selected_records)
        result_total_pages = max(1, (result_total + result_page_size - 1) // result_page_size)
        current_result_page = max(1, min(current_result_page, result_total_pages))
        st.session_state["detail_result_page"] = current_result_page
        st.session_state["detail_result_page_size"] = result_page_size
        result_start = (current_result_page - 1) * result_page_size
        result_end = result_start + result_page_size
        page_records = selected_records[result_start:result_end]

        with tab1:
            st.caption(f"{result_total} images total | Page {current_result_page}/{result_total_pages}")

            with st.expander(f"Selected images ({len(page_records)}/{result_total})", expanded=True):
                cols = st.columns(5, gap="large")
                for idx, record in enumerate(page_records):
                    with cols[idx % 5]:
                        if record["exists"]:
                            st.image(record["path"], width="stretch")
                        else:
                            st.info("Image not found")
                        st.caption(record["filename"])
                        st.caption(f"Prediction: {record['label']}")

            _render_bottom_right_pagination_controls(
                total_items=result_total,
                page_key="detail_result_page",
                page_size_key="detail_result_page_size",
                default_page_size=25,
            )

        with tab2:
            if len(selected_records) < 3:
                st.info("Select at least 3 images to use 3D Visualization.")
            else:
                _render_detail_3d_visualization(selected_records)

        with tab3:
            _render_detail_xai_visualization(selected_records, selected_model_dir)
    else:
        st.info("Select multiple items above to view multiple images.")

    if selected_records:
        st.caption("Interactive fine-tuning can be run on the `Fine-tuning` page.")


configure_page("Detail")
_config, _runs, image_records, _log_entries = load_dashboard_data()
render_detail_page(image_records)
