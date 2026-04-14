from __future__ import annotations

import base64
import csv
import json
import logging
import math
import os
import random
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import toml
try:
    from st_supabase_connection import SupabaseConnection
except ImportError:
    SupabaseConnection = None

from detail_finetune_mcp import (
    CLASSIFICATION_MODEL_ROOT,
    CLASSIFIER_MODEL_DIR,
    DetailFineTunePlan,
    load_available_classes,
    resolve_base_model_dir,
    run_detail_finetune_plan,
)
from local_gemma_model import (
    MODEL_DIR as DEFAULT_LLM_MODEL_DIR,
    are_runtime_dependencies_available,
    generate_response,
    is_model_downloaded,
    list_available_model_dirs,
)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "data" / "semicondotor_seg_data_path.json"
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "log"
APP_LOG_PATH = LOG_DIR / "dashboard_events.jsonl"
PAGE_LINKS = (
    ("Summary", "pages/1_Summary.py"),
    ("Detail", "pages/2_Detail.py"),
    ("Fine-tuning", "pages/3_Fine_tuning.py"),
    ("Setting", "pages/4_Setting.py"),
    ("Log", "pages/5_Log.py"),
   
)
SEVERITY_ORDER = {
    "error": 0,
    "Emergency": 1,
    "Warning": 2,
    "start": 3,
    "done": 4,
    "System done": 5,
    "System start": 6,
    "Model update": 7,
}
DEFAULT_GEMMA_SYSTEM_PROMPT = """당신은 한국어 중심의 제조 대시보드 보조 AI입니다.
답변은 짧고 명확하게 작성하세요.
사용자가 공정, 품질, 알람, 설정 관련 질문을 하면 실행 가능한 다음 행동을 함께 제안하세요.
프롬프트에 현재 대시보드 상태나 Streamlit 실행 컨텍스트가 함께 제공되면 그 정보를 우선 근거로 사용하세요.
컨텍스트에 없는 내용은 추측하지 말고 없다고 명시하세요."""
SUMMARY_ANALYSIS_SYSTEM_PROMPT = """당신은 반도체 검사 결과를 요약하는 품질 분석가입니다.
반드시 한국어로만 답변하세요.
입력된 수치, 추세, 이미지 정보를 근거로 4~6문장 분석 코멘트를 작성하세요.
전체 품질 상태, 주요 결함, 추세 해석, 즉시 권장 조치를 모두 포함하세요.
추정은 최소화하고 데이터에 근거해 간결하게 정리하세요."""
PDF_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
)
CLASS_VISUALIZATION_ORDER = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Local",
    "Near-Full",
    "Normal",
    "Scratch",
)
CLASS_VISUALIZATION_COLORS = {
    "Center": "#FF3B30",
    "Donut": "#FF9500",
    "Edge-Loc": "#34C759",
    "Edge-Ring": "#00C7BE",
    "Local": "#32ADE6",
    "Near-Full": "#007AFF",
    "Normal": "#AF52DE",
    "Scratch": "#FFD60A",
}
DETAIL_PREPROCESSING_LABELS = {
    "none": "전처리 없음",
    "light_augmentation": "가벼운 증강",
    "medium_augmentation": "중간 증강",
    "heavy_augmentation": "강한 증강",
    "histogram_equalization": "히스토그램 균등화",
    "denoise": "잡음 제거",
}
REQUIRED_CLASSIFIER_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "label2id.json",
)
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_MAX_NEW_TOKENS = 512
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SUPABASE_CONNECTION_NAME = "supabase"
SUPABASE_IMAGE_TABLE = "semiconductor"
SUPABASE_IMAGE_COLUMNS = "id,image_path,class,trained,predict,type,created_at"
SUPABASE_QUERY_TTL = "10m"


def _looks_like_project_path(value: str | Path | None) -> bool:
    if value is None:
        return False
    raw_value = str(value).strip()
    if not raw_value:
        return False
    if raw_value.startswith((".", "/", "~")):
        return True
    normalized = raw_value.replace("\\", "/")
    known_prefixes = ("model/", "output/", "data/", "pages/", "scripts/", "log/")
    if normalized.startswith(known_prefixes):
        return True
    return (BASE_DIR / raw_value).exists()


def _resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _to_project_relative_path(value: str | Path | None) -> str:
    if value is None:
        return "-"
    raw_value = str(value).strip()
    if not raw_value:
        return "-"
    path = Path(raw_value).expanduser()
    absolute_path = path if path.is_absolute() else (BASE_DIR / path).resolve()
    return os.path.relpath(str(absolute_path), str(BASE_DIR))


def _format_display_path(value: str | Path | None) -> str:
    if not _looks_like_project_path(value):
        return str(value) if value is not None else "-"
    return _to_project_relative_path(value)


def _normalize_image_path_key(value: str | Path | None) -> str:
    if value is None:
        return ""
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    return str(Path(raw_value).expanduser().resolve())


def _build_log_entry(
    *,
    log_type: str,
    source: str,
    content: str,
    timestamp: datetime | None = None,
    request: str = "",
    response: str = "",
) -> dict[str, str]:
    entry_timestamp = timestamp or datetime.now()
    return {
        "timestamp": entry_timestamp.isoformat(timespec="seconds"),
        "date": entry_timestamp.strftime("%Y-%m-%d"),
        "time": entry_timestamp.strftime("%H:%M:%S"),
        "source": source,
        "log_type": log_type,
        "content": content.strip(),
        "request": request.strip(),
        "response": response.strip(),
    }


def _sort_log_entries(log_entries: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        log_entries,
        key=lambda entry: (
            entry.get("timestamp", ""),
            -SEVERITY_ORDER.get(entry.get("log_type", ""), 99),
        ),
        reverse=True,
    )


def _append_app_log(
    *,
    log_type: str,
    source: str,
    content: str,
    request: str = "",
    response: str = "",
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = _build_log_entry(
        log_type=log_type,
        source=source,
        content=content,
        request=request,
        response=response,
    )
    with APP_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    load_dashboard_data.clear()


def _load_app_logs() -> list[dict[str, str]]:
    if not APP_LOG_PATH.exists():
        return []

    log_entries: list[dict[str, str]] = []
    for raw_line in APP_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        raw_timestamp = str(payload.get("timestamp", "")).strip()
        try:
            entry_timestamp = datetime.fromisoformat(raw_timestamp) if raw_timestamp else datetime.now()
        except ValueError:
            entry_timestamp = datetime.now()
        log_entries.append(
            _build_log_entry(
                log_type=str(payload.get("log_type", "done")),
                source=str(payload.get("source", "App")),
                content=str(payload.get("content", "")),
                timestamp=entry_timestamp,
                request=str(payload.get("request", "")),
                response=str(payload.get("response", "")),
            )
        )
    return _sort_log_entries(log_entries)


def _suppress_transformers_path_alias_warning() -> None:
    logger = logging.getLogger("transformers")
    if any(getattr(current_filter, "_manufacture_path_alias_filter", False) for current_filter in logger.filters):
        return

    class _TransformersPathAliasFilter(logging.Filter):
        _manufacture_path_alias_filter = True

        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "Accessing `__path__`" in message and "alias will be removed in future versions" in message:
                return False
            return True

    logger.addFilter(_TransformersPathAliasFilter())


def configure_page(page_title: str) -> None:
    st.set_page_config(
        page_title=page_title,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_toml_file(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = default or {}
    if not path.exists():
        return fallback
    try:
        return toml.loads(path.read_text(encoding="utf-8"))
    except (toml.TomlDecodeError, OSError):
        return fallback


def _get_supabase_secret_settings() -> dict[str, str]:
    secrets_config = _read_toml_file(SECRETS_PATH, {})
    connection_config = secrets_config.get("connections", {}).get("supabase", {})
    return {
        "url": str(connection_config.get("SUPABASE_URL", "") or "").strip(),
        "key": str(connection_config.get("SUPABASE_KEY", "") or "").strip(),
    }


def _write_supabase_secret_settings(url: str, key: str) -> None:
    secrets_config = _read_toml_file(SECRETS_PATH, {})
    connections = secrets_config.setdefault("connections", {})
    supabase = connections.setdefault("supabase", {})
    supabase["SUPABASE_URL"] = url.strip()
    supabase["SUPABASE_KEY"] = key.strip()
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_PATH.write_text(toml.dumps(secrets_config), encoding="utf-8")


def _mask_secret_value(value: str, *, keep_start: int = 6, keep_end: int = 4) -> str:
    if not value:
        return "-"
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return f"{value[:keep_start]}{'*' * 8}{value[-keep_end:]}"


def _parse_record_timestamp(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value

    raw_value = str(value).strip() if value is not None else ""
    if not raw_value:
        return fallback

    normalized_value = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized_value)
    except ValueError:
        return fallback

    if parsed.tzinfo is not None:
        return parsed.astimezone().replace(tzinfo=None)
    return parsed


def _normalize_db_label(value: Any, fallback: str) -> str:
    raw_value = str(value).strip() if value is not None else ""
    return raw_value or fallback


def _fetch_supabase_semiconductor_rows() -> list[dict[str, Any]]:
    if SupabaseConnection is None:
        raise RuntimeError("st_supabase_connection 패키지를 찾지 못했습니다.")

    connection = st.connection(SUPABASE_CONNECTION_NAME, type=SupabaseConnection)
    last_error: Exception | None = None

    try:
        query_builder = connection.query(
            SUPABASE_IMAGE_COLUMNS,
            table=SUPABASE_IMAGE_TABLE,
            ttl=SUPABASE_QUERY_TTL,
        )
        if hasattr(query_builder, "eq"):
            query_builder = query_builder.eq("trained", False)
        if hasattr(query_builder, "order"):
            query_builder = query_builder.order("created_at", desc=False)
        result = query_builder.execute()
    except Exception as exc:
        last_error = exc
        client = getattr(connection, "client", None)
        if client is None:
            raise

        query_builder = client.table(SUPABASE_IMAGE_TABLE).select(SUPABASE_IMAGE_COLUMNS).eq("trained", False)
        if hasattr(query_builder, "order"):
            query_builder = query_builder.order("created_at", desc=False)
        result = query_builder.execute()

    rows = getattr(result, "data", result)
    if not rows:
        return []

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
            continue
        if hasattr(row, "items"):
            normalized_rows.append(dict(row.items()))
            continue
        normalized_rows.append(dict(row))

    if last_error is not None:
        print(f"Supabase query fallback applied after primary query failed: {last_error}")
    return normalized_rows


def _get_supabase_client() -> Any:
    if SupabaseConnection is None:
        raise RuntimeError("st_supabase_connection 패키지를 찾지 못했습니다.")

    connection = st.connection(SUPABASE_CONNECTION_NAME, type=SupabaseConnection)
    client = getattr(connection, "client", None)
    if client is None:
        raise RuntimeError("Supabase client를 사용할 수 없습니다.")
    return client


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _resolve_training_label_for_record(
    record: dict[str, Any],
    plan: DetailFineTunePlan | None,
    *,
    use_record_labels: bool,
) -> str:
    if use_record_labels:
        label = str(record.get("assigned_label") or record.get("label") or "").strip()
        if label:
            return label

    if plan is not None:
        plan_label = str(plan.new_class_name or plan.target_label or "").strip()
        if plan_label:
            return plan_label

    return str(record.get("assigned_label") or record.get("label") or "").strip()


def _sync_fine_tuning_records_to_supabase(
    selected_records: list[dict[str, Any]],
    plan: DetailFineTunePlan | None,
    *,
    use_record_labels: bool,
) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str]] = set()

    for record in selected_records:
        assigned_label = _resolve_training_label_for_record(record, plan, use_record_labels=use_record_labels)
        if not assigned_label:
            continue

        record_id = record.get("record_id")
        normalized_path = _normalize_image_path_key(record.get("path"))
        if _is_missing_scalar(record_id) and not normalized_path:
            continue

        if _is_missing_scalar(record_id):
            target_key = ("image_path", normalized_path)
            target_value = normalized_path
        else:
            target_key = ("id", str(record_id))
            target_value = record_id

        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        updates.append(
            {
                "target_column": target_key[0],
                "target_value": target_value,
                "image_path": normalized_path,
                "filename": str(record.get("filename") or Path(normalized_path).name),
                "assigned_label": assigned_label,
            }
        )

    summary = {
        "attempted_count": len(updates),
        "updated_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "updated_records": [],
        "skipped_records": [],
        "errors": [],
    }
    if not updates:
        return summary

    client = _get_supabase_client()
    for update in updates:
        payload = {
            "trained": True,
            "class": update["assigned_label"],
        }
        try:
            query_builder = client.table(SUPABASE_IMAGE_TABLE).update(payload)
            query_builder = query_builder.eq(update["target_column"], update["target_value"])
            result = query_builder.execute()
            returned_rows = getattr(result, "data", None)
            if isinstance(returned_rows, list) and len(returned_rows) == 0:
                summary["skipped_count"] += 1
                summary["skipped_records"].append(
                    {
                        "filename": update["filename"],
                        "label": update["assigned_label"],
                        "reason": "no matching semiconductor row",
                    }
                )
                continue

            summary["updated_count"] += 1
            summary["updated_records"].append(
                {
                    "filename": update["filename"],
                    "label": update["assigned_label"],
                    "image_path": update["image_path"],
                }
            )
        except Exception as exc:
            summary["error_count"] += 1
            summary["errors"].append(f"{update['filename']}: {exc}")

    if summary["updated_count"] > 0:
        load_dashboard_data.clear()
    return summary


def _load_supabase_image_candidates(reference_time: datetime) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = _fetch_supabase_semiconductor_rows()
    discovered_images: list[dict[str, Any]] = []
    warning_logs: list[dict[str, str]] = []
    skipped_missing_paths = 0
    skipped_invalid_paths = 0

    for row in rows:
        raw_image_path = str(row.get("image_path") or "").strip()
        if not raw_image_path:
            skipped_invalid_paths += 1
            continue

        image_path = Path(raw_image_path).expanduser()
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            skipped_invalid_paths += 1
            continue
        if not image_path.exists() or not image_path.is_file():
            skipped_missing_paths += 1
            continue

        source_label = _normalize_db_label(row.get("class"), image_path.parent.name)
        database_predict = str(row.get("predict") or "").strip() or None
        created_at = _parse_record_timestamp(row.get("created_at"), reference_time)
        discovered_images.append(
            {
                "record_id": row.get("id"),
                "path": image_path,
                "source_label": source_label,
                "database_predict": database_predict,
                "timestamp": created_at,
                "dataset_type": str(row.get("type") or "").strip() or None,
                "trained": bool(row.get("trained", False)),
            }
        )

    if skipped_missing_paths:
        warning_logs.append(
            _build_log_entry(
                log_type="Warning",
                source="Supabase",
                content=(
                    f"semiconductor 테이블에서 읽은 이미지 중 {skipped_missing_paths}건은 "
                    "로컬 절대경로 파일이 없어 제외했습니다."
                ),
                timestamp=reference_time,
            )
        )
    if skipped_invalid_paths:
        warning_logs.append(
            _build_log_entry(
                log_type="Warning",
                source="Supabase",
                content=(
                    f"semiconductor 테이블에서 읽은 이미지 중 {skipped_invalid_paths}건은 "
                    "image_path 형식이 올바르지 않아 제외했습니다."
                ),
                timestamp=reference_time,
            )
        )

    discovered_images.sort(key=lambda item: (item["timestamp"], str(item["path"])))
    return discovered_images, warning_logs


def _load_local_image_candidates(config: dict[str, Any], reference_time: datetime) -> list[dict[str, Any]]:
    discovered_images: list[dict[str, Any]] = []
    base_dir = Path.cwd()
    data_root = (base_dir / config.get("data_root_path", "")).resolve()
    test_path = config.get("test_data_path", "")
    test_dir = (data_root / test_path).resolve()

    if not test_dir.exists():
        return discovered_images

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        source_label = class_dir.name
        for image_path in class_dir.glob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            discovered_images.append(
                {
                    "record_id": None,
                    "path": image_path,
                    "source_label": source_label,
                    "database_predict": None,
                    "timestamp": reference_time,
                    "dataset_type": "test",
                    "trained": False,
                }
            )

    discovered_images.sort(key=lambda item: (item["source_label"], str(item["path"])))
    return discovered_images


def _get_available_llm_model_dirs() -> list[Path]:
    available_model_dirs = list_available_model_dirs()
    if available_model_dirs:
        return available_model_dirs
    if DEFAULT_LLM_MODEL_DIR.exists():
        return [DEFAULT_LLM_MODEL_DIR]
    return []


def _initialize_llm_runtime_state() -> dict[str, Any]:
    available_model_dirs = _get_available_llm_model_dirs()
    available_values = [str(path) for path in available_model_dirs]
    default_model_value = str(available_model_dirs[0] if available_model_dirs else DEFAULT_LLM_MODEL_DIR)

    selected_model_value = st.session_state.get("llm_model_dir", default_model_value)
    if available_values and selected_model_value not in available_values:
        selected_model_value = default_model_value
    elif not available_values:
        selected_model_value = default_model_value

    temperature = float(st.session_state.get("llm_temperature", DEFAULT_LLM_TEMPERATURE))
    max_new_tokens = int(st.session_state.get("llm_max_new_tokens", DEFAULT_LLM_MAX_NEW_TOKENS))
    st.session_state["llm_model_dir"] = selected_model_value
    st.session_state["llm_temperature"] = max(0.0, temperature)
    st.session_state["llm_max_new_tokens"] = max(1, max_new_tokens)

    if "llm_model_dir_pending" not in st.session_state:
        st.session_state["llm_model_dir_pending"] = selected_model_value
    elif available_values and st.session_state["llm_model_dir_pending"] not in available_values:
        st.session_state["llm_model_dir_pending"] = selected_model_value

    if "llm_temperature_pending" not in st.session_state:
        st.session_state["llm_temperature_pending"] = st.session_state["llm_temperature"]
    if "llm_max_new_tokens_pending" not in st.session_state:
        st.session_state["llm_max_new_tokens_pending"] = st.session_state["llm_max_new_tokens"]

    return {
        "available_model_dirs": available_model_dirs,
        "default_model_value": default_model_value,
    }


def _get_llm_runtime_settings() -> dict[str, Any]:
    runtime_state = _initialize_llm_runtime_state()
    available_model_dirs = runtime_state["available_model_dirs"]
    return {
        "model_dir": st.session_state["llm_model_dir"],
        "temperature": max(0.0, float(st.session_state["llm_temperature"])),
        "max_new_tokens": max(1, int(st.session_state["llm_max_new_tokens"])),
        "available_model_dirs": available_model_dirs,
    }


def _get_pending_llm_runtime_settings() -> dict[str, Any]:
    runtime_state = _initialize_llm_runtime_state()
    available_model_dirs = runtime_state["available_model_dirs"]
    return {
        "model_dir": st.session_state["llm_model_dir_pending"],
        "temperature": max(0.0, float(st.session_state["llm_temperature_pending"])),
        "max_new_tokens": max(1, int(st.session_state["llm_max_new_tokens_pending"])),
        "available_model_dirs": available_model_dirs,
    }


def _get_discrete_class_colors(labels: list[str]) -> dict[str, str]:
    label_set = set(labels)
    ordered_labels = [label for label in CLASS_VISUALIZATION_ORDER if label in label_set]
    ordered_labels.extend(sorted(label_set - set(ordered_labels)))

    fallback_colors = (
        "#FF2D55",
        "#FF9F0A",
        "#64D2FF",
        "#30D158",
        "#5E5CE6",
        "#BF5AF2",
        "#FF375F",
        "#66D4CF",
    )
    color_map: dict[str, str] = {}
    fallback_index = 0
    for label in ordered_labels:
        if label in CLASS_VISUALIZATION_COLORS:
            color_map[label] = CLASS_VISUALIZATION_COLORS[label]
        else:
            color_map[label] = fallback_colors[fallback_index % len(fallback_colors)]
            fallback_index += 1
    return color_map


def parse_timestamp(folder_name: str) -> datetime:
    return datetime.strptime(folder_name, "%Y%m%d_%H%M%S_%f")


def parse_timing_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not path.exists():
        return metrics

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key.startswith("per_image_"):
            continue
        if key == "model_path":
            metrics[key] = value
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def _is_valid_classifier_model_dir(model_dir: Path) -> bool:
    return model_dir.is_dir() and all((model_dir / filename).exists() for filename in REQUIRED_CLASSIFIER_MODEL_FILES)


def _list_available_classifier_model_dirs() -> list[Path]:
    if not CLASSIFICATION_MODEL_ROOT.exists():
        return []
    return sorted(
        (path for path in CLASSIFICATION_MODEL_ROOT.iterdir() if _is_valid_classifier_model_dir(path)),
        key=lambda path: path.name,
    )


def _get_default_detail_inference_model_dir(selected_records: list[dict[str, Any]]) -> Path:
    available_model_dirs = _list_available_classifier_model_dirs()
    preferred_candidates: list[Path] = [
        resolve_base_model_dir(record.get("model_dir"))
        for record in selected_records
    ]

    config = read_json_file(CONFIG_PATH, {})
    configured_model_name = config.get("model_name")
    if configured_model_name:
        preferred_candidates.append(resolve_base_model_dir(configured_model_name))

    preferred_candidates.append(resolve_base_model_dir(CLASSIFIER_MODEL_DIR))

    for candidate in preferred_candidates:
        if _is_valid_classifier_model_dir(candidate):
            return candidate

    if available_model_dirs:
        return available_model_dirs[0]
    return resolve_base_model_dir(CLASSIFIER_MODEL_DIR)


def _render_classifier_model_selector(
    selected_records: list[dict[str, Any]],
    container: Any,
    selector_key: str,
    active_key: str,
    section_title: str | None = None,
    helper_text: str | None = None,
    label: str = "Inference model",
    add_divider: bool = False,
) -> tuple[Path | None, bool]:
    available_model_dirs = _list_available_classifier_model_dirs()
    if add_divider:
        container.divider()
    if section_title:
        container.subheader(section_title)
    if helper_text:
        container.caption(helper_text)

    if not available_model_dirs:
        container.warning(f"분류 모델 폴더를 찾을 수 없습니다: {_to_project_relative_path(CLASSIFICATION_MODEL_ROOT)}")
        return None, False

    default_model_dir = _get_default_detail_inference_model_dir(selected_records)
    available_values = [str(path) for path in available_model_dirs]
    if st.session_state.get(selector_key) not in available_values:
        st.session_state[selector_key] = str(default_model_dir)

    selected_value = container.selectbox(
        label,
        available_values,
        format_func=lambda value: Path(value).name,
        key=selector_key,
    )
    container.caption(f"Model path: {_to_project_relative_path(selected_value)}")

    previous_value = st.session_state.get(active_key)
    model_changed = previous_value is not None and previous_value != selected_value
    st.session_state[active_key] = selected_value
    return Path(selected_value), model_changed


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


@st.cache_data(show_spinner=False)
def _build_thumbnail_data_uri(image_path: str, size: tuple[int, int] = (96, 96)) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    try:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            rgb_image.thumbnail(size)
            buffer = BytesIO()
            rgb_image.save(buffer, format="PNG")
    except Exception:
        return None

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_unique_image_pool(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in image_records:
        path = record["path"]
        if path in seen_paths:
            continue
        unique_records.append(dict(record))
        seen_paths.add(path)
    return unique_records


def _resolve_inference_output_targets(output_path: Path, inference_mode: str) -> tuple[Path, Path, Path]:
    base_output_dir = output_path.parent if output_path.suffix else output_path
    if base_output_dir.exists() and not base_output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {base_output_dir}")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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


def _get_detail_inference_signature(
    selected_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[str, tuple[str, ...]]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    return (
        str(resolved_model_dir),
        tuple(record["path"] for record in selected_records),
    )


def _get_cached_detail_inference_result(
    selected_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[list[dict[str, Any]] | None, list[str], dict[str, str] | None]:
    signature = _get_detail_inference_signature(selected_records, model_dir)
    if st.session_state.get("detail_inference_prediction_signature") != signature:
        return None, [], None

    artifact_paths: dict[str, str] | None = None
    output_dir = st.session_state.get("detail_inference_prediction_output_dir")
    results_path = st.session_state.get("detail_inference_prediction_results_path")
    timing_path = st.session_state.get("detail_inference_prediction_timing_path")
    if output_dir and results_path and timing_path:
        artifact_paths = {
            "output_dir": str(output_dir),
            "results_path": str(results_path),
            "timing_path": str(timing_path),
        }

    return (
        list(st.session_state.get("detail_inference_prediction_records", [])),
        list(st.session_state.get("detail_inference_prediction_errors", [])),
        artifact_paths,
    )


@st.cache_resource(show_spinner=False)
def _load_dashboard_classifier_runtime(model_dir_value: str) -> tuple[Any, Any, str]:
    try:
        import torch

        _suppress_transformers_path_alias_warning()
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError("대시보드 기본 추론에 필요한 torch/transformers 패키지가 없습니다.") from exc

    resolved_model_dir = resolve_base_model_dir(model_dir_value)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(str(resolved_model_dir))
    model = AutoModelForImageClassification.from_pretrained(str(resolved_model_dir)).to(device)
    model.eval()
    return image_processor, model, device


@st.cache_data(show_spinner=False)
def _predict_dashboard_labels(
    image_paths: tuple[str, ...],
    model_dir_value: str,
) -> tuple[dict[str, str], float, float]:
    if not image_paths:
        return {}, 0.0, 0.0

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("대시보드 기본 추론에 필요한 torch/Pillow 패키지가 없습니다.") from exc

    image_processor, model, device_name = _load_dashboard_classifier_runtime(model_dir_value)
    device = torch.device(device_name)

    def _sync_if_needed() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    predicted_labels: dict[str, str] = {}
    total_process_ms = 0.0
    total_inference_ms = 0.0

    with torch.no_grad():
        for image_path in image_paths:
            process_start_ns = time.perf_counter_ns()

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")

            inputs = image_processor(images=rgb_image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            _sync_if_needed()
            inference_start_ns = time.perf_counter_ns()
            logits = model(**inputs).logits
            predicted_index = logits.argmax(dim=-1).item()
            _sync_if_needed()

            inference_ms = (time.perf_counter_ns() - inference_start_ns) / 1_000_000.0
            process_ms = (time.perf_counter_ns() - process_start_ns) / 1_000_000.0
            predicted_labels[image_path] = str(model.config.id2label[predicted_index])
            total_inference_ms += inference_ms
            total_process_ms += process_ms

    average_inference_ms = total_inference_ms / len(image_paths) if image_paths else 0.0
    return predicted_labels, average_inference_ms, total_process_ms


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
    signature = _get_detail_inference_signature(selected_records, resolved_model_dir)

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
    if not resolved_dirs:
        resolved_dirs.append(CLASSIFIER_MODEL_DIR)
    return resolved_dirs


def _get_detail_base_model_dir(selected_records: list[dict[str, Any]]) -> Path:
    return _resolve_selected_model_dirs(selected_records)[0]


def build_log_entries(runs: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, str]]:
    if not runs:
        return _sort_log_entries(
            [
                _build_log_entry(
                    log_type="Warning",
                    source="System",
                    content="No inference results found. Add a run under 05_Manufacutre/output.",
                )
            ]
        )

    latest_run = runs[-1]
    first_run = runs[0]
    label_counts = latest_run["label_counts"]
    top_issue = next(
        ((label, count) for label, count in label_counts.items() if label != "Normal"),
        ("Bad", latest_run["bad_count"]),
    )
    defect_rate = (latest_run["bad_count"] / latest_run["total_count"]) if latest_run["total_count"] else 0.0

    logs: list[dict[str, str]] = []
    if latest_run["bad_count"] > 0:
        logs.append(
            _build_log_entry(
                log_type="Emergency" if defect_rate >= 0.3 else "Warning",
                source="Inference",
                content=f"{top_issue[0]} increasing: {top_issue[1]} images flagged in the latest batch.",
                timestamp=latest_run["timestamp"],
            )
        )

    if latest_run["average_inference_ms"] >= 8:
        logs.append(
            _build_log_entry(
                log_type="Warning",
                source="Inference",
                content=f"Inference latency rose to {latest_run['average_inference_ms']:.2f} ms per image.",
                timestamp=latest_run["timestamp"],
            )
        )

    logs.extend(
        [
            _build_log_entry(
                log_type="System done",
                source="System",
                content=f"Latest inspection batch finished with {latest_run['total_count']} products.",
                timestamp=latest_run["timestamp"],
            ),
            _build_log_entry(
                log_type="System start",
                source="System",
                content=f"Monitoring started with run {first_run['name']}.",
                timestamp=first_run["timestamp"],
            ),
            _build_log_entry(
                log_type="Model update",
                source="System",
                content=f"Current model configuration: {_format_display_path(config.get('model_name', 'unknown model'))}.",
                timestamp=latest_run["timestamp"],
            ),
        ]
    )

    return _sort_log_entries(logs)



@st.cache_data(show_spinner=False, ttl=600)
def load_dashboard_data() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    config = read_json_file(CONFIG_PATH, {})

    runs: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    now = datetime.now()
    default_model_dir = resolve_base_model_dir(CLASSIFIER_MODEL_DIR)
    data_source = "supabase"
    source_warning_logs: list[dict[str, str]] = []
    try:
        discovered_images, source_warning_logs = _load_supabase_image_candidates(now)
    except Exception as exc:
        data_source = "local_filesystem_fallback"
        discovered_images = _load_local_image_candidates(config, now)
        source_warning_logs = [
            _build_log_entry(
                log_type="Warning",
                source="Supabase",
                content=(
                    "Supabase semiconductor 연동에 실패해 로컬 test 디렉터리로 대체했습니다. "
                    f"error={exc}"
                ),
                timestamp=now,
            )
        ]

    config = {
        **config,
        "data_source": data_source,
        "supabase_table": SUPABASE_IMAGE_TABLE,
        "supabase_filter": "trained = false",
    }
    prediction_warning: dict[str, str] | None = None
    predicted_labels_by_path: dict[str, str] = {}
    average_inference_ms = 0.0
    total_process_ms = 0.0

    if discovered_images:
        image_paths = tuple(str(image_record["path"]) for image_record in discovered_images)
        try:
            predicted_labels_by_path, average_inference_ms, total_process_ms = _predict_dashboard_labels(
                image_paths,
                str(default_model_dir),
            )
        except Exception as exc:
            prediction_warning = _build_log_entry(
                log_type="Warning",
                source="Inference",
                content=(
                    "기본 분류 모델 추론에 실패해 저장된 라벨 값으로 대체했습니다. "
                    f"model={_to_project_relative_path(default_model_dir)} | error={exc}"
                ),
                timestamp=now,
            )

    predicted_run_counts: Counter[str] = Counter()
    predicted_good_counts: Counter[str] = Counter()
    predicted_bad_counts: Counter[str] = Counter()
    predicted_latest_timestamps: dict[str, datetime] = {}

    for candidate in discovered_images:
        image_path = Path(candidate["path"])
        source_label = str(candidate["source_label"])
        database_predict = str(candidate.get("database_predict") or "").strip()
        predicted_label = predicted_labels_by_path.get(str(image_path))
        prediction_source = "default_model"
        if not predicted_label:
            if database_predict:
                predicted_label = database_predict
                prediction_source = "supabase_predict_fallback"
            else:
                predicted_label = source_label
                prediction_source = "source_label_fallback"

        display_path = str(image_path)
        record_timestamp = candidate["timestamp"]

        image_records.append(
            {
                "run_name": predicted_label,
                "timestamp": record_timestamp,
                "date": record_timestamp.strftime("%Y-%m-%d"),
                "label": predicted_label,
                "source_label": source_label,
                "prediction_source": prediction_source,
                "path": str(image_path),
                "display_path": display_path,
                "filename": image_path.name,
                "record_id": candidate.get("record_id"),
                "trained": bool(candidate.get("trained", False)),
                "dataset_type": candidate.get("dataset_type"),
                "database_predict": database_predict or None,
                "data_source": data_source,
                "model_dir": str(default_model_dir),
                "model_dir_display": _to_project_relative_path(default_model_dir),
                "exists": True,
            }
        )

        predicted_run_counts[predicted_label] += 1
        previous_timestamp = predicted_latest_timestamps.get(predicted_label)
        if previous_timestamp is None or record_timestamp > previous_timestamp:
            predicted_latest_timestamps[predicted_label] = record_timestamp
        if predicted_label == "Normal":
            predicted_good_counts[predicted_label] += 1
        else:
            predicted_bad_counts[predicted_label] += 1

    per_image_process_ms = (total_process_ms / len(discovered_images)) if discovered_images else 0.0
    for predicted_label in sorted(predicted_run_counts):
        image_count = int(predicted_run_counts[predicted_label])
        runs.append(
            {
                "name": predicted_label,
                "timestamp": predicted_latest_timestamps.get(predicted_label, now),
                "path": "",
                "total_count": image_count,
                "label_counts": {predicted_label: image_count},
                "good_count": int(predicted_good_counts.get(predicted_label, 0)),
                "bad_count": int(predicted_bad_counts.get(predicted_label, 0)),
                "model_dir": str(default_model_dir),
                "average_inference_ms": float(average_inference_ms),
                "total_process_ms": float(per_image_process_ms * image_count),
            }
        )

    runs.sort(key=lambda x: x["name"])
    image_records.sort(key=lambda x: (x["run_name"], x["timestamp"], x["filename"]))

    base_logs = [*_load_app_logs()]
    base_logs.extend(source_warning_logs)
    if prediction_warning:
        base_logs.append(prediction_warning)
    logs = _sort_log_entries(base_logs)
    print(
        "Loaded "
        f"{len(runs)} predicted groups and {len(image_records)} images from {data_source} "
        f"using {default_model_dir.name}."
    )
   
    return config, runs, image_records, logs

def build_report_csv(latest_run: dict[str, Any] | None, runs: list[dict[str, Any]]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Overview"])
    writer.writerow(["metric", "value"])

    if latest_run:
        defect_rate = (
            (latest_run["bad_count"] / latest_run["total_count"]) * 100
            if latest_run["total_count"]
            else 0.0
        )
        writer.writerow(["latest_run", latest_run["timestamp"].strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["total_product", latest_run["total_count"]])
        writer.writerow(["good", latest_run["good_count"]])
        writer.writerow(["bad", latest_run["bad_count"]])
        writer.writerow(["defect_rate_percent", f"{defect_rate:.2f}"])
        writer.writerow(["avg_inference_ms", f"{latest_run['average_inference_ms']:.3f}"])
        writer.writerow(["total_process_ms", f"{latest_run['total_process_ms']:.3f}"])
    else:
        writer.writerow(["latest_run", "No data"])

    writer.writerow([])
    writer.writerow(["Class Distribution"])
    writer.writerow(["label", "count"])
    if latest_run:
        for label, count in latest_run["label_counts"].items():
            writer.writerow([label, count])

    writer.writerow([])
    writer.writerow(["Recent Runs"])
    writer.writerow(["timestamp", "image_count", "good", "bad", "avg_inference_ms", "total_process_ms"])
    for run in reversed(runs[-10:]):
        writer.writerow(
            [
                run["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                run["total_count"],
                run["good_count"],
                run["bad_count"],
                f"{run['average_inference_ms']:.3f}",
                f"{run['total_process_ms']:.3f}",
            ]
        )

    return buffer.getvalue()


def build_overview_frame(latest_run: dict[str, Any] | None) -> pd.DataFrame:
    if not latest_run:
        return pd.DataFrame(
            [
                {"Metric": "Total Product", "Value": 0},
                {"Metric": "Good", "Value": 0},
                {"Metric": "Bad", "Value": 0},
            ]
        )

    return pd.DataFrame(
        [
            {"Metric": "Total Product", "Value": latest_run["total_count"]},
            {"Metric": "Good", "Value": latest_run["good_count"]},
            {"Metric": "Bad", "Value": latest_run["bad_count"]},
        ]
    )


def build_label_distribution_frame(latest_run: dict[str, Any] | None) -> pd.DataFrame:
    if not latest_run:
        return pd.DataFrame({"label": ["No data"], "count": [0]}).set_index("label")

    label_counts = latest_run["label_counts"]
    if not label_counts:
        return pd.DataFrame({"label": ["No data"], "count": [0]}).set_index("label")

    return pd.DataFrame(
        [{"label": label, "count": count} for label, count in label_counts.items()]
    ).set_index("label")


def render_class_distribution_chart(label_frame: pd.DataFrame, container: Any | None = None) -> None:
    target = container or st
    if label_frame.empty:
        target.info("No class distribution data available.")
        return

    labels = [str(index) for index in label_frame.index]
    counts = [int(row["count"]) for _, row in label_frame.iterrows()]
    if labels == ["No data"]:
        target.info("No class distribution data available.")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        target.bar_chart(label_frame, width="stretch")
        return

    color_map = _get_discrete_class_colors(labels)
    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=[color_map.get(label, "#5B8FF9") for label in labels],
                text=counts,
                textposition="outside",
                hovertemplate="Class: %{x}<br>Count: %{y}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    figure.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    target.plotly_chart(figure, width="stretch")


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
    latest_timestamp: str,
    overview_rows: tuple[tuple[str, int], ...],
    label_rows: tuple[tuple[str, int], ...],
    recent_runs_rows: tuple[tuple[str, int, int, int, float, float], ...],
    trend_rows: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
    analysis_comment: str,
    sample_images: tuple[tuple[str, str, str], ...],
) -> bytes:
    plt, PdfPages, FontProperties, Image = _prepare_pdf_runtime()
    font_prop = _get_pdf_font(FontProperties)
    pdf_buffer = BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle(
            "Manufacturing Summary Report",
            fontsize=20,
            fontproperties=font_prop,
            y=0.97,
        )
        fig.text(
            0.08,
            0.93,
            f"Latest run: {latest_timestamp}",
            fontsize=11,
            fontproperties=font_prop,
        )

        overview_ax = fig.add_axes([0.08, 0.68, 0.84, 0.18])
        overview_ax.axis("off")
        overview_ax.set_title("Overview", loc="left", fontsize=14, fontproperties=font_prop)
        overview_table = overview_ax.table(
            cellText=[[metric, value] for metric, value in overview_rows],
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
        )
        overview_table.auto_set_font_size(False)
        overview_table.set_fontsize(11)
        overview_table.scale(1, 1.6)
        if font_prop is not None:
            for cell in overview_table.get_celld().values():
                cell.get_text().set_fontproperties(font_prop)

        label_ax = fig.add_axes([0.08, 0.40, 0.84, 0.20])
        label_ax.set_title("Class Distribution", loc="left", fontsize=14, fontproperties=font_prop)
        labels = [row[0] for row in label_rows]
        counts = [row[1] for row in label_rows]
        label_colors = [_get_discrete_class_colors(labels).get(label, "#5B8FF9") for label in labels]
        label_ax.bar(labels, counts, color=label_colors)
        label_ax.set_ylabel("Count")
        _apply_font_to_axis(label_ax, font_prop)

        runs_ax = fig.add_axes([0.08, 0.08, 0.84, 0.24])
        runs_ax.axis("off")
        runs_ax.set_title("Recent Runs", loc="left", fontsize=14, fontproperties=font_prop)
        runs_table = runs_ax.table(
            cellText=[
                [timestamp, total, good, bad, f"{avg_ms:.2f}", f"{total_ms:.2f}"]
                for name, timestamp, total, good, bad, avg_ms, total_ms in recent_runs_rows
            ],
            colLabels=["Name", "Timestamp", "Total", "Good", "Bad", "Avg(ms)", "Total(ms)"],
            cellLoc="center",
            loc="center",
        )
        runs_table.auto_set_font_size(False)
        runs_table.set_fontsize(8.5)
        runs_table.scale(1, 1.4)
        if font_prop is not None:
            for cell in runs_table.get_celld().values():
                cell.get_text().set_fontproperties(font_prop)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
        fig.suptitle("Trend Graphs", fontsize=18, fontproperties=font_prop, y=0.98)
        for axis, (title, points) in zip(axes, trend_rows):
            x_values = [item[0] for item in points]
            y_values = [item[1] for item in points]
            axis.plot(x_values, y_values, marker="o", color="#00A6A6", linewidth=2)
            axis.set_title(title, fontsize=13, fontproperties=font_prop)
            axis.set_ylabel("Value")
            axis.grid(alpha=0.25)
            _apply_font_to_axis(axis, font_prop)
        axes[-1].set_xlabel("Period")
        _apply_font_to_axis(axes[-1], font_prop)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        images_per_page = 4
        image_chunks = [
            sample_images[index:index + images_per_page]
            for index in range(0, len(sample_images), images_per_page)
        ] or [tuple()]

        for page_index, image_chunk in enumerate(image_chunks, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(
                f"Inference Images ({page_index}/{len(image_chunks)})",
                fontsize=18,
                fontproperties=font_prop,
                y=0.98,
            )
            grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

            for idx in range(images_per_page):
                axis = fig.add_subplot(grid[idx // 2, idx % 2])
                axis.axis("off")
                if idx < len(image_chunk):
                    path, label, filename = image_chunk[idx]
                    try:
                        image = Image.open(path).convert("RGB")
                        axis.imshow(image)
                        axis.set_title(
                            f"{label}\n{filename}",
                            fontsize=9,
                            fontproperties=font_prop,
                        )
                    except Exception:
                        axis.text(
                            0.5,
                            0.5,
                            f"Image load failed\n{filename}",
                            ha="center",
                            va="center",
                            fontproperties=font_prop,
                        )
                else:
                    axis.text(0.5, 0.5, "No image", ha="center", va="center", fontproperties=font_prop)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("AI Analysis Comment", fontsize=18, fontproperties=font_prop, y=0.98)
        text_ax = fig.add_axes([0.08, 0.08, 0.84, 0.82])
        text_ax.axis("off")
        text_ax.set_title("AI Analysis Comment", loc="left", fontsize=14, fontproperties=font_prop)
        text_ax.text(
            0.0,
            0.98,
            _wrap_text_lines(analysis_comment, width=40),
            va="top",
            fontsize=11,
            fontproperties=font_prop,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return pdf_buffer.getvalue()


def build_llm_auto_response(prompt: str, config: dict[str, Any], latest_run: dict[str, Any] | None) -> str:
    if not prompt.strip():
        return ""

    response_lines = [f"User request: {prompt.strip()}"]
    response_lines.append(f"Configured base model: {_format_display_path(config.get('model_name', '-'))}")

    classes = config.get("classes_to_train", [])
    if isinstance(classes, list) and classes:
        response_lines.append("Suggested focus classes: " + ", ".join(str(item) for item in classes[:5]))

    if latest_run:
        total_count = latest_run["total_count"]
        bad_count = latest_run["bad_count"]
        defect_rate = (bad_count / total_count * 100) if total_count else 0.0
        dominant_issue = next(
            ((label, count) for label, count in latest_run["label_counts"].items() if label != "Normal"),
            None,
        )
        response_lines.append(
            f"Latest batch summary: {bad_count} bad out of {total_count} products ({defect_rate:.1f}%)."
        )
        if dominant_issue:
            response_lines.append(
                f"Priority adjustment: increase sampling around {dominant_issue[0]} because it appeared {dominant_issue[1]} times."
            )
        if latest_run["average_inference_ms"] >= 8:
            response_lines.append(
                "Latency note: lighten preprocessing or batch size because inference is trending slower than target."
            )
        else:
            response_lines.append("Latency note: current inference speed is acceptable for iterative tuning.")

    response_lines.append(
        "Recommended next step: validate the updated prompt or threshold on one fresh inspection batch before promoting it."
    )
    return "\n".join(response_lines)


def _summarize_label_counts(label_counts: dict[str, int], limit: int = 6) -> str:
    if not label_counts:
        return "-"

    ordered_items = sorted(label_counts.items(), key=lambda item: (-int(item[1]), item[0]))
    summary = ", ".join(f"{label}={count}" for label, count in ordered_items[:limit])
    if len(ordered_items) > limit:
        summary += ", ..."
    return summary


def _collect_records_for_paths(
    image_records: list[dict[str, Any]],
    selected_paths: list[str],
) -> list[dict[str, Any]]:
    if not selected_paths:
        return []

    selected_path_set = set(selected_paths)
    ordered_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in image_records:
        path = str(record.get("path", ""))
        if path in selected_path_set and path not in seen_paths:
            ordered_records.append(record)
            seen_paths.add(path)
    return ordered_records


def _build_sidebar_runtime_context(current_page_title: str | None = None) -> str:
    config, runs, image_records, log_entries = load_dashboard_data()
    summary_run = build_aggregate_run(runs)
    lines: list[str] = []

    lines.append(f"current_page: {current_page_title or st.session_state.get('current_page_title', '-')}")
    lines.append(f"llm_model: {_to_project_relative_path(st.session_state.get('llm_model_dir'))}")
    lines.append(
        "llm_runtime: "
        f"temperature={float(st.session_state.get('llm_temperature', DEFAULT_LLM_TEMPERATURE)):.1f}, "
        f"max_new_tokens={int(st.session_state.get('llm_max_new_tokens', DEFAULT_LLM_MAX_NEW_TOKENS))}"
    )
    lines.append(f"configured_classifier_model: {_format_display_path(config.get('model_name', '-'))}")
    lines.append(f"data_root_path: {_format_display_path(config.get('data_root_path', '-'))}")
    lines.append(f"test_data_path: {_format_display_path(config.get('test_data_path', '-'))}")

    if summary_run:
        lines.append(
            "dashboard_summary: "
            f"total={int(summary_run['total_count'])}, "
            f"good={int(summary_run['good_count'])}, "
            f"bad={int(summary_run['bad_count'])}, "
            f"avg_inference_ms={float(summary_run['average_inference_ms']):.2f}"
        )
        lines.append(f"class_distribution: {_summarize_label_counts(summary_run['label_counts'])}")
    else:
        lines.append("dashboard_summary: no_data")

    recent_runs = runs[-5:]
    if recent_runs:
        recent_run_summary = "; ".join(
            f"{run['name']}({int(run['total_count'])})"
            for run in recent_runs
        )
        lines.append(f"recent_runs: {recent_run_summary}")

    recent_logs = log_entries[:5]
    if recent_logs:
        log_summary = " | ".join(
            f"{entry.get('time', '-')}/{entry.get('source', '-')}/{entry.get('log_type', '-')}: {entry.get('content', '')}"
            for entry in recent_logs
        )
        lines.append(f"recent_logs: {log_summary}")

    detail_selected_paths_raw = st.session_state.get("detail_multi_select_paths")
    if not isinstance(detail_selected_paths_raw, list):
        detail_selected_paths_raw = st.session_state.get("detail_selected_image_paths", [])
    detail_selected_paths = [str(path) for path in detail_selected_paths_raw if str(path).strip()]
    detail_selected_records = _collect_records_for_paths(image_records, detail_selected_paths)
    lines.append(f"detail_selected_image_count: {len(detail_selected_records)}")
    if detail_selected_records:
        detail_selected_names = ", ".join(record["filename"] for record in detail_selected_records[:6])
        lines.append(f"detail_selected_images: {detail_selected_names}")

        active_detail_model = st.session_state.get("detail_inference_model_selector") or st.session_state.get(
            "detail_inference_model_active"
        )
        if active_detail_model:
            lines.append(f"detail_inference_model: {_to_project_relative_path(active_detail_model)}")
            cached_records, cached_errors, cached_artifacts = _get_cached_detail_inference_result(
                detail_selected_records,
                Path(str(active_detail_model)),
            )
            if cached_records is not None:
                prediction_counts = Counter(str(record.get("label", "-")) for record in cached_records)
                lines.append(f"detail_cached_predictions: {_summarize_label_counts(dict(prediction_counts))}")
                if cached_errors:
                    lines.append(f"detail_cached_prediction_errors: {len(cached_errors)}")
                if cached_artifacts:
                    lines.append(
                        f"detail_inference_results_path: {_to_project_relative_path(cached_artifacts['results_path'])}"
                    )
                    lines.append(
                        f"detail_inference_timing_path: {_to_project_relative_path(cached_artifacts['timing_path'])}"
                    )

    fine_tuning_selected_paths = st.session_state.get("fine_tuning_page_selected_image_paths", [])
    if isinstance(fine_tuning_selected_paths, list):
        lines.append(f"fine_tuning_selected_image_count: {len(fine_tuning_selected_paths)}")

    pending_llm_model = st.session_state.get("llm_model_dir_pending")
    if pending_llm_model:
        lines.append(f"pending_llm_model: {_to_project_relative_path(pending_llm_model)}")

    return "\n".join(lines)


def render_sidebar_llm_panel(current_page_title: str | None = None) -> None:
    st.session_state.setdefault("gemma_sidebar_status", "idle")
    st.session_state.setdefault("gemma_sidebar_notice", "")
    st.session_state.setdefault("gemma_sidebar_response", "")
    st.session_state.setdefault("gemma_sidebar_prompt", "")
    llm_settings = _get_llm_runtime_settings()
    selected_model_dir = Path(str(llm_settings["model_dir"]))
    selected_model_name = selected_model_dir.name

    with st.sidebar:
        # The app uses a pre-downloaded local Gemma model from 05_Manufacutre/model/google__gemma-4-E2B-it.
        # The hidden system instruction is defined in DEFAULT_GEMMA_SYSTEM_PROMPT and is not user-editable.
        st.divider()
        st.subheader(selected_model_name)
        st.caption(
            f"Temp {float(llm_settings['temperature']):.1f} | Max tokens {int(llm_settings['max_new_tokens'])}"
        )
        dependency_ready, dependency_message = are_runtime_dependencies_available()
        model_ready = is_model_downloaded(selected_model_dir)

        prompt = st.text_area(
            "Command",
            key="gemma_sidebar_prompt",
            height=140,
            placeholder="Gemma 4 E2B에 전달할 명령이나 질문을 입력하세요.",
        )
        send_clicked = st.button("Send", key="gemma_sidebar_send", type="primary", width="stretch")

        if send_clicked:
            prompt_text = prompt.strip()
            if not prompt_text:
                st.session_state["gemma_sidebar_status"] = "error"
                st.session_state["gemma_sidebar_notice"] = "질문이나 명령을 먼저 입력해 주세요."
                st.session_state["gemma_sidebar_response"] = ""
                _append_app_log(
                    log_type="error",
                    source="Sidebar LLM",
                    content="Sidebar LLM request failed because the command field was empty.",
                    request="",
                    response=st.session_state["gemma_sidebar_notice"],
                )
            elif not dependency_ready:
                st.session_state["gemma_sidebar_status"] = "error"
                st.session_state["gemma_sidebar_notice"] = dependency_message
                st.session_state["gemma_sidebar_response"] = ""
                _append_app_log(
                    log_type="error",
                    source="Sidebar LLM",
                    content="Sidebar LLM request failed because runtime dependencies are unavailable.",
                    request=prompt_text,
                    response=dependency_message,
                )
            elif not model_ready:
                st.session_state["gemma_sidebar_status"] = "error"
                st.session_state["gemma_sidebar_notice"] = "로컬 모델이 아직 없습니다. 지정된 model 폴더에 모델을 먼저 준비해 주세요."
                st.session_state["gemma_sidebar_response"] = ""
                _append_app_log(
                    log_type="error",
                    source="Sidebar LLM",
                    content=f"Sidebar LLM request failed because model `{_to_project_relative_path(selected_model_dir)}` is unavailable.",
                    request=prompt_text,
                    response=st.session_state["gemma_sidebar_notice"],
                )
            else:
                st.session_state["gemma_sidebar_status"] = "running"
                st.session_state["gemma_sidebar_notice"] = f"{selected_model_name}가 응답을 생성하는 중입니다."
                st.session_state["gemma_sidebar_response"] = ""
                _append_app_log(
                    log_type="start",
                    source="Sidebar LLM",
                    content=f"Sidebar LLM request started with model `{_to_project_relative_path(selected_model_dir)}`.",
                    request=prompt_text,
                )

                try:
                    with st.spinner("LLM 동작 중..."):
                        runtime_context = _build_sidebar_runtime_context(current_page_title)
                        augmented_prompt = (
                            "[Current Streamlit Runtime Context]\n"
                            f"{runtime_context}\n\n"
                            "[User Request]\n"
                            f"{prompt_text}\n\n"
                            "위 현재 Streamlit 대시보드 데이터와 세션 상태를 우선 참고해서 한국어로 답하세요. "
                            "데이터에 없는 내용은 추측하지 말고 없다고 말하세요."
                        )
                        answer = generate_response(
                            prompt=augmented_prompt,
                            system_prompt=DEFAULT_GEMMA_SYSTEM_PROMPT,
                            max_new_tokens=int(llm_settings["max_new_tokens"]),
                            temperature=float(llm_settings["temperature"]),
                            model_dir=selected_model_dir,
                        )
                    st.session_state["gemma_sidebar_status"] = "completed"
                    st.session_state["gemma_sidebar_notice"] = "응답 생성이 완료되었습니다."
                    st.session_state["gemma_sidebar_response"] = answer
                    _append_app_log(
                        log_type="done",
                        source="Sidebar LLM",
                        content=f"Sidebar LLM response completed with model `{_to_project_relative_path(selected_model_dir)}`.",
                        request=prompt_text,
                        response=answer,
                    )
                except Exception as exc:
                    st.session_state["gemma_sidebar_status"] = "error"
                    st.session_state["gemma_sidebar_notice"] = "LLM 호출 중 오류가 발생했습니다."
                    st.session_state["gemma_sidebar_response"] = str(exc)
                    _append_app_log(
                        log_type="error",
                        source="Sidebar LLM",
                        content=f"Sidebar LLM request failed with model `{_to_project_relative_path(selected_model_dir)}`.",
                        request=prompt_text,
                        response=str(exc),
                    )
            st.rerun()

        status = st.session_state["gemma_sidebar_status"]
        if status in {"running", "downloading"}:
            st.info(st.session_state["gemma_sidebar_notice"])
        elif status == "completed":
            st.success(st.session_state["gemma_sidebar_notice"])
        elif status == "error":
            st.error(st.session_state["gemma_sidebar_notice"])
        else:
            st.caption(st.session_state["gemma_sidebar_notice"])

        st.text_area(
            "Response box",
            value=st.session_state["gemma_sidebar_response"],
            height=220,
            disabled=True,
            placeholder="여기에 모델 응답이 표시됩니다.",
        )


def render_page_header(title: str, caption: str | None = None) -> None:
    st.session_state["current_page_title"] = title
    render_sidebar_llm_panel(title)
    st.title(title)
    if caption:
        st.caption(caption)


def render_home_page(config: dict[str, Any], runs: list[dict[str, Any]], log_entries: list[dict[str, str]]) -> None:
    summary_run = build_aggregate_run(runs)
    label_frame = build_label_distribution_frame(summary_run)
    render_page_header("Dashboard Home", "Use Streamlit page navigation in the sidebar to move between screens.")

    if summary_run:
        cols = st.columns(4)
        cols[0].metric("Total Product", f"{summary_run['total_count']:,}")
        cols[1].metric("Good", f"{summary_run['good_count']:,}")
        cols[2].metric("Bad", f"{summary_run['bad_count']:,}")
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
                "good": run["good_count"],
                "bad": run["bad_count"],
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

def build_aggregate_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None

    latest_timestamp = max(run["timestamp"] for run in runs)

    total_count = sum(int(run.get("total_count", 0)) for run in runs)
    good_count = sum(int(run.get("good_count", 0)) for run in runs)
    bad_count = sum(int(run.get("bad_count", 0)) for run in runs)

    average_values = [float(run.get("average_inference_ms", 0.0)) for run in runs]
    total_process_values = [float(run.get("total_process_ms", 0.0)) for run in runs]

    label_counts: dict[str, int] = {}
    for run in runs:
        for label, count in run.get("label_counts", {}).items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

    aggregate_run = {
        "name": "All Data",
        "timestamp": latest_timestamp,
        "path": "",
        "total_count": total_count,
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "good_count": good_count,
        "bad_count": bad_count,
        "model_dir": "",
        "average_inference_ms": sum(average_values) / len(average_values) if average_values else 0.0,
        "total_process_ms": sum(total_process_values),
    }
    return aggregate_run

def render_summary_page(
    latest_run: dict[str, Any] | None,
    runs: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
) -> None:
    render_page_header("Summary")

    summary_run = build_aggregate_run(runs)

    trends = build_trend_data(summary_run)
    overview_frame = build_overview_frame(summary_run)
    label_frame = build_label_distribution_frame(summary_run)
    report_sample_records = select_summary_report_image_records(image_records)

    report_pdf_error = ""

    with st.spinner("Summary 리포트와 AI 코멘트를 준비하는 중입니다..."):
        analysis_comment, sample_records = build_summary_analysis_comment(summary_run, image_records, trends)

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

            try:
                report_pdf = build_summary_pdf_bytes(
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
        good = int(overview_frame.loc[overview_frame["Metric"] == "Good", "Value"].iloc[0])
        bad = int(overview_frame.loc[overview_frame["Metric"] == "Bad", "Value"].iloc[0])

        metric_cols = st.columns(3)
        metric_cols[0].metric("Total", f"{total:,}")
        metric_cols[1].metric("Good", f"{good:,}")
        metric_cols[2].metric("Bad", f"{bad:,}")

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


def _reset_interactive_finetune_session(
    selected_paths: list[str],
    selected_paths_key: str,
    state_prefix: str,
) -> None:
    prefix = f"{state_prefix}_"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            st.session_state.pop(key, None)
    st.session_state[selected_paths_key] = selected_paths
    st.session_state[f"{state_prefix}_chat"] = []
    st.session_state[f"{state_prefix}_plan"] = None
    st.session_state[f"{state_prefix}_execution"] = None


def _reset_detail_finetune_session(selected_paths: list[str]) -> None:
    _reset_interactive_finetune_session(
        selected_paths=selected_paths,
        selected_paths_key="detail_selected_image_paths",
        state_prefix="detail_finetune",
    )


def _reset_fine_tuning_page_session(selected_paths: list[str]) -> None:
    _reset_interactive_finetune_session(
        selected_paths=selected_paths,
        selected_paths_key="fine_tuning_page_selected_image_paths",
        state_prefix="fine_tuning_page",
    )


def _build_path_signature(paths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized_paths = [
        normalized_path
        for normalized_path in (_normalize_image_path_key(path) for path in paths)
        if normalized_path
    ]
    return tuple(sorted(normalized_paths))


def _set_fine_tuning_selection_metadata(
    *,
    selected_paths: list[str],
    strategy: str,
    origin: str,
    notice: str = "",
    selection_percentage: int | None = None,
    model_dir: Path | str | None = None,
) -> None:
    st.session_state["fine_tuning_selection_strategy"] = strategy.strip() or "Manual Selection"
    st.session_state["fine_tuning_selection_origin"] = origin.strip() or "manual"
    st.session_state["fine_tuning_selection_notice"] = notice.strip()
    st.session_state["fine_tuning_selection_percentage_value"] = selection_percentage
    st.session_state["fine_tuning_selection_paths_signature"] = _build_path_signature(selected_paths)
    st.session_state["fine_tuning_selection_model_dir"] = (
        str(resolve_base_model_dir(model_dir)) if model_dir is not None else ""
    )


def _get_fine_tuning_selection_metadata(
    selected_records: list[dict[str, Any]],
    base_model_dir: Path | str,
) -> dict[str, Any]:
    selected_paths = [str(record["path"]) for record in selected_records]
    stored_signature = tuple(st.session_state.get("fine_tuning_selection_paths_signature", ()))
    current_signature = _build_path_signature(selected_paths)
    stored_model_dir = str(st.session_state.get("fine_tuning_selection_model_dir", "")).strip()
    current_model_dir = str(resolve_base_model_dir(base_model_dir))
    strategy = str(st.session_state.get("fine_tuning_selection_strategy", "")).strip() or "Manual Selection"
    origin = str(st.session_state.get("fine_tuning_selection_origin", "")).strip() or "manual"
    notice = str(st.session_state.get("fine_tuning_selection_notice", "")).strip()
    selection_percentage = st.session_state.get("fine_tuning_selection_percentage_value")

    manually_adjusted = bool(stored_signature) and stored_signature != current_signature
    model_changed_after_selection = bool(stored_model_dir) and stored_model_dir != current_model_dir
    if manually_adjusted or model_changed_after_selection:
        origin = "manual"
        strategy = "Manual Selection"
        adjustment_notes: list[str] = []
        if manually_adjusted:
            adjustment_notes.append("selected images were adjusted after auto sampling")
        if model_changed_after_selection:
            adjustment_notes.append("inference model changed after auto sampling")
        if adjustment_notes:
            extra_notice = "; ".join(adjustment_notes)
            notice = f"{notice} | {extra_notice}".strip(" |")

    return {
        "selection_strategy": strategy,
        "selection_origin": origin,
        "selection_notice": notice,
        "selection_percentage": selection_percentage,
        "selection_model_dir": _to_project_relative_path(current_model_dir),
        "selected_path_signature": list(current_signature),
    }


def _predict_image_pool_records_with_model(
    image_pool_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    existing_image_paths = tuple(record["path"] for record in image_pool_records if record.get("exists"))
    predicted_labels_by_path: dict[str, str] = {}
    prediction_errors: list[str] = []

    if existing_image_paths:
        predicted_labels_by_path, _, _ = _predict_dashboard_labels(
            existing_image_paths,
            str(resolved_model_dir),
        )

    predicted_records: list[dict[str, Any]] = []
    for record in image_pool_records:
        updated_record = dict(record)
        updated_record["model_dir"] = str(resolved_model_dir)
        updated_record["model_dir_display"] = _to_project_relative_path(resolved_model_dir)

        predicted_label = predicted_labels_by_path.get(record["path"])
        if predicted_label:
            updated_record["label"] = predicted_label
        elif record.get("exists"):
            prediction_errors.append(f"{Path(record['path']).name}: prediction result missing")

        updated_record["predicted_label"] = str(updated_record.get("label", ""))
        predicted_records.append(updated_record)

    return predicted_records, prediction_errors


def _sync_records_with_model_dir(records: list[dict[str, Any]], model_dir: Path | str) -> list[dict[str, Any]]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    synchronized_records: list[dict[str, Any]] = []
    for record in records:
        updated_record = dict(record)
        updated_record["model_dir"] = str(resolved_model_dir)
        updated_record["model_dir_display"] = _to_project_relative_path(resolved_model_dir)
        synchronized_records.append(updated_record)
    return synchronized_records


def _select_margin_sampling_paths_for_fine_tuning(
    image_pool_records: list[dict[str, Any]],
    base_model_dir: Path | str,
    selection_percentage: int,
) -> tuple[list[str], pd.DataFrame]:
    scripts_dir = BASE_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from margin_sampling import select_margin_sampling_paths

    selectable_paths = [record["path"] for record in image_pool_records if record.get("exists")]
    sampled_paths, margin_frame = select_margin_sampling_paths(
        image_paths=selectable_paths,
        base_model_dir=resolve_base_model_dir(base_model_dir),
        selection_percentage=selection_percentage,
    )
    if not sampled_paths:
        return [], margin_frame

    sampled_path_keys = {_normalize_image_path_key(path) for path in sampled_paths}
    selected_paths = [
        record["path"]
        for record in image_pool_records
        if _normalize_image_path_key(record.get("path")) in sampled_path_keys
    ]
    return selected_paths, margin_frame


def _select_boundary_sampling_paths_for_fine_tuning(
    image_pool_records: list[dict[str, Any]],
    base_model_dir: Path | str,
    selection_percentage: int,
) -> tuple[list[str], pd.DataFrame]:
    scripts_dir = BASE_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from boundary_sampling import select_boundary_sampling_paths

    selectable_paths = [record["path"] for record in image_pool_records if record.get("exists")]
    sampled_paths, boundary_frame = select_boundary_sampling_paths(
        image_paths=selectable_paths,
        base_model_dir=resolve_base_model_dir(base_model_dir),
        selection_percentage=selection_percentage,
    )
    if not sampled_paths:
        return [], boundary_frame

    sampled_path_keys = {_normalize_image_path_key(path) for path in sampled_paths}
    selected_paths = [
        record["path"]
        for record in image_pool_records
        if _normalize_image_path_key(record.get("path")) in sampled_path_keys
    ]
    return selected_paths, boundary_frame


def _get_fine_tuning_label_overrides() -> dict[str, str]:
    stored_value = st.session_state.get("fine_tuning_label_overrides", {})
    if not isinstance(stored_value, dict):
        return {}
    return {
        str(path): str(label).strip()
        for path, label in stored_value.items()
        if str(path).strip() and str(label).strip()
    }


def _apply_fine_tuning_label_overrides(selected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_overrides = _get_fine_tuning_label_overrides()
    labeled_records: list[dict[str, Any]] = []
    for record in selected_records:
        predicted_label = str(record.get("predicted_label") or record.get("label") or "").strip()
        assigned_label = label_overrides.get(record["path"], predicted_label)
        labeled_record = dict(record)
        labeled_record["predicted_label"] = predicted_label
        labeled_record["assigned_label"] = assigned_label
        labeled_record["label"] = assigned_label
        labeled_records.append(labeled_record)
    return labeled_records


def _save_fine_tuning_label_overrides(
    selected_records: list[dict[str, Any]],
    assigned_labels_by_path: dict[str, str],
) -> None:
    label_overrides = _get_fine_tuning_label_overrides()
    for record in selected_records:
        path = str(record["path"])
        predicted_label = str(record.get("predicted_label") or record.get("label") or "").strip()
        assigned_label = str(assigned_labels_by_path.get(path, predicted_label)).strip()
        if not assigned_label or assigned_label == predicted_label:
            label_overrides.pop(path, None)
        else:
            label_overrides[path] = assigned_label
    st.session_state["fine_tuning_label_overrides"] = label_overrides


def _build_fine_tuning_plan_from_labels(
    selected_records: list[dict[str, Any]],
    epochs: float,
    learning_rate: float,
    repeat_count: int,
    preprocessing_method: str,
) -> DetailFineTunePlan:
    label_counts = Counter(str(record.get("label") or "-") for record in selected_records)
    label_summary = ", ".join(f"{label}: {count}" for label, count in sorted(label_counts.items()))
    return DetailFineTunePlan(
        assistant_reply=f"저장된 이미지 라벨로 바로 학습합니다. 라벨 분포: {label_summary}.",
        target_label=None,
        epochs=float(epochs),
        learning_rate=float(learning_rate),
        repeat_count=int(repeat_count),
        ready_to_train=bool(selected_records),
        create_new_class=False,
        new_class_name=None,
        preprocessing_method=preprocessing_method,
        notes="fine-tuning page saved labels",
    )


def _render_fine_tuning_training_panel(
    image_pool_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    base_model_dir: Path,
    panel: Any | None = None,
    heading: str = "Interactive Fine-tuning",
    has_unsaved_label_changes: bool = False,
) -> None:
    panel = panel or st
    state_prefix = "fine_tuning_page"
    widget_token = f"{state_prefix}__{abs(hash(tuple(record['path'] for record in selected_records)))}"
    execution_state_key = f"{state_prefix}_execution"
    manual_epochs_key = f"{state_prefix}_manual_epochs_{widget_token}"
    manual_learning_rate_key = f"{state_prefix}_manual_learning_rate_{widget_token}"
    manual_repeat_count_key = f"{state_prefix}_manual_repeat_count_{widget_token}"
    manual_preprocessing_key = f"{state_prefix}_manual_preprocessing_{widget_token}"

    panel.subheader(heading)
    panel.caption(f"Base model: {_to_project_relative_path(base_model_dir)}")
    panel.caption(f"Labeled samples: {len(selected_records)}")

    if selected_records:
        label_counts = Counter(str(record.get("label") or "-") for record in selected_records)
        panel.caption("Saved labels: " + ", ".join(f"{label} {count}장" for label, count in sorted(label_counts.items())))

    panel.divider()
    panel.write("Active Learning")
    panel.caption("선택한 inference model을 기준으로 random, margin sampling, boundary sampling으로 이미지를 자동 선택합니다.")
    active_learning_notice = str(st.session_state.get(f"{state_prefix}_active_learning_notice", "")).strip()
    if active_learning_notice:
        panel.caption(active_learning_notice)

    strategy_col, slider_col = panel.columns([0.9, 1.1], gap="large")
    active_learning_strategy = strategy_col.selectbox(
        "Selection strategy",
        options=["Random", "Margin Sampling"],
        key=f"{state_prefix}_active_learning_strategy",
    )
    selection_percentage = slider_col.slider(
        "Selection rate (%)",
        min_value=1,
        max_value=100,
        value=int(st.session_state.get(f"{state_prefix}_active_learning_percentage", 10)),
        step=1,
        key=f"{state_prefix}_active_learning_percentage",
    )

    selectable_records = [record for record in image_pool_records if record.get("exists")]
    if panel.button(
        "Start active learning",
        key=f"{state_prefix}_active_learning_start",
        disabled=not selectable_records,
        width="stretch",
    ):
        try:
            if active_learning_strategy == "Margin Sampling":
                selected_paths, margin_frame = _select_margin_sampling_paths_for_fine_tuning(
                    image_pool_records=selectable_records,
                    base_model_dir=base_model_dir,
                    selection_percentage=selection_percentage,
                )
                if not selected_paths:
                    panel.warning("Margin sampling으로 선택할 이미지가 없습니다.")
                else:
                    min_margin = float(margin_frame["margin_score"].min()) if not margin_frame.empty else 0.0
                    selection_notice = (
                        f"Margin Sampling으로 {len(selected_paths)}장을 선택했습니다. "
                        f"최소 margin={min_margin:.4f}"
                    )
                    _reset_fine_tuning_page_session(selected_paths)
                    st.session_state[f"{state_prefix}_active_learning_percentage"] = selection_percentage
                    st.session_state[f"{state_prefix}_active_learning_notice"] = selection_notice
                    _set_fine_tuning_selection_metadata(
                        selected_paths=selected_paths,
                        strategy="Margin Sampling",
                        origin="active_learning",
                        notice=selection_notice,
                        selection_percentage=selection_percentage,
                        model_dir=base_model_dir,
                    )
                    st.rerun()
            
            else:
                sample_count = max(1, math.ceil(len(selectable_records) * (selection_percentage / 100.0)))
                sample_count = min(sample_count, len(selectable_records))
                sampled_paths = set(random.sample([record["path"] for record in selectable_records], sample_count))
                selected_paths = [record["path"] for record in image_pool_records if record["path"] in sampled_paths]
                selection_notice = f"Random sampling으로 {len(selected_paths)}장을 선택했습니다."
                _reset_fine_tuning_page_session(selected_paths)
                st.session_state[f"{state_prefix}_active_learning_percentage"] = selection_percentage
                st.session_state[f"{state_prefix}_active_learning_notice"] = selection_notice
                _set_fine_tuning_selection_metadata(
                    selected_paths=selected_paths,
                    strategy="Random",
                    origin="active_learning",
                    notice=selection_notice,
                    selection_percentage=selection_percentage,
                    model_dir=base_model_dir,
                )
                st.rerun()
        except Exception as exc:
            panel.warning(f"{active_learning_strategy} 실행 중 오류가 발생했습니다: {exc}")

    panel.divider()
    panel.write("Manual settings")
    manual_defaults = _get_detail_manual_setting_defaults(base_model_dir)
    if manual_epochs_key not in st.session_state:
        st.session_state[manual_epochs_key] = float(manual_defaults["epochs"])
    if manual_learning_rate_key not in st.session_state:
        st.session_state[manual_learning_rate_key] = float(manual_defaults["learning_rate"])
    if manual_repeat_count_key not in st.session_state:
        st.session_state[manual_repeat_count_key] = int(manual_defaults["repeat_count"])
    if manual_preprocessing_key not in st.session_state:
        st.session_state[manual_preprocessing_key] = manual_defaults["preprocessing_method"]

    manual_epochs = panel.number_input(
        "Epochs",
        min_value=1.0,
        max_value=5.0,
        value=float(st.session_state.get(manual_epochs_key, manual_defaults["epochs"])),
        step=0.5,
        key=manual_epochs_key,
    )
    manual_learning_rate = panel.number_input(
        "Learning rate",
        min_value=1e-6,
        max_value=1e-4,
        value=float(st.session_state.get(manual_learning_rate_key, manual_defaults["learning_rate"])),
        step=1e-6,
        format="%.6f",
        key=manual_learning_rate_key,
    )
    manual_repeat_count = panel.number_input(
        "Repeat count",
        min_value=1,
        max_value=64,
        value=int(st.session_state.get(manual_repeat_count_key, manual_defaults["repeat_count"])),
        step=1,
        key=manual_repeat_count_key,
    )
    preprocessing_options = list(DETAIL_PREPROCESSING_LABELS.keys())
    current_preprocessing = str(
        st.session_state.get(manual_preprocessing_key, manual_defaults["preprocessing_method"])
    )
    if current_preprocessing not in DETAIL_PREPROCESSING_LABELS:
        current_preprocessing = "none"
    manual_preprocessing = panel.selectbox(
        "Preprocessing",
        preprocessing_options,
        index=preprocessing_options.index(current_preprocessing),
        format_func=lambda method: DETAIL_PREPROCESSING_LABELS.get(method, method),
        key=manual_preprocessing_key,
    )

    if has_unsaved_label_changes:
        panel.info("왼쪽 `Selected images`의 라벨 변경사항을 저장한 뒤 fine-tuning을 시작하세요.")

    train_disabled = not selected_records or has_unsaved_label_changes
    if panel.button(
        "Start fine-tuning",
        key=f"{state_prefix}_start_{widget_token}",
        type="primary",
        disabled=train_disabled,
        width="stretch",
    ):
        live_log_placeholder = panel.empty()

        def _update_live_finetune_log(log_text: str) -> None:
            live_log_placeholder.text_area(
                "Training log",
                value=log_text[-12000:],
                height=220,
                disabled=True,
            )

        effective_plan = _build_fine_tuning_plan_from_labels(
            selected_records=selected_records,
            epochs=float(manual_epochs),
            learning_rate=float(manual_learning_rate),
            repeat_count=int(manual_repeat_count),
            preprocessing_method=manual_preprocessing,
        )
        selection_metadata = _get_fine_tuning_selection_metadata(selected_records, base_model_dir)
        request_summary = (
            f"base_model={_to_project_relative_path(base_model_dir)}, "
            f"epochs={float(manual_epochs):.1f}, "
            f"learning_rate={float(manual_learning_rate):.6f}, "
            f"repeat_count={int(manual_repeat_count)}, "
            f"preprocessing={manual_preprocessing}, "
            f"selection_strategy={selection_metadata['selection_strategy']}, "
            f"selected_images={len(selected_records)}"
        )
        _append_app_log(
            log_type="start",
            source="Fine-tuning",
            content="Interactive fine-tuning started from saved labels.",
            request=request_summary,
            response="\n".join(
                (
                    f"{record.get('display_path', _to_project_relative_path(record['path']))} | "
                    f"predicted={record.get('predicted_label', '-')} | label={record['label']}"
                )
                for record in selected_records[:20]
            ),
        )
        with st.spinner("저장된 이미지 라벨을 기준으로 MobileViT 모델을 파인튜닝하는 중입니다..."):
            execution_result = run_detail_finetune_plan(
                effective_plan,
                selected_records,
                [],
                base_model_dir=base_model_dir,
                manual_target_class_input=None,
                selected_class_option=None,
                log_callback=_update_live_finetune_log,
                use_record_labels=True,
                selection_metadata=selection_metadata,
            )
        supabase_sync_summary = None
        if execution_result.success:
            try:
                supabase_sync_summary = _sync_fine_tuning_records_to_supabase(
                    selected_records,
                    effective_plan,
                    use_record_labels=True,
                )
                _append_app_log(
                    log_type="done" if supabase_sync_summary["error_count"] == 0 else "Warning",
                    source="Supabase",
                    content=(
                        "Fine-tuning 학습 이미지 상태를 semiconductor 테이블에 반영했습니다."
                        if supabase_sync_summary["error_count"] == 0
                        else "Fine-tuning 학습 이미지 상태를 semiconductor 테이블에 일부만 반영했습니다."
                    ),
                    request=(
                        f"trained=True, class=assigned_label, updated={supabase_sync_summary['updated_count']}, "
                        f"skipped={supabase_sync_summary['skipped_count']}, errors={supabase_sync_summary['error_count']}"
                    ),
                    response="\n".join(
                        [
                            *(
                                f"{item['filename']} -> {item['label']}"
                                for item in supabase_sync_summary["updated_records"][:20]
                            ),
                            *supabase_sync_summary["errors"][:10],
                        ]
                    ),
                )
            except Exception as exc:
                supabase_sync_summary = {
                    "attempted_count": 0,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "error_count": 1,
                    "updated_records": [],
                    "skipped_records": [],
                    "errors": [str(exc)],
                }
                _append_app_log(
                    log_type="error",
                    source="Supabase",
                    content="Fine-tuning 후 semiconductor 테이블 업데이트에 실패했습니다.",
                    request="trained=True, class=assigned_label",
                    response=str(exc),
                )
        st.session_state[execution_state_key] = {
            "success": execution_result.success,
            "command": execution_result.command,
            "output_dir": execution_result.output_dir,
            "llm_comment_path": execution_result.llm_comment_path,
            "selected_images_path": execution_result.selected_images_path,
            "context_json_path": execution_result.context_json_path,
            "audit_log_json_path": execution_result.audit_log_json_path,
            "audit_log_text_path": execution_result.audit_log_text_path,
            "supabase_sync_summary": supabase_sync_summary,
            "stdout": execution_result.stdout,
            "stderr": execution_result.stderr,
            "returncode": execution_result.returncode,
        }
        _append_app_log(
            log_type="done" if execution_result.success else "error",
            source="Fine-tuning",
            content=(
                "Interactive fine-tuning completed successfully."
                if execution_result.success
                else f"Interactive fine-tuning failed with return code {execution_result.returncode}."
            ),
            request=" ".join(execution_result.command),
            response="\n".join(
                filter(
                    None,
                    [
                        (execution_result.stdout or execution_result.stderr or "").strip(),
                        f"audit_json={execution_result.audit_log_json_path or '-'}",
                        f"audit_txt={execution_result.audit_log_text_path or '-'}",
                    ],
                )
            ),
        )

    execution_result = st.session_state.get(execution_state_key)
    if execution_result:
        with panel.expander("Fine-tuning result", expanded=not execution_result["success"]):
            if execution_result["success"]:
                panel.success("파인튜닝이 완료되었습니다.")
                if execution_result.get("output_dir"):
                    panel.caption(f"Output directory: {_format_display_path(execution_result['output_dir'])}")
                if execution_result.get("llm_comment_path"):
                    panel.caption(f"LLM comment file: {_format_display_path(execution_result['llm_comment_path'])}")
                if execution_result.get("selected_images_path"):
                    panel.caption(f"Selected images file: {_format_display_path(execution_result['selected_images_path'])}")
                if execution_result.get("context_json_path"):
                    panel.caption(f"Context json file: {_format_display_path(execution_result['context_json_path'])}")
            else:
                panel.error(f"파인튜닝 실행에 실패했습니다. return code: {execution_result['returncode']}")
            if execution_result.get("audit_log_json_path"):
                panel.caption(f"Audit json file: {_format_display_path(execution_result['audit_log_json_path'])}")
            if execution_result.get("audit_log_text_path"):
                panel.caption(f"Audit text file: {_format_display_path(execution_result['audit_log_text_path'])}")
            supabase_sync_summary = execution_result.get("supabase_sync_summary")
            if isinstance(supabase_sync_summary, dict):
                panel.caption(
                    "Supabase sync: "
                    f"updated={int(supabase_sync_summary.get('updated_count', 0))}, "
                    f"skipped={int(supabase_sync_summary.get('skipped_count', 0))}, "
                    f"errors={int(supabase_sync_summary.get('error_count', 0))}"
                )
                error_messages = supabase_sync_summary.get("errors") or []
                if error_messages:
                    panel.warning("Supabase update error: " + "; ".join(str(message) for message in error_messages[:3]))
            panel.code(" ".join(execution_result["command"]), language="bash")
            if execution_result["stdout"]:
                panel.text_area(
                    "stdout",
                    value=execution_result["stdout"],
                    height=180,
                    disabled=True,
                )
            if execution_result["stderr"]:
                panel.text_area(
                    "stderr",
                    value=execution_result["stderr"],
                    height=180,
                    disabled=True,
                )


def _build_direct_finetune_plan(
    base_model_dir: Path,
    manual_target_label: str | None,
    available_classes: list[str],
) -> DetailFineTunePlan | None:
    if not manual_target_label:
        return None

    dataset_config = read_json_file(resolve_base_model_dir(base_model_dir) / "dataset_config.json", {})
    epochs = float(dataset_config.get("num_epochs", 2.0))
    learning_rate = float(dataset_config.get("learning_rate", 1e-5))
    repeat_count = int(dataset_config.get("interactive_repeat_count", 16))
    normalized_target = manual_target_label.strip()
    existing_class = normalized_target in available_classes
    new_class_name = None if existing_class else normalized_target.replace(" ", "_").replace("-", "_")

    return DetailFineTunePlan(
        assistant_reply=(
            f"LLM 계획 생성 없이 바로 학습합니다. 목표 클래스: "
            f"{normalized_target if existing_class else new_class_name}."
        ),
        target_label=normalized_target if existing_class else None,
        epochs=max(1.0, min(5.0, epochs)),
        learning_rate=max(1e-6, min(1e-4, learning_rate)),
        repeat_count=max(4, min(64, repeat_count)),
        ready_to_train=True,
        create_new_class=not existing_class,
        new_class_name=new_class_name,
        preprocessing_method="none",
        notes="direct fine-tuning without llm plan",
    )


def _get_detail_manual_setting_defaults(base_model_dir: Path) -> dict[str, Any]:
    dataset_config = read_json_file(resolve_base_model_dir(base_model_dir) / "dataset_config.json", {})
    return {
        "epochs": max(1.0, min(5.0, float(dataset_config.get("num_epochs", 2.0)))),
        "learning_rate": max(1e-6, min(1e-4, float(dataset_config.get("learning_rate", 1e-5)))),
        "repeat_count": max(4, min(64, int(dataset_config.get("interactive_repeat_count", 16)))),
        "preprocessing_method": str(dataset_config.get("interactive_preprocessing_method", "none")),
    }


def _render_interactive_finetune_panel(
    selected_records: list[dict[str, Any]],
    state_prefix: str,
    panel: Any | None = None,
    heading: str = "Interactive Fine-tuning",
) -> None:
    panel = panel or st

    widget_token = f"{state_prefix}__" + "__".join(
        record["run_name"] + "_" + record["filename"] for record in selected_records
    )
    base_model_dir = _get_detail_base_model_dir(selected_records)
    selected_model_dirs = _resolve_selected_model_dirs(selected_records)
    execution_state_key = f"{state_prefix}_execution"
    manual_epochs_key = f"{state_prefix}_manual_epochs_{widget_token}"
    manual_learning_rate_key = f"{state_prefix}_manual_learning_rate_{widget_token}"
    manual_repeat_count_key = f"{state_prefix}_manual_repeat_count_{widget_token}"
    manual_preprocessing_key = f"{state_prefix}_manual_preprocessing_{widget_token}"

    panel.subheader(heading)
    panel.caption(f"Base model: {_to_project_relative_path(base_model_dir)}")
    panel.caption(f"Training samples: {len(selected_records)}")
    if len(selected_model_dirs) > 1:
        panel.warning("선택한 이미지들이 서로 다른 추론 모델에서 생성되었습니다. 가장 첫 번째 모델 기준으로 계획과 파인튜닝을 진행합니다.")

    available_classes: list[str]
    try:
        available_classes = load_available_classes(base_model_dir)
    except Exception as exc:
        available_classes = []
        panel.warning(f"클래스 정보를 불러오지 못했습니다: {exc}")

    with panel.container():
        panel.write("Target class")
        if available_classes:
            panel.caption("Available classes: " + ", ".join(available_classes))

        manual_class = panel.text_input(
            "Manual target class 추가",
            placeholder="가능한 클래스 이름을 입력하세요",
            key=f"{state_prefix}_new_class_{widget_token}",
        )
        target_class_options = ["(자동)"] + available_classes
        chosen_class = panel.selectbox(
            "Select target class",
            target_class_options,
            key=f"{state_prefix}_target_class_{widget_token}",
        )

        manual_target_label = None
        manual_target_class_input = manual_class.strip() if manual_class and manual_class.strip() else None
        if manual_class:
            manual_target_label = manual_class.strip()
        elif chosen_class != "(자동)":
            manual_target_label = chosen_class

        if manual_target_label:
            panel.caption(f"선택된 target class: {manual_target_label}")
            if available_classes and manual_target_label not in available_classes:
                panel.warning(
                    "입력한 클래스는 현재 모델 클래스 목록에 없습니다. 실제 파인튜닝 실행 시 에러가 발생할 수 있습니다."
                )

        manual_defaults = _get_detail_manual_setting_defaults(base_model_dir)
        if manual_epochs_key not in st.session_state:
            st.session_state[manual_epochs_key] = float(manual_defaults["epochs"])
        if manual_learning_rate_key not in st.session_state:
            st.session_state[manual_learning_rate_key] = float(manual_defaults["learning_rate"])
        if manual_repeat_count_key not in st.session_state:
            st.session_state[manual_repeat_count_key] = int(manual_defaults["repeat_count"])
        if manual_preprocessing_key not in st.session_state:
            st.session_state[manual_preprocessing_key] = manual_defaults["preprocessing_method"]

        panel.divider()
        panel.write("Manual settings")
        manual_epochs = panel.number_input(
            "Epochs",
            min_value=1.0,
            max_value=5.0,
            value=float(st.session_state.get(manual_epochs_key, manual_defaults["epochs"])),
            step=0.5,
            key=manual_epochs_key,
        )
        manual_learning_rate = panel.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1e-4,
            value=float(st.session_state.get(manual_learning_rate_key, manual_defaults["learning_rate"])),
            step=1e-6,
            format="%.6f",
            key=manual_learning_rate_key,
        )
        preprocessing_options = list(DETAIL_PREPROCESSING_LABELS.keys())
        current_preprocessing = str(st.session_state.get(manual_preprocessing_key, manual_defaults["preprocessing_method"]))
        if current_preprocessing not in DETAIL_PREPROCESSING_LABELS:
            current_preprocessing = "none"
        manual_preprocessing = panel.selectbox(
            "Preprocessing",
            preprocessing_options,
            index=preprocessing_options.index(current_preprocessing),
            format_func=lambda method: DETAIL_PREPROCESSING_LABELS.get(method, method),
            key=manual_preprocessing_key,
        )

        effective_plan = None
        if manual_target_label:
            effective_plan = _build_direct_finetune_plan(base_model_dir, manual_target_label, available_classes)
            if effective_plan:
                effective_plan.epochs = float(manual_epochs)
                effective_plan.learning_rate = float(manual_learning_rate)
                effective_plan.repeat_count = int(st.session_state.get(manual_repeat_count_key, manual_defaults["repeat_count"]))
                effective_plan.preprocessing_method = manual_preprocessing

        train_disabled = (
            effective_plan is None
            or not effective_plan.ready_to_train
            or not (effective_plan.target_label or effective_plan.new_class_name)
        )
        if panel.button(
            "Start fine-tuning",
            key=f"{state_prefix}_start_{widget_token}",
            type="primary",
            disabled=train_disabled,
            width="stretch",
        ):
            live_log_placeholder = panel.empty()

            def _update_live_finetune_log(log_text: str) -> None:
                live_log_placeholder.text_area(
                    "Training log",
                    value=log_text[-12000:],
                    height=220,
                    disabled=True,
                )

            request_summary = (
                f"base_model={_to_project_relative_path(base_model_dir)}, "
                f"target_class={manual_target_label}, "
                f"epochs={float(manual_epochs):.1f}, "
                f"learning_rate={float(manual_learning_rate):.6f}, "
                f"preprocessing={manual_preprocessing}, "
                f"selected_images={len(selected_records)}"
            )
            _append_app_log(
                log_type="start",
                source="Fine-tuning",
                content="Interactive fine-tuning started.",
                request=request_summary,
                response="\n".join(
                    record.get("display_path", _to_project_relative_path(record["path"]))
                    for record in selected_records[:20]
                ),
            )
            selection_metadata = {
                "selection_strategy": "Manual Selection",
                "selection_origin": "manual",
                "selection_notice": "사용자가 detail 페이지에서 직접 선택한 이미지로 학습했습니다.",
                "selection_percentage": None,
            }
            with st.spinner("선택 이미지를 기준으로 MobileViT 모델을 파인튜닝하는 중입니다..."):
                execution_result = run_detail_finetune_plan(
                    effective_plan,
                    selected_records,
                    [],
                    base_model_dir=base_model_dir,
                    manual_target_class_input=manual_target_class_input,
                    selected_class_option=None if chosen_class == "(자동)" else chosen_class,
                    log_callback=_update_live_finetune_log,
                    selection_metadata=selection_metadata,
                )
            supabase_sync_summary = None
            if execution_result.success:
                try:
                    supabase_sync_summary = _sync_fine_tuning_records_to_supabase(
                        selected_records,
                        effective_plan,
                        use_record_labels=False,
                    )
                    _append_app_log(
                        log_type="done" if supabase_sync_summary["error_count"] == 0 else "Warning",
                        source="Supabase",
                        content=(
                            "Detail fine-tuning 학습 이미지 상태를 semiconductor 테이블에 반영했습니다."
                            if supabase_sync_summary["error_count"] == 0
                            else "Detail fine-tuning 학습 이미지 상태를 semiconductor 테이블에 일부만 반영했습니다."
                        ),
                        request=(
                            f"trained=True, class=target_label, updated={supabase_sync_summary['updated_count']}, "
                            f"skipped={supabase_sync_summary['skipped_count']}, errors={supabase_sync_summary['error_count']}"
                        ),
                        response="\n".join(
                            [
                                *(
                                    f"{item['filename']} -> {item['label']}"
                                    for item in supabase_sync_summary["updated_records"][:20]
                                ),
                                *supabase_sync_summary["errors"][:10],
                            ]
                        ),
                    )
                except Exception as exc:
                    supabase_sync_summary = {
                        "attempted_count": 0,
                        "updated_count": 0,
                        "skipped_count": 0,
                        "error_count": 1,
                        "updated_records": [],
                        "skipped_records": [],
                        "errors": [str(exc)],
                    }
                    _append_app_log(
                        log_type="error",
                        source="Supabase",
                        content="Detail fine-tuning 후 semiconductor 테이블 업데이트에 실패했습니다.",
                        request="trained=True, class=target_label",
                        response=str(exc),
                    )
            st.session_state[execution_state_key] = {
                "success": execution_result.success,
                "command": execution_result.command,
                "output_dir": execution_result.output_dir,
                "llm_comment_path": execution_result.llm_comment_path,
                "selected_images_path": execution_result.selected_images_path,
                "context_json_path": execution_result.context_json_path,
                "audit_log_json_path": execution_result.audit_log_json_path,
                "audit_log_text_path": execution_result.audit_log_text_path,
                "supabase_sync_summary": supabase_sync_summary,
                "stdout": execution_result.stdout,
                "stderr": execution_result.stderr,
                "returncode": execution_result.returncode,
            }
            _append_app_log(
                log_type="done" if execution_result.success else "error",
                source="Fine-tuning",
                content=(
                    "Interactive fine-tuning completed successfully."
                    if execution_result.success
                    else f"Interactive fine-tuning failed with return code {execution_result.returncode}."
                ),
                request=" ".join(execution_result.command),
                response="\n".join(
                    filter(
                        None,
                        [
                            (execution_result.stdout or execution_result.stderr or "").strip(),
                            f"audit_json={execution_result.audit_log_json_path or '-'}",
                            f"audit_txt={execution_result.audit_log_text_path or '-'}",
                        ],
                    )
                ),
            )

    execution_result = st.session_state.get(execution_state_key)
    if execution_result:
        with panel.expander("Fine-tuning result", expanded=not execution_result["success"]):
            if execution_result["success"]:
                panel.success("파인튜닝이 완료되었습니다.")
                if execution_result.get("output_dir"):
                    panel.caption(f"Output directory: {_format_display_path(execution_result['output_dir'])}")
                if execution_result.get("llm_comment_path"):
                    panel.caption(f"LLM comment file: {_format_display_path(execution_result['llm_comment_path'])}")
                if execution_result.get("selected_images_path"):
                    panel.caption(f"Selected images file: {_format_display_path(execution_result['selected_images_path'])}")
                if execution_result.get("context_json_path"):
                    panel.caption(f"Context json file: {_format_display_path(execution_result['context_json_path'])}")
            else:
                panel.error(f"파인튜닝 실행에 실패했습니다. return code: {execution_result['returncode']}")
            if execution_result.get("audit_log_json_path"):
                panel.caption(f"Audit json file: {_format_display_path(execution_result['audit_log_json_path'])}")
            if execution_result.get("audit_log_text_path"):
                panel.caption(f"Audit text file: {_format_display_path(execution_result['audit_log_text_path'])}")
            supabase_sync_summary = execution_result.get("supabase_sync_summary")
            if isinstance(supabase_sync_summary, dict):
                panel.caption(
                    "Supabase sync: "
                    f"updated={int(supabase_sync_summary.get('updated_count', 0))}, "
                    f"skipped={int(supabase_sync_summary.get('skipped_count', 0))}, "
                    f"errors={int(supabase_sync_summary.get('error_count', 0))}"
                )
                error_messages = supabase_sync_summary.get("errors") or []
                if error_messages:
                    panel.warning("Supabase update error: " + "; ".join(str(message) for message in error_messages[:3]))
            panel.code(" ".join(execution_result["command"]), language="bash")
            if execution_result["stdout"]:
                panel.text_area(
                    "stdout",
                    value=execution_result["stdout"],
                    height=180,
                    disabled=True,
                )
            if execution_result["stderr"]:
                panel.text_area(
                    "stderr",
                    value=execution_result["stderr"],
                    height=180,
                    disabled=True,
                )


def _render_detail_finetune_sidebar(selected_records: list[dict[str, Any]], use_sidebar: bool = True) -> None:
    target_panel = st.sidebar if use_sidebar else st
    _render_interactive_finetune_panel(
        selected_records=selected_records,
        state_prefix="detail_finetune",
        panel=target_panel,
        heading="Interactive Fine-tuning",
    )


def _extract_features_from_images(
    image_paths: list[str],
    model_dir: Path,
) -> tuple[Any, list[str]]:
    """Extract feature vectors from images using the classifier model.
    
    Args:
        image_paths: List of image file paths
        model_dir: Path to the classifier model directory
        
    Returns:
        Tuple of (features array, processed image path list)
    """
    try:
        import numpy as np
        import torch
        from PIL import Image
        _suppress_transformers_path_alias_warning()
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        st.error("필요한 라이브러리가 없습니다. torch, transformers, Pillow를 설치해주세요.")
        return None, []
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_processor = AutoImageProcessor.from_pretrained(str(model_dir))
        model = AutoModelForImageClassification.from_pretrained(str(model_dir)).to(device)
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = True

        def _to_feature_vector(output_tensor: Any) -> np.ndarray:
            array = output_tensor.detach().cpu().numpy()[0]
            if array.ndim == 1:
                return array.astype(np.float32)

            feature_axis = int(np.argmax(array.shape))
            reduce_axes = tuple(index for index in range(array.ndim) if index != feature_axis)
            reduced = array.mean(axis=reduce_axes)
            return np.asarray(reduced, dtype=np.float32).reshape(-1)

        features_list = []
        processed_paths = []
        
        model.eval()
        with torch.no_grad():
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = image_processor(images=[image], return_tensors="pt").to(device)
                    
                    # Prefer richer hidden representations when available, then fall back to logits.
                    try:
                        outputs = model(**inputs, output_hidden_states=True)
                    except TypeError:
                        outputs = model(**inputs)
                    hidden_states = getattr(outputs, "hidden_states", None)
                    if hidden_states:
                        feature_vector = _to_feature_vector(hidden_states[-1])
                    else:
                        feature_vector = _to_feature_vector(outputs.logits)
                    features_list.append(feature_vector)
                    processed_paths.append(image_path)
                except Exception as e:
                    st.warning(f"특성 추출 실패 {Path(image_path).name}: {e}")
                    continue
        
        if not features_list:
            st.error("어떤 이미지도 처리할 수 없습니다.")
            return None, []
        
        features_array = np.array(features_list, dtype=np.float32)
        
        # Ensure 2D shape: (num_samples, feature_dim)
        if features_array.ndim > 2:
            features_array = features_array.reshape(features_array.shape[0], -1)
        elif features_array.ndim == 1:
            features_array = features_array.reshape(-1, 1)
        
        return features_array, processed_paths
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, []


def _render_detail_3d_visualization(selected_records: list[dict[str, Any]]) -> None:
    """Render 3D visualization of model predictions using dimensionality reduction."""
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
    
    # Extract features
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
    
    # Ensure features is 2D: (num_samples, feature_dim)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    elif features.ndim == 1:
        features = features.reshape(-1, 1)
    
    # Dimensionality reduction options
    col1, col2 = st.columns(2)
    with col1:
        reduction_method = st.selectbox(
            "차원 축소 방법",
            ["PCA", "t-SNE", "UMAP"],
            help="고차원 특성을 3D로 축소하는 방법을 선택하세요"
        )
    with col2:
        show_labels = st.checkbox("라벨 표시", value=False)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply dimensionality reduction
    if reduction_method == "PCA":
        reducer = PCA(n_components=3, random_state=42)
        features_3d = reducer.fit_transform(features_scaled)
        reduction_info = f"PCA (설명된 분산: {sum(reducer.explained_variance_ratio_):.2%})"
    elif reduction_method == "t-SNE":
        try:
            from sklearn.manifold import TSNE
            with st.spinner("t-SNE 계산 중... (시간이 걸릴 수 있습니다)"):
                reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features)-1))
                features_3d = reducer.fit_transform(features_scaled)
            reduction_info = "t-SNE"
        except Exception as e:
            st.error(f"t-SNE 계산 실패: {e}")
            return
    else:  # UMAP
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

    # Normalize each axis for display so depth is easier to perceive in the 3D view.
    features_3d_display = np.asarray(features_3d, dtype=np.float32)
    axis_std = features_3d_display.std(axis=0)
    axis_std[axis_std == 0] = 1.0
    features_3d_display = (features_3d_display - features_3d_display.mean(axis=0)) / axis_std
    
    # Create discrete class colors and traces for legend clarity
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

        fig.add_trace(go.Scatter3d(
            x=features_3d_display[class_indices, 0],
            y=features_3d_display[class_indices, 1],
            z=features_3d_display[class_indices, 2],
            mode='markers+text' if show_labels else 'markers',
            name=label,
            marker=dict(
                size=8,
                color=color_map[label],
                line=dict(width=0.5, color='white'),
            ),
            text=[label] * len(class_indices) if show_labels else None,
            textposition="top center",
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
        ))
    
    # Update layout
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
        hovermode='closest',
        dragmode="orbit",
        legend=dict(title="Class"),
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Show class distribution
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
    
    # Show image details
    with st.expander("이미지 상세 정보", expanded=False):
        details_df = pd.DataFrame({
            "Filename": [record["filename"] for record in plotted_records],
            "Prediction": labels,
            "Path": [record["path"] for record in plotted_records],
        })
        st.dataframe(details_df, width="stretch")


def render_detail_page(image_records: list[dict[str, Any]]) -> None:
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

    select_options: list[str] = []
    seen_paths: set[str] = set()
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

    selected_records: list[dict[str, Any]] = []
    if selected_paths:
        raw_selected_records = _get_detail_selected_records(image_records)
        selected_model_dir, model_changed, start_infer = _render_detail_inference_model_selector(raw_selected_records)
        if model_changed:
            _reset_detail_finetune_session(selected_paths)

        selected_records = raw_selected_records
        prediction_errors: list[str] = []
        artifact_paths: dict[str, str] | None = None
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
            st.warning(
                "일부 이미지 재추론에 실패했습니다: " + "; ".join(prediction_errors[:3])
            )
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


def render_fine_tuning_page(image_records: list[dict[str, Any]]) -> None:
    render_page_header("Fine-tuning", "이미지 pool에서 학습할 샘플을 선택하고 Interactive Fine-tuning을 실행하세요.")
    image_pool_records = _build_unique_image_pool(image_records)
    if not image_pool_records:
        st.info("Fine-tuning에 사용할 이미지 pool이 없습니다.")
        return

    predicted_pool_records = image_pool_records
    selected_records: list[dict[str, Any]] = []
    has_unsaved_label_changes = False
    selected_model_dir: Path | None = None
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        with st.container(border=True):
            selected_model_dir, model_changed = _render_classifier_model_selector(
                selected_records=image_pool_records,
                container=st,
                selector_key="fine_tuning_inference_model_selector",
                active_key="fine_tuning_inference_model_active",
                section_title="Image Pool",
                helper_text="모델을 선택하면 아래 이미지 pool 전체를 해당 모델로 다시 예측합니다.",
                label="Inference model",
            )
            if selected_model_dir is None:
                return
            selected_model_dir = resolve_base_model_dir(selected_model_dir)

            previous_selected_paths = list(st.session_state.get("fine_tuning_page_selected_image_paths", []))
            available_paths = {record["path"] for record in image_pool_records}
            previous_selected_paths = [path for path in previous_selected_paths if path in available_paths]
            if model_changed:
                _reset_fine_tuning_page_session(previous_selected_paths)
                if previous_selected_paths:
                    _set_fine_tuning_selection_metadata(
                        selected_paths=previous_selected_paths,
                        strategy="Manual Selection",
                        origin="manual",
                        notice="inference model changed after selection",
                        selection_percentage=None,
                        model_dir=selected_model_dir,
                    )

            try:
                predicted_pool_records, prediction_errors = _predict_image_pool_records_with_model(
                    image_pool_records,
                    selected_model_dir,
                )
            except Exception as exc:
                st.warning(str(exc))
                predicted_pool_records = image_pool_records
                prediction_errors = []

            if prediction_errors:
                st.warning("일부 이미지 재추론에 실패했습니다: " + "; ".join(prediction_errors[:3]))

            available_classes: list[str] = []
            try:
                available_classes = load_available_classes(selected_model_dir)
            except Exception as exc:
                st.warning(f"클래스 정보를 불러오지 못했습니다: {exc}")

            records_by_index = {
                index: record for index, record in enumerate(predicted_pool_records, start=1)
            }
            table_rows = [
                {
                    "Index": index,
                    "Raw Image": _build_thumbnail_data_uri(record["path"]) if record["exists"] else None,
                    "Trained": bool(record.get("trained", False)),
                    "Predicted Class": record.get("predicted_label", record["label"]),
                    "Select": record["path"] in previous_selected_paths,
                }
                for index, record in records_by_index.items()
            ]

            edited_table = st.data_editor(
                pd.DataFrame(table_rows),
                hide_index=True,
                use_container_width=True,
                height=760,
                disabled=["Index", "Raw Image", "Trained", "Predicted Class"],
                column_order=["Index", "Raw Image", "Trained", "Predicted Class", "Select"],
                column_config={
                    "Index": st.column_config.NumberColumn("Index", format="%d", width="small"),
                    "Raw Image": st.column_config.ImageColumn("Raw Image", width="medium"),
                    "Trained": st.column_config.CheckboxColumn(
                        "Trained",
                        help="Supabase 기준으로 이미 학습에 반영된 데이터인지 표시합니다.",
                        width="small",
                    ),
                    "Predicted Class": st.column_config.TextColumn("Predicted Class", width="medium"),
                    "Select": st.column_config.CheckboxColumn("Select", help="선택된 이미지만 fine-tuning에 사용됩니다."),
                },
                key="fine_tuning_image_pool_editor",
            )

            selected_indices = [
                int(row["Index"]) for _, row in edited_table.iterrows() if bool(row["Select"])
            ]
            raw_selected_records = [records_by_index[index] for index in selected_indices if index in records_by_index]
            selected_paths = [record["path"] for record in raw_selected_records]
            if previous_selected_paths != selected_paths:
                _reset_fine_tuning_page_session(selected_paths)
                _set_fine_tuning_selection_metadata(
                    selected_paths=selected_paths,
                    strategy="Manual Selection",
                    origin="manual",
                    notice="사용자가 fine-tuning 이미지 pool에서 직접 선택했습니다.",
                    selection_percentage=None,
                    model_dir=selected_model_dir,
                )
            else:
                st.session_state["fine_tuning_page_selected_image_paths"] = selected_paths

            selected_records = _sync_records_with_model_dir(
                _apply_fine_tuning_label_overrides(raw_selected_records),
                selected_model_dir,
            )
            st.caption(f"전체 이미지 {len(predicted_pool_records)}장 중 {len(selected_records)}장을 선택했습니다.")
            if selected_records:
                with st.expander(f"Selected images ({len(selected_records)})", expanded=True):
                    selected_records_widget_token = str(abs(hash(tuple(record["path"] for record in selected_records))))
                    if available_classes:
                        st.caption(
                            "Available classes: "
                            + ", ".join(available_classes)
                            + " | 새 class는 `Label` 칸에 직접 입력하세요."
                        )
                    label_editor_rows = [
                        {
                            "Raw Image": _build_thumbnail_data_uri(record["path"]) if record["exists"] else None,
                            "Filename": record["filename"],
                            "Predicted Class": record.get("predicted_label", record["label"]),
                            "Label": record["label"],
                        }
                        for record in selected_records
                    ]
                    label_column_config: dict[str, Any] = {
                        "Raw Image": st.column_config.ImageColumn("Raw Image", width="medium"),
                        "Filename": st.column_config.TextColumn("Filename", width="medium"),
                        "Predicted Class": st.column_config.TextColumn("Predicted Class", width="medium"),
                        "Label": st.column_config.TextColumn(
                            "Label",
                            width="medium",
                            required=True,
                            help="기존 class 이름을 입력하거나 새 class를 직접 입력하세요.",
                        ),
                    }

                    edited_labels = st.data_editor(
                        pd.DataFrame(label_editor_rows),
                        hide_index=True,
                        use_container_width=True,
                        height=min(420, 120 + len(label_editor_rows) * 70),
                        disabled=["Raw Image", "Filename", "Predicted Class"],
                        column_order=["Raw Image", "Filename", "Predicted Class", "Label"],
                        column_config=label_column_config,
                        key=f"fine_tuning_selected_label_editor__{selected_records_widget_token}",
                    )
                    assigned_labels_by_path = {
                        record["path"]: str(edited_labels.iloc[index]["Label"]).strip()
                        for index, record in enumerate(selected_records)
                    }
                    saved_labels_by_path = {
                        record["path"]: str(record["label"]).strip()
                        for record in selected_records
                    }
                    has_unsaved_label_changes = assigned_labels_by_path != saved_labels_by_path

                    if st.button("Save labels", key="fine_tuning_save_labels", width="stretch"):
                        _save_fine_tuning_label_overrides(raw_selected_records, assigned_labels_by_path)
                        selected_records = _sync_records_with_model_dir(
                            _apply_fine_tuning_label_overrides(raw_selected_records),
                            selected_model_dir,
                        )
                        has_unsaved_label_changes = False
                        st.success("선택 이미지 라벨을 저장했습니다.")
                    elif has_unsaved_label_changes:
                        st.info("라벨 변경 후 `Save labels`를 누르면 오른쪽 fine-tuning에 반영됩니다.")
            else:
                st.info("왼쪽 Image Pool에서 `Select` 체크박스로 fine-tuning에 사용할 이미지를 선택하세요.")

    with right_col:
        with st.container(border=True):
            _render_fine_tuning_training_panel(
                image_pool_records=predicted_pool_records,
                selected_records=selected_records,
                base_model_dir=resolve_base_model_dir(selected_model_dir),
                panel=st,
                heading="Interactive Fine-tuning",
                has_unsaved_label_changes=has_unsaved_label_changes,
            )


def render_setting_page(config: dict[str, Any], latest_run: dict[str, Any] | None) -> None:
    render_page_header("Setting")
    top_left, _ = st.columns([1.1, 1.9])
    current_supabase_settings = _get_supabase_secret_settings()
    st.session_state.setdefault("db_setting_panel_open", False)
    st.session_state.setdefault("db_setting_status", "idle")
    st.session_state.setdefault("db_setting_notice", "")
    st.session_state.setdefault("supabase_url_pending", current_supabase_settings["url"])
    st.session_state.setdefault("supabase_key_pending", current_supabase_settings["key"])

    with top_left:
        if st.button("DB setting", width="stretch"):
            st.session_state["db_setting_panel_open"] = not bool(st.session_state["db_setting_panel_open"])
            st.session_state["supabase_url_pending"] = current_supabase_settings["url"]
            st.session_state["supabase_key_pending"] = current_supabase_settings["key"]
            st.session_state["db_setting_status"] = "idle"
            st.session_state["db_setting_notice"] = ""
            st.rerun()

    if st.session_state["db_setting_panel_open"]:
        with st.container(border=True):
            st.subheader("Supabase DB")
            st.caption(
                "`.streamlit/secrets.toml`의 `connections.supabase` 값을 UI에서 수정합니다. "
                "저장 후에는 앱을 다시 실행하면 가장 확실합니다."
            )
            st.text_input(
                "Supabase URL",
                key="supabase_url_pending",
                placeholder="http://127.0.0.1:54321",
            )
            st.text_input(
                "Supabase KEY",
                key="supabase_key_pending",
                type="password",
                placeholder="service_role or anon key",
            )
            button_col, close_col = st.columns(2, gap="large")
            with button_col:
                save_db_clicked = st.button(
                    "Save DB setting",
                    key="db_setting_apply",
                    type="primary",
                    width="stretch",
                )
            with close_col:
                close_db_clicked = st.button(
                    "Close",
                    key="db_setting_close",
                    width="stretch",
                )

            if save_db_clicked:
                try:
                    supabase_url = str(st.session_state.get("supabase_url_pending", "")).strip()
                    supabase_key = str(st.session_state.get("supabase_key_pending", "")).strip()
                    if not supabase_url:
                        raise RuntimeError("Supabase URL을 입력해 주세요.")
                    if not supabase_key:
                        raise RuntimeError("Supabase KEY를 입력해 주세요.")

                    _write_supabase_secret_settings(supabase_url, supabase_key)
                    load_dashboard_data.clear()
                    st.cache_data.clear()
                    st.session_state["db_setting_status"] = "done"
                    st.session_state["db_setting_notice"] = "Supabase DB 설정이 저장되었습니다."
                    _append_app_log(
                        log_type="done",
                        source="Setting",
                        content="Supabase DB settings were saved.",
                        request=f"url={supabase_url}",
                        response=f"key={_mask_secret_value(supabase_key)}",
                    )
                    st.rerun()
                except Exception as exc:
                    st.session_state["db_setting_status"] = "error"
                    st.session_state["db_setting_notice"] = "Supabase DB 설정 저장에 실패했습니다."
                    _append_app_log(
                        log_type="error",
                        source="Setting",
                        content="Supabase DB settings could not be saved.",
                        request=f"url={str(st.session_state.get('supabase_url_pending', '')).strip()}",
                        response=str(exc),
                    )
                    st.rerun()

            if close_db_clicked:
                st.session_state["db_setting_panel_open"] = False
                st.session_state["db_setting_status"] = "idle"
                st.session_state["db_setting_notice"] = ""
                st.rerun()

            db_status = st.session_state["db_setting_status"]
            if db_status == "done":
                st.success(st.session_state["db_setting_notice"])
            elif db_status == "error":
                st.error(st.session_state["db_setting_notice"])

            st.caption(f"Secrets file: {_to_project_relative_path(SECRETS_PATH)}")
            st.caption(f"Saved URL: {current_supabase_settings['url'] or '-'}")
            st.caption(f"Saved KEY: {_mask_secret_value(current_supabase_settings['key'])}")

    applied_llm_settings = _get_llm_runtime_settings()
    pending_llm_settings = _get_pending_llm_runtime_settings()
    available_model_dirs = applied_llm_settings["available_model_dirs"]
    available_model_values = [str(path) for path in available_model_dirs]
    st.session_state.setdefault("llm_setting_status", "idle")
    st.session_state.setdefault("llm_setting_notice", "")

    with st.container(border=True):
        st.subheader("LLM Runtime")
        st.caption("사이드바 LLM 응답과 Summary 분석 코멘트에 사용할 모델 설정입니다.")

        if available_model_values:
            current_model_value = str(pending_llm_settings["model_dir"])
            if current_model_value not in available_model_values:
                current_model_value = available_model_values[0]

            model_col, temp_col, token_col = st.columns(3, gap="large")
            with model_col:
                selected_model = st.selectbox(
                    "LLM model",
                    options=available_model_values,
                    index=available_model_values.index(current_model_value),
                    format_func=lambda value: Path(value).name,
                    key="llm_model_dir_pending",
                )
            with temp_col:
                st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.5,
                    value=float(pending_llm_settings["temperature"]),
                    step=0.1,
                    key="llm_temperature_pending",
                )
            with token_col:
                st.number_input(
                    "Max tokens",
                    min_value=64,
                    max_value=4096,
                    value=int(pending_llm_settings["max_new_tokens"]),
                    step=64,
                    key="llm_max_new_tokens_pending",
                )

            apply_clicked = st.button("Set", key="llm_runtime_apply", type="primary", width="stretch")
            if apply_clicked:
                try:
                    pending_model_dir = Path(str(st.session_state["llm_model_dir_pending"]))
                    if not available_model_values:
                        raise RuntimeError("사용 가능한 로컬 LLM 모델이 없습니다.")
                    if str(pending_model_dir) not in available_model_values:
                        raise RuntimeError("선택한 LLM 모델을 찾을 수 없습니다.")
                    if not is_model_downloaded(pending_model_dir):
                        raise RuntimeError("선택한 LLM 모델 파일이 완전히 준비되지 않았습니다.")

                    st.session_state["llm_model_dir"] = str(pending_model_dir)
                    st.session_state["llm_temperature"] = max(
                        0.0,
                        float(st.session_state["llm_temperature_pending"]),
                    )
                    st.session_state["llm_max_new_tokens"] = max(
                        1,
                        int(st.session_state["llm_max_new_tokens_pending"]),
                    )
                    st.session_state["llm_setting_status"] = "done"
                    st.session_state["llm_setting_notice"] = "LLM runtime 설정이 적용되었습니다."
                    _append_app_log(
                        log_type="done",
                        source="Setting",
                        content="LLM runtime settings were applied successfully.",
                        request=(
                            f"model={_to_project_relative_path(pending_model_dir)}, "
                            f"temperature={float(st.session_state['llm_temperature']):.1f}, "
                            f"max_tokens={int(st.session_state['llm_max_new_tokens'])}"
                        ),
                    )
                except Exception as exc:
                    st.session_state["llm_setting_status"] = "error"
                    st.session_state["llm_setting_notice"] = "LLM runtime 설정 적용에 실패했습니다."
                    _append_app_log(
                        log_type="error",
                        source="Setting",
                        content="LLM runtime settings could not be applied.",
                        request=(
                            f"model={_format_display_path(st.session_state.get('llm_model_dir_pending'))}, "
                            f"temperature={float(st.session_state.get('llm_temperature_pending', DEFAULT_LLM_TEMPERATURE)):.1f}, "
                            f"max_tokens={int(st.session_state.get('llm_max_new_tokens_pending', DEFAULT_LLM_MAX_NEW_TOKENS))}"
                        ),
                        response=str(exc),
                    )
                st.rerun()

            status = st.session_state["llm_setting_status"]
            if status == "done":
                st.success(st.session_state["llm_setting_notice"])
            elif status == "error":
                st.error(st.session_state["llm_setting_notice"])

            st.caption(
                "Applied setting: "
                f"{Path(str(applied_llm_settings['model_dir'])).name} | "
                f"Temp {float(applied_llm_settings['temperature']):.1f} | "
                f"Max tokens {int(applied_llm_settings['max_new_tokens'])}"
            )
            st.caption(f"Applied model path: {_to_project_relative_path(applied_llm_settings['model_dir'])}")
            st.caption(f"Pending model path: {_to_project_relative_path(selected_model)}")
        else:
            st.warning(f"사용 가능한 로컬 LLM 모델을 찾지 못했습니다: {_to_project_relative_path(DEFAULT_LLM_MODEL_DIR.parent)}")

    
    with st.expander("Current configuration", expanded=False):
        config_frame = pd.DataFrame(
            [
                {
                    "Setting": key,
                    "Value": _format_display_path(value) if _looks_like_project_path(value) else str(value),
                }
                for key, value in config.items()
                if key != "classes_to_train"
            ]
        )
        st.dataframe(config_frame, width="stretch", hide_index=True)
        if isinstance(config.get("classes_to_train"), list):
            st.write("Classes To Train")
            st.write(", ".join(str(item) for item in config["classes_to_train"]))


def render_log_page(log_entries: list[dict[str, str]]) -> None:
    render_page_header("Log")
    if not log_entries:
        st.info("No log entries found.")
        return

    key_log = sorted(
        log_entries,
        key=lambda item: (
            SEVERITY_ORDER.get(item["log_type"], 99),
            item.get("timestamp", ""),
        ),
    )[0]
    with st.container(border=True):
        st.subheader("Key Log")
        st.dataframe(
            pd.DataFrame([key_log])[["date", "time", "source", "log_type", "content", "request", "response"]],
            width="stretch",
            hide_index=True,
        )

    all_dates = ["All dates"] + sorted({entry["date"] for entry in log_entries}, reverse=True)
    all_types = ["All log types"] + sorted(
        {entry["log_type"] for entry in log_entries},
        key=lambda value: SEVERITY_ORDER.get(value, 99),
    )

    filter_cols = st.columns(2, gap="large")
    with filter_cols[0]:
        selected_date = st.selectbox("Date filter", all_dates)
    with filter_cols[1]:
        selected_type = st.selectbox("Log type filter", all_types)

    filtered_logs = log_entries
    if selected_date != "All dates":
        filtered_logs = [entry for entry in filtered_logs if entry["date"] == selected_date]
    if selected_type != "All log types":
        filtered_logs = [entry for entry in filtered_logs if entry["log_type"] == selected_type]

    with st.container(border=True):
        st.subheader("Log History")
        if filtered_logs:
            log_frame = pd.DataFrame(filtered_logs)
            ordered_columns = ["date", "time", "source", "log_type", "content", "request", "response"]
            available_columns = [column for column in ordered_columns if column in log_frame.columns]
            st.dataframe(log_frame[available_columns], width="stretch", hide_index=True)
        else:
            st.info("No log entries found for this filter.")


def main() -> None:
    configure_page("Dashboard Home")
    config, runs, _image_records, log_entries = load_dashboard_data()
    render_home_page(config, runs, log_entries)


if __name__ == "__main__":
    main()
