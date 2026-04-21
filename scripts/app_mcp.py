from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st
import toml

from scripts.local_gemma_model import generate_response

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None


BASE_DIR = Path(__file__).resolve().parents[1]
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"

DETAIL_PAGE = "pages/2_Detail.py"
SUMMARY_PAGE = "pages/1_Summary.py"
SETTING_PAGE = "pages/4_Setting.py"

DETAIL_REDUCTION_METHODS = ("PCA", "t-SNE", "UMAP")
DETAIL_XAI_METHODS = ("grad", "sg", "itg", "ig", "lime", "shap", "control")
MCP_TOOL_NAMES = {
    "summary_download_report",
    "detail_set_class_filter",
    "detail_set_3d_reduction_method",
    "detail_set_xai_method",
    "setting_open_db_settings",
    "setting_save_db_settings",
}

app_mcp = FastMCP("Manufacturing Inspection App", json_response=True) if FastMCP else None


def _mcp_tool(func):
    if app_mcp is None:
        return func
    return app_mcp.tool()(func)


def _normalize_token(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.strip().lower())


def _normalize_reduction_method(method: str) -> str | None:
    normalized = _normalize_token(method)
    aliases = {
        "pca": "PCA",
        "tsne": "t-SNE",
        "t-sne": "t-SNE",
        "umap": "UMAP",
    }
    return aliases.get(normalized)


def _normalize_xai_method(method: str) -> str | None:
    normalized = _normalize_token(method)
    aliases = {
        "gradient": "grad",
        "gradientmethod": "grad",
        "grad": "grad",
        "smoothgrad": "sg",
        "sg": "sg",
        "inputxgradient": "itg",
        "inputgradient": "itg",
        "itg": "itg",
        "integratedgradient": "ig",
        "integratedgradients": "ig",
        "ig": "ig",
        "lime": "lime",
        "shap": "shap",
        "control": "control",
    }
    return aliases.get(normalized)


def _read_toml_file(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = default or {}
    if not path.exists():
        return fallback
    try:
        return toml.loads(path.read_text(encoding="utf-8"))
    except (toml.TomlDecodeError, OSError):
        return fallback


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


@_mcp_tool
def summary_download_report() -> dict[str, Any]:
    """Request generation of the Summary page PDF report and open the Summary page."""
    st.session_state["summary_download_report_requested"] = True
    st.session_state["mcp_last_action"] = "Summary report generation requested."
    return {
        "status": "ok",
        "tool": "summary_download_report",
        "message": "Summary report generation was requested. Opening the Summary page.",
        "target_page": SUMMARY_PAGE,
        "requires_rerun": True,
    }


@_mcp_tool
def detail_set_class_filter(class_name: str) -> dict[str, Any]:
    """Set the Detail page class filter."""
    requested_class = str(class_name or "").strip()
    if not requested_class:
        return {
            "status": "error",
            "tool": "detail_set_class_filter",
            "message": "Class name is required.",
            "requires_rerun": False,
        }

    st.session_state["detail_class_filter_requested"] = requested_class
    st.session_state["detail_selected_image_paths"] = []
    st.session_state.pop("detail_multi_select_paths", None)
    st.session_state["mcp_last_action"] = f"Detail class filter requested: {requested_class}"
    return {
        "status": "ok",
        "tool": "detail_set_class_filter",
        "message": f"Detail class filter was set to `{requested_class}`.",
        "target_page": DETAIL_PAGE,
        "requires_rerun": True,
    }


@_mcp_tool
def detail_set_3d_reduction_method(method: str) -> dict[str, Any]:
    """Set the Detail page 3D visualization dimensionality reduction method."""
    normalized_method = _normalize_reduction_method(str(method or ""))
    if normalized_method is None:
        return {
            "status": "error",
            "tool": "detail_set_3d_reduction_method",
            "message": "Reduction method must be one of PCA, t-SNE, or UMAP.",
            "requires_rerun": False,
        }

    st.session_state["detail_3d_reduction_requested"] = normalized_method
    st.session_state["mcp_last_action"] = f"3D reduction method requested: {normalized_method}"
    return {
        "status": "ok",
        "tool": "detail_set_3d_reduction_method",
        "message": f"3D visualization reduction method was set to `{normalized_method}`.",
        "target_page": DETAIL_PAGE,
        "requires_rerun": True,
    }


@_mcp_tool
def detail_set_xai_method(method: str) -> dict[str, Any]:
    """Set the Detail page OpenXAI method."""
    normalized_method = _normalize_xai_method(str(method or ""))
    if normalized_method is None:
        return {
            "status": "error",
            "tool": "detail_set_xai_method",
            "message": "XAI method must be one of grad, sg, itg, ig, lime, shap, or control.",
            "requires_rerun": False,
        }

    st.session_state["detail_xai_method_requested"] = normalized_method
    st.session_state["mcp_last_action"] = f"XAI method requested: {normalized_method}"
    return {
        "status": "ok",
        "tool": "detail_set_xai_method",
        "message": f"OpenXAI method was set to `{normalized_method}`.",
        "target_page": DETAIL_PAGE,
        "requires_rerun": True,
    }


@_mcp_tool
def setting_open_db_settings() -> dict[str, Any]:
    """Open the Setting page DB Settings panel."""
    st.session_state["db_setting_panel_open"] = True
    st.session_state["db_setting_status"] = "idle"
    st.session_state["db_setting_notice"] = ""
    st.session_state["mcp_last_action"] = "DB Settings panel opened."
    return {
        "status": "ok",
        "tool": "setting_open_db_settings",
        "message": "DB Settings panel was opened.",
        "target_page": SETTING_PAGE,
        "requires_rerun": True,
    }


@_mcp_tool
def setting_save_db_settings(url: str, key: str) -> dict[str, Any]:
    """Save Supabase DB settings to .streamlit/secrets.toml."""
    supabase_url = str(url or "").strip()
    supabase_key = str(key or "").strip()
    if not supabase_url:
        return {
            "status": "error",
            "tool": "setting_save_db_settings",
            "message": "Supabase URL is required.",
            "requires_rerun": False,
        }
    if not supabase_key:
        return {
            "status": "error",
            "tool": "setting_save_db_settings",
            "message": "Supabase key is required.",
            "requires_rerun": False,
        }

    _write_supabase_secret_settings(supabase_url, supabase_key)
    st.session_state["db_setting_panel_open"] = True
    st.session_state["supabase_url_pending"] = supabase_url
    st.session_state["supabase_key_pending"] = supabase_key
    st.session_state["db_setting_status"] = "done"
    st.session_state["db_setting_notice"] = "The Supabase DB settings have been saved."
    st.session_state["mcp_last_action"] = "Supabase DB settings saved."
    return {
        "status": "ok",
        "tool": "setting_save_db_settings",
        "message": f"Supabase DB settings were saved. Key: {_mask_secret_value(supabase_key)}",
        "target_page": SETTING_PAGE,
        "requires_rerun": True,
        "clear_dashboard_cache": True,
    }


def execute_app_mcp_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = arguments or {}
    if tool_name == "summary_download_report":
        return summary_download_report()
    if tool_name == "detail_set_class_filter":
        return detail_set_class_filter(str(args.get("class_name") or args.get("class") or args.get("label") or ""))
    if tool_name == "detail_set_3d_reduction_method":
        return detail_set_3d_reduction_method(str(args.get("method", "")))
    if tool_name == "detail_set_xai_method":
        return detail_set_xai_method(str(args.get("method", "")))
    if tool_name == "setting_open_db_settings":
        return setting_open_db_settings()
    if tool_name == "setting_save_db_settings":
        return setting_save_db_settings(
            str(args.get("url") or args.get("supabase_url") or ""),
            str(args.get("key") or args.get("supabase_key") or ""),
        )
    return {
        "status": "error",
        "tool": tool_name,
        "message": f"Unknown MCP tool: {tool_name}",
        "requires_rerun": False,
    }


def _parse_supabase_settings_from_text(text: str) -> dict[str, str] | None:
    url_match = re.search(r"https?://[^\s,]+", text)
    if not url_match:
        return None

    key_match = re.search(
        r"(?:key|anon|service[_\s-]*role|supabase[_\s-]*key|키)\s*[:=]\s*([^\s,]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not key_match:
        return None

    return {"url": url_match.group(0), "key": key_match.group(1)}


def route_app_command_deterministically(user_prompt: str, current_page_title: str | None = None) -> dict[str, Any] | None:
    raw = user_prompt.strip()
    lowered = raw.lower()

    if any(term in lowered for term in ("download report", "report download", "summary report")) or (
        any(term in raw for term in ("리포트", "보고서"))
        and any(term in raw for term in ("발행", "다운로드", "생성", "만들", "뽑"))
    ):
        return {"tool": "summary_download_report", "arguments": {}}

    parsed_db_settings = _parse_supabase_settings_from_text(raw)
    if parsed_db_settings and any(term in lowered for term in ("db", "supabase", "database", "setting", "settings")):
        return {"tool": "setting_save_db_settings", "arguments": parsed_db_settings}

    if any(term in lowered for term in ("db setting", "db settings", "database setting", "database settings", "supabase")) or any(
        term in raw for term in ("DB 설정", "DB 세팅", "데이터베이스 설정", "디비 설정")
    ):
        return {"tool": "setting_open_db_settings", "arguments": {}}

    class_match = re.search(
        r"(normal|center|donut|edge[-\s]?loc|edge[-\s]?ring|local|near[-\s]?full|scratch)",
        lowered,
        flags=re.IGNORECASE,
    )
    if class_match and any(term in lowered for term in ("class", "filter", "only", "show")) or (
        class_match and any(term in raw for term in ("클래스", "필터", "보여", "만"))
    ):
        return {"tool": "detail_set_class_filter", "arguments": {"class_name": class_match.group(1)}}

    for method in DETAIL_REDUCTION_METHODS:
        normalized_method = _normalize_reduction_method(method)
        if normalized_method and _normalize_token(normalized_method) in _normalize_token(raw):
            if any(term in lowered for term in ("3d", "reduction", "visualization", "dimension")) or any(
                term in raw for term in ("3D", "축소", "시각화", "차원")
            ):
                return {"tool": "detail_set_3d_reduction_method", "arguments": {"method": normalized_method}}

    for alias in ("smoothgrad", "integrated gradients", "input x gradient"):
        if alias in lowered:
            normalized_method = _normalize_xai_method(alias)
            if normalized_method:
                return {"tool": "detail_set_xai_method", "arguments": {"method": normalized_method}}

    for method in DETAIL_XAI_METHODS:
        if method in lowered and ("xai" in lowered or "openxai" in lowered or "method" in lowered or "방법" in raw):
            return {"tool": "detail_set_xai_method", "arguments": {"method": method}}

    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def route_app_command_with_llm(
    *,
    user_prompt: str,
    runtime_context: str,
    model_dir: str | Path,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any] | None:
    router_prompt = f"""
You are an MCP tool router for a Streamlit manufacturing inspection app.
Return only a JSON object. Do not include Markdown.

Available MCP tools:
- summary_download_report(): use when the user asks to issue, generate, publish, or download the Summary report.
- detail_set_class_filter(class_name): use when the user asks to show/filter a class on the Detail page.
- detail_set_3d_reduction_method(method): use when the user asks to set Detail 3D visualization reduction. method is one of PCA, t-SNE, UMAP.
- detail_set_xai_method(method): use when the user asks to set Detail XAI/OpenXAI method. method is one of grad, sg, itg, ig, lime, shap, control.
- setting_open_db_settings(): use when the user asks to open DB settings.
- setting_save_db_settings(url, key): use only when both Supabase URL and key are explicitly present.

If no app-control tool is appropriate, return {{"tool": null, "arguments": {{}}, "assistant_message": ""}}.
If a tool is appropriate, return {{"tool": "<tool_name>", "arguments": {{...}}, "assistant_message": "<short Korean confirmation>"}}.

[Current App Context]
{runtime_context}

[User Request]
{user_prompt}
""".strip()

    raw_response = generate_response(
        prompt=router_prompt,
        system_prompt="You map user requests to MCP tools and return strict JSON only.",
        max_new_tokens=min(max(int(max_new_tokens), 64), 384),
        temperature=max(0.0, float(temperature)),
        model_dir=model_dir,
    )
    parsed = _extract_json_object(raw_response)
    if not parsed:
        return None

    tool_name = parsed.get("tool")
    if tool_name in (None, "", "null"):
        return None
    if str(tool_name) not in MCP_TOOL_NAMES:
        return None
    arguments = parsed.get("arguments")
    return {
        "tool": str(tool_name),
        "arguments": arguments if isinstance(arguments, dict) else {},
        "assistant_message": str(parsed.get("assistant_message", "") or ""),
    }


def _looks_like_app_control_request(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    hints = (
        "report",
        "download",
        "filter",
        "class",
        "3d",
        "pca",
        "t-sne",
        "tsne",
        "umap",
        "reduction",
        "visualization",
        "xai",
        "openxai",
        "grad",
        "smoothgrad",
        "integrated",
        "method",
        "db",
        "database",
        "supabase",
        "setting",
        "settings",
    )
    korean_hints = ("리포트", "보고서", "다운로드", "발행", "필터", "클래스", "시각화", "축소", "방법", "설정", "세팅")
    return any(hint in lowered for hint in hints) or any(hint in user_prompt for hint in korean_hints)


def route_app_command(
    *,
    user_prompt: str,
    current_page_title: str | None,
    runtime_context: str,
    model_dir: str | Path | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    allow_llm: bool = True,
) -> dict[str, Any] | None:
    deterministic_route = route_app_command_deterministically(user_prompt, current_page_title)
    if deterministic_route is not None:
        return deterministic_route

    if not allow_llm or model_dir is None:
        return None
    if not _looks_like_app_control_request(user_prompt):
        return None

    return route_app_command_with_llm(
        user_prompt=user_prompt,
        runtime_context=runtime_context,
        model_dir=model_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


if __name__ == "__main__":
    if app_mcp is None:
        raise RuntimeError("The `mcp` package is not installed. Install `mcp[cli]` first.")
    app_mcp.run(transport="streamable-http")
