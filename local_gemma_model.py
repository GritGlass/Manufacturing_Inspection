from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any


MODEL_ID = "google/gemma-4-E2B-it"
BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "model"
MODEL_DIR = MODEL_ROOT / "google__gemma-4-E2B-it"
MODEL_SIZE_NOTE = "about 10.3 GB"
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
WEIGHT_FILE_GLOBS = ("*.safetensors", "*.safetensors.index.json")


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


def _import_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        _suppress_transformers_path_alias_warning()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "로컬 Gemma 실행에 필요한 패키지가 없습니다. "
            "먼저 `pip install -r 05_Manufacutre/requirements.txt`를 실행해 주세요."
        ) from exc

    return torch, AutoModelForImageTextToText, AutoProcessor


def are_runtime_dependencies_available() -> tuple[bool, str]:
    try:
        _import_runtime_dependencies()
    except RuntimeError as exc:
        return False, str(exc)
    return True, "로컬 실행 패키지가 준비되어 있습니다."


def _resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    candidate = Path(model_dir).expanduser() if model_dir else MODEL_DIR
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate.resolve()


def _is_valid_model_dir(model_dir: Path) -> bool:
    required_files_ready = all((model_dir / filename).exists() for filename in REQUIRED_MODEL_FILES)
    weight_files_ready = any(next(model_dir.glob(pattern), None) is not None for pattern in WEIGHT_FILE_GLOBS)
    return required_files_ready and weight_files_ready


def list_available_model_dirs() -> list[Path]:
    if not MODEL_ROOT.exists():
        return []
    return sorted(
        path for path in MODEL_ROOT.iterdir() if path.is_dir() and _is_valid_model_dir(path)
    )


def is_model_downloaded(model_dir: str | Path | None = None) -> bool:
    return _is_valid_model_dir(_resolve_model_dir(model_dir))


@lru_cache(maxsize=1)
def load_model(model_dir: str | Path | None = None) -> tuple[Any, Any, Any]:
    torch, AutoModelForImageTextToText, AutoProcessor = _import_runtime_dependencies()
    resolved_model_dir = _resolve_model_dir(model_dir)

    if not is_model_downloaded(resolved_model_dir):
        raise RuntimeError(
            "로컬 모델 파일이 없습니다. "
            f"`{resolved_model_dir}` 경로에 모델을 먼저 준비해 주세요."
        )

    processor = AutoProcessor.from_pretrained(resolved_model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model_dir,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    return torch, processor, model


def unload_model() -> None:
    try:
        torch, _auto_model, _auto_processor = _import_runtime_dependencies()
    except RuntimeError:
        load_model.cache_clear()
        return

    load_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_response_text(processor: Any, output_tokens: Any, input_len: int) -> str:
    raw_text = processor.decode(output_tokens[0][input_len:], skip_special_tokens=False).strip()

    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(raw_text)
        except Exception:
            parsed = None
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("text", "response", "content", "answer", "final"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    cleaned_text = processor.decode(output_tokens[0][input_len:], skip_special_tokens=True).strip()
    return cleaned_text or raw_text


def generate_response(
    prompt: str,
    system_prompt: str,
    image_paths: list[str] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    model_dir: str | Path | None = None,
) -> str:
    if not prompt.strip():
        return ""

    resolved_model_dir = _resolve_model_dir(model_dir)
    torch, processor, model = load_model(str(resolved_model_dir))
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}],
            }
        )

    user_content: list[dict[str, str]] = []
    for image_path in image_paths or []:
        resolved_path = Path(image_path)
        if not resolved_path.is_absolute():
            resolved_path = (BASE_DIR / resolved_path).resolve()
        if resolved_path.exists():
            user_content.append({"type": "image", "path": str(resolved_path.resolve())})
    user_content.append({"type": "text", "text": prompt.strip()})
    messages.append({"role": "user", "content": user_content})

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    generation_temperature = max(0.0, float(temperature))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": generation_temperature > 0.0,
    }
    if generation_temperature > 0.0:
        generation_kwargs["temperature"] = generation_temperature

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    decoded = _extract_response_text(processor, outputs, input_len).strip()
    return decoded or "응답이 비어 있습니다. 프롬프트나 모델 상태를 확인해 주세요."
