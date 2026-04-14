from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

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


_suppress_transformers_path_alias_warning()
from transformers import AutoModelForImageClassification, set_seed

try:
    from transformers import AutoImageProcessor
except ImportError:  # Compatibility with older Transformers releases.
    from transformers import AutoFeatureExtractor as AutoImageProcessor

from model_train import (
    FolderImageClassificationDataset,
    ImageClassificationCollator,
    build_trainer,
    build_training_arguments,
    build_training_args_payload,
    compute_class_weights,
    ensure_output_dir,
    evaluate_and_save_split,
    replace_training_args_bin_with_json,
    resolve_project_path,
    save_json,
    save_records_csv,
    to_project_relative_path,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally fine-tune the MobileViT classifier with selected Detail image(s).",
    )
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--selected-image", type=Path, required=False)
    parser.add_argument("--selected-images", type=Path, nargs="+", action="append", required=False)
    parser.add_argument("--selected-records-path", type=Path, required=False)
    parser.add_argument("--predicted-label", type=str, default="")
    parser.add_argument("--target-label", type=str, required=False)
    parser.add_argument("--create-new-class", action="store_true", default=False)
    parser.add_argument("--new-class-name", type=str, required=False)
    parser.add_argument("--preprocessing-method", type=str, default="none", required=False)
    parser.add_argument("--manual-target-class-input", type=str, default="", required=False)
    parser.add_argument("--selected-class-option", type=str, default="", required=False)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--repeat-count", type=int, default=16)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR / "model",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{"path": row["path"], "label": row["label"]} for row in reader]


def load_selected_records_manifest(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("selected records manifest 형식이 올바르지 않습니다.")

    selected_records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("selected records manifest 항목 형식이 올바르지 않습니다.")

        resolved_image_path = resolve_project_path(item.get("path"))
        label = str(item.get("label") or "").strip()
        predicted_label = str(item.get("predicted_label") or "").strip() or None
        if resolved_image_path is None:
            raise ValueError("selected records manifest에서 이미지 경로를 해석할 수 없습니다.")
        if not label:
            raise ValueError("selected records manifest에 비어 있는 라벨이 있습니다.")

        selected_records.append(
            {
                "path": resolved_image_path,
                "label": label,
                "predicted_label": predicted_label,
            }
        )
    return selected_records


def build_interactive_config(base_model_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = load_json(base_model_dir / "dataset_config.json")
    dataset_config["model_name"] = to_project_relative_path(base_model_dir)
    dataset_config["num_epochs"] = float(args.epochs)
    dataset_config["learning_rate"] = float(args.learning_rate)
    dataset_config["train_batch_size"] = min(int(dataset_config.get("train_batch_size", 16)), 8)
    dataset_config["eval_batch_size"] = min(int(dataset_config.get("eval_batch_size", 16)), 8)
    dataset_config["num_workers"] = 0
    dataset_config["logging_steps"] = 1
    dataset_config["save_total_limit"] = 1
    dataset_config["gradient_accumulation_steps"] = 1
    return dataset_config


def build_label_mappings(base_model_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    label2id = load_json(base_model_dir / "label2id.json")
    id2label = {int(index): label for label, index in label2id.items()}
    return label2id, id2label


def ensure_selected_labels_in_mappings(
    selected_records: list[dict[str, Any]],
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> list[str]:
    added_labels: list[str] = []
    next_id = (max((int(index) for index in label2id.values()), default=-1) + 1) if label2id else 0

    for selected_record in selected_records:
        selected_label = str(selected_record.get("label") or "").strip()
        if not selected_label:
            raise ValueError("선택 이미지 라벨이 비어 있습니다.")

        selected_record["label"] = selected_label
        if selected_label in label2id:
            continue

        label2id[selected_label] = next_id
        id2label[next_id] = selected_label
        added_labels.append(selected_label)
        next_id += 1

    return added_labels


def build_augmented_train_records(
    base_train_records: list[dict[str, str]],
    selected_records: list[dict[str, Any]],
    repeat_count: int,
) -> list[dict[str, str]]:
    augmented_records = list(base_train_records)
    for selected_record in selected_records:
        selected_image = Path(selected_record["path"])
        target_label = str(selected_record["label"]).strip()
        for _ in range(max(1, int(repeat_count))):
            augmented_records.append({"path": str(selected_image), "label": target_label})
    return augmented_records


def main() -> None:
    args = parse_args()
    args.base_model_dir = resolve_project_path(args.base_model_dir)
    args.output_root = resolve_project_path(args.output_root)
    args.selected_records_path = resolve_project_path(args.selected_records_path)
    if args.base_model_dir is None or args.output_root is None:
        raise ValueError("base-model-dir 또는 output-root 경로를 해석할 수 없습니다.")
    if not args.base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory does not exist: {args.base_model_dir}")
    label2id, id2label = build_label_mappings(args.base_model_dir)
    original_num_labels = len(label2id)

    target_label = None
    if args.create_new_class:
        if not args.new_class_name:
            raise ValueError("--create-new-class 옵션을 사용할 때는 --new-class-name을 반드시 지정해야 합니다.")
        new_class_name = args.new_class_name.replace(" ", "_").replace("-", "_")
        if new_class_name in label2id:
            raise ValueError(f"클래스 '{new_class_name}'은(는) 이미 존재합니다.")
        next_id = max(int(id) for id in label2id.values()) + 1
        label2id[new_class_name] = next_id
        id2label[next_id] = new_class_name
        target_label = new_class_name
        print(f"새로운 클래스 생성: {new_class_name} (ID: {next_id})")
    else:
        if not args.selected_records_path and not args.target_label:
            raise ValueError("--target-label을 반드시 지정해야 합니다.")
        target_label = args.target_label
        if target_label and target_label not in label2id:
            raise ValueError(
                f"Unknown target label '{target_label}'. Available labels: {sorted(label2id.keys())}"
            )

    selected_records: list[dict[str, Any]] = []
    if args.selected_records_path:
        if not args.selected_records_path.exists():
            raise FileNotFoundError(f"Selected records manifest does not exist: {args.selected_records_path}")
        selected_records = load_selected_records_manifest(args.selected_records_path)
    else:
        selected_images: list[Path | None] = []
        if args.selected_images:
            selected_images = [resolve_project_path(path) for group in args.selected_images for path in group]
        elif args.selected_image:
            resolved_selected_image = resolve_project_path(args.selected_image)
            selected_images = [resolved_selected_image]
        else:
            raise ValueError(
                "--selected-records-path, --selected-image 또는 --selected-images 중 하나를 반드시 지정해야 합니다."
            )
        if any(path is None for path in selected_images):
            raise ValueError("선택 이미지 경로를 해석할 수 없습니다.")
        selected_records = [
            {
                "path": path,
                "label": target_label,
                "predicted_label": args.predicted_label.strip() or None,
            }
            for path in selected_images
            if path is not None
        ]

    if not selected_records:
        raise ValueError("학습에 사용할 선택 이미지가 없습니다.")

    selected_images = [Path(record["path"]) for record in selected_records]
    for selected_image in selected_images:
        if not selected_image.exists():
            raise FileNotFoundError(f"Selected image does not exist: {selected_image}")

    added_selected_labels = ensure_selected_labels_in_mappings(selected_records, label2id, id2label)
    for added_label in added_selected_labels:
        print(f"새로운 클래스 생성: {added_label} (selected record label)")

    config = build_interactive_config(args.base_model_dir, args)
    set_seed(int(config.get("seed", 42)))

    train_records = load_records_csv(args.base_model_dir / "train_split.csv")
    valid_records = load_records_csv(args.base_model_dir / "valid_split.csv")
    test_records = load_records_csv(args.base_model_dir / "test_split.csv")
    train_records = build_augmented_train_records(
        base_train_records=train_records,
        selected_records=selected_records,
        repeat_count=args.repeat_count,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ensure_output_dir(args.output_root / run_name)
    print(f"OUTPUT_DIR={to_project_relative_path(output_dir)}")

    image_processor = AutoImageProcessor.from_pretrained(args.base_model_dir)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=len(label2id) != original_num_labels,
        problem_type="single_label_classification",
    )

    train_dataset = FolderImageClassificationDataset(train_records, label2id)
    valid_dataset = FolderImageClassificationDataset(valid_records, label2id)
    test_dataset = FolderImageClassificationDataset(test_records, label2id)
    collator = ImageClassificationCollator(image_processor, preprocessing_method=args.preprocessing_method)
    class_weights = compute_class_weights(train_records, label2id)
    train_batch_size = max(1, int(config["train_batch_size"]))
    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"TRAIN_DEVICE={device_name}")
    print(f"TRAIN_DATASET_SIZE={len(train_dataset)}")
    print(f"VALID_DATASET_SIZE={len(valid_dataset)}")
    print(f"TEST_DATASET_SIZE={len(test_dataset)}")
    print(f"TRAIN_BATCH_SIZE={train_batch_size}")
    print(f"STEPS_PER_EPOCH={steps_per_epoch}")
    if device_name == "cpu":
        print("WARNING: CUDA를 사용할 수 없어 CPU로 학습합니다. 시간이 오래 걸릴 수 있습니다.")

    training_args = build_training_arguments(config, output_dir)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        image_processor=image_processor,
        collator=collator,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        class_weights=class_weights,
    )

    print("TRAINING_STARTED")
    train_result = trainer.train()
    print("TRAINING_FINISHED")
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    training_args_payload = build_training_args_payload(
        training_args=training_args,
        config=config,
        config_path=args.base_model_dir / "dataset_config.json",
    )
    replace_training_args_bin_with_json(output_dir, training_args_payload)

    save_records_csv(train_records, output_dir / "train_split.csv")
    save_records_csv(valid_records, output_dir / "valid_split.csv")
    save_records_csv(test_records, output_dir / "test_split.csv")
    save_json(label2id, output_dir / "label2id.json")
    save_json(config, output_dir / "dataset_config.json")
    save_json(train_result.metrics, output_dir / "train_summary.json")
    save_json(training_args_payload, output_dir / "training_args.json")
    save_json(
        {
            "selected_images": [to_project_relative_path(path) for path in selected_images],
            "selected_records": [
                {
                    "path": to_project_relative_path(record["path"]),
                    "label": str(record["label"]).strip(),
                    "predicted_label": str(record.get("predicted_label") or "").strip() or None,
                }
                for record in selected_records
            ],
            "predicted_label": args.predicted_label,
            "target_label": target_label,
            "create_new_class": args.create_new_class,
            "new_class_name": args.new_class_name,
            "added_selected_labels": added_selected_labels,
            "preprocessing_method": args.preprocessing_method,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "repeat_count": args.repeat_count,
            "base_model_dir": to_project_relative_path(args.base_model_dir),
            "manual_target_class_input": args.manual_target_class_input or None,
            "selected_class_option": args.selected_class_option or None,
            "saved_model_dir": to_project_relative_path(output_dir),
        },
        output_dir / "interactive_request.json",
    )

    valid_metrics = trainer.evaluate(valid_dataset)
    trainer.save_metrics("valid", valid_metrics)
    evaluate_and_save_split(
        trainer=trainer,
        dataset=valid_dataset,
        id2label=id2label,
        output_dir=output_dir,
        split_name="valid",
    )

    test_metrics = trainer.evaluate(test_dataset)
    trainer.save_metrics("test", test_metrics)
    evaluate_and_save_split(
        trainer=trainer,
        dataset=test_dataset,
        id2label=id2label,
        output_dir=output_dir,
        split_name="test",
        save_report=True,
    )

    save_json(
        {
            "train_metrics": train_result.metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        output_dir / "all_results.json",
    )
    print("Interactive fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
