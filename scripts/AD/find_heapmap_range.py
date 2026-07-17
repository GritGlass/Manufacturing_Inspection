"""
memory bank + 정상 이미지 feature들로
  1) anomaly heatmap 색 스케일(vmin/vmax)
  2) Normal/Abnormal 판정 threshold (image_score 기준)
를 모두 percentile 방식으로 계산해 json으로 저장하는 스크립트.

anomaly_score.py의 calibrate_score_range()(patch score용)와
calibrate_image_score_threshold()(image_score용)를 그대로 재사용한다.
여기서 계산한 값들을 pages/2_Analysis.py가 그대로 읽어서 쓰므로, 하드코딩된
threshold 없이 memory bank가 바뀔 때마다 다시 계산해서 갱신하면 된다.

실행 예시:
    python3 scripts/AD/find_heapmap_range.py \
        --memory_bank data/memory_bank/MB.npy \
        --calibrate_glob "data/features/wideresnet_layer3/agg/*.npy" \
        --calibrate_csv data/semiconductor_AD_df.csv \
        --calibrate_label Normal \
        --output data/memory_bank/heatmap_range.json
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from anomaly_score import (
    PatchCoreScorer,
    calibrate_image_score_threshold,
    calibrate_score_range,
    load_memory_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="memory bank + 정상 이미지 feature로 heatmap vmin/vmax와 "
                     "anomaly threshold를 계산해 json으로 저장",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # ---- anomaly_score.py의 입력 인자를 그대로 참고 ----
    parser.add_argument("--memory_bank", type=str, required=True,
                        help="memory bank (M, C) npy 경로")
    parser.add_argument("--calibrate_glob", type=str, required=True,
                        help="정상 이미지의 aggregation feature(.npy) glob 패턴 "
                             "(예: 'data/features/wideresnet_layer3/agg/*.npy')")
    parser.add_argument("--calibrate_sample_n", type=int, default=30,
                        help="캘리브레이션에 쓸 정상 이미지 최대 샘플 수 (기본 30)")
    parser.add_argument("--vmin_percentile", type=float, default=0.5, help="캘리브레이션 vmin percentile")
    parser.add_argument("--vmax_percentile", type=float, default=99.5, help="캘리브레이션 vmax percentile")
    parser.add_argument("--vmax_margin", type=float, default=1.5,
                        help="vmax percentile 값에 곱할 여유 배수 (기본 1.5)")
    parser.add_argument("--k_nn", type=int, default=3,
                        help="PatchCoreScorer 생성용 인자 (vmin/vmax 계산에는 영향 없음)")
    parser.add_argument("--seed", type=int, default=0, help="캘리브레이션 샘플링 시드")

    # ---- Normal/Abnormal threshold (image_score 기준) ----
    parser.add_argument("--calibrate_csv", type=str, default=None,
                        help="정상 이미지의 원본 이미지 경로를 찾기 위한 CSV "
                             "(data_path, labels 컬럼 필요, 예: data/semiconductor_AD_df.csv). "
                             "지정하지 않으면 threshold 계산을 건너뛴다.")
    parser.add_argument("--calibrate_label", type=str, default="Normal",
                        help="--calibrate_csv에서 정상으로 취급할 labels 값 (기본 Normal)")
    parser.add_argument("--threshold_percentile", type=float, default=99.5,
                        help="정상 image_score 분포에서 threshold로 쓸 percentile (기본 99.5)")
    parser.add_argument("--threshold_margin", type=float, default=1.0,
                        help="threshold percentile 값에 곱할 여유 배수 (기본 1.0)")

    # ---- 출력 ----
    parser.add_argument("--output", "-o", type=str, required=True,
                        help='결과 json 저장 경로 (예: {"vmin": 0.0, "vmax": 11.70, "anomaly_threshold": 9.5})')
    return parser.parse_args()


def _resolve_normal_pairs(calibrate_glob: str, csv_path: str, label: str) -> list[tuple[Path, Path]]:
    """calibrate_glob의 feature npy들과 csv의 원본 이미지 경로를 파일명으로 매칭."""
    df = pd.read_csv(csv_path)
    normal_rows = df[df["labels"] == label]
    stem_to_image = {Path(p).stem: Path(p) for p in normal_rows["data_path"]}

    pairs: list[tuple[Path, Path]] = []
    for feature_path in sorted(Path(p) for p in glob.glob(calibrate_glob)):
        image_stem = feature_path.stem.removesuffix("_local")
        image_path = stem_to_image.get(image_stem)
        if image_path is not None and image_path.exists():
            pairs.append((feature_path, image_path))
    return pairs


def main() -> None:
    args = parse_args()

    bank = load_memory_bank(Path(args.memory_bank))
    print(f"memory bank : {tuple(bank.shape)}")

    normal_paths = sorted(Path(p) for p in glob.glob(args.calibrate_glob))
    if not normal_paths:
        raise FileNotFoundError(f"--calibrate_glob에 해당하는 파일이 없습니다: {args.calibrate_glob}")

    scorer = PatchCoreScorer(bank, k_nn=args.k_nn)
    vmin, vmax = calibrate_score_range(
        scorer, normal_paths,
        sample_n=args.calibrate_sample_n,
        vmin_percentile=args.vmin_percentile,
        vmax_percentile=args.vmax_percentile,
        vmax_margin=args.vmax_margin,
        seed=args.seed,
    )

    result = {"vmin": vmin, "vmax": vmax}

    if args.calibrate_csv:
        normal_pairs = _resolve_normal_pairs(args.calibrate_glob, args.calibrate_csv, args.calibrate_label)
        if not normal_pairs:
            raise FileNotFoundError(
                f"--calibrate_csv({args.calibrate_csv})에서 --calibrate_glob과 매칭되는 "
                f"'{args.calibrate_label}' 이미지를 찾지 못했습니다."
            )
        anomaly_threshold = calibrate_image_score_threshold(
            scorer, normal_pairs,
            sample_n=args.calibrate_sample_n,
            percentile=args.threshold_percentile,
            margin=args.threshold_margin,
            seed=args.seed,
        )
        result["anomaly_threshold"] = anomaly_threshold
    else:
        print("[건너뜀] --calibrate_csv 미지정 -> anomaly_threshold 계산 생략")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[저장] {output_path}  {result}")


if __name__ == "__main__":
    main()
