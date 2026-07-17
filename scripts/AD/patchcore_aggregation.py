"""
PatchCore-style local neighborhood aggregation.

공식 구현(amazon-research/patchcore-inspection)의 PatchMaker + MeanMapper 로직을 따름:
  1) Unfold로 각 위치의 p x p 이웃 feature를 추출 (stride=1, padding=p//2 -> 해상도 유지)
  2) 각 위치의 (C * p * p) 벡터를 F.adaptive_avg_pool1d로 목표 차원 d로 압축

d == C 로 두면 채널별 3x3 평균 = 일반 spatial average pooling과 수치적으로 동일.
d != C 로 두면 채널 방향으로도 pooling이 일어남 (이게 일반 AvgPool2d와의 차이).

#단일 (default: LocalNeighborhoodAggregation만 수행)
cd /mnt/data/github/AI-portfolio/3D-Anomaly-Detection
python3 Scripts/patchcore_aggregation.py \
  --input1 Data/features/masked_img_view1/raw \
  --output Data/features/masked_img_view1/agg

  #hierarchy 까지 수행 (--hierarchy 옵션 + --input2 둘 다 필요)
  python3 Scripts/patchcore_aggregation.py \
  --input1 Data/features/masked_img_view1/raw \
  --input2 <layer31_디렉토리> \
  --hierarchy \
  --output Data/features/masked_img_view1/agg

"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class LocalNeighborhoodAggregation(torch.nn.Module):
    """단일 feature map에 대한 locally aware patch feature 생성.

    Args:
        patchsize: 이웃 크기 p (논문 기본값 3)
        stride: 위치 샘플링 stride (논문 기본값 1)
        output_dim: 목표 feature 차원 d. None이면 입력 채널 수 C를 그대로 사용.
    """

    def __init__(self, patchsize: int = 3, stride: int = 1, output_dim: int | None = None):
        super().__init__()
        self.patchsize = patchsize
        self.stride = stride
        self.output_dim = output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] backbone feature map
        Returns:
            [B, d, H', W']  (stride=1이면 H'=H, W'=W)
        """
        B, C, H, W = features.shape
        p = self.patchsize
        d = self.output_dim or C

        # 1) p x p 이웃 추출: [B, C*p*p, L], L = H' * W'
        padding = (p - 1) // 2
        unfolded = F.unfold(features, kernel_size=p, stride=self.stride, padding=padding)
        H_out = (H + 2 * padding - p) // self.stride + 1
        W_out = (W + 2 * padding - p) // self.stride + 1
        L = unfolded.shape[-1]

        # 2) 위치별 (C*p*p) 벡터로 정리: [B*L, 1, C*p*p]
        unfolded = unfolded.permute(0, 2, 1).reshape(B * L, 1, C * p * p)

        # 3) adaptive avg pooling으로 d차원으로 압축: [B*L, d]
        aggregated = F.adaptive_avg_pool1d(unfolded, d).squeeze(1)

        # 4) feature map 형태로 복원: [B, d, H', W']
        return aggregated.reshape(B, H_out, W_out, d).permute(0, 3, 1, 2)


class MultiHierarchyAggregation(torch.nn.Module):
    """두 hierarchy의 feature를 PatchCore 방식으로 결합.

    각 레벨에 이웃 aggregation 적용 -> 깊은 레벨을 얕은 레벨 해상도로 bilinear upsample
    -> 채널 concat -> 다시 adaptive pooling으로 최종 차원 d로 통합.
    """

    def __init__(self, patchsize: int = 3, per_level_dim: int = 1024, target_dim: int = 1024):
        super().__init__()
        self.agg = LocalNeighborhoodAggregation(patchsize=patchsize, stride=1,
                                                output_dim=per_level_dim)
        self.target_dim = target_dim

    def forward(self, feat_low: torch.Tensor, feat_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_low:  얕은 레벨 [B, C1, H1, W1]  (고해상도)
            feat_high: 깊은 레벨 [B, C2, H2, W2]  (저해상도)
        Returns:
            [B, target_dim, H1, W1]
        """
        f1 = self.agg(feat_low)                          # [B, d1, H1, W1]
        f2 = self.agg(feat_high)                         # [B, d1, H2, W2]
        f2 = F.interpolate(f2, size=f1.shape[-2:],
                           mode="bilinear", align_corners=False)

        combined = torch.cat([f1, f2], dim=1)            # [B, 2*d1, H1, W1]

        # 최종 통합 pooling (공식 구현의 Aggregator에 해당)
        B, C, H, W = combined.shape
        combined = combined.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        combined = F.adaptive_avg_pool1d(combined, self.target_dim).squeeze(1)
        return combined.reshape(B, H, W, self.target_dim).permute(0, 3, 1, 2)


def _load_feature(path: Path) -> torch.Tensor:
    array = np.load(path)
    return torch.from_numpy(array).float()


def run_aggregation(input_dir: str, output_dir: str, input_dir2: str | None = None,
                     use_hierarchy: bool = False, patchsize: int = 3,
                     output_dim: int | None = None, per_level_dim: int | None = None,
                     target_dim: int | None = None) -> Path:
    """input_dir의 .npy feature들에 aggregation을 적용해 output_dir에 저장.

    - use_hierarchy=False (default): 각 파일에 LocalNeighborhoodAggregation만 적용
      (예: SAM3 layer15).
    - use_hierarchy=True: input_dir2(다른 hierarchy 레벨, 동일 파일명 필요)가 반드시 있어야 하며,
      위 결과에 더해 두 레벨을 결합하는 MultiHierarchyAggregation을 추가로 수행한다.

    두 경우 모두 처리한 모든 샘플의 input/output shape 메타 정보를 output_dir 아래
    단일 JSON 파일(aggregation_metadata.json)로 저장한다.
    """
    if use_hierarchy and not input_dir2:
        raise ValueError("--hierarchy를 사용하려면 --input2가 필요합니다.")

    input_path = Path(input_dir)
    input_path2 = Path(input_dir2) if input_dir2 else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    local_agg = LocalNeighborhoodAggregation(patchsize=patchsize, output_dim=output_dim)
    multi_agg = None
    if use_hierarchy:
        sample_files = sorted(input_path.glob("*.npy"))
        if not sample_files:
            raise FileNotFoundError(f"{input_path}에 .npy 파일이 없습니다.")
        primary_channels = _load_feature(sample_files[0]).shape[1]
        resolved_per_level_dim = per_level_dim or primary_channels
        resolved_target_dim = target_dim or resolved_per_level_dim
        multi_agg = MultiHierarchyAggregation(patchsize=patchsize,
                                              per_level_dim=resolved_per_level_dim,
                                              target_dim=resolved_target_dim)

    metadata = {
        "config": {
            "mode": "multi_hierarchy" if use_hierarchy else "local_neighborhood",
            "input_dir": str(input_path),
            "input_dir2": str(input_path2) if input_path2 else None,
            "output_dir": str(output_path),
            "patchsize": patchsize,
            "output_dim": output_dim,
            "per_level_dim": multi_agg.agg.output_dim if multi_agg else None,
            "target_dim": multi_agg.target_dim if multi_agg else None,
        },
        "samples": {},
    }

    files = sorted(input_path.glob("*.npy"))
    for f in files:
        name = f.stem
        feat1 = _load_feature(f)
        sample_meta = {"input_shape": list(feat1.shape)}

        if not use_hierarchy:
            with torch.no_grad():
                local_out = local_agg(feat1)
            local_file = f"{name}_local.npy"
            np.save(output_path / local_file, local_out.numpy())
            sample_meta["local_output_shape"] = list(local_out.shape)
            sample_meta["local_output_file"] = local_file
        else:
            assert input_path2 is not None
            f2 = input_path2 / f.name
            if not f2.exists():
                sample_meta["multi_error"] = f"{f2}에 대응 파일이 없어 multi-hierarchy aggregation을 건너뜀"
            else:
                feat2 = _load_feature(f2)
                sample_meta["input2_shape"] = list(feat2.shape)
                with torch.no_grad():
                    multi_out = multi_agg(feat1, feat2)
                multi_file = f"{name}_multi.npy"
                np.save(output_path / multi_file, multi_out.numpy())
                sample_meta["multi_output_shape"] = list(multi_out.shape)
                sample_meta["multi_output_file"] = multi_file

        metadata["samples"][name] = sample_meta

    meta_path = output_path / "aggregation_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    return meta_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input1", "-i1", required=True,
                        help="1차 feature .npy 디렉토리 (예: SAM3 layer15). 항상 LocalNeighborhoodAggregation 대상")
    parser.add_argument("--input2", "-i2", default=None,
                        help="2차 feature .npy 디렉토리 (예: layer31). --hierarchy와 함께 사용")
    parser.add_argument("--hierarchy", action="store_true",
                        help="설정 시 --input2를 이용해 MultiHierarchyAggregation을 추가로 수행 (default: 미수행)")
    parser.add_argument("--output", "-o", required=True, help="결과 npy와 메타 json을 저장할 디렉토리")
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--output-dim", type=int, default=None,
                        help="LocalNeighborhoodAggregation 목표 채널 d. 미지정 시 입력 채널 수 유지")
    parser.add_argument("--per-level-dim", type=int, default=None,
                        help="MultiHierarchyAggregation에서 레벨별 압축 차원. 미지정 시 첫 입력의 채널 수 사용")
    parser.add_argument("--target-dim", type=int, default=None,
                        help="MultiHierarchyAggregation 최종 출력 차원. 미지정 시 per-level-dim과 동일")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    meta_path = run_aggregation(
        input_dir=args.input1,
        input_dir2=args.input2,
        use_hierarchy=args.hierarchy,
        output_dir=args.output,
        patchsize=args.patchsize,
        output_dim=args.output_dim,
        per_level_dim=args.per_level_dim,
        target_dim=args.target_dim,
    )
    print(f"메타 정보 저장 완료: {meta_path}")