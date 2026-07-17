"""
torchvision의 wide_resnet101_2를 사용해서
이미지 폴더의 원본 이미지 -> feature map(.npy)을 추출하는 코드

- point cloud/projection/masking 없이 원본 이미지를 그대로 resize하여 인코더에 입력
- 네트워크를 초반/중반/끝 5구간으로 나눠 각 구간 끝에서 나오는 5개의 중간 feature 중
  --layer(1-indexed, 기본 3)번째 feature를 저장
  (1->stem, 2->layer1, 3->layer2, 4->layer3, 5->layer4)

실행 예시:
  # 1) CSV로 이미지 목록 지정
  python Scripts/wide_resnet_img_feature_extract.py \\
      --csv_path data/semiconductor_AD_df.csv \\
      --label_filter Normal \\
      --output_dir Data/output/features/good/wide_resnet_img_1008 \\
      --layer 3

  # 2) 이미지 디렉토리를 그대로 사용 (하위 폴더 포함, 모든 이미지 파일)
  python Scripts/wide_resnet_img_feature_extract.py \\
      --img_dir data/images/good \\
      --output_dir Data/output/features/good/wide_resnet_img_1008 \\
      --layer 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import Wide_ResNet101_2_Weights, wide_resnet101_2


class WideResNetFeatureExtractor(nn.Module):
    """wide_resnet101_2를 stem/layer1~4 5구간으로 나눠 각 구간의 출력을 반환."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """5구간(stem, layer1, layer2, layer3, layer4)의 출력 feature map을 순서대로 반환."""
        outputs = []
        x = self.stem(x)
        outputs.append(x)
        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)
        return outputs


def build_wide_resnet_encoder(
    pretrained_weights_path: str | None = None,
    imagenet_pretrained: bool = True,
) -> WideResNetFeatureExtractor:
    """
    wide_resnet101_2 image encoder를 생성.

    Args:
        pretrained_weights_path: 로컬 checkpoint(.pt) 경로. 지정하면 torchvision의
                                  ImageNet weight 대신 이 checkpoint를 로드한다.
        imagenet_pretrained: pretrained_weights_path가 None일 때 torchvision의
                              ImageNet 사전학습 weight를 사용할지 여부.
                              False면 random 초기화 weight 사용 (구조 테스트용).
    """
    weights = Wide_ResNet101_2_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
    backbone = wide_resnet101_2(weights=None if pretrained_weights_path is not None else weights)

    if pretrained_weights_path is not None:
        with open(pretrained_weights_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu")
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]

        missing, unexpected = backbone.load_state_dict(ckpt, strict=False)
        print(f"로드 완료. missing keys: {len(missing)}개, unexpected keys: {len(unexpected)}개")

    encoder = WideResNetFeatureExtractor(backbone)
    encoder.eval()
    return encoder


def preprocess_image_array(image_bgr: np.ndarray, img_size: int = 640) -> torch.Tensor:
    """BGR 이미지 배열을 wide_resnet101_2 입력 형식으로 전처리."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_np = img_rgb.astype(np.float32) / 255.0  # 0~255 -> 0~1 정규화

    # ImageNet 표준 mean/std로 정규화 (wide_resnet101_2가 ImageNet으로 사전학습됨)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std

    # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor


def preprocess_image(image_path: str, img_size: int = 640) -> torch.Tensor:
    """이미지 파일 경로를 읽어 wide_resnet101_2 입력 형식으로 전처리."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    return preprocess_image_array(image_bgr, img_size=img_size)


def collect_img_paths_from_csv(
    csv_path,
    label_filter: str,
    path_col: str = "data_path",
    label_col: str = "labels",
) -> list[Path]:
    """csv_path를 읽어 label_col == label_filter인 행만 남긴 뒤 path_col 경로 목록을 반환."""
    df = pd.read_csv(csv_path)
    if label_col not in df.columns or path_col not in df.columns:
        raise ValueError(
            f"CSV에 필요한 컬럼이 없습니다. 필요: [{path_col}, {label_col}], 실제: {list(df.columns)}"
        )
    filtered = df[df[label_col] == label_filter]
    img_paths = sorted(Path(p) for p in filtered[path_col])
    return img_paths


DEFAULT_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_img_paths_from_dir(
    img_dir,
    extensions: tuple[str, ...] = DEFAULT_IMG_EXTENSIONS,
) -> list[Path]:
    """img_dir 이하(하위 폴더 포함)에서 확장자가 extensions인 이미지 파일을 모두 수집."""
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(f"이미지 디렉토리를 찾을 수 없습니다: {img_dir}")
    img_paths = sorted(
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not img_paths:
        raise FileNotFoundError(f"{img_dir}에서 이미지 파일을 찾지 못했습니다 (확장자: {extensions})")
    return img_paths


def process_images(
    img_paths: list[Path],
    output_dir,
    encoder,
    img_size: int = 640,
    layer: int = 3,
):
    """
    img_paths 이미지들을 원본 그대로 resize하여 wide_resnet101_2 feature를 추출하고
    {output_dir}/{원본 파일명 stem}.npy로 저장한다.

    layer: stem/layer1~4 5구간 중 몇 번째(1-indexed) feature를 저장할지.
           기본값 3 -> layer2의 feature.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not img_paths:
        raise FileNotFoundError("처리할 이미지가 없습니다.")

    print(f"[대상] {len(img_paths)}개 이미지")
    print(f"[설정] img_size={img_size}, layer={layer}")

    for img_path in tqdm(img_paths, desc="wide_resnet101_2 feature 추출", unit="img"):
        input_tensor = preprocess_image(str(img_path), img_size=img_size)

        with torch.no_grad():
            features = encoder(input_tensor)

        if not (1 <= layer <= len(features)):
            raise ValueError(f"layer={layer}는 유효 범위(1~{len(features)})를 벗어났습니다.")

        feat = features[layer - 1].numpy()
        save_path = output_dir / f"{img_path.stem}.npy"
        np.save(save_path, feat)

    print(f"[완료] {len(img_paths)}개 feature -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="wide_resnet101_2 image encoder로 이미지 폴더 -> feature npy 추출",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--csv_path", default=None,
        help="이미지 경로 목록을 담은 CSV 파일 경로. --label_col == --label_filter인 행의 "
             "--path_col 컬럼 값을 이미지 경로로 사용. --img_dir과 동시 사용 불가.",
    )
    input_group.add_argument(
        "--img_dir", default=None,
        help="이미지가 있는 디렉토리 경로. 지정하면 하위 폴더를 포함해 모든 이미지 파일을 사용한다 "
             "(--path_col/--label_col/--label_filter는 무시됨). --csv_path와 동시 사용 불가.",
    )
    parser.add_argument("--path_col", default="data_path", help="CSV에서 이미지 경로가 담긴 컬럼명 (--csv_path 사용 시)")
    parser.add_argument("--label_col", default="labels", help="CSV에서 라벨이 담긴 컬럼명 (--csv_path 사용 시)")
    parser.add_argument(
        "--label_filter", default="Normal",
        help="--csv_path 사용 시, --label_col 값이 이 값과 일치하는 행만 사용 (기본값 Normal)",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/data/github/AI-portfolio/3D-Anomaly-Detection/Data/output/features/good/wide_resnet_img_1008",
        help="feature npy 저장 폴더 경로",
    )
    parser.add_argument(
        "--layer", type=int, default=3,
        help="저장할 중간 feature 순번(1-indexed). "
             "1->stem, 2->layer1, 3->layer2, 4->layer3, 5->layer4 (기본값 3).",
    )
    parser.add_argument("--img_size", type=int, default=640, help="wide_resnet101_2 인코더 입력 이미지 크기")
    parser.add_argument(
        "--pretrained_weights_path", default=None,
        help="로컬 checkpoint(.pt) 경로. 지정하지 않으면 torchvision ImageNet weight 사용",
    )
    parser.add_argument(
        "--no_imagenet_pretrained", action="store_true",
        help="지정하면 ImageNet 사전학습 weight 대신 random 초기화 weight 사용 (구조 테스트용)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    encoder = build_wide_resnet_encoder(
        pretrained_weights_path=args.pretrained_weights_path,
        imagenet_pretrained=not args.no_imagenet_pretrained,
    )

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"wide_resnet101_2 encoder 전체 파라미터 수: {total_params:,}")
    print()

    if args.csv_path:
        img_paths = collect_img_paths_from_csv(
            args.csv_path,
            label_filter=args.label_filter,
            path_col=args.path_col,
            label_col=args.label_col,
        )
        print(f"[CSV] {args.csv_path}  ({args.label_col}=={args.label_filter})  ->  {len(img_paths)}개 이미지")
    else:
        img_paths = collect_img_paths_from_dir(args.img_dir)
        print(f"[IMG_DIR] {args.img_dir}  ->  {len(img_paths)}개 이미지")

    process_images(
        img_paths,
        args.output_dir,
        encoder,
        img_size=args.img_size,
        layer=args.layer,
    )
