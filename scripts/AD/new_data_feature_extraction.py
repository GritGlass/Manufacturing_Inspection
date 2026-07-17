#!/usr/bin/env python3
"""
새 데이터(이미지 + 포인트 클라우드) 폴더를 입력받아
- 이미지: SAM3 ViT encoder feature 추출 (sam3_img_feature_extract.py)
- PLY:    Utonia encoder feature 추출 (utonia_3DPC_feature_extract.py)
를 각각 수행하고 결과 npy를 출력 폴더에 저장.

실행 예:
    python new_data_feature_extraction.py \
        --img_input Data/point_cloud/bad/test_sample/img \
        --pc_input Data/point_cloud/bad/test_sample/ply \
        --img_output Data/output/features/bad/img \
        --pc_output Data/output/features/bad/ply
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


from sam3_img_feature_extract import build_sam3_vit_encoder, preprocess_image
from utonia_3DPC_feature_extract import (
    extract_pc_features,
    load_point_cloud,
    load_utonia_model,
    prepare_utonia_data,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def extract_image_features(
    input_dir: Path,
    output_dir: Path,
    layer: int = 1,
    img_size: int = 640,
    pretrained_weights_path: str | None = None,
) -> None:
    img_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        print(f"[이미지] 입력 폴더에 이미지가 없습니다: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = build_sam3_vit_encoder(pretrained_weights_path=pretrained_weights_path)

    for img_path in img_paths:
        raw_image = preprocess_image(str(img_path), img_size=img_size)
        with torch.no_grad():
            features = encoder(raw_image)

        feat = features[layer].detach().cpu().numpy()
        out_path = output_dir / f"{img_path.stem}.npy"
        np.save(out_path, feat)
        print(f"[이미지] {img_path.name} -> {out_path}  shape={feat.shape}")


def extract_pc_features_batch(
    input_dir: Path,
    output_dir: Path,
    layer: int = 2,
    checkpoint: str | None = None,
    no_flash: bool = False,
    device_str: str = "cuda",
) -> None:
    ply_paths = sorted(input_dir.glob("*.ply"))
    if not ply_paths:
        print(f"[PC] 입력 폴더에 ply 파일이 없습니다: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_utonia_model(checkpoint or "utonia", no_flash, device)

    for ply_path in ply_paths:
        points_xyz = load_point_cloud(str(ply_path))
        data = prepare_utonia_data(points_xyz)
        feat = extract_pc_features(model, data, layer, device)

        out_path = output_dir / f"{ply_path.stem}.npy"
        np.save(out_path, feat)
        print(f"[PC] {ply_path.name} -> {out_path}  shape={feat.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이미지/PLY feature 추출 후 npy로 저장")
    parser.add_argument("--img_input", type=str, required=True, help="입력 이미지 폴더 경로")
    parser.add_argument("--pc_input", type=str, required=True, help="입력 PLY 폴더 경로")
    parser.add_argument("--img_output", type=str, required=True, help="이미지 feature npy 출력 폴더 경로")
    parser.add_argument("--pc_output", type=str, required=True, help="PC feature npy 출력 폴더 경로")
    parser.add_argument("--img_layer", type=int, default=1, help="SAM3 encoder feature list 중 저장할 layer index")
    parser.add_argument("--pc_layer", type=int, default=2, help="Utonia encoder feature layer index")
    parser.add_argument("--sam3_checkpoint", type=str, default=None, help="SAM3 ViT encoder pretrained weight 경로")
    parser.add_argument("--utonia_checkpoint", type=str, default=None, help="Utonia 모델 checkpoint 이름")
    parser.add_argument("--no_flash", action="store_true", help="Utonia flash attention 비활성화")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="PC feature 추출에 사용할 device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extract_image_features(
        input_dir=Path(args.img_input),
        output_dir=Path(args.img_output),
        layer=args.img_layer,
        pretrained_weights_path=args.sam3_checkpoint,
    )

    extract_pc_features_batch(
        input_dir=Path(args.pc_input),
        output_dir=Path(args.pc_output),
        layer=args.pc_layer,
        checkpoint=args.utonia_checkpoint,
        no_flash=args.no_flash,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
