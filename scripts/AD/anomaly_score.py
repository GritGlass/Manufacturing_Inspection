"""
PatchCore anomaly scoring (feature extraction 이후 단계만).

입력:
    test_feat   : (P, C) 또는 (B, P, C)   -- 추출 완료된 test patch feature
                  P = Hf * Wf (예: 72*72 = 5184), C = feature 차원 (예: 1024)
    memory_bank : (M, C)                  -- 구축 완료된 coreset memory bank

출력:
    patch_scores : (B, Hf, Wf)  patch별 anomaly score (= 1-NN 거리)
    image_scores : (B,)         softmax re-weighting 적용된 이미지 score (Eq. 7)
    anomaly_map  : (B, H, W)    원본 해상도로 upsample + Gaussian blur

사용 예:
    bank = torch.load("memory_bank.pt")["memory_bank"].to(DEVICE)
    scorer = PatchCoreScorer(bank, k_nn=3)
    result = scorer.score(test_feat, grid=(72, 72), out_size=(1008, 1008))

    python Scripts/anomaly_score.py \
        --memory_bank /mnt/data/github/Manufacutring_Inspection/data/memory_bank/MB.npy \
        --test_image /mnt/data/github/Manufacutring_Inspection/data/images/image_Normal_34014_png_jpg.rf.99baa8fde27fdf9f55dc4cb74ff2a259.jpg \
        --output_dir /mnt/data/github/Manufacutring_Inspection/outputs/AD

    (feature는 --test_feature로 넘기지 않고, data/features/app/agg/{test_image의 stem}_local.npy
     에서 자동으로 찾는다.)

    
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_MEMORY_BANK = PROJECT_DIR / "Data" / "memory_bank" / "masked_img_view1" / "MB.npy"
DEFAULT_TEST_IMAGE = (
    PROJECT_DIR / "Data" / "features" / "test" / "bad" / "masked_img"
    / "masked_rubbish_bin_01_bad_01_0.png"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Data" / "features" / "test" / "bad" / "output"
DEFAULT_HEATMAP_RANGE = Path("/mnt/data/github/Manufacutring_Inspection/data/memory_bank/heatmap_range.json")
# memory bank와 grid가 항상 일치하는 최신 feature 저장소 (pages/2_Analysis.py가 채움).
# test_image의 feature는 항상 여기서 찾는다 (다른 img_size로 추출된 stale feature 혼입 방지).
APP_AGG_DIR = Path("/mnt/data/github/Manufacutring_Inspection/data/features/app/agg")


class PatchCoreScorer:
    def __init__(self, memory_bank: torch.Tensor, k_nn: int = 3,
                 blur_sigma: float = 4.0, chunk: int = 2048):
        """
        Args:
            memory_bank: (M, C) coreset memory bank
            k_nn: re-weighting에 사용할 bank 내 neighbor 수 (논문의 b)
            blur_sigma: anomaly map Gaussian smoothing sigma
            chunk: cdist 메모리 절약용 chunk 크기
        """
        self.bank = memory_bank.to(DEVICE)          # (M, C)
        self.k_nn = min(k_nn, self.bank.shape[0])
        self.blur_sigma = blur_sigma
        self.chunk = chunk

    # ------------------------------------------------------------------
    # Step 1. patch별 1-NN 거리 (= patch anomaly score)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _nn_distance(self, flat: torch.Tensor):
        """
        flat: (P_total, C)
        Returns:
            dmin (P_total,)  각 patch에서 bank까지의 최소 거리
            imin (P_total,)  그 최근접 bank 점의 index (re-weighting에 필요)
        """
        dists, idxs = [], []
        for i in range(0, flat.shape[0], self.chunk):
            d = torch.cdist(flat[i:i + self.chunk], self.bank)   # (chunk, M)
            dmin, imin = d.min(dim=1)
            dists.append(dmin)
            idxs.append(imin)
        return torch.cat(dists), torch.cat(idxs)

    # ------------------------------------------------------------------
    # Step 2. 이미지 score: softmax re-weighting (논문 Eq. 7)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _image_score(self, feat_b: torch.Tensor,
                     patch_dist_b: torch.Tensor,
                     nn_idx_b: torch.Tensor) -> torch.Tensor:
        """
        feat_b       : (P, C)  한 이미지의 patch feature
        patch_dist_b : (P,)    patch별 1-NN 거리
        nn_idx_b     : (P,)    patch별 최근접 bank index

        s = (1 - exp(s*) / sum_{m in N_b(m*)} exp(d(x*, m))) * s*

        직관: 가장 anomalous한 patch x*의 최근접 bank 점 m*가
        bank 내에서 '외딴 점'이면(이웃이 멀면) 확신을 낮추고,
        '밀집 영역의 점'이면 s*를 거의 그대로 사용.
        """
        # (1) 가장 anomalous한 test patch x* 선택
        p_star = torch.argmax(patch_dist_b)
        s_star = patch_dist_b[p_star]                # 최대 patch 거리
        x_star = feat_b[p_star:p_star + 1]           # (1, C)
        m_star = nn_idx_b[p_star]                    # x*의 최근접 bank 점

        # (2) m*의 bank 내 b-nearest neighbors N_b(m*)
        d_bank = torch.cdist(self.bank[m_star:m_star + 1], self.bank).squeeze(0)
        nb_idx = torch.topk(d_bank, k=self.k_nn, largest=False).indices

        # (3) x*에서 그 neighbor들까지의 거리로 softmax 가중
        d_nb = torch.cdist(x_star, self.bank[nb_idx]).squeeze(0)   # (k_nn,)

        # 수치 안정화: 최대값 빼고 exp (오버플로 방지, 비율은 동일)
        m = torch.maximum(s_star, d_nb.max())
        w = 1 - torch.exp(s_star - m) / torch.exp(d_nb - m).sum()
        return w * s_star

    # ------------------------------------------------------------------
    # Step 3. anomaly map: 72x72 -> 원본 해상도 + Gaussian blur
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _make_anomaly_map(self, patch_scores: torch.Tensor,
                          out_size: tuple[int, int]) -> torch.Tensor:
        amap = patch_scores.unsqueeze(1)                         # (B,1,Hf,Wf)
        amap = F.interpolate(amap, size=out_size, mode="bilinear",
                             align_corners=False).squeeze(1)     # (B,H,W)

        sigma = self.blur_sigma
        ks = int(4 * sigma + 1) | 1
        coords = torch.arange(ks, device=amap.device) - ks // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, -1)
        x = amap.unsqueeze(1)
        x = F.conv2d(x, g.unsqueeze(2), padding=(0, ks // 2))
        x = F.conv2d(x, g.unsqueeze(3), padding=(ks // 2, 0))
        return x.squeeze(1)

    # ------------------------------------------------------------------
    # 전체 scoring 파이프라인
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(self, test_feat: torch.Tensor, grid: tuple[int, int],
              out_size: tuple[int, int] | None = None,
              fg_mask: torch.Tensor | None = None) -> dict:
        """
        Args:
            test_feat: (P, C) 단일 이미지 또는 (B, P, C) 배치
            grid: (Hf, Wf), 예: (72, 72)
            out_size: anomaly map 출력 해상도 (None이면 map 생략)
            fg_mask: (B, Hf, Wf) bool, True=foreground.
                     지정 시 background patch score를 0으로 마스킹
                     (image score의 argmax에서도 제외됨)
        Returns:
            dict(patch_scores, image_scores[, anomaly_map])
        """
        if test_feat.dim() == 2:
            test_feat = test_feat.unsqueeze(0)       # (1, P, C)
        test_feat = test_feat.to(DEVICE)
        B, P, C = test_feat.shape
        Hf, Wf = grid
        assert P == Hf * Wf, f"P={P} != Hf*Wf={Hf * Wf}"

        # ---- Step 1: patch scores ----
        flat = test_feat.reshape(-1, C)              # (B*P, C)
        dmin, imin = self._nn_distance(flat)
        patch_dist = dmin.reshape(B, P)              # (B, P)
        nn_idx = imin.reshape(B, P)

        # ---- foreground masking (선택) ----
        if fg_mask is not None:
            mask_flat = fg_mask.reshape(B, P).to(DEVICE)
            patch_dist = patch_dist * mask_flat      # bg patch -> 0

        # ---- Step 2: image scores (re-weighting) ----
        image_scores = torch.empty(B, device=DEVICE)
        for b in range(B):
            image_scores[b] = self._image_score(
                test_feat[b], patch_dist[b], nn_idx[b])

        out = {
            "patch_scores": patch_dist.reshape(B, Hf, Wf).cpu(),
            "image_scores": image_scores.cpu(),
        }

        # ---- Step 3: anomaly map (선택) ----
        if out_size is not None:
            out["anomaly_map"] = self._make_anomaly_map(
                patch_dist.reshape(B, Hf, Wf), out_size).cpu()
        return out


# ----------------------------------------------------------------------------
# 입출력 helper
# ----------------------------------------------------------------------------
def load_memory_bank(path: Path) -> torch.Tensor:
    """(M, C) memory bank npy -> tensor"""
    bank = np.load(path)
    return torch.from_numpy(bank).float()


def load_test_feature(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """(1, C, Hf, Wf) test feature npy -> (1, Hf*Wf, C) tensor, (Hf, Wf)"""
    feat = np.load(path)
    if feat.ndim != 4 or feat.shape[0] != 1:
        raise ValueError(f"예상하지 못한 shape: {feat.shape}  (기대: (1, C, Hf, Wf))")
    _, c, hf, wf = feat.shape
    flat = feat[0].transpose(1, 2, 0).reshape(1, hf * wf, c)   # (1, P, C)
    return torch.from_numpy(np.ascontiguousarray(flat)).float(), (hf, wf)


def foreground_mask_from_image(image: Image.Image, threshold: int = 10) -> np.ndarray:
    """masked_img(배경=검정)에서 foreground mask 추출.

    Returns:
        (H, W) bool array, True=foreground (밝기 > threshold)
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return gray > threshold


def calibrate_score_range(
    scorer: "PatchCoreScorer",
    normal_feature_paths: list[Path],
    sample_n: int = 30,
    vmin_percentile: float = 0.5,
    vmax_percentile: float = 99.5,
    vmax_margin: float = 1.5,
    seed: int = 0,
) -> tuple[float, float]:
    """정상 이미지들의 patch score 분포로 히트맵에 쓸 고정 (vmin, vmax)를 계산.

    normal_feature_paths 중 최대 sample_n개를 뽑아 patch별 1-NN 거리를 모은 뒤
    그 분포의 vmin_percentile 지점을 vmin으로, vmax_percentile 지점에
    vmax_margin을 곱한 값을 vmax로 쓴다.

    margin이 필요한 이유: 정상 이미지 한 장 안에서도 patch score는 자체적으로
    상당한 편차가 있어(예: 중앙값이 이미 자기 최댓값의 70%대), vmax를 정상
    데이터의 상한(vmax_percentile) 그대로 쓰면 그 자연스러운 편차만으로도
    color scale의 위쪽(노랑/빨강)을 채워버려 정상 이미지가 빨갛게 보인다.
    vmax에 여유를 둬야 정상 patch 전체가 scale 하단(파랑/초록)에 눌리고,
    실제로 그 상한을 벗어나는 결함 patch만 빨갛게 튀어나온다.
    """
    if not normal_feature_paths:
        raise ValueError("calibrate_score_range: normal_feature_paths가 비어 있습니다.")

    rng = np.random.default_rng(seed)
    paths = list(normal_feature_paths)
    if len(paths) > sample_n:
        idx = rng.choice(len(paths), size=sample_n, replace=False)
        paths = [paths[i] for i in idx]

    all_scores = []
    for p in paths:
        test_feat, _ = load_test_feature(p)
        flat = test_feat.reshape(-1, test_feat.shape[-1]).to(DEVICE)
        dmin, _ = scorer._nn_distance(flat)
        all_scores.append(dmin.cpu().numpy())

    scores = np.concatenate(all_scores)
    vmin = float(np.percentile(scores, vmin_percentile))
    vmax = float(np.percentile(scores, vmax_percentile)) * vmax_margin
    print(f"[캘리브레이션] 정상 이미지 {len(paths)}장, patch {scores.size}개 "
          f"-> vmin={vmin:.4f} (p{vmin_percentile}), "
          f"vmax={vmax:.4f} (p{vmax_percentile} x {vmax_margin})")
    return vmin, vmax


def calibrate_image_score_threshold(
    scorer: "PatchCoreScorer",
    normal_pairs: list[tuple[Path, Path]],
    sample_n: int = 30,
    percentile: float = 99.5,
    margin: float = 1.0,
    seed: int = 0,
) -> float:
    """정상 이미지들의 image_score 분포로 Normal/Abnormal 판정 threshold를 계산.

    calibrate_score_range()와 같은 percentile 방식이지만, 히트맵용 patch score가
    아니라 실제 화면에서 Normal/Abnormal을 가르는 데 쓰이는 image_score(재가중
    적용된 이미지 단위 score, scorer.score()의 결과)의 분포에 적용한다.
    fg_mask도 실제 스코어링과 동일하게 적용해야 분포가 일치한다.

    normal_pairs: (feature_path, image_path) 쌍의 리스트. image_path는 fg_mask
                  계산에 쓰인다.
    """
    if not normal_pairs:
        raise ValueError("calibrate_image_score_threshold: normal_pairs가 비어 있습니다.")

    rng = np.random.default_rng(seed)
    pairs = list(normal_pairs)
    if len(pairs) > sample_n:
        idx = rng.choice(len(pairs), size=sample_n, replace=False)
        pairs = [pairs[i] for i in idx]

    image_scores = []
    for feature_path, image_path in pairs:
        test_feat, grid = load_test_feature(feature_path)
        image = Image.open(image_path)
        fg_mask_full = foreground_mask_from_image(image)
        fg_mask_grid = F.adaptive_max_pool2d(
            torch.from_numpy(fg_mask_full).float()[None, None], output_size=grid
        ).squeeze(1).bool()
        result = scorer.score(test_feat, grid=grid, fg_mask=fg_mask_grid)
        image_scores.append(float(result["image_scores"][0]))

    scores = np.array(image_scores)
    threshold = float(np.percentile(scores, percentile)) * margin
    print(f"[캘리브레이션] 정상 이미지 {len(pairs)}장 image_score 분포 "
          f"-> threshold={threshold:.4f} (p{percentile} x {margin})")
    return threshold


def save_result(image_path: Path, anomaly_map: np.ndarray, image_score: float,
                 out_dir: Path, stem: str, fg_mask: np.ndarray | None = None,
                 vmin: float | None = None, vmax: float | None = None) -> None:
    """anomaly map 히트맵(원본 사진 없이 raw score 그대로)과 raw anomaly map npy를 out_dir에 저장.

    raw score 값을 파란색(낮음)~빨간색(높음) jet 컬러맵으로 직접 매핑한다.
    vmin/vmax를 지정하면 모든 이미지에 동일한 고정 색 스케일을 쓰므로, 정상 이미지는
    실제로 낮은 raw score만큼 파랗게, 결함 이미지는 그보다 높은 부분만 빨갛게 나온다.
    지정하지 않으면 matplotlib이 이미지 자신의 min~max로 자동 스케일한다(이미지 간 비교 불가).

    fg_mask: (H, W) bool, True=foreground. 지정 시 background 영역은 회색으로 마스킹.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    npy_path = out_dir / f"{stem}_anomaly_map.npy"
    np.save(npy_path, anomaly_map)

    display_map = np.ma.masked_array(anomaly_map, mask=~fg_mask) if fg_mask is not None else anomaly_map

    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(display_map, cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(f"image_score={image_score:.4f}")
    ax.axis("off")
    fig.tight_layout()

    png_path = out_dir / f"{stem}_anomaly_map.png"
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    print(f"[저장] {npy_path}")
    print(f"[저장] {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="memory bank + test feature -> PatchCore anomaly score/map 계산 및 저장",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--memory_bank", type=str, default=str(DEFAULT_MEMORY_BANK),
                        help="memory bank (M, C) npy 경로")
    parser.add_argument("--test_image", type=str, default=str(DEFAULT_TEST_IMAGE),
                        help="test sample 원본 이미지 경로. feature는 "
                             f"{APP_AGG_DIR}/{{test_image의 stem}}_local.npy 에서 자동으로 찾는다.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="결과 저장 상위 디렉토리 (test feature 파일명으로 하위 폴더 생성)")
    parser.add_argument("--k_nn", type=int, default=3, help="image score re-weighting neighbor 수")
    parser.add_argument("--blur_sigma", type=float, default=4.0, help="anomaly map Gaussian blur sigma")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    memory_bank_path = Path(args.memory_bank)
    test_image_path = Path(args.test_image)
    output_dir = Path(args.output_dir)

    # ---- test feature 경로: data/features/app/agg에 미리 추출된 feature만 사용 ----
    # (다른 img_size로 추출된 stale feature를 잘못 넘겨서 memory bank와 grid가
    #  어긋나는 문제를 원천 차단하기 위해, 경로를 test_image로부터 항상 여기서 계산한다)
    test_feature_path = APP_AGG_DIR / f"{test_image_path.stem}_local.npy"
    if not test_feature_path.exists():
        raise FileNotFoundError(
            f"app/agg에 feature가 없습니다: {test_feature_path}\n"
            f"먼저 feature extraction 파이프라인으로 {APP_AGG_DIR}에 해당 이미지의 "
            f"feature를 생성하세요."
        )
    print(f"[feature] {test_feature_path}")

    # ---- 1. bank + test feature 로드 ----
    bank = load_memory_bank(memory_bank_path)
    print(f"memory bank : {tuple(bank.shape)}")

    test_feat, grid = load_test_feature(test_feature_path)
    print(f"test feature: {tuple(test_feat.shape)}  grid={grid}")

    image = Image.open(test_image_path)
    out_size = (image.height, image.width)

    # ---- fg_mask: masked_img(배경=검정)에서 foreground 영역 추출 ----
    fg_mask_full = foreground_mask_from_image(image)                        # (H, W) bool
    fg_mask_grid = F.adaptive_max_pool2d(
        torch.from_numpy(fg_mask_full).float()[None, None], output_size=grid
    ).squeeze(1).bool()                                                     # (1, Hf, Wf)

    # ---- 2. scoring ----
    scorer = PatchCoreScorer(bank, k_nn=args.k_nn, blur_sigma=args.blur_sigma)
    result = scorer.score(test_feat, grid=grid, out_size=out_size, fg_mask=fg_mask_grid)

    image_score = float(result["image_scores"][0])
    anomaly_map = result["anomaly_map"][0].numpy()
    print(f"image_score : {image_score:.4f}")
    print(f"anomaly_map : {anomaly_map.shape}")

    # ---- 2b. 히트맵 색 스케일(vmin/vmax): find_heapmap_range.py가 미리 계산해둔 json에서 읽음 ----
    with open(DEFAULT_HEATMAP_RANGE) as f:
        heatmap_range = json.load(f)
    vmin, vmax = heatmap_range["vmin"], heatmap_range["vmax"]
    print(f"heatmap range: vmin={vmin:.4f}, vmax={vmax:.4f}  (출처: {DEFAULT_HEATMAP_RANGE})")

    # ---- 3. 저장: output_dir/{test feature 파일명}/ 에 img + feature map npy ----
    stem = test_feature_path.stem
    save_result(test_image_path, anomaly_map, image_score, output_dir / stem, stem,
                fg_mask=fg_mask_full, vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    main()