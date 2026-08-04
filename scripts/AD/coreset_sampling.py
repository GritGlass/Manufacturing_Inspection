"""
PatchCore greedy coreset subsampling (논문 Algorithm 1, Eq. 5).

npy 파일로 저장된 patch feature bank [N, d]를 입력받아,
minimax facility location 기준의 greedy 근사로 coreset을 선택한다.

논문 구성 요소:
  - Eq. 5: M*_C = argmin_{M_C} max_{m in M} min_{n in M_C} ||m - n||_2  (NP-Hard)
  - 반복적 greedy 근사 [48]: 매 스텝 "현재 coreset에서 가장 먼 점"을 추가
  - Johnson-Lindenstrauss [11]: 랜덤 선형 사영 ψ: R^d -> R^d* (d* < d)로
    거리 계산 차원을 줄여 선택 속도 향상 (공식 구현 기본 d* = 128)

메모리 처리:
  --input 디렉토리의 .npy 파일들을 한 번에 RAM에 올려 concatenate하면
  전체 patch 수가 많을 때 (리스트 보관 + concatenate 복사) 두 배의 메모리를
  순간적으로 요구해 OOM이 발생할 수 있다. 이를 피하기 위해 파일을 하나씩
  스트리밍으로 읽어 원본 차원 patch bank는 디스크 memmap에 채우고, greedy
  선택에 쓰는 JL 사영 결과(d* 차원, 원본 대비 훨씬 작음)만 RAM에 유지한다.

사용 예:
  cd /mnt/data/github/AI-portfolio/3D-Anomaly-Detection
  python3 Scripts/coreset_sampling.py \
    --input Data/features/masked_img_view1/agg \
    --output Data/features/masked_img_view1/coreset \
    --percentage 0.01 \
    --proj-dim 128
  # --input 디렉토리 안의 .npy 확장자 파일들을 모두 모아 하나의 patch memory bank로 합친 뒤 coreset을 선택.
  # --output 생략 시 기본값: <input>과 같은 레벨의 'coreset' 디렉토리
  # 결과: coreset.npy, coreset_indices.npy
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def greedy_coreset_indices(
    work: torch.Tensor,
    target_size: int,
    seed: int = 0,
    verbose: bool = True,
) -> torch.Tensor:
    """Greedy minimax facility location coreset 선택.

    Args:
        work: [N, d*] 선택 기준으로 쓸 feature 행렬 (JL 사영된 것 또는 원본)
        target_size: 선택할 coreset 크기 l
        seed: 시작점 재현용 시드
    Returns:
        [target_size] 선택된 인덱스 (LongTensor)
    """
    N = work.shape[0]
    device = work.device
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # --- greedy 선택 ---
    # min_dists[i] = i번째 점에서 현재까지 선택된 coreset까지의 최소 거리 제곱.
    # 매 스텝 argmax(min_dists)를 추가하고, 새 점과의 거리로 min_dists를 갱신.
    # ||a-b||^2 = ||a||^2 - 2*a·b + ||b||^2 전개로 계산한다: torch.cdist는 매 스텝
    # [N, d*] 크기의 임시 버퍼를 새로 할당해 GPU에서 OOM을 유발하므로, 대신
    # 행렬-벡터 곱(GEMV)만 써서 O(N) 메모리로 거리를 구한다.
    # norms_sq도 (work * work)를 통째로 계산하면 work와 같은 크기(N x d*)의 임시
    # 버퍼가 순간적으로 추가 할당되어 GPU 메모리가 튀므로, 행 청크 단위로 나눠 계산한다.
    norms_sq = torch.empty(N, dtype=work.dtype, device=device)
    _norm_chunk = 200_000
    for _s in range(0, N, _norm_chunk):
        _e = min(_s + _norm_chunk, N)
        norms_sq[_s:_e] = (work[_s:_e] * work[_s:_e]).sum(dim=1)

    def sq_dists_to(idx: int) -> torch.Tensor:
        cross = work @ work[idx]  # [N]
        return (norms_sq - 2.0 * cross + norms_sq[idx]).clamp_(min=0)

    start = int(torch.randint(N, (1,), generator=gen).item())
    selected = torch.empty(target_size, dtype=torch.long, device=device)
    selected[0] = start

    min_dists = sq_dists_to(start)

    for i in range(1, target_size):
        idx = int(torch.argmax(min_dists).item())
        selected[i] = idx
        new_dists = sq_dists_to(idx)
        torch.minimum(min_dists, new_dists, out=min_dists)
        min_dists[idx] = -1.0  # 이미 선택된 점은 다시 뽑히지 않도록

        if verbose and (i % max(1, target_size // 10) == 0):
            # coverage radius: 아직 커버되지 않은 점 중 가장 먼 거리 (Eq. 5의 max-min 항)
            print(f"  [{i}/{target_size}] coverage radius^2 = {min_dists.max().item():.4f}")

    return selected


def scan_patch_files(input_dir: Path) -> tuple[list[Path], int, int]:
    """.npy 파일들의 헤더만 읽어 (전체 로드 없이) 총 patch 수 N과 차원 d를 파악."""
    npy_files = sorted(p for p in input_dir.iterdir() if p.suffix == ".npy")
    if not npy_files:
        raise FileNotFoundError(f"{input_dir}에 .npy 확장자 파일이 없습니다.")

    total = 0
    dim: int | None = None
    for f in npy_files:
        arr = np.load(f, mmap_mode="r")
        if arr.ndim == 4:
            B, C, H, W = arr.shape
            n = B * H * W
        elif arr.ndim == 2:
            n, C = arr.shape
        else:
            raise ValueError(f"{f.name}: 지원하지 않는 shape {arr.shape} (2D 또는 4D[B,C,H,W]만 지원)")
        if dim is None:
            dim = C
        elif dim != C:
            raise ValueError(f"{f.name}: 차원 불일치 (기대 {dim}, 실제 {C})")
        total += n
    assert dim is not None
    return npy_files, total, dim


def _pca_chunk_bounds(total: int, batch_size: int, min_size: int) -> list[tuple[int, int]]:
    """IncrementalPCA용 청크 경계 목록. 마지막 청크가 min_size(=n_components)보다
    작아지지 않도록 직전 청크와 합친다 (IncrementalPCA는 배치 크기가
    n_components보다 작으면 에러를 낸다)."""
    bounds = [(s, min(s + batch_size, total)) for s in range(0, total, batch_size)]
    if len(bounds) >= 2 and (bounds[-1][1] - bounds[-1][0]) < min_size:
        prev_start, _ = bounds[-2]
        _, last_end = bounds[-1]
        bounds = bounds[:-2] + [(prev_start, last_end)]
    return bounds


def stream_build_bank(
    npy_files: list[Path],
    total: int,
    dim: int,
    bank_path: Path,
    proj_dim: int | None,
    seed: int,
    proj_type: str = "JL",
    pca_batch_size: int = 4096,
) -> tuple[np.memmap, torch.Tensor]:
    """파일을 하나씩 읽어 원본 차원 patch bank는 디스크 memmap에 채우고,
    저차원 사영 결과(work)는 RAM에 누적한다.

    한 시점에 RAM에 올라오는 원본 데이터는 파일 하나 분량뿐이므로,
    전체 patch 수가 많아도 메모리 사용량은 O(파일 하나 크기 + N x proj_dim)로 유지된다.

    proj_type='JL': 랜덤 선형 사영이라 파일을 읽는 동시에 바로 적용 가능.
    proj_type='PCA': fit에 전체 데이터가 필요하므로, patch bank를 디스크에 다 쓴
      뒤 IncrementalPCA로 그 memmap을 청크 단위로만 다시 읽어 fit/transform한다
      (한 번에 batch_size행만 RAM에 올라오므로 전체를 한 번에 올리는 것보다 안전).
    """
    bank = np.lib.format.open_memmap(bank_path, mode="w+", dtype=np.float32, shape=(total, dim))

    use_proj = proj_dim is not None and proj_dim < dim
    psi: torch.Tensor
    work: torch.Tensor
    if use_proj and proj_dim is not None:
        work = torch.empty((total, proj_dim), dtype=torch.float32)
        if proj_type == "JL":
            gen = torch.Generator(device="cpu").manual_seed(seed)
            psi = torch.randn(dim, proj_dim, generator=gen) / np.sqrt(proj_dim)

    offset = 0
    for f in npy_files:
        arr = np.load(f)
        if arr.ndim == 4:
            B, C, H, W = arr.shape
            patches = arr.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            patches = arr
        n = patches.shape[0]
        bank[offset:offset + n] = patches
        if use_proj and proj_type == "JL":
            work[offset:offset + n] = torch.from_numpy(patches).float() @ psi
        print(f"  {f.name}: {arr.shape} -> {n} patches ({offset + n}/{total})")
        offset += n

    bank.flush()

    if use_proj and proj_type == "PCA" and proj_dim is not None:
        from sklearn.decomposition import IncrementalPCA
        chunks = _pca_chunk_bounds(total, pca_batch_size, min_size=proj_dim)

        print(f"  PCA fit 중 (IncrementalPCA, {len(chunks)}개 청크)...")
        ipca = IncrementalPCA(n_components=proj_dim, batch_size=pca_batch_size)
        for start, end in chunks:
            ipca.partial_fit(bank[start:end])

        print("  PCA transform 적용 중...")
        for start, end in chunks:
            work[start:end] = torch.from_numpy(ipca.transform(bank[start:end]).astype(np.float32))
        print(f"  설명된 분산 비율 합: {ipca.explained_variance_ratio_.sum():.4f}")

    if not use_proj:
        # 원본 차원을 그대로 선택 공간으로 사용 (memmap 기반, page cache로 관리되어
        # concatenate 방식보다 안전하지만 여전히 RAM 부담이 클 수 있음)
        work = torch.from_numpy(bank)
    return bank, work


def main():
    parser = argparse.ArgumentParser(description="PatchCore greedy coreset subsampling")
    parser.add_argument("--input", "-i", required=True,
                        help="patch feature .npy 파일들이 있는 디렉토리 (확장자가 .npy인 파일만 선택)")
    parser.add_argument("--percentage", type=float, default=0.01,
                        help="subsampling 비율 (예: 0.01 = PatchCore-1%%)")
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="랜덤/PCA 사영 차원 d*. 0이면 사영 생략")
    parser.add_argument("--proj-type", default="JL", choices=["JL", "PCA"],
                        help="사영 방식. JL=랜덤 선형 사영(빠름, 스트리밍 중 바로 적용). "
                             "PCA=IncrementalPCA로 청크 단위 fit/transform(느리지만 분산 보존 기준으로 선택)")
    parser.add_argument("--pca-batch-size", type=int, default=4096,
                        help="PCA fit/transform 시 한 번에 메모리에 올릴 patch 수")
    parser.add_argument("--output", default=None,
                        help="coreset 저장 디렉토리 (기본: <input>과 같은 레벨의 'coreset' 디렉토리)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="greedy 선택 연산 장치. auto=cuda 가능하면 cuda, 아니면 cpu. "
                             "디스플레이를 겸하는 GPU에서 RC watchdog(Xid 8)으로 죽는다면 cpu로 강제 지정.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    npy_files, N, d = scan_patch_files(input_dir)
    target = max(1, int(N * args.percentage))
    proj = args.proj_dim if args.proj_dim and args.proj_dim > 0 else None
    print(f"입력: 총 {N} patches x {d} dims -> coreset {target}개 선택 "
          f"({args.percentage:.1%}), proj_dim={proj or '없음'}")
    if proj is None:
        print("  [경고] proj-dim 생략: 선택 단계에서 원본 차원 전체를 사용하므로 메모리 부담이 큽니다.")

    output_dir = Path(args.output) if args.output else input_dir.parent / "coreset"
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(dir=output_dir))
    bank_path = tmp_dir / "patch_bank.npy"
    try:
        print(f"patch bank를 디스크에 스트리밍 저장 중: {bank_path}")
        bank, work = stream_build_bank(npy_files, N, d, bank_path, proj, args.seed,
                                        proj_type=args.proj_type, pca_batch_size=args.pca_batch_size)

        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        print(f"선택 연산 장치: {device}")
        work = work.to(device)

        indices = greedy_coreset_indices(work, target, seed=args.seed)
        idx_np = indices.cpu().numpy()

        coreset = np.array(bank[idx_np])  # 원본 차원 feature를 memmap에서 필요한 행만 읽어 복사
        del bank, work
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    coreset_path = output_dir / "MB.npy"
    indices_path = output_dir / "MB_indices.npy"
    np.save(coreset_path, coreset)
    np.save(indices_path, idx_np)
    print(f"저장 완료: {coreset_path}  shape={coreset.shape}")
    print(f"인덱스 저장 완료: {indices_path}")


if __name__ == "__main__":
    main()
