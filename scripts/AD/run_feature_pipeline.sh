#!/usr/bin/env bash
# =============================================================================
# SAM3 feature 추출 → PatchCore aggregation → coreset sampling(memory bank)까지
# 3단계를 순서대로 실행하는 파이프라인
#
# Step 1: sam3_img_feature_extract.py  이미지 폴더 -> layer feature npy
# Step 2: patchcore_aggregation.py     feature npy -> local aggregation npy
# Step 3: coreset_sampling.py          aggregation npy -> memory bank(MB.npy)
#         (--build_memory_bank 옵션을 켰을 때만 실행. 기본은 건너뜀)
#
# 사용법:
#   bash Scripts/run_feature_pipeline.sh [OPTIONS]
#
# 예시 1) CSV로 good 이미지 목록 지정해서 memory bank 생성 (Step 1,2,3):
#   bash Scripts/run_feature_pipeline.sh \
#       --input_mode   csv \
#       --csv_path     data/semiconductor_AD_df.csv \
#       --label_filter Normal \
#       --feature_root Data/features/masked_img_crop_view1 \
#       --build_memory_bank \
#       --memory_bank_dir Data/memory_bank/masked_crop_img_view1
#
# 예시 2) 이미지 디렉토리를 그대로 사용 (하위 폴더 포함, CSV 불필요):
#   bash Scripts/run_feature_pipeline.sh \
#       --input_mode   img_dir \
#       --img_dir      data/images/good \
#       --feature_root Data/features/masked_img_crop_view1 \
#       --build_memory_bank \
#       --memory_bank_dir Data/memory_bank/masked_crop_img_view1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- 기본값 ----------
INPUT_MODE="img_dir"           # csv: --csv_path로 이미지 목록 지정 / img_dir: --img_dir 디렉토리 내 이미지 전체 사용
CSV_PATH="/mnt/data/github/Manufacutring_Inspection/data/semiconductor_AD_df.csv"  # 이미지 목록을 구성할 CSV (INPUT_MODE=csv)
PATH_COL="data_path"       # CSV 내 이미지 경로 컬럼명
LABEL_COL="labels"         # CSV 내 라벨 컬럼명
LABEL_FILTER="Normal"      # LABEL_COL == LABEL_FILTER 인 행만 사용
IMG_DIR="/mnt/data/github/Manufacutring_Inspection/data/images"                 # 이미지 디렉토리 경로 (INPUT_MODE=img_dir일 때 필수, 하위 폴더 포함 전체 이미지 사용)
FEATURE_ROOT="$PROJECT_ROOT/data/features/test_img"
RAW_DIR="$PROJECT_ROOT/data/features/test_img/raw"         # 기본: $FEATURE_ROOT/raw
AGG_DIR="$PROJECT_ROOT/data/features/test_img/agg"           # 기본: $FEATURE_ROOT/agg

LAYER=3
IMG_SIZE=1008
PRETRAINED_WEIGHTS_PATH=""

# Step 2 (patchcore aggregation)
RAW_INPUT1="$PROJECT_ROOT/data/features/test_img/raw"    # --input1 (미지정 시 RAW_DIR 사용)
RAW_INPUT2=""  # --input2 (hierarchy 시 사용)
AGG_OUTPUT="$PROJECT_ROOT/data/features/test_img/agg"    # --output (미지정 시 AGG_DIR 사용)
HIERARCHY=0
PATCHSIZE=3
OUTPUT_DIM=""
PER_LEVEL_DIM=""
TARGET_DIM=""

# Step 3 (coreset sampling / memory bank 생성)
MEMORY_BANK_DIR="$PROJECT_ROOT/data/memory_bank"
CORESET_PERCENTAGE=0.01
CORESET_PROJ_DIM=128
CORESET_PROJ_TYPE="PCA"  # JL/PCA. PCA는 IncrementalPCA로 청크 단위 fit (느리지만 OOM 안전)
CORESET_PCA_BATCH_SIZE=4096
CORESET_SEED=0
CORESET_DEVICE="auto"  # auto/cuda/cpu. 디스플레이 겸용 GPU에서 RC watchdog(Xid 8)으로 죽으면 cpu로 지정

RUN_EXTRACT=1       #0 : Step1 실행 안함, 1: Step1 실행
RUN_AGGREGATION=1   #0 : Step2 실행 안함, 1: Step2 실행
BUILD_MEMORY_BANK=0 #0 : Step3 실행 안함, 1: Step3 실행

# Step3(coreset sampling)만 실행하려면:
#   RUN_EXTRACT=0, RUN_AGGREGATION=0, BUILD_MEMORY_BANK=1 로 설정
#   (단, Step2 결과물인 $AGG_DIR 의 aggregation npy가 이미 존재해야 함)

# ---------- 인수 파싱 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_mode)             INPUT_MODE="$2";             shift 2 ;;
        --csv_path)              CSV_PATH="$2";               shift 2 ;;
        --path_col)               PATH_COL="$2";               shift 2 ;;
        --label_col)              LABEL_COL="$2";              shift 2 ;;
        --label_filter)           LABEL_FILTER="$2";           shift 2 ;;
        --img_dir)                IMG_DIR="$2";                shift 2 ;;
        --feature_root)         FEATURE_ROOT="$2";           shift 2 ;;
        --raw_dir)               RAW_DIR="$2";                shift 2 ;;
        --agg_dir)               AGG_DIR="$2";                shift 2 ;;
        --layer)                 LAYER="$2";                  shift 2 ;;
        --img_size)              IMG_SIZE="$2";               shift 2 ;;
        --pretrained_weights_path) PRETRAINED_WEIGHTS_PATH="$2"; shift 2 ;;

        --raw_input1)            RAW_INPUT1="$2";             shift 2 ;;
        --raw_input2)            RAW_INPUT2="$2";             shift 2 ;;
        --agg_output)            AGG_OUTPUT="$2";             shift 2 ;;
        --hierarchy)             HIERARCHY=1;                 shift   ;;
        --patchsize)              PATCHSIZE="$2";              shift 2 ;;
        --output_dim)             OUTPUT_DIM="$2";             shift 2 ;;
        --per_level_dim)          PER_LEVEL_DIM="$2";          shift 2 ;;
        --target_dim)             TARGET_DIM="$2";             shift 2 ;;

        --build_memory_bank)      BUILD_MEMORY_BANK=1;         shift   ;;
        --memory_bank_dir)        MEMORY_BANK_DIR="$2";        shift 2 ;;
        --coreset_percentage)     CORESET_PERCENTAGE="$2";     shift 2 ;;
        --coreset_proj_dim)       CORESET_PROJ_DIM="$2";       shift 2 ;;
        --proj-type)              CORESET_PROJ_TYPE="$2";      shift 2 ;;
        --pca-batch-size)         CORESET_PCA_BATCH_SIZE="$2"; shift 2 ;;
        --coreset_seed)           CORESET_SEED="$2";           shift 2 ;;
        --coreset_device)         CORESET_DEVICE="$2";         shift 2 ;;

        --run_extract)            RUN_EXTRACT="$2";            shift 2 ;;
        --run_aggregation)        RUN_AGGREGATION="$2";        shift 2 ;;
        *)
            echo "[오류] 알 수 없는 인수: $1"
            exit 1
            ;;
    esac
done

# ---------- 파생 경로 기본값 ----------
[[ -z "$RAW_DIR" ]] && RAW_DIR="$FEATURE_ROOT/raw"
[[ -z "$AGG_DIR" ]] && AGG_DIR="$FEATURE_ROOT/agg"
[[ -z "$RAW_INPUT1" ]] && RAW_INPUT1="$RAW_DIR"
[[ -z "$AGG_OUTPUT" ]] && AGG_OUTPUT="$AGG_DIR"

# ---------- 입력 모드 검증 ----------
if [[ "$INPUT_MODE" != "csv" && "$INPUT_MODE" != "img_dir" ]]; then
    echo "[오류] --input_mode는 csv 또는 img_dir 이어야 합니다 (현재: $INPUT_MODE)"
    exit 1
fi
if [[ $RUN_EXTRACT -eq 1 && "$INPUT_MODE" == "img_dir" && -z "$IMG_DIR" ]]; then
    echo "[오류] --input_mode img_dir 사용 시 --img_dir 경로를 지정해야 합니다."
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo " Feature 파이프라인 설정"
if [[ "$INPUT_MODE" == "csv" ]]; then
    echo "   input_mode         : csv ($CSV_PATH, $LABEL_COL==$LABEL_FILTER, path_col=$PATH_COL)"
else
    echo "   input_mode         : img_dir ($IMG_DIR)"
fi
echo "   raw_dir(Step1)     : $RAW_DIR"
echo "   raw_input1(Step2)  : $RAW_INPUT1"
echo "   raw_input2(Step2)  : ${RAW_INPUT2:-없음}"
echo "   agg_output(Step2)  : $AGG_OUTPUT"
echo "   layer              : $LAYER"
echo "   build_memory_bank  : $BUILD_MEMORY_BANK"
[[ $BUILD_MEMORY_BANK -eq 1 ]] && echo "   memory_bank_dir    : $MEMORY_BANK_DIR"
echo "════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────
# Step 1: Wide ResNet 이미지 feature 추출
# ─────────────────────────────────────────────
if [[ $RUN_EXTRACT -eq 1 ]]; then
    echo ""
    echo "[Step 1] Wide ResNet feature 추출"

    if [[ "$INPUT_MODE" == "csv" ]]; then
        EXTRACT_ARGS=(
            --csv_path      "$CSV_PATH"
            --path_col      "$PATH_COL"
            --label_col     "$LABEL_COL"
            --label_filter  "$LABEL_FILTER"
            --output_dir    "$RAW_DIR"
            --layer         "$LAYER"
            --img_size      "$IMG_SIZE"
        )
    else
        EXTRACT_ARGS=(
            --img_dir       "$IMG_DIR"
            --output_dir    "$RAW_DIR"
            --layer         "$LAYER"
            --img_size      "$IMG_SIZE"
        )
    fi
    [[ -n "$PRETRAINED_WEIGHTS_PATH" ]] && EXTRACT_ARGS+=(--pretrained_weights_path "$PRETRAINED_WEIGHTS_PATH")

    python "$SCRIPT_DIR/wide_resnet_img_feature_extract.py" "${EXTRACT_ARGS[@]}"
else
    echo ""
    echo "[Step 1 건너뜀] 기존 raw feature 사용: $RAW_DIR"
fi

# ─────────────────────────────────────────────
# Step 2: PatchCore local neighborhood aggregation
# ─────────────────────────────────────────────
if [[ $RUN_AGGREGATION -eq 1 ]]; then
    echo ""
    echo "[Step 2] PatchCore aggregation"

    AGG_ARGS=(
        --input1 "$RAW_INPUT1"
        --output "$AGG_OUTPUT"
        --patchsize "$PATCHSIZE"
    )
    [[ -n "$RAW_INPUT2" ]] && AGG_ARGS+=(--input2 "$RAW_INPUT2")
    [[ $HIERARCHY -eq 1 ]] && AGG_ARGS+=(--hierarchy)
    [[ -n "$OUTPUT_DIM" ]] && AGG_ARGS+=(--output-dim "$OUTPUT_DIM")
    [[ -n "$PER_LEVEL_DIM" ]] && AGG_ARGS+=(--per-level-dim "$PER_LEVEL_DIM")
    [[ -n "$TARGET_DIM" ]] && AGG_ARGS+=(--target-dim "$TARGET_DIM")

    python3 "$SCRIPT_DIR/patchcore_aggregation.py" "${AGG_ARGS[@]}"
else
    echo ""
    echo "[Step 2 건너뜀] 기존 aggregation feature 사용: $AGG_OUTPUT"
fi

# ─────────────────────────────────────────────
# Step 3: coreset sampling으로 memory bank 생성 (옵션)
# ─────────────────────────────────────────────
if [[ $BUILD_MEMORY_BANK -eq 1 ]]; then
    echo ""
    echo "[Step 3] Coreset sampling (memory bank 생성)"

    python3 "$SCRIPT_DIR/coreset_sampling.py" \
        --input "$AGG_OUTPUT" \
        --output "$MEMORY_BANK_DIR" \
        --percentage "$CORESET_PERCENTAGE" \
        --proj-dim "$CORESET_PROJ_DIM" \
        --proj-type "$CORESET_PROJ_TYPE" \
        --pca-batch-size "$CORESET_PCA_BATCH_SIZE" \
        --seed "$CORESET_SEED" \
        --device "$CORESET_DEVICE"
else
    echo ""
    echo "[Step 3 건너뜀] (--build_memory_bank 미지정)"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo " [전체 완료]"
echo "════════════════════════════════════════════════════════"
