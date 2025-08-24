#!/bin/bash

# exp_000: github上のquick startを動作確認

# --- 変更すべき部分 ---
EXP_NAME="exp_001"
EXP_TYPE="test"  # val test

# --- 変更しない部分 ---

# --- setup ---
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/mnt/data/.cache/huggingface"

PROJECT_ROOT="/home_dir_path/pjt/cure"
cd "$PROJECT_ROOT"
source .venv/bin/activate

EXP_DIR="${PROJECT_ROOT}/experiments/${EXP_NAME}"
cd "$EXP_DIR"
echo "Current directory: $(pwd)"

# --- setup config ---
PYTHON_SCRIPT="${EXP_DIR}/run.py"
CONFIG_FILE="${EXP_DIR}/config_${EXP_TYPE}.json"

RESULTS_DIR="${PROJECT_ROOT}/results/${EXP_NAME}"
TIMESTAMP=$(TZ=Asia/Tokyo date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/${EXP_NAME}_${TIMESTAMP}.log"
OUTPUT_FILE="${RESULTS_DIR}/competition_${EXP_TYPE}_results/submission_${EXP_NAME}_${TIMESTAMP}.csv"
mkdir -p "${RESULTS_DIR}/competition_${EXP_TYPE}_results"

# --- run ---
nohup python "$PYTHON_SCRIPT" \
    --config "$CONFIG_FILE" \
    --output-file "$OUTPUT_FILE" \
    --timestamp "$TIMESTAMP" \
    --results-dir "$RESULTS_DIR" \
    --exp-name "$EXP_NAME" \
    --exp-type "$EXP_TYPE" > "$LOG_FILE" 2>&1 &


# --- 終了メッセージ ---
echo "Running script in the background. PID: $!"
echo "Output is being recorded in $LOG_FILE."
echo "The script has completed (running in the background)."