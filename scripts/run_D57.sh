#!/usr/bin/env bash
# D5～D7 服务器单次运行。
# 用法：bash run_D57.sh 任务 种子 GPU 标记名
# 任务：d5（带提示词 LoRA）、d6（增强 LoRA）、d7（少样本提示）
# 例：  bash run_D57.sh d5 0 0 D5_s0
# 可选环境变量：QWEN7B（必需）、PYTHON、GPU_WAIT_MB（默认 28000）。
# d5/d6 加 --grad_ckpt；d7 先 --select 再评测，评测加 --bench。
# 失败写 results/FAILED_<标记名>，成功写 results/DONE_<标记名>。
set -eo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 4 ]; then
  echo "用法: bash run_D57.sh d5|d6|d7 种子 GPU 标记名" >&2
  exit 2
fi
TASK=$1
SEED=$2
GPU=$3
MARK=$4
R=results
PY=${PYTHON:-python}
MODEL=${QWEN7B:-}
WAIT_MB=${GPU_WAIT_MB:-28000}
mkdir -p "$R" logs
rm -f "$R/DONE_$MARK" "$R/FAILED_$MARK"

case "$TASK" in
  d5|d6|d7) ;;
  *) echo "未知任务 $TASK，可选 d5/d6/d7" >&2; exit 2 ;;
esac

fail() { echo "FAILED $MARK: $*" > "$R/FAILED_$MARK"; exit 1; }
on_exit() {
  local c=$?
  if [ "$c" -ne 0 ] && [ ! -f "$R/FAILED_$MARK" ]; then
    echo "FAILED $MARK: exit $c" > "$R/FAILED_$MARK"
  fi
}
trap on_exit EXIT

[ -n "$MODEL" ] || fail "环境变量 QWEN7B 为空"
[ -d "$MODEL" ] || fail "模型目录不存在：$MODEL"

wait_gpu() {
  local g=$1 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      -i "$g" | tr -d ' ' || true)
    if [[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge "$WAIT_MB" ]; then break; fi
    echo "GPU $g 空闲 ${free:-未知} MB < $WAIT_MB，等待 60s"
    sleep 60
  done
}

{ date '+开始 %F %T'; nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader; } > "logs/env_D57_$MARK.log"

wait_gpu "$GPU"
export CUDA_VISIBLE_DEVICES=$GPU
EXTRA=mhc_zh,toxicn_test,cloak_pinyin_orig,cloak_pinyin_edit,cloak_homophone_orig,cloak_homophone_edit,cloak_chaizi_orig,cloak_chaizi_edit,cloak_symbol_orig,cloak_symbol_edit,cloak_delete_orig,cloak_delete_edit,swap_orig,swap_edit

check_status() {
  "$PY" - "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    sys.exit(0 if json.load(f).get("status") == "ok" else 1)
PY
}

case "$TASK" in
  d5)
    ARGS=(src/finetune_llm_prompt.py --model_path "$MODEL" --name qwen7b
          --seed "$SEED" --grad_ckpt --extra_eval "$EXTRA"
          --save_preds --save_val_preds --bench --device cuda --out_dir "$R")
    RESULT="$R/cold_llmftp_qwen7b_s${SEED}.json"
    ;;
  d6)
    AUG="data_selfmade/aug_all_s${SEED}.csv"
    [ -f "$AUG" ] || fail "增强文件不存在：$AUG"
    ARGS=(src/finetune_llm.py --model_path "$MODEL" --name qwen7b_aug
          --seed "$SEED" --aug_csv "$AUG" --grad_ckpt --extra_eval "$EXTRA"
          --save_preds --save_val_preds --bench --device cuda --out_dir "$R")
    RESULT="$R/cold_llmft_qwen7b_aug_s${SEED}.json"
    ;;
  d7)
    ARGS=(src/llm_zeroshot.py --model_path "$MODEL" --name qwen7b
          --shots 8 --shot_seed "$SEED" --batch_size 16 --dtype float16
          --device cuda --out_dir "$R")
    RESULT="$R/cold_llm_qwen7b_fs${SEED}.json"
    ;;
esac

if [ "$TASK" = "d7" ]; then
  echo "运行选择: $PY ${ARGS[*]} --select"
  if ! "$PY" "${ARGS[@]}" --select; then
    fail "$PY ${ARGS[*]} --select"
  fi
  SELECTION="$R/D3_选择_qwen7b_fs${SEED}.json"
  [ -f "$SELECTION" ] || fail "选择结果未生成：$SELECTION"
  # D7 提示词约长 6 倍，只评所需集合（不含互换集与删词对照）
  EXTRA_D7=mhc_zh,toxicn_test,cloak_pinyin_orig,cloak_pinyin_edit,cloak_homophone_orig,cloak_homophone_edit,cloak_chaizi_orig,cloak_chaizi_edit,cloak_symbol_orig,cloak_symbol_edit
  echo "运行评测: $PY ${ARGS[*]} --eval $EXTRA_D7 --bench"
  if ! "$PY" "${ARGS[@]}" --eval "$EXTRA_D7" --bench; then
    fail "$PY ${ARGS[*]} --eval"
  fi
else
  echo "运行: $PY ${ARGS[*]}"
  if ! "$PY" "${ARGS[@]}"; then
    fail "$PY ${ARGS[*]}"
  fi
fi

[ -f "$RESULT" ] || fail "结果 JSON 未生成：$RESULT"
if [ "$TASK" != "d7" ]; then
  check_status "$RESULT" || fail "结果 JSON 状态不是 ok：$RESULT"
fi

echo "完成 $MARK $(date '+%F %T')" > "$R/DONE_$MARK"
{ date '+结束 %F %T'; nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader; } >> "logs/env_D57_$MARK.log"
echo "完成 $MARK"
