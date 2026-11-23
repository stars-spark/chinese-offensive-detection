#!/usr/bin/env bash
# D4 服务器单次运行。
# 用法：bash run_D4.sh 名字 模型目录 种子 GPU 标记名
# 例：  bash run_D4.sh qwen7b "$QWEN7B" 0 0 D4_qwen7b_s0
# 可选环境变量：PYTHON、GPU_WAIT_MB（默认 28000）。
# 7B 自动加 --grad_ckpt，1.5B 不加；失败写 results/FAILED_<标记名>，
# 成功写 results/DONE_<标记名>。
set -eo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 5 ]; then
  echo "用法: bash run_D4.sh 名字 模型目录 种子 GPU 标记名" >&2
  exit 2
fi
NAME=$1
MODEL=$2
SEED=$3
GPU=$4
MARK=$5
R=results
PY=${PYTHON:-python}
WAIT_MB=${GPU_WAIT_MB:-28000}
mkdir -p "$R" logs
rm -f "$R/DONE_$MARK" "$R/FAILED_$MARK"

fail() { echo "FAILED $MARK: $*" > "$R/FAILED_$MARK"; exit 1; }
on_exit() {
  local c=$?
  if [ "$c" -ne 0 ] && [ ! -f "$R/FAILED_$MARK" ]; then
    echo "FAILED $MARK: exit $c" > "$R/FAILED_$MARK"
  fi
}
trap on_exit EXIT

[ -n "$MODEL" ] || fail "模型目录为空"
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
    --format=csv,noheader; } > "logs/env_D4_$MARK.log"

wait_gpu "$GPU"
export CUDA_VISIBLE_DEVICES=$GPU

GRAD_CKPT=0
case "${NAME,,}${MODEL,,}" in
  *7b*) GRAD_CKPT=1 ;;
esac
EXTRA=mhc_zh,toxicn_test,cloak_pinyin_orig,cloak_pinyin_edit,cloak_homophone_orig,cloak_homophone_edit,cloak_chaizi_orig,cloak_chaizi_edit,cloak_symbol_orig,cloak_symbol_edit,cloak_delete_orig,cloak_delete_edit,swap_orig,swap_edit

ARGS=(src/finetune_llm.py --model_path "$MODEL" --name "$NAME" --seed "$SEED"
      --extra_eval "$EXTRA" --save_preds --save_val_preds --bench
      --device cuda --out_dir "$R")
if [ "$GRAD_CKPT" = 1 ]; then
  ARGS+=(--grad_ckpt)
  echo "7B：已加 --grad_ckpt"
else
  echo "非 7B：不加 --grad_ckpt"
fi

echo "运行: $PY ${ARGS[*]}"
if ! "$PY" "${ARGS[@]}"; then
  fail "$PY ${ARGS[*]}"
fi
RESULT="$R/cold_llmft_${NAME}_s${SEED}.json"
[ -f "$RESULT" ] || fail "结果 JSON 未生成：$RESULT"
"$PY" - "$RESULT" <<'PY' || fail "结果 JSON 状态不是 ok：$RESULT"
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    sys.exit(0 if json.load(f).get("status") == "ok" else 1)
PY

echo "完成 $MARK $(date '+%F %T')" > "$R/DONE_$MARK"
{ date '+结束 %F %T'; nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader; } >> "logs/env_D4_$MARK.log"
echo "完成 $MARK"
