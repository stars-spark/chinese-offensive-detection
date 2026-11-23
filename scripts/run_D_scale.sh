#!/bin/bash
# D3b 模型规模曲线的队列。
# 用法：bash run_D_scale.sh 名字 模型目录 GPU 先等的完成标记 模型就绪标记 本队列标记
# 例：bash run_D_scale.sh qwen3b models/Qwen2.5-3B-Instruct 1 results/DONE_D_q2 logs/QWEN3B_READY D_q4
# 显存不够时 batch 依次降到 8、4，用到的 batch 写进日志。
set -u
cd "$(dirname "$0")/.."
NAME=$1; MP=$2; G=$3; PREV=$4; READY=$5; MARK=$6
PY=${PYTHON:-python}
EXTRA=mhc_zh,toxicn_test,cloak_pinyin_orig,cloak_pinyin_edit,cloak_homophone_orig,cloak_homophone_edit,cloak_chaizi_orig,cloak_chaizi_edit,cloak_symbol_orig,cloak_symbol_edit,cloak_delete_orig,cloak_delete_edit,swap_orig,swap_edit
rm -f results/DONE_$MARK results/FAILED_$MARK
fail() { echo "FAILED $MARK: $*" > results/FAILED_$MARK; exit 1; }
until [ -f "$PREV" ]; do sleep 60; done
until [ -f "$READY" ]; do echo "等待模型就绪 $READY"; sleep 60; done
export CUDA_VISIBLE_DEVICES=$G
for bs in 16 8 4; do
  echo "尝试 batch $bs"
  [ -f results/D3_选择_$NAME.json ] || $PY src/llm_zeroshot.py --model_path "$MP" --name $NAME --select --batch_size $bs --dtype float16 || continue
  [ -f results/cold_llm_$NAME.json ] || $PY src/llm_zeroshot.py --model_path "$MP" --name $NAME --eval "$EXTRA" --batch_size $bs --dtype float16 --bench || continue
  echo "完成 $NAME batch=$bs $(date '+%F %T')" > results/DONE_$MARK; exit 0
done
fail "batch 16/8/4 都失败"
