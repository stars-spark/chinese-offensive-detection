#!/usr/bin/env bash
# D 组服务器队列。
# 用法：bash run_D.sh q1|q2|q3
# 可选环境变量：OUT_DIR（默认 results）、PYTHON、QWEN7B、QWEN15B、
# CKPT_DIR（默认 checkpoints）、GPU_WAIT_MB（默认 12000）。
# 每个队列开始前等目标卡空闲显存 >= GPU_WAIT_MB；任一步失败写
# results/FAILED_D_<队列> 并退出，全部完成才 touch results/DONE_D_<队列>。
# q3 在 1 号卡上等 q2 的完成标记后再开始。
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
QUEUE=${1:?用法: run_D.sh q1|q2|q3}
R=${OUT_DIR:-results}
CKPT=${CKPT_DIR:-checkpoints}
WAIT_MB=${GPU_WAIT_MB:-12000}
MARK=D_$QUEUE
mkdir -p "$R" logs
rm -f "results/DONE_$MARK" "results/FAILED_$MARK"

fail() { echo "FAILED $QUEUE: $*" > "results/FAILED_$MARK"; exit 1; }
trap 'c=$?; if [ $c -ne 0 ]; then echo "FAILED $QUEUE: line $LINENO (exit $c)" > "results/FAILED_$MARK"; fi' ERR

case "$QUEUE" in
  q1) G=${Q1_GPU:-0} ;;
  q2) G=${Q2_GPU:-1} ;;
  q3) G=${Q3_GPU:-1} ;;
  *) echo "未知队列 $QUEUE，可选 q1/q2/q3" >&2; exit 2 ;;
esac
{ date '+开始 %F %T'; nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader; } > "logs/env_$MARK.log"

wait_gpu() {
  local g=$1 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" \
      | tr -d ' ' || true)
    if [ -n "$free" ] && [ "$free" -ge "$WAIT_MB" ]; then break; fi
    echo "GPU $g 空闲 ${free:-未知} MB < $WAIT_MB，等待 60s"
    sleep 60
  done
}

run() {  # 已存在主结果就跳过；失败写 FAILED
  local json=$1; shift
  if [ -n "$json" ] && [ -f "$json" ]; then
    echo "跳过（已存在）: $json"
    return 0
  fi
  echo "运行: $*"
  if ! "$@"; then fail "$*"; fi
}

run_force() {  # 必须重训的步骤，不按结果文件跳过
  echo "运行: $*"
  if ! "$@"; then fail "$*"; fi
}

EXTRA=mhc_zh,toxicn_test,cloak_pinyin_orig,cloak_pinyin_edit,cloak_homophone_orig,cloak_homophone_edit,cloak_chaizi_orig,cloak_chaizi_edit,cloak_symbol_orig,cloak_symbol_edit,cloak_delete_orig,cloak_delete_edit,swap_orig,swap_edit

bert_aug() {  # condition seed
  local cond=$1 seed=$2 suffix="aug_$1"
  local name="cold_bert_${suffix}_s${seed}"
  run "$R/$name.json" "$PY" src/finetune_bert.py --dataset cold \
    --model_path models/rbt3 --seed "$seed" --run_suffix "$suffix" \
    --aug_csv "data_selfmade/aug_${cond}_s${seed}.csv" \
    --extra_eval "$EXTRA" --save_preds --save_val_preds --decontaminate
}

tfidf_char() {  # seed suffix [aug_csv]
  local seed=$1 suffix=$2 aug=${3:-}
  local name="cold_tfidf_main_${suffix}_s${seed}"
  run "$R/$name.json" "$PY" src/baseline_tfidf.py --dataset cold \
    --tokenizer char --seed "$seed" --run_suffix "$suffix" \
    ${aug:+--aug_csv "$aug"} --extra_eval "$EXTRA" \
    --save_preds --save_val_preds --decontaminate
}

qwen() {  # name model_path
  local name=$1 path=$2
  if [ -z "$path" ]; then fail "环境变量未设置：$name"; fi
  # 模型下载与 sha256 校验完成后才开始
  if [ -n "${QWEN_READY:-}" ]; then
    until [ -f "$QWEN_READY" ]; do echo "等待模型就绪：$QWEN_READY"; sleep 60; done
  fi
  run "$R/D3_选择_${name}.json" "$PY" src/llm_zeroshot.py \
    --model_path "$path" --name "$name" --select --batch_size 16 --dtype float16
  run "$R/cold_llm_${name}.json" "$PY" src/llm_zeroshot.py \
    --model_path "$path" --name "$name" --eval "$EXTRA" \
    --batch_size 16 --dtype float16 --bench
}

case "$QUEUE" in
  q1)
    export CUDA_VISIBLE_DEVICES=$G
    wait_gpu "$G"
    run_force "$PY" src/make_aug.py
    for seed in 0 1 2; do
      bert_aug all "$seed"
      bert_aug held "$seed"
    done
    tfidf_char 0 char ""
    tfidf_char 0 aug_all "data_selfmade/aug_all_s0.csv"
    tfidf_char 0 aug_held "data_selfmade/aug_held_s0.csv"
    ;;
  q2)
    export CUDA_VISIBLE_DEVICES=$G
    wait_gpu "$G"
    # 只补检查点：输出放 $R/retrain_d，避免覆盖 results/ 下 C 组已有 JSON
    for seed in 1 2; do
      ckpt="$CKPT/cold_bert_rbt3_s$seed"
      if [ -f "$ckpt/config.json" ]; then
        echo "跳过（已有检查点）: $ckpt"
        continue
      fi
      run_force "$PY" src/finetune_bert.py --dataset cold \
        --model_path models/rbt3 --seed "$seed" --run_suffix rbt3 \
        --extra_eval "$EXTRA" --save_preds --save_val_preds --save_model \
        --out_dir "$R/retrain_d" --decontaminate
      [ -f "$ckpt/config.json" ] || fail "检查点未生成：$ckpt"
    done
    for seed in 0 1 2; do
      model="$CKPT/cold_bert_rbt3_s$seed"
      if [ "$seed" = 0 ]; then model="$CKPT/cold_bert_demo_rbt3_s0"; fi
      run "$R/D2_归因_s${seed}.json" "$PY" src/attribution.py \
        --model_path "$model" --seed "$seed" --device cuda --check_ref
    done
    qwen qwen15b "$QWEN15B"
    ;;
  q3)
    export CUDA_VISIBLE_DEVICES=$G
    until [ -f "results/DONE_D_q2" ]; do
      if [ -f "results/FAILED_D_q2" ]; then fail "q2 失败，q3 不启动"; fi
      echo "等待 q2 完成（results/DONE_D_q2）"
      sleep 60
    done
    wait_gpu "$G"
    qwen qwen7b "$QWEN7B"
    ;;
esac

echo "完成 $QUEUE $(date '+%F %T')" > "results/DONE_$MARK"
{ date '+结束 %F %T'; nvidia-smi --query-gpu=index,memory.used,memory.free \
    --format=csv,noheader; } >> "logs/env_$MARK.log"
echo "完成 $QUEUE"
