#!/usr/bin/env bash
# C 组四个队列；每个队列在一张卡上串行跑，队列之间可并行。
# 用法：./run_C.sh q1|q2|q3|q4
# 可选环境变量：OUT_DIR（默认 results）、TEST_LIMIT、TRAIN_N、EPOCHS、
# Q1_GPU/Q2_GPU/Q3_GPU/Q4_GPU（默认 0/1/2/3）。
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
QUEUE=${1:?用法: run_C.sh q1|q2|q3|q4}
R=${OUT_DIR:-results}
mkdir -p "$R" logs

EXTRA=mhc_zh,toxicn_test,cloak_pinyin_orig,cloak_pinyin_edit,cloak_homophone_orig,cloak_homophone_edit,cloak_chaizi_orig,cloak_chaizi_edit,cloak_symbol_orig,cloak_symbol_edit,cloak_delete_orig,cloak_delete_edit,swap_orig,swap_edit

COMMON=()
if [ -n "${TEST_LIMIT:-}" ]; then COMMON+=(--test_limit "$TEST_LIMIT"); fi
if [ -n "${TRAIN_N:-}" ]; then COMMON+=(--train_n "$TRAIN_N"); fi
if [ -n "${EPOCHS:-}" ]; then COMMON+=(--epochs "$EPOCHS"); fi
if [ -n "${OUT_DIR:-}" ]; then COMMON+=(--out_dir "$OUT_DIR"); fi

run() {
  local json=$1; shift
  if [ -f "$json" ]; then
    echo "跳过（已存在）: $json"
    return 0
  fi
  echo "运行: $*"
  "$@"
}

neural() {  # dataset model tokenizer seed suffix
  local ds=$1 model=$2 tok=$3 seed=$4 suffix=$5
  local name="${ds}_${model}_main_${suffix}_s${seed}"
  run "$R/$name.json" "$PY" src/train.py --dataset "$ds" --model "$model" \
    --tokenizer "$tok" --seed "$seed" --run_suffix "$suffix" \
    --extra_eval "$EXTRA" --save_preds --save_val_preds \
    --decontaminate "${COMMON[@]}"
}

tfidf() {  # dataset tokenizer seed suffix
  local ds=$1 tok=$2 seed=$3 suffix=$4
  local name="${ds}_tfidf_main_${suffix}_s${seed}"
  run "$R/$name.json" "$PY" src/baseline_tfidf.py --dataset "$ds" \
    --tokenizer "$tok" --seed "$seed" --run_suffix "$suffix" \
    --extra_eval "$EXTRA" --save_preds --save_val_preds \
    --decontaminate "${COMMON[@]}"
}

bert() {  # dataset model_path model_tag seed suffix
  local ds=$1 path=$2 tag=$3 seed=$4 suffix=$5
  local name="${ds}_bert_${suffix}_s${seed}"
  run "$R/$name.json" "$PY" src/finetune_bert.py --dataset "$ds" \
    --model_path "$path" --seed "$seed" --run_suffix "$suffix" \
    --extra_eval "$EXTRA" --save_preds --save_val_preds \
    --decontaminate "${COMMON[@]}"
}

case "$QUEUE" in
  q1)
    export CUDA_VISIBLE_DEVICES=${Q1_GPU:-0}
    # 五个从头训练的网络：TextCNN/BiLSTM 各字、词，Transformer 字
    for seed in 0 1 2; do
      neural cold textcnn char "$seed" char
      neural cold textcnn word "$seed" word
      neural cold bilstm char "$seed" char
      neural cold bilstm word "$seed" word
      neural cold transformer char "$seed" char
    done
    # TF-IDF 确定性，只跑种子 0
    tfidf cold char 0 char
    tfidf cold word 0 word
    ;;
  q2)
    export CUDA_VISIBLE_DEVICES=${Q2_GPU:-1}
    # 学习曲线只用 n=1000/3000/10000；全量复用 q1/q3
    # LC_NS 可指定要跑的 n，便于拆到两张卡上并行。
    BASE=("${COMMON[@]}"); LC_NS=${LC_NS:-1000,3000,10000}
    for n in ${LC_NS//,/ }; do
      COMMON=("${BASE[@]}" --train_n "$n")
      for seed in 0 1 2; do
        tfidf cold char "$seed" "n${n}_char"
        neural cold textcnn char "$seed" "n${n}_char"
        neural cold transformer char "$seed" "n${n}_char"
        bert cold models/rbt3 rbt3 "$seed" "n${n}_rbt3"
      done
    done
    ;;
  q3)
    export CUDA_VISIBLE_DEVICES=${Q3_GPU:-2}
    for seed in 0 1 2; do
      bert cold models/rbt3 rbt3 "$seed" rbt3
      bert cold models/chinese-roberta-wwm-ext roberta "$seed" roberta
    done
    ;;
  q4)
    export CUDA_VISIBLE_DEVICES=${Q4_GPU:-3}
    run "$R/bench_cascade_cold_rbt3.json" "$PY" src/bench_cascade.py \
      --dataset cold --model_path models/rbt3 --max_lens 192 \
      --out "$R/bench_cascade_cold_rbt3.json" ${TEST_LIMIT:+--test_limit "$TEST_LIMIT"}
    run "$R/bench_cascade_cold_roberta.json" "$PY" src/bench_cascade.py \
      --dataset cold --model_path models/chinese-roberta-wwm-ext --max_lens 192 \
      --out "$R/bench_cascade_cold_roberta.json" ${TEST_LIMIT:+--test_limit "$TEST_LIMIT"}
    ;;
  *)
    echo "未知队列 $QUEUE，可选 q1/q2/q3/q4" >&2
    exit 2
    ;;
esac
