"""D3 大模型零样本判断：比较第一个输出词元是“是”还是“否”的概率。

不自由生成，也不解析文本：提示词末尾加上 generation prompt 后，取最后一个
位置的 logits，只比较“是”“否”两个单字词元，softmax 得到攻击性概率。
两版提示词固定，不改字。

用法：
    # 选提示词与阈值（COLD 验证集去重后随机 1000 条）
    python src/llm_zeroshot.py --model_path "$QWEN7B" --name qwen7b --select
    # 正式评测 + 测速
    python src/llm_zeroshot.py --model_path "$QWEN7B" --name qwen7b --bench
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import (
    clean_text,
    deduplicate_cold_splits,
    load_eval_set,
    load_splits,
    normalize_for_cold_match,
)
from engine import classification_metrics
from finetune_bert import fine_grained_stats
from result_io import now_iso, preds_path, save_json, setup_logger

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SYSTEM_PROMPT = "你是社交平台的内容审核员。"
PROMPTS = {
    "P1": "判断下面这条评论是否带有攻击性。只回答“是”或“否”。\n评论：{text}",
    "P2": (
        "判断下面这条评论是否带有攻击性。带有攻击性指辱骂、贬低或歧视某个人"
        "或某个群体，包括地域、性别、种族歧视，也包括用讽刺、刻板印象等含蓄"
        "方式表达的贬低。替某个群体辩护、批评或反对偏见的言论不算攻击性，"
        "即使其中提到了偏见的内容。只回答“是”或“否”。\n评论：{text}"
    ),
}
# 与 run_C.sh 的 EXTRA 完全相同
DEFAULT_EXTRA = (
    "mhc_zh,toxicn_test,"
    "cloak_pinyin_orig,cloak_pinyin_edit,"
    "cloak_homophone_orig,cloak_homophone_edit,"
    "cloak_chaizi_orig,cloak_chaizi_edit,"
    "cloak_symbol_orig,cloak_symbol_edit,"
    "cloak_delete_orig,cloak_delete_edit,swap_orig,swap_edit"
)
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}


def clean_comment(text: str, max_chars: int = 192) -> str:
    """按 C 组规则清洗，再按字数截断到 192 个字。"""
    return clean_text(str(text), "cold")[:max_chars]


def build_messages(prompt_key: str, text: str,
                   examples: list[dict] | None = None) -> list[dict]:
    """系统提示 + 可选少样本对 + 最后一条待判断评论。

    ``examples`` 为空（默认）时输出与 D3 完全相同：只有 system 与一条 user。
    D7 的每个例子是一对 user/assistant，助手内容为“是”或“否”。
    """
    if prompt_key not in PROMPTS:
        raise ValueError(f"未知提示词 {prompt_key!r}，可选 {sorted(PROMPTS)}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples or []:
        messages.append({
            "role": "user",
            "content": PROMPTS[prompt_key].format(
                text=clean_comment(example["text"])),
        })
        messages.append({"role": "assistant", "content": example["answer"]})
    messages.append({
        "role": "user",
        "content": PROMPTS[prompt_key].format(text=clean_comment(text)),
    })
    return messages


def select_shot_examples(shots: int = 8, shot_seed: int = 0,
                         splits: dict | None = None) -> list[dict]:
    """按标签分层从去重训练集抽 ``shots/2`` 条攻击性、同样多安全例子。

    用独立的 ``random.Random(shot_seed)``，不碰全局随机状态；与验证集或
    测试集归一化后相同的训练行先排除，等价于“抽到重复就跳过重抽”。
    返回顺序已打乱，并记录去重后训练集的位置 ``train_index``。
    """
    if shots <= 0 or shots % 2:
        raise ValueError("shots 必须为正偶数")
    per_class = shots // 2
    if splits is None:
        splits, _ = deduplicate_cold_splits(load_splits("cold"))
    train, val, test = splits["train"], splits["val"], splits["test"]
    blocked = set(val["text"].map(normalize_for_cold_match))
    blocked |= set(test["text"].map(normalize_for_cold_match))
    rng = random.Random(shot_seed)
    chosen: list[dict] = []
    for label in (1, 0):
        pool = []
        for index, (text, value) in enumerate(
                zip(train["text"].astype(str), train["label"])):
            if int(value) != label:
                continue
            if normalize_for_cold_match(text) in blocked:
                continue
            pool.append(index)
        if len(pool) < per_class:
            raise ValueError(
                f"标签 {label} 只有 {len(pool)} 条可用，不够抽 {per_class} 条")
        rng.shuffle(pool)
        for index in pool[:per_class]:
            chosen.append({
                "train_index": int(index),
                "text": clean_comment(train["text"].iloc[index]),
                "label": int(label),
                "answer": "是" if label == 1 else "否",
            })
    rng.shuffle(chosen)
    return chosen


def single_token_id(tokenizer, word: str) -> int:
    """断言 word 是单个词元，返回其 id。"""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} 不是单个词元，实际切成 {ids}")
    return int(ids[0])


def yes_prob_from_logits(logits: torch.Tensor, yes_id: int,
                         no_id: int) -> torch.Tensor:
    """对 [logit_是, logit_否] 做 softmax，返回“是”的概率。

    ``logits`` 最后一维是词表；可以是一维（单个位置）或二维（batch）。
    """
    pair = torch.stack([logits[..., yes_id], logits[..., no_id]], dim=-1).float()
    return torch.softmax(pair, dim=-1)[..., 0]


def load_model(model_path: str, dtype: torch.dtype, device: torch.device):
    """只读本地目录加载因果语言模型；没有缓存时给出明确提示。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype, local_files_only=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, local_files_only=True)
    except OSError as exc:
        raise SystemExit(
            f"本地没有 {model_path} 的缓存，跳过冒烟；正式运行请设置模型路径。\n{exc}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def score_texts(model, tokenizer, texts: list[str], prompt_key: str,
                batch_size: int, device: torch.device,
                yes_id: int, no_id: int,
                examples: list[dict] | None = None,
                max_length: int = 1024,
                length_stats: dict | None = None) -> np.ndarray:
    """批量算攻击性概率；左填充，取最后一个位置的 logits。

    ``examples`` 为空时与 D3 完全相同；D7 把它作为多轮对话的前缀。
    ``length_stats`` 若给出，会累计本次所有批次的最长非 padding 词元数。
    """
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        chats = [
            tokenizer.apply_chat_template(
                build_messages(prompt_key, text, examples),
                tokenize=False, add_generation_prompt=True,
            )
            for text in chunk
        ]
        enc = tokenizer(chats, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        # Qwen 没有 token_type_ids；用别的分词器做冒烟时也不把它传给模型
        enc.pop("token_type_ids", None)
        if length_stats is not None:
            longest = int(enc["attention_mask"].sum(dim=1).max())
            length_stats["max_input_tokens"] = max(
                length_stats.get("max_input_tokens", 0), longest)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :]
        out.append(yes_prob_from_logits(logits, yes_id, no_id).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


CLASS_NAMES = ["安全", "攻击性"]


def metrics_for(labels, probs: np.ndarray, threshold: float):
    """按阈值把概率转成预测，返回 (指标, 预测)。"""
    preds = (np.asarray(probs) >= threshold).astype(np.int64)
    return classification_metrics(np.asarray(labels), preds, CLASS_NAMES), preds


def best_threshold(labels, probs: np.ndarray) -> tuple[dict, list[dict]]:
    """在 0.01～0.99 步长 0.01 上取宏 F1 最大的阈值；同分取更小的阈值。"""
    best = {"threshold": 0.5, "macro_f1": -1.0}
    curve = []
    for step in range(1, 100):
        threshold = step / 100.0
        metrics, _ = metrics_for(labels, probs, threshold)
        f1 = float(metrics["macro_f1"])
        curve.append({"threshold": threshold, "macro_f1": f1})
        if f1 > best["macro_f1"]:
            best = {"threshold": threshold, "macro_f1": f1}
    return best, curve


def load_selection(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"选择结果不存在: {path}，先跑 --select")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def result_stem(args) -> str:
    """D3 用模型名；D7 在名字后加 ``_fs{shot_seed}``。"""
    if args.shots:
        return f"{args.name}_fs{args.shot_seed}"
    return args.name


def run_selection(args, tokenizer, model, yes_id: int, no_id: int,
                  device: torch.device, logger=None) -> dict:
    """COLD 验证集去重后抽 1000 条，选提示词与阈值。

    D7（``--shots > 0``）只跑 P2，例子用 ``--shot_seed`` 抽。
    """
    splits = load_splits("cold")
    dedup, _ = deduplicate_cold_splits(splits)
    val = dedup["val"]
    n = min(1000, len(val))
    if args.limit is not None:
        n = min(n, args.limit)
    sample = val.sample(n=n, random_state=0).reset_index(drop=True)
    texts = sample["text"].astype(str).tolist()
    labels = sample["label"].to_numpy()
    examples = (select_shot_examples(args.shots, args.shot_seed, dedup)
                if args.shots else None)
    max_length = 4096 if args.shots else 1024
    prompt_keys = ("P2",) if args.shots else ("P1", "P2")
    stats: dict = {}
    per_prompt = {}
    for prompt_key in prompt_keys:
        probs = score_texts(model, tokenizer, texts, prompt_key,
                            args.batch_size, device, yes_id, no_id,
                            examples=examples, max_length=max_length,
                            length_stats=stats)
        metrics, _ = metrics_for(labels, probs, 0.5)
        best, curve = best_threshold(labels, probs)
        per_prompt[prompt_key] = {
            "macro_f1_at_0.5": float(metrics["macro_f1"]),
            "accuracy_at_0.5": float(metrics["accuracy"]),
            "best_threshold": best["threshold"],
            "best_macro_f1": best["macro_f1"],
            "threshold_curve": curve,
        }
    selected = "P2" if args.shots else "P1"
    if not args.shots and (per_prompt["P2"]["macro_f1_at_0.5"]
                           > per_prompt["P1"]["macro_f1_at_0.5"]):
        selected = "P2"
    payload = {
        "task": "D3_选择",
        "model_name": args.name,
        "model_path": str(args.model_path),
        "seed": args.seed,
        "run_at": now_iso(),
        "val_n": int(n),
        "sample": "COLD 验证集去重后按 random_state=0 随机抽",
        "prompts": {"P1": PROMPTS["P1"], "P2": PROMPTS["P2"],
                    "system": SYSTEM_PROMPT},
        "per_prompt": per_prompt,
        "selected_prompt": selected,
        "selected_threshold": per_prompt[selected]["best_threshold"],
        "selected_macro_f1": per_prompt[selected]["best_macro_f1"],
    }
    if args.shots:
        payload.update({
            "shots": int(args.shots),
            "shot_seed": int(args.shot_seed),
            "shot_examples": examples or [],
            "max_input_tokens": stats.get("max_input_tokens"),
            "max_length": max_length,
        })
    path = Path(args.out_dir) / f"D3_选择_{result_stem(args)}.json"
    save_json(path, payload)
    if logger is not None and args.shots:
        logger.info("选择集最长输入 %s 词元（max_length=%d）",
                    stats.get("max_input_tokens"), max_length)
    print(f"选择结果写入 {path}")
    if args.shots:
        print(f"P2={per_prompt['P2']['macro_f1_at_0.5']:.4f} -> P2，"
              f"阈值={payload['selected_threshold']:.2f}")
    else:
        print(f"P1={per_prompt['P1']['macro_f1_at_0.5']:.4f} "
              f"P2={per_prompt['P2']['macro_f1_at_0.5']:.4f} -> {selected}，"
              f"阈值={payload['selected_threshold']:.2f}")
    return payload


def _save_preds(path: Path, index, labels, preds, probs: np.ndarray,
                frame: pd.DataFrame | None) -> None:
    from result_io import save_predictions
    matrix = np.column_stack([1.0 - probs, probs])
    pair_ids = (frame["pair_id"].to_numpy()
                if frame is not None and "pair_id" in frame else None)
    save_predictions(path, np.asarray(index), np.asarray(labels),
                     np.asarray(preds), matrix, CLASS_NAMES, pair_ids=pair_ids)


def run_eval(args, tokenizer, model, prompt_key: str, threshold: float,
             yes_id: int, no_id: int, device: torch.device,
             logger, examples: list[dict] | None = None) -> dict:
    """COLD 测试集 + 14 个额外评测集，逐条预测与指标。"""
    out_dir = Path(args.out_dir)
    max_length = 4096 if args.shots else 1024
    stats: dict = {}
    test = load_splits("cold")["test"]
    if args.limit is not None:
        test = test.iloc[:args.limit].reset_index(drop=True)
    texts = test["text"].astype(str).tolist()
    labels = test["label"].to_numpy()
    probs = score_texts(model, tokenizer, texts, prompt_key,
                        args.batch_size, device, yes_id, no_id,
                        examples=examples, max_length=max_length,
                        length_stats=stats)
    metrics, preds = metrics_for(labels, probs, threshold)
    metrics_05, preds_05 = metrics_for(labels, probs, 0.5)
    run_name = f"cold_llm_{result_stem(args)}"
    _save_preds(preds_path(out_dir, run_name), np.arange(len(test)),
                labels, preds, probs, test)
    logger.info("测试集 n=%d 阈值=%.2f 宏F1=%.4f 准确率=%.4f",
                len(test), threshold, metrics["macro_f1"], metrics["accuracy"])

    extra = {}
    for name in args.eval_sets:
        try:
            frame = load_eval_set(name)
        except FileNotFoundError as exc:
            logger.info("额外评测 %s: 缺文件（%s），跳过", name, exc)
            continue
        if args.limit is not None:
            frame = frame.iloc[:args.limit].reset_index(drop=True)
        e_probs = score_texts(model, tokenizer, frame["text"].astype(str).tolist(),
                              prompt_key, args.batch_size, device, yes_id, no_id,
                              examples=examples, max_length=max_length,
                              length_stats=stats)
        e_metrics, e_preds = metrics_for(frame["label"].to_numpy(), e_probs, threshold)
        extra[name] = {
            "accuracy": float(e_metrics["accuracy"]),
            "macro_f1": float(e_metrics["macro_f1"]),
            "n": int(len(frame)),
            "pos_precision": float(e_metrics["pos_precision"]),
            "pos_recall": float(e_metrics["pos_recall"]),
        }
        _save_preds(preds_path(out_dir, f"{run_name}__{name}"),
                    np.arange(len(frame)), frame["label"].to_numpy(),
                    e_preds, e_probs, frame)
        logger.info("额外评测 %s: n=%d 宏F1=%.4f", name, len(frame),
                    e_metrics["macro_f1"])

    if args.shots:
        logger.info("评测最长输入 %s 词元（max_length=%d）",
                    stats.get("max_input_tokens"), max_length)
    payload = {
        "dataset": "cold",
        "model": "llm_causal",
        "model_name": args.name,
        "model_path": str(args.model_path),
        "seed": args.seed,
        "run_name": run_name,
        "run_at": now_iso(),
        "prompt": prompt_key,
        "threshold": float(threshold),
        "config": {
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "device": str(device),
            "prompt": prompt_key,
            "threshold": float(threshold),
            "max_chars": 192,
            "n_test": int(len(test)),
            "eval_sets": list(args.eval_sets),
            "system_prompt": SYSTEM_PROMPT,
        },
        "n_test": int(len(test)),
        "test_accuracy": float(metrics["accuracy"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "test_per_class_f1": metrics["per_class_f1"],
        "test_pos_precision": float(metrics["pos_precision"]),
        "test_pos_recall": float(metrics["pos_recall"]),
        "test_fine_grained": fine_grained_stats(test, preds),
        "test_accuracy_at_0.5": float(metrics_05["accuracy"]),
        "test_macro_f1_at_0.5": float(metrics_05["macro_f1"]),
        "test_pos_precision_at_0.5": float(metrics_05["pos_precision"]),
        "test_pos_recall_at_0.5": float(metrics_05["pos_recall"]),
        "test_fine_grained_at_0.5": fine_grained_stats(test, preds_05),
        "extra_eval": extra,
        "ms_per_1k": None,
    }
    if args.shots:
        payload.update({
            "shots": int(args.shots),
            "shot_seed": int(args.shot_seed),
            "shot_examples": examples or [],
            "max_input_tokens": stats.get("max_input_tokens"),
            "max_length": max_length,
        })
        payload["config"].update({
            "max_length": max_length,
            "shots": int(args.shots),
            "shot_seed": int(args.shot_seed),
        })
    return payload


def run_bench(args, tokenizer, model, prompt_key: str, yes_id: int,
              no_id: int, device: torch.device,
              examples: list[dict] | None = None) -> dict:
    """预热 2 个 batch，对测试集前 1000 条计时 3 次取中位数。"""
    max_length = 4096 if args.shots else 1024
    test = load_splits("cold")["test"]
    if args.limit is not None:
        test = test.iloc[:args.limit]
    texts = test["text"].astype(str).tolist()[:1000]
    if not texts:
        return {"n": 0, "ms_per_1k": None, "reps_sec": []}
    warm = min(2 * args.batch_size, len(texts))
    score_texts(model, tokenizer, texts[:warm], prompt_key,
                args.batch_size, device, yes_id, no_id,
                examples=examples, max_length=max_length)
    reps = []
    for _ in range(3):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        score_texts(model, tokenizer, texts, prompt_key,
                    args.batch_size, device, yes_id, no_id,
                    examples=examples, max_length=max_length)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reps.append(time.perf_counter() - start)
    median_sec = float(np.median(reps))
    return {
        "n": len(texts),
        "batch_size": args.batch_size,
        "warmup_batches": 2,
        "reps": 3,
        "reps_sec": reps,
        "median_sec": median_sec,
        "ms_per_1k": median_sec / len(texts) * 1_000_000.0,  # 秒/条 × 10^6 = 毫秒/千条
        "prompt": prompt_key,
        "dtype": args.dtype,
        "device": str(device),
    }


def derive_name(model_path: str) -> str:
    """从模型目录名猜一个短名字；run_D.sh 会显式传 --name。"""
    import re
    low = Path(model_path).name.lower()
    if "1.5b" in low or "1_5b" in low:
        return "qwen15b"
    if "7b" in low:
        return "qwen7b"
    return re.sub(r"[^a-z0-9]+", "_", low).strip("_") or "llm"


def main() -> None:
    parser = argparse.ArgumentParser(description="D3 大模型零样本判断")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--name", default="", help="结果名，如 qwen7b / qwen15b")
    parser.add_argument("--prompt", choices=("P1", "P2"), default=None)
    parser.add_argument("--eval", default=DEFAULT_EXTRA,
                        help="逗号分隔的额外评测集，默认与 run_C.sh 的 EXTRA 相同")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="每个评测集只取前 N 条；默认不限，冒烟用")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=0,
                        help="D7 少样本例子数，默认 0 与 D3 完全相同")
    parser.add_argument("--shot_seed", type=int, default=0,
                        help="抽少样本例子的随机种子")
    parser.add_argument("--select", action="store_true",
                        help="只在验证集上选提示词与阈值")
    parser.add_argument("--bench", action="store_true", help="评测后测速")
    parser.add_argument("--selection_file", default="")
    parser.add_argument("--out_dir", default=str(RESULTS))
    parser.add_argument("--log_dir", default=str(ROOT / "logs"))
    args = parser.parse_args()
    args.name = args.name or derive_name(args.model_path)
    if args.shots < 0 or (args.shots > 0 and args.shots % 2):
        raise SystemExit("--shots 必须为 0 或正偶数")
    args.eval_sets = [x.strip() for x in args.eval.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    args.out_dir = str(out_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("要求 CUDA 但当前环境不可用；冒烟请显式传 --device cpu")
    device = torch.device(args.device)
    stem = result_stem(args)
    logger = setup_logger(Path(args.log_dir) / f"cold_llm_{stem}.log")
    logger.info("参数: %s", vars(args))

    tokenizer, model = load_model(args.model_path, DTYPES[args.dtype], device)
    yes_id = single_token_id(tokenizer, "是")
    no_id = single_token_id(tokenizer, "否")
    logger.info("是/否 词元 id: %d / %d", yes_id, no_id)

    if args.select:
        run_selection(args, tokenizer, model, yes_id, no_id, device, logger)
        return

    selection_path = Path(args.selection_file) if args.selection_file else (
        out_dir / f"D3_选择_{stem}.json")
    selection = load_selection(selection_path) if selection_path.exists() else {}
    prompt_key = args.prompt or selection.get("selected_prompt")
    threshold = (args.threshold if args.threshold is not None
                 else selection.get("selected_threshold"))
    if prompt_key is None or threshold is None:
        raise SystemExit("没有 --prompt/--threshold，也找不到选择结果；先跑 --select")
    shot_examples = None
    if args.shots:
        shot_examples = selection.get("shot_examples") or select_shot_examples(
            args.shots, args.shot_seed)
        logger.info("少样本：%d 条，shot_seed=%d，最长选择输入=%s",
                    len(shot_examples), args.shot_seed,
                    selection.get("max_input_tokens"))
    logger.info("使用提示词 %s，阈值 %.2f（选择文件 %s）",
                prompt_key, threshold, selection_path)

    payload = run_eval(args, tokenizer, model, prompt_key, float(threshold),
                       yes_id, no_id, device, logger, examples=shot_examples)
    if args.bench:
        bench = run_bench(args, tokenizer, model, prompt_key,
                          yes_id, no_id, device, examples=shot_examples)
        payload["ms_per_1k"] = bench["ms_per_1k"]
        payload["bench"] = bench
        bench_path = out_dir / f"bench_llm_{stem}.json"
        save_json(bench_path, {"task": "D3_bench", **bench})
        logger.info("测速: %.2f ms/千条（%d 条）",
                    bench["ms_per_1k"] or float("nan"), bench["n"])
    result_path = out_dir / f"cold_llm_{stem}.json"
    save_json(result_path, payload)
    logger.info("结果写入 %s", result_path)
    print(json.dumps({
        "result": str(result_path),
        "prompt": prompt_key,
        "threshold": threshold,
        "macro_f1": payload["test_macro_f1"],
        "macro_f1_at_0.5": payload["test_macro_f1_at_0.5"],
        "ms_per_1k": payload["ms_per_1k"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
