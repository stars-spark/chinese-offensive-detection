"""实验 L：级联两级的推理测速（只测速，不训练）。

DistilBERT 从本地 ``models/distilbert-base-uncased`` 载入预训练权重，
二分类头随机初始化；不训练。截断 256、512 各测一次，批大小 64，FP32，
计时包含分词；先跑 1 个批次预热，再完整测 3 次取中位数。另测 chunk：
按 ``finetune_bert.predict_chunks`` 的规则（510 子词一段、相邻段重叠
128）切段，全部段按批大小 64 推理，计时含分词与切段，测 3 次取中位数。

TF-IDF 用 ``baseline_tfidf.build_vectorizer`` 的同一套向量化设置，在 IMDB
训练集上拟合一次（不计时）；只计测试集“向量化 + 预测概率”的时间，单进程，
预热一次后测 3 次取中位数。成本公式需要概率，所以预测用 ``predict_proba``。

输出 JSON 含每千条毫秒数、GPU/CPU 型号、torch 与 sklearn 版本、线程数；
chunk 另含总段数与平均每条评论的段数。
``--test_limit`` 与 ``--out`` 给冒烟用，正式运行不传。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline_tfidf import build_vectorizer
from data import load_splits
from finetune_bert import encode_split, predict_chunks
from result_io import now_iso

ROOT = Path(__file__).resolve().parents[1]
CHUNK_MAX_LEN = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="级联的 BERT 与 TF-IDF 推理测速")
    p.add_argument("--dataset", choices=("imdb", "cold"), default="imdb")
    p.add_argument("--max_lens", type=int, nargs="+", default=None,
                   help="默认 IMDB [256,512]，COLD [192]")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup_batches", type=int, default=1,
                   help="BERT 预热批次数，不计时")
    p.add_argument("--device", default="cuda")
    p.add_argument("--test_limit", type=int, default=None,
                   help="只测测试集前 N 条；默认不限，冒烟用")
    p.add_argument("--model_path", default=None,
                   help="本地预训练模型目录；IMDB 默认 distilbert，COLD 默认 rbt3")
    p.add_argument("--model_name", default=None,
                   help="写进 JSON 的模型名，默认取 model_path 目录名")
    p.add_argument("--no_chunk", action="store_true",
                   help="不测 chunk；COLD 默认不测")
    p.add_argument("--out", default=None,
                   help="默认 results/bench_cascade.json；冒烟可指到 results/smoke_c/")
    return p.parse_args()


def cpu_model_name() -> str:
    """优先读 /proc/cpuinfo 的 model name；拿不到时退回 platform。"""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _time_distilbert(model, tokenizer, texts: list[str], max_len: int,
                     batch_size: int, device: torch.device,
                     warmup_batches: int, reps: int) -> dict:
    """测完整测试集：分词 + 逐批前向，返回每次总耗时和每千条耗时。"""
    def one_pass() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        ids, mask, _ = encode_split(tokenizer, texts, max_len, "head")
        for begin in range(0, len(texts), batch_size):
            batch_ids = torch.from_numpy(ids[begin:begin + batch_size]).to(device)
            batch_mask = torch.from_numpy(mask[begin:begin + batch_size]).to(device)
            model(input_ids=batch_ids, attention_mask=batch_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    # 预热只看前 warmup_batches 个批次，分词和整个流程都走一遍
    warm_n = min(len(texts), batch_size * max(1, warmup_batches))
    if warm_n:
        warm_ids, warm_mask, _ = encode_split(tokenizer, texts[:warm_n], max_len, "head")
        for begin in range(0, warm_n, batch_size):
            batch_ids = torch.from_numpy(warm_ids[begin:begin + batch_size]).to(device)
            batch_mask = torch.from_numpy(warm_mask[begin:begin + batch_size]).to(device)
            model(input_ids=batch_ids, attention_mask=batch_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del warm_ids, warm_mask

    times = [one_pass() for _ in range(reps)]
    median = statistics.median(times)
    return {
        "n": len(texts),
        "max_len": max_len,
        "batch_size": batch_size,
        "reps_sec": [round(x, 6) for x in times],
        "total_sec_median": median,
        "ms_per_1k": median / max(1, len(texts)) * 1_000_000.0,
    }


def _time_tfidf(texts: list[str], vectorizer, model, reps: int) -> dict:
    """只计“向量化 + predict_proba”；拟合和一次预热都不计时。"""
    def one_pass() -> float:
        start = time.perf_counter()
        matrix = vectorizer.transform(texts)
        model.predict_proba(matrix)
        return time.perf_counter() - start

    one_pass()  # 预热，不计时
    times = [one_pass() for _ in range(reps)]
    median = statistics.median(times)
    return {
        "n": len(texts),
        "reps_sec": [round(x, 6) for x in times],
        "total_sec_median": median,
        "ms_per_1k": median / max(1, len(texts)) * 1_000_000.0,
    }


def _time_chunk(model, tokenizer, texts: list[str], max_len: int,
                batch_size: int, device: torch.device, reps: int) -> dict:
    """按 head512 的 chunk 规则切段并推理，计时含分词与切段。"""
    def one_pass() -> tuple[float, np.ndarray]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _, _, counts = predict_chunks(
            model, tokenizer, texts, max_len, batch_size, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start, counts

    runs = [one_pass() for _ in range(reps)]
    times = [run[0] for run in runs]
    counts = runs[-1][1]
    median = statistics.median(times)
    return {
        "n": len(texts),
        "max_len": max_len,
        "chunk_size": max_len - 2,
        "overlap": 128,
        "batch_size": batch_size,
        "n_segments": int(counts.sum()),
        "segments_per_review": float(counts.mean()),
        "max_segments": int(counts.max()),
        "reps_sec": [round(x, 6) for x in times],
        "total_sec_median": median,
        "ms_per_1k": median / max(1, len(texts)) * 1_000_000.0,
    }


def main() -> None:
    args = parse_args()
    if args.max_lens is None:
        args.max_lens = [192] if args.dataset == "cold" else [256, 512]
    if args.test_limit is not None and args.test_limit <= 0:
        raise SystemExit("--test_limit 必须为正整数或省略")
    if args.batch_size <= 0 or args.reps <= 0:
        raise SystemExit("--batch_size 与 --reps 必须为正整数")
    if any(n <= 0 for n in args.max_lens):
        raise SystemExit("--max_lens 必须为正整数")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("要求 CUDA 但找不到显卡；CPU 冒烟请显式传 --device cpu")
    device = torch.device(args.device)
    torch.manual_seed(0)
    chinese = args.dataset == "cold"
    default_model = "rbt3" if chinese else "distilbert-base-uncased"
    model_path = args.model_path or str(ROOT / "models" / default_model)
    model_name = args.model_name or Path(model_path).name
    do_chunk = (not args.no_chunk) and not chinese

    splits = load_splits(args.dataset)
    train = splits["train"]
    test = splits["test"]
    if args.test_limit is not None:
        test = test.iloc[:args.test_limit].reset_index(drop=True)
    texts = test["text"].tolist()
    print(f"测试集 {len(texts)} 条；训练集 {len(train)} 条用于拟合 TF-IDF", flush=True)

    # TF-IDF：向量化器与 baseline_tfidf 相同（COLD 字 1~3 gram），C 取 1.0
    tokenizer_kind = "char" if chinese else "word"
    vectorizer = build_vectorizer(tokenizer_kind, args.dataset)
    start = time.perf_counter()
    x_train = vectorizer.fit_transform(train["text"].tolist())
    classifier = LogisticRegression(
        C=1.0, max_iter=2000, solver="liblinear", random_state=0
    )
    classifier.fit(x_train, train["label"].to_numpy())
    print(f"TF-IDF 拟合完成：{x_train.shape[1]} 个特征，耗时 {time.perf_counter() - start:.1f}s",
          flush=True)
    del x_train
    tfidf_info = _time_tfidf(texts, vectorizer, classifier, args.reps)
    print(f"TF-IDF: {tfidf_info['ms_per_1k']:.2f} ms/千条", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, local_files_only=True,
    )
    model.to(device)
    model.eval()
    model.float()  # 明确 FP32，不用 autocast
    bert_info = {}
    with torch.no_grad():
        for max_len in args.max_lens:
            print(f"[{model_name}] max_len={max_len} ...", flush=True)
            info = _time_distilbert(
                model, tokenizer, texts, max_len, args.batch_size, device,
                args.warmup_batches, args.reps,
            )
            bert_info[str(max_len)] = info
            print(f"  {info['ms_per_1k']:.2f} ms/千条", flush=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        chunk_info = None
        if do_chunk:
            print(f"[DistilBERT] chunk（{CHUNK_MAX_LEN - 2} 子词一段，重叠 128）...",
                  flush=True)
            chunk_info = _time_chunk(
                model, tokenizer, texts, CHUNK_MAX_LEN, args.batch_size, device,
                args.reps,
            )
            print(f"  {chunk_info['ms_per_1k']:.2f} ms/千条，"
                  f"{chunk_info['n_segments']} 段，平均 {chunk_info['segments_per_review']:.4f} 段/条",
                  flush=True)

    ms = {"tfidf": tfidf_info["ms_per_1k"]}
    if chunk_info is not None:
        ms["chunk"] = chunk_info["ms_per_1k"]
    ms.update({f"distilbert_{k}": v["ms_per_1k"] for k, v in bert_info.items()})
    if chinese:
        ms.update({f"bert_{k}": v["ms_per_1k"] for k, v in bert_info.items()})
    payload = {
        "generated_at": now_iso(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cpu": cpu_model_name(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "reps": args.reps,
        "test_limit": args.test_limit,
        "test_n": len(texts),
        "fp32": True,
        "distilbert": bert_info,
        "chunk": chunk_info,
        "tfidf": {**tfidf_info, "C": 1.0,
                  "vectorizer": "baseline_tfidf.build_vectorizer"},
        "ms_per_1k": ms,
    }
    if chinese:
        payload["dataset"] = args.dataset
        payload["model_name"] = model_name
        payload["model_path"] = model_path
        payload["tfidf"]["tokenizer"] = tokenizer_kind
        payload["bert"] = bert_info
    out = Path(args.out) if args.out else ROOT / "results" / "bench_cascade.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
