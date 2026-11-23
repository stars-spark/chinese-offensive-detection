"""D8 全量：试点放行后，在 COLD 测试集与各额外评测集上跑 API 零样本。

1. 选阈值：与试点同一个 1000 条池子（前 200 条沿用试点结果），提示词固定 P2，
   用 llm_zeroshot.best_threshold 选阈值，与 D3 同一函数。
2. 评测：COLD 测试集 + D3 的 14 个额外评测集。逐条原始结果存
   preds/D8_raw__{集}.parquet，不含评论原文，可断点续跑。
3. 被拦截、报错或无概率的条目在主结果里按“判为攻击性”计（概率记 1.0），
   另报剔除这些条目后的结果。
输出与 D3 同格式，供 analyze_D 读：results/cold_llm_deepseek.json、
preds/cold_llm_deepseek[__集].parquet、results/D8_选择_deepseek.json。
高峰时段在请求前等到谷时；密钥只从环境变量 DEEPSEEK_API_KEY 读。
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import llm_api as A
from data import load_eval_set, load_splits
from finetune_bert import fine_grained_stats
from llm_zeroshot import DEFAULT_EXTRA, PROMPTS, SYSTEM_PROMPT, _save_preds, best_threshold, metrics_for
from result_io import now_iso, preds_path, save_json

NAME = "deepseek"
RUN_NAME = f"cold_llm_{NAME}"


def raw_path(tag: str) -> Path:
    return A.RESULTS / "preds" / f"D8_raw__{tag}.parquet"


def score_frame(client, frame: pd.DataFrame, tag: str, args, logger):
    """逐条打分（续跑跳过已有 index），返回按原顺序排好的原始结果与本次新跑的计时。"""
    sample = pd.DataFrame({"index": np.arange(len(frame), dtype=np.int64),
                           "text": frame["text"].astype(str).to_numpy(),
                           "label": frame["label"].astype(int).to_numpy()})
    before = len(A.load_results(raw_path(tag)))
    start = time.perf_counter()
    out = A.run_pending(client, sample, raw_path(tag), workers=args.workers, flush_every=50,
                        logger=logger, peak_check=A.make_peak_check("wait", logger))
    wall = time.perf_counter() - start
    out = out.drop_duplicates("index", keep="last").set_index("index").loc[sample["index"]].reset_index()
    return out, {"n_new": int(len(out) - before), "wall_sec": float(wall)}


def probs_of(raw: pd.DataFrame):
    p = pd.to_numeric(raw["prob_pos"], errors="coerce").to_numpy(dtype=float)
    has = ~np.isnan(p)
    return np.where(has, p, 1.0), has


def status_counts(raw: pd.DataFrame) -> dict:
    return {s: int((raw["status"] == s).sum()) for s in A.STATUSES}


def brief(m: dict, n: int) -> dict:
    return {"n": int(n), "accuracy": float(m["accuracy"]), "macro_f1": float(m["macro_f1"]),
            "pos_precision": float(m["pos_precision"]), "pos_recall": float(m["pos_recall"])}


def excluded_block(labels, probs, has, thr) -> dict | None:
    """剔除拦截、报错、无概率条目后的指标；全部可评时与主结果相同。"""
    if not has.any():
        return None
    m, _ = metrics_for(labels[has], probs[has], thr)
    return brief(m, int(has.sum()))


def run_select(client, args, logger) -> dict:
    pool, meta = A.pilot_sample(limit=A.POOL_N, seed=0)
    pilot = A.RESULTS / "preds" / "D8_试点.parquet"
    if not raw_path("val1000").exists() and pilot.exists():
        shutil.copy(pilot, raw_path("val1000"))  # 试点就是这 1000 条的前 200 条，index 相同
    raw, timing = score_frame(client, pool, "val1000", args, logger)
    probs, has = probs_of(raw)
    labels = pool["label"].to_numpy()
    m05, _ = metrics_for(labels, probs, 0.5)
    best, curve = best_threshold(labels, probs)
    payload = {
        "task": "D8_选择", "model_name": NAME, "api_model": args.model, "endpoint": args.endpoint,
        "run_at": now_iso(), "val_n": int(len(pool)), "sample": meta["method"],
        "prompts": {"P2": PROMPTS["P2"], "system": SYSTEM_PROMPT}, "selected_prompt": "P2",
        "macro_f1_at_0.5": float(m05["macro_f1"]), "accuracy_at_0.5": float(m05["accuracy"]),
        "selected_threshold": best["threshold"], "selected_macro_f1": best["macro_f1"],
        "threshold_curve": curve, "status_counts": status_counts(raw),
        "n_distinct_prob": int(pd.Series(probs[has]).round(4).nunique()), "timing": timing,
    }
    save_json(A.RESULTS / f"D8_选择_{NAME}.json", payload)
    logger.info("选择：0.5 处宏F1=%.4f，选出阈值=%.2f（宏F1=%.4f）",
                m05["macro_f1"], best["threshold"], best["macro_f1"])
    return payload


def usage_cost(frames: list[pd.DataFrame]) -> dict:
    allr = pd.concat(frames, ignore_index=True)
    pt = pd.to_numeric(allr["prompt_tokens"], errors="coerce").mean()
    ct = pd.to_numeric(allr["completion_tokens"], errors="coerce").mean()
    c = A.PRICING_CN
    return {"n_calls": int(len(allr)), "avg_prompt_tokens": float(pt), "avg_completion_tokens": float(ct),
            "cost_per_1k_cny_off_peak_cache_miss": A.cost_per_1k(pt, ct, c["input_cache_miss_off_peak"], c["output_off_peak"]),
            "cost_per_1k_cny_peak_cache_miss": A.cost_per_1k(pt, ct, c["input_cache_miss_peak"], c["output_peak"]),
            "pricing_source": A.PRICING_SOURCE_CN, "raw_cn": A.PRICING_RAW_CN}


def run_eval(client, args, thr: float, logger) -> dict:
    test = load_splits("cold")["test"].reset_index(drop=True)
    raw, timing = score_frame(client, test, "test", args, logger)
    labels = test["label"].to_numpy()
    probs, has = probs_of(raw)
    m, preds = metrics_for(labels, probs, thr)
    m05, preds05 = metrics_for(labels, probs, 0.5)
    _save_preds(preds_path(A.RESULTS, RUN_NAME), np.arange(len(test)), labels, preds, probs, test)
    logger.info("测试集 n=%d 阈值=%.2f 宏F1=%.4f", len(test), thr, m["macro_f1"])
    raws, extra, timings = [raw], {}, {"test": timing}
    for name in args.eval_sets:
        frame = load_eval_set(name).reset_index(drop=True)
        r, t = score_frame(client, frame, name, args, logger)
        lab = frame["label"].to_numpy()
        p, h = probs_of(r)
        em, ep = metrics_for(lab, p, thr)
        extra[name] = {**brief(em, len(frame)), "status_counts": status_counts(r),
                       "excluded": excluded_block(lab, p, h, thr)}
        _save_preds(preds_path(A.RESULTS, f"{RUN_NAME}__{name}"), np.arange(len(frame)), lab, ep, p, frame)
        raws.append(r); timings[name] = t
        logger.info("额外评测 %s: n=%d 宏F1=%.4f", name, len(frame), em["macro_f1"])
    lat = pd.to_numeric(raw["latency_ms"], errors="coerce")
    payload = {
        "dataset": "cold", "model": "llm_api", "model_name": NAME, "run_name": RUN_NAME, "seed": 0,
        "run_at": now_iso(), "prompt": "P2", "threshold": float(thr),
        "api": {"model": args.model, "endpoint": args.endpoint, "temperature": 0, "max_tokens": 1,
                "top_logprobs": A.MAX_TOP_LOGPROBS, "thinking": "disabled", "workers": args.workers},
        "config": {"prompt": "P2", "threshold": float(thr), "max_chars": 192, "n_test": int(len(test)),
                   "eval_sets": list(args.eval_sets), "system_prompt": SYSTEM_PROMPT,
                   "missing_as_attack": True},
        "n_test": int(len(test)),
        "test_accuracy": float(m["accuracy"]), "test_macro_f1": float(m["macro_f1"]),
        "test_per_class_f1": m["per_class_f1"],
        "test_pos_precision": float(m["pos_precision"]), "test_pos_recall": float(m["pos_recall"]),
        "test_fine_grained": fine_grained_stats(test, preds),
        "test_accuracy_at_0.5": float(m05["accuracy"]), "test_macro_f1_at_0.5": float(m05["macro_f1"]),
        "test_pos_precision_at_0.5": float(m05["pos_precision"]),
        "test_pos_recall_at_0.5": float(m05["pos_recall"]),
        "test_fine_grained_at_0.5": fine_grained_stats(test, preds05),
        "test_status_counts": status_counts(raw),
        "test_excluded": excluded_block(labels, probs, has, thr),
        "extra_eval": extra,
        "ms_per_1k": None,
        "api_latency": {"mean_ms": float(lat.mean()), "median_ms": float(lat.median()),
                        "note": "单条请求往返延迟，含网络；与本地 GPU 的每千条耗时不可比，照报"},
        "timing": timings, "cost": usage_cost(raws),
    }
    save_json(A.RESULTS / f"{RUN_NAME}.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D8 全量：选阈值 + 评测")
    ap.add_argument("--stage", choices=("select", "eval", "all"), default="all")
    ap.add_argument("--model", default=A.DEFAULT_MODEL)
    ap.add_argument("--base_url", default=A.DEFAULT_BASE_URL)
    ap.add_argument("--workers", type=int, default=A.MAX_WORKERS)
    ap.add_argument("--eval", default=",".join(DEFAULT_EXTRA.split(",")))
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= A.MAX_WORKERS:
        ap.error("并发须在 1～4")
    args.endpoint = A.resolve_endpoint(args.base_url, "")
    args.eval_sets = [s for s in args.eval.split(",") if s]
    key = A.load_api_key()
    logger = A.make_logger(A.ROOT / "logs" / "D8_全量.log", key)
    client = A.DeepSeekClient(key, model=args.model, endpoint=args.endpoint, logger=logger)
    sel_path = A.RESULTS / f"D8_选择_{NAME}.json"
    if args.stage in ("select", "all"):
        sel = run_select(client, args, logger)
    else:
        import json
        sel = json.load(open(sel_path, encoding="utf-8"))
    if args.stage in ("eval", "all"):
        out = run_eval(client, args, float(sel["selected_threshold"]), logger)
        print(f"测试宏F1={out['test_macro_f1']:.4f} 阈值={out['threshold']:.2f} "
              f"状态={out['test_status_counts']} 每千条谷时费用={out['cost']['cost_per_1k_cny_off_peak_cache_miss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
