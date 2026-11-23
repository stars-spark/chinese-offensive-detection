"""D2 反歧视误判的积分梯度归因。

对 rbt3 的**词向量层输出**做积分梯度：基线把正文词元换成 [PAD] 的词向量，
[CLS]、[SEP] 保留原词向量；目标为 logit[1] − logit[0]（攻击性方向）。
归因按 tokenizer 的 offset_mapping 分摊回原文字符，再计算群体词字的富集比、
完整性误差和删字忠实性检查。

用法示例：
    python src/attribution.py --model_path checkpoints/cold_bert_demo_rbt3_s0 \
        --seed 0 --device cuda --check_ref
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_splits
from make_swap import GROUP_WORDS
from make_swap import find_matches as find_group_matches
from result_io import now_iso, preds_path, save_json, setup_logger

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
MAX_LEN = 192
BATCH_SIZE = 32
COMPLETENESS_THRESHOLD = 0.05
GROUP_NAMES = {
    "A": "细粒度 3 且判为攻击性",
    "B": "细粒度 3 且判为安全",
    "C": "细粒度 2 且判为攻击性",
}


def encode_with_offsets(tokenizer, text: str, max_len: int = MAX_LEN):
    """按 finetune_bert 的 head 截断编码一条文本，并保留字符偏移。

    不用 tokenizer 自带的 truncation，而是先不加特殊符号编码、取前
    ``max_len-2`` 个词元，再手工加 [CLS]/[SEP]，保证与 C 组训练/评测
    看到完全相同的词元序列。
    """
    raw = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    body = raw["input_ids"][:max_len - 2]
    offsets = raw["offset_mapping"][:max_len - 2]
    ids = [tokenizer.cls_token_id] + body + [tokenizer.sep_token_id]
    mask = [1] * len(ids)
    offsets = [(0, 0)] + list(offsets) + [(0, 0)]
    return ids, mask, offsets


def body_positions(ids: list[int], tokenizer) -> list[int]:
    """正文位置：排除 [CLS]、[SEP]、[PAD]；[UNK] 仍算正文。"""
    skip = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    return [i for i, x in enumerate(ids) if x not in skip]


def _target_diff(model, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logit[1] − logit[0]，输入是词向量序列。"""
    logits = model(inputs_embeds=embeds, attention_mask=mask).logits
    return logits[0, 1] - logits[0, 0]


def integrated_gradients(model, input_embeds: torch.Tensor,
                         baseline_embeds: torch.Tensor, mask: torch.Tensor,
                         n_steps: int = 50) -> torch.Tensor:
    """梯形法积分梯度，返回每个词元一个标量归因（对嵌入维求和）。

    ``input_embeds`` / ``baseline_embeds`` 形状为 ``[1, L, H]``。每个插值点
    单独前向反向，避免 200 步时一次塞进 batch 把显存撑爆。
    """
    if n_steps < 1:
        raise ValueError("n_steps 必须 >= 1")
    alphas = torch.linspace(0.0, 1.0, n_steps + 1,
                            device=input_embeds.device, dtype=input_embeds.dtype)
    grads = []
    for alpha in alphas:
        embeds = (baseline_embeds + alpha * (input_embeds - baseline_embeds))
        embeds = embeds.detach().requires_grad_(True)
        target = _target_diff(model, embeds, mask)
        grad = torch.autograd.grad(target, embeds)[0]
        grads.append(grad.detach())
    # 梯形公式：端点各算一半，中间点全算
    total = (grads[0] + grads[-1]) * 0.5 + sum(grads[1:-1])
    avg_grad = total / n_steps
    attr = ((input_embeds - baseline_embeds) * avg_grad).sum(dim=-1)[0]
    return attr.detach()


def token_completeness(model, input_embeds: torch.Tensor,
                       baseline_embeds: torch.Tensor, mask: torch.Tensor,
                       token_attrs: torch.Tensor) -> tuple[float, float, float]:
    """返回 (完整性相对误差, F(x), F(基线))。"""
    with torch.no_grad():
        f_x = float(_target_diff(model, input_embeds, mask))
        f_base = float(_target_diff(model, baseline_embeds, mask))
    diff = f_x - f_base
    err = abs(float(token_attrs.sum()) - diff)
    rel = err / max(abs(diff), 1e-12)
    return rel, f_x, f_base


def token_attrs_to_chars(text: str, offsets, token_attrs: torch.Tensor):
    """把词元归因按字符偏移分摊回原文，返回 (字符归因, 覆盖掩码)。

    一个词元覆盖多个字符时平均分摊，因此字符归因之和等于词元归因之和。
    特殊符号的偏移是 ``(0, 0)``，跳过。
    """
    chars = np.zeros(len(text), dtype=np.float64)
    covered = np.zeros(len(text), dtype=bool)
    values = token_attrs.detach().cpu().numpy()
    for (start, end), value in zip(offsets, values):
        if end <= start:
            continue
        start, end = max(0, int(start)), min(len(text), int(end))
        if end <= start:
            continue
        share = float(value) / (end - start)
        for i in range(start, end):
            chars[i] += share
            covered[i] = True
    return chars, covered


def group_char_mask(text: str, covered: np.ndarray) -> np.ndarray:
    """三个话题的群体词在原文中覆盖到的字；只保留模型实际看到的字。"""
    mask = np.zeros(len(text), dtype=bool)
    for topic in GROUP_WORDS:
        for start, end, _ in find_group_matches(text, topic):
            for i in range(max(0, start), min(len(text), end)):
                mask[i] = True
    return mask & covered


def enrichment(char_attrs: np.ndarray, group_mask: np.ndarray,
               covered: np.ndarray) -> dict:
    """群体词字的正归因份额 ÷ 群体词字占正文字数的份额。

    正归因指朝攻击性方向的归因，负值记 0。没有正文或没有正归因时
    富集比无定义，返回 ``None``。
    """
    body = covered
    n_body = int(body.sum())
    n_group = int((group_mask & body).sum())
    if n_body == 0 or n_group == 0:
        return {"enrichment": None, "n_body": n_body, "n_group": n_group,
                "body_attr": 0.0, "group_attr": 0.0}
    positive = np.maximum(char_attrs, 0.0) * body
    body_attr = float(positive.sum())
    group_attr = float(positive[group_mask & body].sum())
    if body_attr <= 0.0:
        value = None
    else:
        value = (group_attr / body_attr) / (n_group / n_body)
    return {"enrichment": value, "n_body": n_body, "n_group": n_group,
            "body_attr": body_attr, "group_attr": group_attr}


def load_model(model_path: str, device: torch.device):
    """加载分类检查点与分词器；只读本地目录，不联网。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, local_files_only=True)
    model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def predict_probs(model, tokenizer, texts: list[str], device: torch.device,
                  batch_size: int = BATCH_SIZE) -> np.ndarray:
    """按同一 head 截断批量预测，返回攻击性概率。"""
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        rows = [encode_with_offsets(tokenizer, t)[:2] for t in chunk]
        max_len = max(len(x[0]) for x in rows)
        ids = torch.full((len(rows), max_len), tokenizer.pad_token_id,
                         dtype=torch.long)
        mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, (row_ids, row_mask) in enumerate(rows):
            ids[i, :len(row_ids)] = torch.tensor(row_ids)
            mask[i, :len(row_ids)] = torch.tensor(row_mask)
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def summarize(values) -> dict:
    """中位数、四分位数与均值；无有效值时全部为 None。"""
    arr = np.asarray([float(v) for v in values
                      if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return {"n": int(len(arr)), "median": float(med), "q1": float(q1),
            "q3": float(q3), "mean": float(arr.mean())}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_whitney_greater(a: list[float], b: list[float]) -> dict:
    """单侧 Mann-Whitney U（A 大于 B）。优先 scipy，缺了就正态近似。"""
    # 富集比无定义（没有正归因）的条目记为 None，不参与检验，条数单独报告
    n_a_raw, n_b_raw = len(a), len(b)
    a = [float(x) for x in a if x is not None and np.isfinite(float(x))]
    b = [float(x) for x in b if x is not None and np.isfinite(float(x))]
    dropped = {"a_undefined": n_a_raw - len(a), "b_undefined": n_b_raw - len(b)}
    if not a or not b:
        return {"u": None, "p": None, "method": "none", "n_a": len(a), "n_b": len(b), **dropped}
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(a, b, alternative="greater")
        return {"u": float(res.statistic), "p": float(res.pvalue),
                "method": "scipy", "n_a": len(a), "n_b": len(b), **dropped}
    except ImportError:
        # 秩和 U 的正态近似，带结点校正；这里没有 scipy 时才会走到
        combined = a + b
        order = np.argsort(combined, kind="mergesort")
        ranks = np.empty(len(combined), dtype=np.float64)
        sorted_vals = np.asarray(combined)[order]
        i = 0
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        n1, n2 = len(a), len(b)
        u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
        mu = n1 * n2 / 2.0
        _, counts = np.unique(combined, return_counts=True)
        tie = float(((counts ** 3) - counts).sum())
        n = n1 + n2
        var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
        z = (u - mu - 0.5) / math.sqrt(max(var, 1e-12))
        return {"u": float(u), "p": float(1.0 - _normal_cdf(z)),
                "method": "normal_approx", "n_a": n1, "n_b": n2, **dropped}


def select_groups(frame: pd.DataFrame, probs: np.ndarray) -> dict[str, np.ndarray]:
    """按细粒度标签与预测选 A/B/C 三组，返回每组的测试集行号。"""
    fine = frame["fine-grained-label"].to_numpy()
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "A": np.flatnonzero((fine == 3) & (pred == 1)),
        "B": np.flatnonzero((fine == 3) & (pred == 0)),
        "C": np.flatnonzero((fine == 2) & (pred == 1)),
    }


def check_reference(seed: int, macro_f1: float, logger) -> dict:
    """核对检查点测试宏 F1 与 C 组 JSON；差超过 0.1 个百分点就停下。"""
    ref_path = RESULTS / f"cold_bert_rbt3_s{seed}.json"
    if not ref_path.exists():
        logger.info("参考 JSON 不存在，跳过宏 F1 核对: %s", ref_path)
        return {"ref_path": str(ref_path), "ref_macro_f1": None, "diff": None}
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)
    ref_f1 = float(ref["test_macro_f1"])
    diff = macro_f1 - ref_f1
    if abs(diff) > 0.001:
        raise SystemExit(
            f"检查点宏 F1 {macro_f1:.6f} 与 {ref_path.name} 的 {ref_f1:.6f} "
            f"相差 {diff:+.6f}，超过 0.1 个百分点，先查原因。")
    logger.info("宏 F1 核对通过：本次 %.6f，参考 %.6f", macro_f1, ref_f1)
    return {"ref_path": str(ref_path), "ref_macro_f1": ref_f1, "diff": diff}


def attribute_one(model, tokenizer, text: str, device: torch.device,
                  n_steps: int) -> dict:
    """一条评论的积分梯度归因、完整性误差与字符级结果。"""
    ids, mask, offsets = encode_with_offsets(tokenizer, text)
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    mask_t = torch.tensor([mask], dtype=torch.long, device=device)
    embed = model.get_input_embeddings()
    input_embeds = embed(ids_t).detach()
    pad_embed = embed(torch.tensor([tokenizer.pad_token_id], device=device)).detach()
    baseline = input_embeds.clone()
    body = body_positions(ids, tokenizer)
    if body:
        baseline[:, body, :] = pad_embed[0]
    attrs = integrated_gradients(model, input_embeds, baseline, mask_t, n_steps)
    rel, f_x, f_base = token_completeness(
        model, input_embeds, baseline, mask_t, attrs)
    chars, covered = token_attrs_to_chars(text, offsets, attrs)
    return {
        "ids_len": len(ids),
        "token_attrs": attrs.detach().cpu().numpy(),
        "char_attrs": chars,
        "covered": covered,
        "completeness": rel,
        "f_x": f_x,
        "f_base": f_base,
    }


def offensive_prob(model, tokenizer, text: str, device: torch.device) -> float:
    """单条文本的攻击性概率，用于忠实性检查。"""
    return float(predict_probs(model, tokenizer, [text], device)[0])


def _delete_chars(text: str, positions) -> str:
    drop = set(int(i) for i in positions)
    return "".join(ch for i, ch in enumerate(text) if i not in drop)


def faithfulness_check(record: dict, text: str, model, tokenizer,
                       device: torch.device, rng: random.Random) -> dict:
    """A 组每条：删正归因最高的 3 个字 vs 随机删 3 个字。"""
    chars = record["_char_attrs_array"]
    covered = np.flatnonzero(record["_covered"])
    n_delete = min(3, len(covered))
    if n_delete == 0:
        return {"drop_top3": None, "drop_rand3": None,
                "deleted_top3": "", "deleted_rand3": ""}
    # 正归因优先（负值记 0），同分按字在文中的先后，保证可复现
    ranked = sorted(covered, key=lambda i: (-max(float(chars[i]), 0.0), i))
    top = ranked[:n_delete]
    rand = rng.sample(list(covered), n_delete)
    p_orig = record["prob_off"]
    p_top = offensive_prob(model, tokenizer, _delete_chars(text, top), device)
    p_rand = offensive_prob(model, tokenizer, _delete_chars(text, rand), device)
    return {
        "drop_top3": float(p_orig - p_top),
        "drop_rand3": float(p_orig - p_rand),
        "deleted_top3": "".join(text[i] for i in sorted(top)),
        "deleted_rand3": "".join(text[i] for i in sorted(rand)),
    }


def _has_group_word(text: str, tokenizer, max_len: int = MAX_LEN) -> bool:
    _, _, offsets = encode_with_offsets(tokenizer, text, max_len)
    covered = np.zeros(len(text), dtype=bool)
    for start, end in offsets:
        if end > start:
            covered[max(0, int(start)):min(len(text), int(end))] = True
    return bool(group_char_mask(text, covered).any())


def _json_list(values: np.ndarray) -> str:
    return json.dumps([round(float(x), 6) for x in values], ensure_ascii=False)


def _record_row(index: int, group: str, frame: pd.DataFrame, prob: float,
                result: dict, text: str) -> dict:
    mask = group_char_mask(text, result["covered"])
    stat = enrichment(result["char_attrs"], mask, result["covered"])
    return {
        "index": int(index),
        "group": group,
        "text": text,
        "label": int(frame.iloc[index]["label"]),
        "fine_grained": int(frame.iloc[index]["fine-grained-label"]),
        "pred": 1 if prob >= 0.5 else 0,
        "prob_off": float(prob),
        "enrichment": stat["enrichment"],
        "n_group_chars": stat["n_group"],
        "n_body_chars": stat["n_body"],
        "group_attr_sum": stat["group_attr"],
        "body_attr_sum": stat["body_attr"],
        "completeness": float(result["completeness"]),
        "n_tokens": int(result["ids_len"]),
        "chars": text,
        "char_attrs": _json_list(result["char_attrs"]),
        "group_mask": "".join("1" if x else "0" for x in mask),
        "_char_attrs_array": result["char_attrs"],
        "_covered": result["covered"],
    }


def attribute_all(model, tokenizer, frame: pd.DataFrame,
                  groups: dict[str, list[int]], probs: np.ndarray,
                  device: torch.device, n_steps: int, logger) -> list[dict]:
    """对 A/B/C 三组逐条做归因，返回带内部数组的记录列表。"""
    records = []
    for group in ("A", "B", "C"):
        for idx in groups[group]:
            text = str(frame.iloc[idx]["text"])
            result = attribute_one(model, tokenizer, text, device, n_steps)
            row = _record_row(idx, group, frame, float(probs[idx]), result, text)
            row["_group"] = group
            records.append(row)
        logger.info("%s 组完成 %d 条（%d 步）", group, len(groups[group]), n_steps)
    return records


def group_summary(records: list[dict], group: str) -> dict:
    sub = [r for r in records if r["_group"] == group]
    return {
        "group": group,
        "name": GROUP_NAMES[group],
        "n": len(sub),
        "enrichment": summarize([r["enrichment"] for r in sub]),
        "completeness": summarize([r["completeness"] for r in sub]),
    }


def run_faithfulness(records: list[dict], model, tokenizer,
                     device: torch.device) -> dict:
    """A 组每条删 3 个字，随机删除用 ``random.Random(0)``。"""
    rng = random.Random(0)
    for row in records:
        if row["_group"] != "A":
            continue
        row.update(faithfulness_check(
            row, row["text"], model, tokenizer, device, rng))
    top = [r.get("drop_top3") for r in records if r["_group"] == "A"]
    rand = [r.get("drop_rand3") for r in records if r["_group"] == "A"]
    return {
        "n": len(top),
        "top3": summarize(top),
        "random3": summarize(rand),
        "top3_median_drop": summarize(top)["median"],
        "random3_median_drop": summarize(rand)["median"],
    }


PUBLIC_COLUMNS = [
    "index", "group", "text", "label", "fine_grained", "pred", "prob_off",
    "enrichment", "n_group_chars", "n_body_chars", "group_attr_sum",
    "body_attr_sum", "completeness", "n_tokens", "chars", "char_attrs",
    "group_mask", "drop_top3", "drop_rand3", "deleted_top3", "deleted_rand3",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="D2 积分梯度归因")
    parser.add_argument("--model_path", required=True, help="rbt3 检查点目录")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="每组最多分析多少条；默认全部，冒烟用")
    parser.add_argument("--test_limit", type=int, default=None,
                        help="只取测试集前 N 条；默认全部，冒烟用")
    parser.add_argument("--steps", type=int, default=50, help="初始积分步数")
    parser.add_argument("--escalate_steps", type=int, default=200,
                        help="完整性中位数超过 5%% 后改用的步数")
    parser.add_argument("--check_ref", action="store_true",
                        help="核对测试宏 F1 与 results/cold_bert_rbt3_s{seed}.json")
    parser.add_argument("--out_dir", default=str(RESULTS))
    parser.add_argument("--log_dir", default=str(ROOT / "logs"))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("要求 CUDA 但当前环境不可用；冒烟请显式传 --device cpu")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    logger = setup_logger(Path(args.log_dir) / f"D2_归因_s{args.seed}.log")
    logger.info("参数: %s", vars(args))

    tokenizer, model = load_model(args.model_path, device)
    frame = load_splits("cold")["test"]
    if args.test_limit is not None:
        frame = frame.iloc[:args.test_limit].reset_index(drop=True)
    texts = frame["text"].astype(str).tolist()
    probs = predict_probs(model, tokenizer, texts, device)
    labels = frame["label"].to_numpy()
    preds = (probs >= 0.5).astype(np.int64)
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    accuracy = float((preds == labels).mean())
    ref = {"ref_path": None, "ref_macro_f1": None, "diff": None}
    if args.check_ref and args.test_limit is None:
        ref = check_reference(args.seed, macro_f1, logger)

    groups = select_groups(frame, probs)
    candidates = {}
    for group, inds in groups.items():
        keep = [int(i) for i in inds if _has_group_word(texts[int(i)], tokenizer)]
        if args.limit is not None:
            keep = keep[:args.limit]
        candidates[group] = keep
    logger.info("候选条数: %s", {k: len(v) for k, v in candidates.items()})
    if not any(candidates.values()):
        raise SystemExit("A/B/C 三组都没有可用样本，检查检查点是否加载正确")

    records = attribute_all(model, tokenizer, frame, candidates, probs,
                            device, args.steps, logger)
    comp50 = summarize([r["completeness"] for r in records])
    escalated = comp50["median"] is not None and comp50["median"] > COMPLETENESS_THRESHOLD
    n_steps_used = args.steps
    if escalated:
        logger.info("完整性中位数 %.4f > %.2f，改用 %d 步重算",
                    comp50["median"], COMPLETENESS_THRESHOLD, args.escalate_steps)
        records = attribute_all(model, tokenizer, frame, candidates, probs,
                                device, args.escalate_steps, logger)
        n_steps_used = args.escalate_steps
    comp_used = summarize([r["completeness"] for r in records])
    faith = run_faithfulness(records, model, tokenizer, device)

    rows = [{k: r.get(k) for k in PUBLIC_COLUMNS} for r in records]
    pred_path = out_dir / "preds" / f"D2_归因_s{args.seed}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PUBLIC_COLUMNS).to_parquet(pred_path, index=False)

    by_group = {g: group_summary(records, g) for g in ("A", "B", "C")}
    a_vals = [r["enrichment"] for r in records if r["_group"] == "A"]
    b_vals = [r["enrichment"] for r in records if r["_group"] == "B"]
    payload = {
        "task": "D2",
        "checkpoint": str(args.model_path),
        "seed": args.seed,
        "device": str(device),
        "run_at": now_iso(),
        "method": {
            "name": "integrated_gradients",
            "attribution_target": "词向量层输出",
            "target": "logit[1] - logit[0]",
            "baseline": "正文位置换成 [PAD] 词向量，[CLS]/[SEP] 保留",
            "integration": "梯形法",
            "steps": n_steps_used,
        },
        "test": {
            "n_total": int(len(load_splits("cold")["test"])),
            "n_used": int(len(frame)),
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "reference": ref,
        },
        "groups": by_group,
        "mann_whitney": {
            "A_vs_B": mann_whitney_greater(a_vals, b_vals),
        },
        "completeness": {
            "threshold": COMPLETENESS_THRESHOLD,
            "median_at_initial_steps": comp50["median"],
            "initial_steps": args.steps,
            "escalated": bool(escalated),
            "steps_used": n_steps_used,
            "median_used": comp_used["median"],
            "q1_used": comp_used["q1"],
            "q3_used": comp_used["q3"],
            "n": comp_used["n"],
        },
        "faithfulness": faith,
        "n_candidates": {k: len(v) for k, v in candidates.items()},
    }
    json_path = out_dir / f"D2_归因_s{args.seed}.json"
    save_json(json_path, payload)
    logger.info("逐条结果写入 %s", pred_path)
    logger.info("结果写入 %s", json_path)
    print(json.dumps({
        "json": str(json_path),
        "parquet": str(pred_path),
        "n_steps": n_steps_used,
        "completeness_median": comp_used["median"],
        "A_enrichment_median": by_group["A"]["enrichment"]["median"],
        "B_enrichment_median": by_group["B"]["enrichment"]["median"],
        "mann_whitney_p": payload["mann_whitney"]["A_vs_B"]["p"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
