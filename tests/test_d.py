"""D 组新增代码的单元测试。

覆盖：D1 增强（标签、一条源评论一个变体、held 类型、先去重）、
``--aug_csv`` 为空时两个训练入口读到的训练集不变、D2 积分梯度完整性与
线性模型手算、富集比、D3 是/否词元打分方向、D4 LoRA 可训练范围、
NaN 损失停止与 prob_pos 方向。
"""
from __future__ import annotations

import json
import logging
import random
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification

import data
from data import append_aug, deduplicate_cold_splits
from make_aug import build_augment, load_resources


def _frame(texts, labels):
    return pd.DataFrame({
        "text": list(texts),
        "label": list(labels),
        "topic": ["region"] * len(texts),
    })


def _synthetic_splits():
    """训练集含两条会被 dev/test 去重掉的评论，另有两条含词表词。"""
    return {
        "train": _frame(
            ["你是垃圾", "我不是垃圾", "重复攻击", "重复安全", "正常话"],
            [1, 0, 1, 0, 0],
        ),
        "val": _frame(["重复攻击"], [1]),
        "test": _frame(["重复安全"], [0]),
    }


@pytest.fixture(scope="module")
def resources():
    return load_resources()


def test_aug_keeps_label_and_one_variant_per_source(resources):
    splits = _synthetic_splits()
    rows, stats = build_augment(splits, "all", 0, resources)
    assert stats["n_matched"] == 2
    assert len(rows) == 2 and stats["n_skipped"] == 0
    assert {r["src_text"] for r in rows} == {"你是垃圾", "我不是垃圾"}
    # 每条源评论恰好一行，标签保留原评论标签
    by_src = {r["src_text"]: r for r in rows}
    assert by_src["你是垃圾"]["label"] == 1
    assert by_src["我不是垃圾"]["label"] == 0
    assert len({r["src_text"] for r in rows}) == len(rows)


def test_aug_dedup_before_augment(resources):
    splits = _synthetic_splits()
    rows, _ = build_augment(splits, "all", 0, resources)
    src = {r["src_text"] for r in rows}
    # 这两条分别与 val/test 重复，必须先去重，不能再被增强
    assert "重复攻击" not in src and "重复安全" not in src
    dedup, _ = deduplicate_cold_splits(splits)
    assert src <= set(dedup["train"]["text"])


def test_held_only_pinyin_and_symbol(resources):
    rows, _ = build_augment(_synthetic_splits(), "held", 0, resources)
    assert {r["kind"] for r in rows} <= {"pinyin", "symbol"}
    assert len(rows) == 2


def test_aug_is_deterministic(resources):
    a, _ = build_augment(_synthetic_splits(), "all", 1, resources)
    b, _ = build_augment(_synthetic_splits(), "all", 1, resources)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_append_aug_empty_returns_same_rows(tmp_path):
    frame = _frame(["a", "b"], [0, 1])
    out, n = append_aug(frame, "")
    assert n == 0
    assert out is frame or out.equals(frame)
    path = tmp_path / "aug.csv"
    pd.DataFrame({"text": ["c"], "label": [1]}).to_csv(path, index=False)
    out2, n2 = append_aug(frame, str(path))
    assert n2 == 1
    assert out2["text"].tolist() == ["a", "b", "c"]
    assert out2["label"].tolist() == [0, 1, 1]


def test_both_scripts_share_same_append_helper():
    import baseline_tfidf
    import finetune_bert

    assert finetune_bert.append_aug is data.append_aug
    assert baseline_tfidf.append_aug is data.append_aug


class _LinearModel(nn.Module):
    """logit[1] − logit[0] 对嵌入是线性的，积分梯度可以手算。"""

    def __init__(self, weight, bias=0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.bias = float(bias)

    def forward(self, inputs_embeds, attention_mask=None):
        per_token = (inputs_embeds * self.weight).sum(dim=-1)
        target = per_token.sum(dim=-1) + self.bias
        logits = torch.stack([torch.zeros_like(target), target], dim=-1)
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)


def test_integrated_gradients_linear_closed_form():
    from attribution import integrated_gradients, token_completeness

    torch.manual_seed(0)
    weight = torch.randn(1, 6, 4)
    model = _LinearModel(weight)
    x = torch.randn(1, 6, 4)
    baseline = torch.randn(1, 6, 4)
    mask = torch.ones(1, 6, dtype=torch.long)
    attrs = integrated_gradients(model, x, baseline, mask, n_steps=10)
    expected = ((x - baseline) * weight).sum(dim=-1)[0]
    assert torch.allclose(attrs, expected, atol=1e-5)
    rel, f_x, f_base = token_completeness(model, x, baseline, mask, attrs)
    assert rel < 1e-5


def _tiny_bert():
    config = BertConfig(
        vocab_size=40, hidden_size=24, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=48,
        max_position_embeddings=24, type_vocab_size=2, num_labels=2,
    )
    return BertForSequenceClassification(config).eval()


def test_integrated_gradients_completeness_two_layer_bert():
    from attribution import integrated_gradients, token_completeness

    torch.manual_seed(0)
    model = _tiny_bert()
    embed = model.get_input_embeddings()
    pad = embed(torch.tensor([0])).detach()
    for seed in range(3):
        torch.manual_seed(seed)
        ids = torch.randint(1, 40, (1, 12))
        mask = torch.ones_like(ids)
        x = embed(ids).detach()
        baseline = x.clone()
        baseline[:, 1:-1] = pad[0]
        attrs = integrated_gradients(model, x, baseline, mask, n_steps=200)
        rel, _, _ = token_completeness(model, x, baseline, mask, attrs)
        assert rel < 0.05, f"seed={seed} 完整性误差 {rel:.4f}"


def test_enrichment_ratio_two_of_four():
    from attribution import enrichment

    attrs = np.array([1.0, 1.0, 0.0, 0.0])
    group = np.array([True, True, False, False])
    covered = np.ones(4, dtype=bool)
    stat = enrichment(attrs, group, covered)
    assert stat["enrichment"] == 2.0


def test_yes_no_prob_and_threshold_direction():
    from llm_zeroshot import PROMPTS, yes_prob_from_logits

    logits = torch.tensor([0.0, 2.0])
    p_yes = float(yes_prob_from_logits(logits, yes_id=1, no_id=0))
    assert abs(p_yes - 0.8807970779778823) < 1e-6
    assert p_yes >= 0.5 and not p_yes >= 0.99
    p_no = float(yes_prob_from_logits(torch.tensor([2.0, 0.0]),
                                     yes_id=1, no_id=0))
    assert abs(p_no - (1.0 - 0.8807970779778823)) < 1e-6
    assert p_no < 0.5
    assert PROMPTS["P1"].startswith("判断下面这条评论是否带有攻击性。只回答“是”或“否”。")


def _null_logger(name):
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def test_finetune_aug_csv_empty_reads_same_train(monkeypatch, tmp_path):
    import finetune_bert
    from transformers import AutoTokenizer

    splits = _synthetic_splits()
    monkeypatch.setattr(finetune_bert, "load_splits", lambda ds: splits)
    monkeypatch.setattr(finetune_bert, "setup_logger",
                        lambda p: _null_logger("test_d_finetune"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(data.ROOT / "models/rbt3"), local_files_only=True)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=24, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=48,
        max_position_embeddings=64, type_vocab_size=2, num_labels=2,
    )
    tiny = BertForSequenceClassification(config)
    monkeypatch.setattr(finetune_bert.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: tokenizer))
    monkeypatch.setattr(
        finetune_bert.AutoModelForSequenceClassification, "from_pretrained",
        staticmethod(lambda *a, **k: tiny))
    calls = []
    original = finetune_bert.encode_split

    def spy(tok, texts, max_len, *args, **kwargs):
        calls.append(list(texts))
        return original(tok, texts, max_len, *args, **kwargs)

    monkeypatch.setattr(finetune_bert, "encode_split", spy)
    monkeypatch.setattr(sys, "argv", [
        "finetune_bert.py", "--dataset", "cold", "--device", "cpu",
        "--epochs", "1", "--max_len", "16", "--batch_size", "4",
        "--out_dir", str(tmp_path),
    ])
    finetune_bert.main()
    expected = deduplicate_cold_splits(splits)[0]["train"]["text"].tolist()
    assert calls[0] == expected
    result = json.loads((tmp_path / "cold_bert_s0.json").read_text(encoding="utf-8"))
    assert result["aug_n_appended"] == 0
    assert result["aug_csv"] is None


def test_baseline_aug_csv_empty_reads_same_train(monkeypatch, tmp_path):
    import baseline_tfidf

    splits = _synthetic_splits()
    monkeypatch.setattr(baseline_tfidf, "load_splits", lambda ds: splits)
    monkeypatch.setattr(baseline_tfidf, "setup_logger",
                        lambda p: _null_logger("test_d_baseline"))
    captured = {}
    original = baseline_tfidf.fit_and_select

    def spy(train_texts, *args, **kwargs):
        captured.setdefault("train", list(train_texts))
        return original(train_texts, *args, **kwargs)

    monkeypatch.setattr(baseline_tfidf, "fit_and_select", spy)
    monkeypatch.setattr(sys, "argv", [
        "baseline_tfidf.py", "--dataset", "cold", "--test_limit", "2",
        "--out_dir", str(tmp_path),
    ])
    baseline_tfidf.main()
    expected = deduplicate_cold_splits(splits)[0]["train"]["text"].tolist()
    assert captured["train"] == expected
    result = json.loads(
        (tmp_path / "cold_tfidf_main_s0.json").read_text(encoding="utf-8"))
    assert result["aug_n_appended"] == 0
    assert result["aug_csv"] is None




def test_mann_whitney_skips_undefined():
    """富集比无定义的条目（None）不参与检验，并计数。"""
    from attribution import mann_whitney_greater
    res = mann_whitney_greater([3.0, 4.0, None, 5.0], [None, 1.0, 2.0])
    assert res["n_a"] == 3 and res["n_b"] == 2
    assert res["a_undefined"] == 1 and res["b_undefined"] == 1
    assert res["p"] is not None and res["p"] < 0.5


def _tiny_qwen(vocab_size=128):
    from transformers import Qwen2Config, Qwen2ForSequenceClassification

    config = Qwen2Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, num_labels=2, pad_token_id=0,
    )
    return Qwen2ForSequenceClassification(config)


def test_lora_only_adapters_and_head_trainable():
    """2 层随机小模型：基座全冻结，可训练的只有 LoRA 与分类头。"""
    import finetune_llm

    wrapped = finetune_llm.wrap_lora(_tiny_qwen(), r=4, alpha=8, dropout=0.05)
    names = [(n, p) for n, p in wrapped.named_parameters()]
    trainable = [n for n, p in names if p.requires_grad]
    assert trainable, "至少要有 LoRA 参数可训练"
    assert all(finetune_llm.is_trainable_adapter(n) for n in trainable)
    assert any("lora_A" in n or "lora_B" in n for n in trainable)
    assert any("modules_to_save" in n and "score" in n for n in trainable)
    # 基座参数（含原始分类头）的 requires_grad 必须为 False
    base = [(n, p) for n, p in names
            if not finetune_llm.is_trainable_adapter(n)]
    assert any("q_proj" in n for n, _ in base)
    assert all(not p.requires_grad for _, p in base)
    # LoRA 与 modules_to_save 的分类头显式转成了 fp32
    assert all(p.dtype == torch.float32 for n, p in names if p.requires_grad)


def test_nan_loss_stops_and_writes_status(monkeypatch, tmp_path):
    """分类头权重人造为 NaN 时，脚本在第 1 步停下并写 nan_loss。"""
    import finetune_llm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(data.ROOT / "models/rbt3"), local_files_only=True)
    base = _tiny_qwen(vocab_size=tokenizer.vocab_size)
    with torch.no_grad():
        base.score.weight.fill_(float("nan"))
    split_texts = {
        "train": ["攻击一", "安全一", "攻击二", "安全二"],
        "val": ["验证攻击", "验证安全"],
        "test": ["测试攻击", "测试安全"],
    }
    labels = {"train": [1, 0, 1, 0], "val": [1, 0], "test": [1, 0]}
    splits = {
        name: pd.DataFrame({
            "text": texts, "label": labels[name],
            **({"fine-grained-label": labels[name]} if name == "test" else {}),
        })
        for name, texts in split_texts.items()
    }
    monkeypatch.setattr(finetune_llm, "load_splits", lambda ds: splits)
    monkeypatch.setattr(finetune_llm, "load_base_model",
                        lambda *a, **k: base)
    monkeypatch.setattr(finetune_llm.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: tokenizer))
    monkeypatch.setattr(finetune_llm, "setup_logger",
                        lambda p: _null_logger("test_d4_nan"))
    monkeypatch.setattr(sys, "argv", [
        "finetune_llm.py", "--model_path", str(tmp_path / "unused"),
        "--name", "nan", "--seed", "0", "--device", "cpu",
        "--base_dtype", "float32", "--batch_size", "2",
        "--grad_accum", "2", "--train_limit", "4", "--test_limit", "2",
        "--out_dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as exc:
        finetune_llm.main()
    assert exc.value.code == 1
    result = json.loads(
        (tmp_path / "cold_llmft_nan_s0.json").read_text(encoding="utf-8"))
    assert result["status"] == "nan_loss"
    assert result["nan_loss_step"] == 1
    assert result["stopped_epoch"] == 0


def test_eval_prob_pos_direction():
    """softmax 第 1 类概率大于 0.5 时预测为 1，prob_pos 就是第 1 类概率。"""
    import finetune_llm
    from types import SimpleNamespace

    class _Fixed(nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(
                logits=torch.tensor([[0.0, 2.0], [2.0, 0.0]]))

    batch = {
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([1, 0], dtype=torch.long),
        "index": torch.tensor([0, 1], dtype=torch.long),
    }
    _, acc, preds, labels, probs = finetune_llm.evaluate(
        _Fixed(), [batch], torch.device("cpu"), "float32")
    assert preds.tolist() == [1, 0]
    assert labels.tolist() == [1, 0] and acc == 1.0
    expected = 1.0 / (1.0 + np.exp(-2.0))
    assert np.allclose(probs[:, 1], [expected, 1.0 - expected], atol=1e-6)
    assert (probs[:, 1] > 0.5).astype(int).tolist() == preds.tolist()


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content']"
    " + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def _chat_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(data.ROOT / "models/rbt3"), local_files_only=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def test_d5_labels_only_at_answer_and_right_padding():
    from finetune_llm_prompt import encode_example, make_collate, pad_right_batch
    from llm_zeroshot import single_token_id

    tokenizer = _chat_tokenizer()
    yes_id = single_token_id(tokenizer, "是")
    no_id = single_token_id(tokenizer, "否")
    expected = {1: yes_id, 0: no_id}
    for label, answer_id in expected.items():
        ids, labels = encode_example(
            tokenizer, "这是一条测试评论", label, yes_id, no_id, max_len=512)
        assert ids[-1] == answer_id
        assert labels[-1] == answer_id
        assert all(x == -100 for x in labels[:-1])

    input_ids, mask, padded = pad_right_batch(
        [[1, 2, 3], [4, 5]], [[-100, -100, yes_id], [-100, no_id]], 0)
    assert input_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert padded.tolist() == [[-100, -100, yes_id], [-100, no_id, -100]]

    collate = make_collate(tokenizer, yes_id, no_id, 512)
    batch = collate([
        {"text": "短评论", "label": 1, "index": 0},
        {"text": "稍微长一点的评论", "label": 0, "index": 1},
    ])
    for row in range(2):
        n = int(batch["attention_mask"][row].sum())
        assert bool(batch["attention_mask"][row, :n].all())
        assert not bool(batch["attention_mask"][row, n:].any())
        assert int((batch["labels"][row] != -100).sum()) == 1
        assert batch["labels"][row, n - 1] == batch["input_ids"][row, n - 1]
        assert all(int(x) == -100 for x in batch["labels"][row, :n - 1])
        assert all(int(x) == -100 for x in batch["labels"][row, n:])


def test_finetune_llm_aug_csv_empty_reads_same_train(monkeypatch, tmp_path):
    import finetune_llm
    from transformers import AutoTokenizer

    splits = _synthetic_splits()
    monkeypatch.setattr(finetune_llm, "load_splits", lambda ds: splits)
    monkeypatch.setattr(finetune_llm, "setup_logger",
                        lambda p: _null_logger("test_d6"))
    monkeypatch.setattr(finetune_llm, "ROOT", tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(data.ROOT / "models/rbt3"), local_files_only=True)
    base = _tiny_qwen(vocab_size=tokenizer.vocab_size)
    monkeypatch.setattr(finetune_llm.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: tokenizer))
    monkeypatch.setattr(finetune_llm, "load_base_model", lambda *a, **k: base)
    calls = []
    original = finetune_llm.make_loader

    def spy(texts, *args, **kwargs):
        calls.append(list(texts))
        return original(texts, *args, **kwargs)

    monkeypatch.setattr(finetune_llm, "make_loader", spy)
    monkeypatch.setattr(sys, "argv", [
        "finetune_llm.py", "--model_path", str(tmp_path / "unused"),
        "--name", "augempty", "--seed", "0", "--device", "cpu",
        "--base_dtype", "float32", "--batch_size", "2", "--grad_accum", "2",
        "--test_limit", "2", "--out_dir", str(tmp_path),
    ])
    finetune_llm.main()
    expected = deduplicate_cold_splits(splits)[0]["train"]["text"].tolist()
    assert calls[0] == expected
    result = json.loads(
        (tmp_path / "cold_llmft_augempty_s0.json").read_text(encoding="utf-8"))
    assert result["aug_n_appended"] == 0
    assert result["aug_csv"] is None
    assert result["n_train_used"] == len(expected)


def test_d7_shots_zero_messages_identical():
    from llm_zeroshot import PROMPTS, SYSTEM_PROMPT, build_messages, clean_comment

    text = "这条评论在说什么"
    expected = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPTS["P2"].format(
            text=clean_comment(text))},
    ]
    assert build_messages("P2", text) == expected
    assert build_messages("P2", text, examples=[]) == expected


def test_d7_shot_examples_and_message_rounds():
    from llm_zeroshot import (
        PROMPTS, SYSTEM_PROMPT, build_messages, result_stem,
        select_shot_examples,
    )
    from types import SimpleNamespace

    splits, _ = deduplicate_cold_splits(data.load_splits("cold"))
    examples = select_shot_examples(8, 0, splits)
    assert len(examples) == 8
    assert sum(ex["label"] == 1 for ex in examples) == 4
    assert sum(ex["label"] == 0 for ex in examples) == 4
    blocked = set(splits["val"]["text"].map(data.normalize_for_cold_match))
    blocked |= set(splits["test"]["text"].map(data.normalize_for_cold_match))
    for ex in examples:
        assert data.normalize_for_cold_match(ex["text"]) not in blocked
        assert data.clean_text(
            str(splits["train"]["text"].iloc[ex["train_index"]]), "cold"
        )[:192] == ex["text"]
    assert select_shot_examples(8, 0, splits) == examples
    assert select_shot_examples(8, 1, splits) != examples

    messages = build_messages("P2", "待判断的评论", examples)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in messages] == (
        ["system"] + ["user", "assistant"] * 8 + ["user"])
    for i, ex in enumerate(examples):
        assert messages[1 + 2 * i]["content"] == PROMPTS["P2"].format(
            text=ex["text"])
        assert messages[2 + 2 * i]["content"] == ex["answer"]
    assert messages[-1]["content"] == PROMPTS["P2"].format(
        text="待判断的评论")
    assert result_stem(SimpleNamespace(name="qwen7b", shots=0,
                                       shot_seed=0)) == "qwen7b"
    assert result_stem(SimpleNamespace(name="qwen7b", shots=8,
                                       shot_seed=2)) == "qwen7b_fs2"


# ---------------- D8 DeepSeek API 试点 ----------------

def _d8_top(*pairs):
    return [{"token": token, "logprob": logprob} for token, logprob in pairs]


def test_d8_prob_pos_ok_partial_no_prob():
    import math

    from llm_api import prob_pos_from_top_logprobs

    prob, status = prob_pos_from_top_logprobs(
        _d8_top(("是", -0.2), ("否", -1.7)))
    assert status == "ok"
    assert prob == pytest.approx(1.0 / (1.0 + math.exp(-1.5)), abs=1e-12)

    # 只出现一个时，另一个按 -30 计，并标 partial
    only_yes, status_yes = prob_pos_from_top_logprobs(_d8_top(("是", -0.2)))
    assert status_yes == "partial"
    assert only_yes > 0.999999
    only_no, status_no = prob_pos_from_top_logprobs(_d8_top(("否", -0.2)))
    assert status_no == "partial"
    assert only_no < 1e-12

    # 两个都不在候选里，prob_pos 为空
    none_prob, status_none = prob_pos_from_top_logprobs(
        _d8_top(("好", -0.1), ("坏", -2.0)))
    assert none_prob is None and status_none == "no_prob"

    # 词元前后有空白也按“是/否”匹配；重复候选取较大的对数概率
    spaced, spaced_status = prob_pos_from_top_logprobs(
        _d8_top((" 是", -0.1), ("否", -2.0)))
    assert spaced_status == "ok" and spaced > 0.8
    repeated, repeated_status = prob_pos_from_top_logprobs(
        _d8_top(("是", -2.0), ("是", -0.5), ("否", -1.0)))
    assert repeated_status == "ok" and repeated == pytest.approx(
        1.0 / (1.0 + math.exp(-0.5)), abs=1e-12)


def test_d8_parse_completion_fields():
    from llm_api import parse_completion

    payload = {
        "choices": [{
            "finish_reason": "length",
            "message": {"content": "是"},
            "logprobs": {"content": [{
                "token": "是",
                "top_logprobs": _d8_top(("是", -0.1), ("否", -2.2)),
            }]},
        }],
        "usage": {"prompt_tokens": 300, "completion_tokens": 1},
    }
    result = parse_completion(payload, latency_ms=12, attempts=2)
    assert result.status == "ok"
    assert result.out_token == "是"
    assert result.prompt_tokens == 300
    assert result.completion_tokens == 1
    assert result.latency_ms == 12
    assert result.attempts == 2


def test_d8_blocked_400_not_retried():
    import io
    import json as json_module
    import urllib.error

    import llm_api

    calls = []
    body = json_module.dumps({
        "error": {
            "message": "Content Risk detected in input",
            "type": "invalid_request_error",
            "code": "content_filter",
        }
    }).encode("utf-8")

    def opener(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    client = llm_api.DeepSeekClient(
        "fake-key", opener=opener, sleep_fn=lambda _: None)
    result = client.score([{"role": "user", "content": "评论原文不要进日志"}])
    assert result.status == "blocked"
    assert len(calls) == 1
    assert result.error_code and "400" in result.error_code
    assert "评论原文" not in (result.error_message or "")


def test_d8_retries_5xx_then_ok():
    import io
    import json as json_module
    import math
    import urllib.error
    from types import SimpleNamespace

    import llm_api

    calls = []
    good = json_module.dumps({
        "choices": [{
            "finish_reason": "length",
            "message": {"content": "否"},
            "logprobs": {"content": [{
                "token": "否",
                "top_logprobs": _d8_top(("否", -0.1), ("是", -2.2)),
            }]},
        }],
        "usage": {"prompt_tokens": 200, "completion_tokens": 1},
    }).encode("utf-8")

    def opener(request, timeout=None):
        calls.append(1)
        if len(calls) <= 2:
            raise urllib.error.HTTPError(
                request.full_url, 503, "Service Unavailable", {},
                io.BytesIO(b'{"error":{"type":"server_error"}}'))
        return SimpleNamespace(status=200, headers={},
                               read=lambda: good, close=lambda: None)

    client = llm_api.DeepSeekClient(
        "fake-key", opener=opener, sleep_fn=lambda _: None, max_retries=3)
    result = client.score([{"role": "user", "content": "x"}])
    assert result.status == "ok"
    assert len(calls) == 3
    assert result.attempts == 3
    assert result.prob_pos == pytest.approx(1.0 / (1.0 + math.exp(2.1)))


def test_d8_log_redacts_key(tmp_path):
    from llm_api import make_logger

    secret = "sk-D8-FAKE-1234567890"
    path = tmp_path / "D8_日志测试.log"
    logger = make_logger(path, secret)
    logger.info("Authorization: Bearer %s", secret)
    for handler in logger.handlers:
        handler.flush()
    text = path.read_text(encoding="utf-8")
    assert secret not in text
    assert "***" in text


def test_d8_missing_key():
    import llm_api

    with pytest.raises(SystemExit) as exc_info:
        llm_api.load_api_key({})
    assert "未取到密钥" in str(exc_info.value)


def test_d8_pilot_sample_matches_select():
    from llm_api import pilot_sample

    sample, meta = pilot_sample(200, 0)
    splits, _ = deduplicate_cold_splits(data.load_splits("cold"))
    expected = (splits["val"].sample(n=1000, random_state=0)
                .reset_index(drop=True).head(200))
    assert meta["pool_n"] == 1000 and meta["pilot_n"] == 200
    assert sample["index"].tolist() == list(range(200))
    assert sample["label"].tolist() == expected["label"].tolist()
    assert sample["text"].tolist() == expected["text"].tolist()
    assert meta["label_counts"] == {"0": 113, "1": 87}


def test_d8_parquet_has_no_comment(tmp_path):
    from llm_api import PARQUET_COLUMNS, write_results

    rows = [{
        "index": 0, "label": 1, "prob_pos": 0.9, "status": "ok",
        "out_token": "是", "prompt_tokens": 10, "completion_tokens": 1,
        "latency_ms": 5, "error_code": None, "error_type": None,
        "error_message": None,
    }]
    path = tmp_path / "D8_测试.parquet"
    frame = write_results(path, rows)
    assert list(frame.columns) == PARQUET_COLUMNS
    assert "text" not in frame.columns
    back = pd.read_parquet(path)
    assert "text" not in back.columns
    assert int(back.loc[0, "index"]) == 0
    assert float(back.loc[0, "prob_pos"]) == pytest.approx(0.9)


def test_d8_release_conditions_boundary():
    from llm_api import release_verdict

    # 10/200 = 5%，190/200 = 95%，两条都刚好压线
    verdict = release_verdict(
        {"ok": 180, "partial": 10, "no_prob": 0, "blocked": 9, "error": 1}, 200)
    assert verdict["items"][0]["pass"] is True
    assert verdict["items"][1]["pass"] is True
    assert verdict["overall_pass"] is True

    # 11/200 = 5.5%，189/200 = 94.5%，两条都不满足
    verdict = release_verdict(
        {"ok": 179, "partial": 10, "no_prob": 0, "blocked": 10, "error": 1}, 200)
    assert verdict["items"][0]["pass"] is False
    assert verdict["items"][1]["pass"] is False
    assert verdict["overall_pass"] is False


def test_d8_cost_per_1k():
    from llm_api import cost_per_1k

    value = cost_per_1k(1000, 1, 1.0, 4.0)
    assert value == pytest.approx((1000 * 1.0 + 1 * 4.0) / 1_000_000 * 1000)
    assert cost_per_1k(1000, 1, 2.0, 8.0) == pytest.approx(value * 2)


def test_d8_peak_windows():
    from datetime import datetime, timezone

    from llm_api import is_peak, next_offpeak

    monday = datetime(2026, 11, 23, 1, 30, tzinfo=timezone.utc)
    sunday = datetime(2026, 11, 22, 1, 30, tzinfo=timezone.utc)
    assert is_peak(monday) is True
    assert is_peak(sunday) is False
    assert next_offpeak(monday).hour == 4
    assert is_peak(datetime(2026, 11, 23, 4, 0, tzinfo=timezone.utc)) is False
    assert is_peak(datetime(2026, 11, 23, 9, 59, tzinfo=timezone.utc)) is True
    assert is_peak(datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)) is False


def test_d8_summarize_reports_fields():
    import json as json_module
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    from llm_api import summarize

    rows = []
    for index in range(10):
        label = index % 2
        rows.append({
            "index": index, "label": label,
            "prob_pos": 0.9 if label == 1 else 0.1,
            "status": "ok", "out_token": "是" if label == 1 else "否",
            "prompt_tokens": 100 + index, "completion_tokens": 1,
            "latency_ms": 10 + index, "error_code": None,
            "error_type": None, "error_message": None,
        })
    frame = pd.DataFrame(rows)
    meta = {"pilot_n": 10, "pool_n": 1000, "seed": 0,
            "label_counts": {"0": 5, "1": 5}}
    args = SimpleNamespace(
        workers=4, max_retries=3, model="deepseek-flash",
        base_url="https://api.deepseek.com",
        endpoint="https://api.deepseek.com/chat/completions")
    started = datetime.now().astimezone()
    payload = summarize(frame, meta, args, started, started + timedelta(seconds=2))
    assert payload["n"] == 10
    assert payload["status_counts"]["ok"] == 10
    assert payload["has_prob"]["ratio"] == 1.0
    assert payload["release_conditions"]["overall_pass"] is True
    assert payload["metrics_at_0.5"]["accuracy"] == 1.0
    assert payload["metrics_at_0.5"]["macro_f1"] == 1.0
    costs = payload["pricing"]["cost_per_1k"]
    assert costs["cost_per_1k_cny_off_peak_cache_miss"] > 0
    assert costs["cost_per_1k_cny_peak_cache_miss"] == pytest.approx(
        costs["cost_per_1k_cny_off_peak_cache_miss"] * 2)
    json_module.dumps(payload, ensure_ascii=False)  # 不留 numpy 标量


def test_d8_resume_skips_existing_indices(tmp_path):
    from llm_api import ScoreResult, run_pending

    sample = pd.DataFrame({
        "index": [0, 1, 2],
        "text": ["第一条", "第二条", "第三条"],
        "label": [1, 0, 1],
    })

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def score(self, messages):
            self.calls += 1
            return ScoreResult("ok", prob_pos=0.7, out_token="是",
                               prompt_tokens=3, completion_tokens=1,
                               latency_ms=1)

    path = tmp_path / "D8_续跑.parquet"
    first = FakeClient()
    frame = run_pending(first, sample, path, workers=2, flush_every=2)
    assert first.calls == 3
    assert len(frame) == 3
    assert sorted(frame["index"].tolist()) == [0, 1, 2]

    second = FakeClient()
    frame2 = run_pending(second, sample, path, workers=2, flush_every=2)
    assert second.calls == 0
    assert len(frame2) == 3


# === D8 TESTS END ===

