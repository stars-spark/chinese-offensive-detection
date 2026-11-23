"""C 组中文数据：COLD 去重、char 分词、评测集标签映射与自制集载入。"""

import pandas as pd
import pytest

import data
from data import (
    char_tokenize,
    deduplicate_cold_splits,
    load_eval_set,
    normalize_for_cold_match,
)


def test_char_tokenizer_contract():
    assert char_tokenize("你好abc，世界") == ["你", "好", "abc", "，", "世", "界"]
    # 连续字母数字整体一个 token，标点各自一个
    assert char_tokenize("A1 中文!") == ["a1", "中", "文", "!"]


def test_cold_dedup_counts_on_synthetic_data():
    def frame(texts, labels):
        return pd.DataFrame({"text": texts, "label": labels})

    splits = {
        "train": frame(["重复", "训练独有", "ＤＥＤＵＰ　空格"], [1, 0, 1]),
        "val": frame(["重复", "验证独有", "dedup 空格"], [1, 0, 1]),
        "test": frame(["q", "验证独有"], [1, 0]),
    }
    # 归一化后 "ＤＥＤＵＰ　空格" == "dedup 空格"（NFKC + 去空白 + 小写）
    assert normalize_for_cold_match("ＤＥＤＵＰ　空格") == "dedup空格"
    out, removed = deduplicate_cold_splits(splits)
    assert removed == {"train": 2, "val": 1, "test": 0}
    assert out["train"]["text"].tolist() == ["训练独有"]
    assert out["val"]["text"].tolist() == ["重复", "dedup 空格"]
    assert len(out["test"]) == 2


def test_mhc_label_mapping():
    frame = load_eval_set("mhc_zh")
    assert set(frame.columns) == {"text", "label", "functionality"}
    assert len(frame) == 3378
    assert int(frame["label"].sum()) == 2265
    assert set(frame["label"].unique()) == {0, 1}
    assert frame["functionality"].nunique() == 25


def test_toxicn_label_mapping_and_cold_dedup():
    frame = load_eval_set("toxicn_test")
    assert {"text", "label", "platform", "expression"} <= set(frame.columns)
    # 原始 2411 条，剔除与 COLD 重复的 1 条
    assert len(frame) == 2410
    assert frame.attrs["n_removed_cold"] == 1
    assert int(frame["label"].sum()) == 1273
    assert set(frame["platform"]) == {"zhihu", "tieba"}


def test_selfmade_cloak_and_swap_load(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SELF_MADE_ROOT", tmp_path)
    cloak = pd.DataFrame({
        "pair_id": [0, 1, 2, 3, 4],
        "orig": ["你是傻子", "滚出去", "正常话", "别来了", "真好笑"],
        "edit": ["你是shazi", "滚出qu", "正常hua", "别lai了", "真hao笑"],
        "label": [1, 1, 0, 1, 0],
        "kind": ["pinyin"] * 5,
    })
    cloak.to_csv(tmp_path / "cloak_pinyin.csv", index=False)
    swap = pd.DataFrame({
        "pair_id": [0, 1, 2, 3, 4],
        "orig": ["河南人不好", "男人不行", "白人不差", "北方人很好", "普通人"],
        "edit": ["山东人不好", "女人不行", "黑人不好", "南方人很好", "普通人"],
        "label": [1, 1, 0, 0, 0],
        "kind": ["swap"] * 5,
        "topic": ["region", "gender", "race", "region", "other"],
    })
    swap.to_csv(tmp_path / "swap.csv", index=False)

    orig = load_eval_set("cloak_pinyin_orig")
    edit = load_eval_set("cloak_pinyin_edit")
    assert len(orig) == 5 and len(edit) == 5
    assert orig["text"].tolist()[0] == "你是傻子"
    assert edit["text"].tolist()[0] == "你是shazi"
    assert orig["pair_id"].tolist() == edit["pair_id"].tolist()
    assert set(orig["kind"]) == {"pinyin"}

    s_orig = load_eval_set("swap_orig")
    s_edit = load_eval_set("swap_edit")
    assert s_orig["text"].tolist()[0] == "河南人不好"
    assert s_edit["text"].tolist()[0] == "山东人不好"
    assert s_orig["topic"].tolist()[0] == "region"
    assert s_orig["label"].tolist() == [1, 1, 0, 0, 0]


def test_selfmade_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SELF_MADE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_eval_set("swap_orig")
