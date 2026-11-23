"""自制规避集与群体词互换集的单元测试。"""
from __future__ import annotations

import random

from make_cloak import (
    SYMBOLS,
    apply_transform,
    build_homophone_map,
    build_word_index,
    chaizi_word,
    char_pinyin,
    find_matches,
    homophone_word,
    load_chaizi,
    load_lexicon,
    to_pinyin,
)
from make_swap import find_matches as find_group_matches
from make_swap import make_variants


def test_longest_match_heilongjiang():
    """“黑龙江人”整体命中，不被拆成更短的群体词。"""
    matches = find_group_matches("我是黑龙江人", "region")
    assert [word for _, _, word in matches] == ["黑龙江人"]
    assert matches[0][:2] == (2, 6)


def test_group_word_twice_no_variant():
    """群体词合计出现两次时不生成变体。"""
    assert make_variants("男人和女人", "gender", random.Random(0)) == []
    assert make_variants("男人和男人", "gender", random.Random(0)) == []


def test_pinyin_laji():
    assert to_pinyin("垃圾") == "laji"


def test_symbol_removable():
    """去掉插入的符号后能还原原句。"""
    text = "这是垃圾"
    by_len, lengths = build_word_index(load_lexicon())
    matches = find_matches(text, by_len, lengths)
    edit, changed_words, _ = apply_transform(
        text, matches, "symbol", rng=random.Random(0))
    assert changed_words == ["垃圾"]
    stripped = edit
    for symbol in SYMBOLS:
        stripped = stripped.replace(symbol, "")
    assert stripped == text


def test_homophone_same_pinyin_different_char():
    mapping = build_homophone_map()
    new, changed = homophone_word("垃圾", mapping)
    assert changed and new != "垃圾"
    assert len(new) == len("垃圾")
    for old, rep in zip("垃圾", new):
        if old != rep:
            assert char_pinyin(old) == char_pinyin(rep)


def test_chaizi_ma():
    assert chaizi_word("妈", load_chaizi()) == ("女马", True)


def test_chaizi_zhi_not_strokes():
    """“直”只有四笔画的拆法，不是两个部件，不能拆。"""
    assert chaizi_word("直", load_chaizi()) == ("直", False)


def test_no_match_inside_jieba_word():
    """“男同”落在 jieba 词“男同学”内部，不能算匹配。"""
    by_len, lengths = build_word_index(load_lexicon())
    assert find_matches("男同学", by_len, lengths) == []
