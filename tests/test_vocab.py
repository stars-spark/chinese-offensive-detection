"""词表与编码的测试：截断、补齐、mask、截断比例。"""

import numpy as np

from vocab import Vocab, build_vocab, encode_dataset


def make_vocab():
    # 显式指定 id，方便断言
    return Vocab(["<pad>", "<unk>", "good", "movie", "bad"])


def test_encode_pads_and_masks():
    vocab = make_vocab()
    ids, mask = vocab.encode(["good", "bad", "good"], max_len=4)
    assert ids == [2, 4, 2, 0]
    assert mask == [1, 1, 1, 0]


def test_encode_truncates_keeps_head():
    vocab = make_vocab()
    ids, mask = vocab.encode(["good", "bad", "good", "movie", "bad"], max_len=3)
    assert ids == [2, 4, 2]  # 保留开头三个词
    assert mask == [1, 1, 1]


def test_encode_unknown_token():
    vocab = make_vocab()
    ids, mask = vocab.encode(["never-seen"], max_len=2)
    assert ids == [1, 0]
    assert mask == [1, 0]


def test_encode_dataset_truncation_rate():
    vocab = make_vocab()
    # 第一条 5 个词 > max_len=3；第二条 2 个词刚好不截断
    token_lists = [["good", "bad", "good", "movie", "bad"], ["good", "movie"]]
    ids, mask, stats = encode_dataset(token_lists, vocab, max_len=3)
    assert ids.shape == (2, 3)
    assert mask.shape == (2, 3)
    assert stats["n_truncated"] == 1
    assert stats["truncation_rate"] == 0.5
    assert mask[1].tolist() == [1, 1, 0]
    assert np.array_equal(ids[1], np.array([2, 3, 0]))


def test_build_vocab_min_freq_and_order():
    # "a" 出现 3 次，"b" 2 次，"c" 1 次（低于 min_freq 被丢掉）
    vocab = build_vocab([["a", "b", "a"], ["a", "b", "c"]], max_size=10, min_freq=2)
    assert vocab.itos == ["<pad>", "<unk>", "a", "b"]
