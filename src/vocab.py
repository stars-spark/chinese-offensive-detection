"""词表与编码（流程第 2 步）。

职责很窄：
1. 用训练部分的 token 统计词频，建词表（按词频取前 30000，词频 >= 2）；
2. 把 token 列表变成固定长度的 id 序列，同时给出 mask；
3. 统计截断比例，写进结果 JSON 和数据统计。

约定：``<pad>`` 的 id 是 0，``<unk>`` 是 1；mask 为 1 表示真实词，
为 0 表示补齐。后面所有模型都靠这个 mask 忽略 padding。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

PAD_ID = 0
UNK_ID = 1


class Vocab:
    """id <-> 词的映射。0 固定是 <pad>，1 固定是 <unk>。"""

    def __init__(self, itos: list[str]):
        if len(itos) < 2:
            raise ValueError("词表至少要包含 <pad> 和 <unk>")
        self.itos = list(itos)
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> tuple[list[int], list[int]]:
        """截断（保留开头）并补齐到 max_len，返回 (ids, mask)。

        保留开头而不是结尾：IMDB 的情绪线索贯穿全文，AG News 的
        导语在第一句，开头的信息密度更高。截断和补齐都只影响
        表示，真实长度由 mask 记住。
        """
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokens[:max_len]]
        mask = [1] * len(ids)
        n_pad = max_len - len(ids)
        if n_pad > 0:
            ids.extend([PAD_ID] * n_pad)
            mask.extend([0] * n_pad)
        return ids, mask


def build_vocab(
    token_lists: list[list[str]], max_size: int = 30000, min_freq: int = 2
) -> Vocab:
    """只用训练部分的 token 建词表。

    词频 < min_freq 的词只出现一两次，学不到可靠的向量，统一进 <unk>。
    排序时用 (-词频, 词) 而不是只按词频，是为了让频次相同的词有稳定的
    先后顺序，保证同一种子下词表可复现。
    """
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    # max_size 包含 <pad> 和 <unk> 两个特殊符号
    n_normal = max(0, max_size - 2)
    kept = [(word, freq) for word, freq in counter.items() if freq >= min_freq]
    kept.sort(key=lambda item: (-item[1], item[0]))
    itos = ["<pad>", "<unk>"] + [word for word, _ in kept[:n_normal]]
    return Vocab(itos)


def encode_dataset(
    token_lists: list[list[str]], vocab: Vocab, max_len: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """批量编码，返回 ids、mask 和截断统计。

    ids:  (n, max_len) int64
    mask: (n, max_len) int64，1=真实词，0=padding
    """
    n = len(token_lists)
    ids = np.zeros((n, max_len), dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.int64)
    n_truncated = 0
    for i, tokens in enumerate(token_lists):
        if len(tokens) > max_len:
            n_truncated += 1
        row_ids, row_mask = vocab.encode(tokens, max_len)
        ids[i] = row_ids
        mask[i] = row_mask
    stats = {
        "n_samples": n,
        "n_truncated": n_truncated,
        # 截断比例 = 词数超过 max_len 的样本占比
        "truncation_rate": n_truncated / n if n else 0.0,
    }
    return ids, mask, stats
