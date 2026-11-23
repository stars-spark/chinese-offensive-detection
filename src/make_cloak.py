"""生成 COLD 自制规避集：pinyin / homophone / chaizi / symbol 四种变换。

原料是 COLD 测试集中标为攻击性、且含 ToxiCN 词表词的评论。词表只用于定位，
不参与训练；匹配时最长优先、不重叠，且起止位置都得落在 jieba 词边界上，
只改匹配到的词，其余原样保留。
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import jieba
import pandas as pd
from pypinyin import Style, lazy_pinyin

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
COLD_DIR = EXT / "coldataset" / "raw" / "COLDataset"
LEXICON_DIR = EXT / "toxicn" / "raw" / "ToxiCN_ex" / "ToxiCN" / "lexicon"
OUT_DIR = ROOT / "data_selfmade"

KINDS = ("pinyin", "homophone", "chaizi", "symbol")
# 对照变换：直接删掉匹配词，衡量"这个词对判断是否必要"；不进抽查
CONTROL_KINDS = ("delete",)
SYMBOLS = ["*", "、", " ", "."]
SEED = 2026
HAN_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def load_lexicon() -> list[str]:
    """合并 5 个词表，只留长度 >= 2 且全为汉字的词。"""
    words: set[str] = set()
    for path in sorted(LEXICON_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        words.update(str(w) for w in data)
    return sorted((w for w in words if len(w) >= 2 and HAN_RE.match(w)),
                  key=lambda w: (-len(w), w))


def build_word_index(words: list[str]) -> tuple[dict[int, set[str]], list[int]]:
    """按词长分桶，长度从大到小，供最长优先匹配用。"""
    by_len: dict[int, set[str]] = {}
    for word in words:
        by_len.setdefault(len(word), set()).add(word)
    return by_len, sorted(by_len, reverse=True)


def word_boundaries(text: str) -> set[int]:
    """返回 jieba 精确模式的词边界，含 0 和文本末尾。"""
    bounds = {0, len(text)}
    for _, start, end in jieba.tokenize(text, mode="default"):
        bounds.add(start)
        bounds.add(end)
    return bounds


def find_matches(text: str, by_len: dict[int, set[str]],
                 lengths: list[int]) -> list[tuple[int, int, str]]:
    """从左到右扫描，每个词边界上取最长词，命中后跳过整个词。

    词表词的起止位置都必须在 jieba 词边界上。因此它等于一个 jieba 词，
    或等于若干相邻 jieba 词的拼接；落在某个 jieba 词内部的子串不算。
    """
    out: list[tuple[int, int, str]] = []
    bounds = word_boundaries(text)
    i, n = 0, len(text)
    while i < n:
        if i not in bounds:
            i += 1
            continue
        for length in lengths:
            end = i + length
            if end <= n and end in bounds and text[i:end] in by_len[length]:
                out.append((i, end, text[i:end]))
                i = end
                break
        else:
            i += 1
    return out


@lru_cache(maxsize=8192)
def char_pinyin(ch: str) -> str:
    """单字的无声调拼音；多音字取 pypinyin 的默认读音。"""
    return lazy_pinyin(ch, style=Style.NORMAL)[0]


def to_pinyin(word: str) -> str:
    """整词转无声调拼音，字与字之间不加空格。"""
    return "".join(lazy_pinyin(word, style=Style.NORMAL))


@lru_cache(maxsize=1)
def ranked_common_chars(top_n: int = 3500) -> tuple[str, ...]:
    """COLD 训练集汉字按字频降序，同频按码位，取前 top_n 个。"""
    train = pd.read_csv(COLD_DIR / "train.csv", usecols=["TEXT"])
    counts = Counter(
        ch for text in train["TEXT"].astype(str) for ch in text
        if "\u4e00" <= ch <= "\u9fff"
    )
    return tuple(ch for ch, _ in
                 sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n])


def build_common_chars(top_n: int = 3500) -> frozenset[str]:
    return frozenset(ranked_common_chars(top_n))


@lru_cache(maxsize=1)
def load_chaizi() -> dict[str, str]:
    """只保留“恰好两个部件、且两个部件都是常用字”的第一种拆法。

    拆字表一行里用制表符分隔多种拆法，按出现顺序取第一种满足条件的。
    不满足条件的字不进表，变换时保持原样。
    """
    common = build_common_chars()
    table: dict[str, str] = {}
    path = EXT / "chaizi" / "raw" / "chaizi-jt.txt"
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        ch = fields[0]
        if ch in table:
            continue
        for field in fields[1:]:
            comps = field.split()
            if len(comps) == 2 and all(comp in common for comp in comps):
                table[ch] = "".join(comps)
                break
    return table


@lru_cache(maxsize=1)
def build_homophone_map(top_n: int = 3500) -> dict[str, str | None]:
    """按 COLD 训练集字频取前 top_n 个字，给每个字找同音的最高频他字。"""
    ranked = ranked_common_chars(top_n)
    groups: dict[str, list[str]] = {}
    for ch in ranked:
        groups.setdefault(char_pinyin(ch), []).append(ch)
    mapping: dict[str, str | None] = {}
    for group in groups.values():
        for ch in group:
            mapping[ch] = next((x for x in group if x != ch), None)
    return mapping


def pinyin_word(word: str) -> tuple[str, bool]:
    new = to_pinyin(word)
    return new, new != word


def homophone_word(word: str, mapping: dict[str, str | None]) -> tuple[str, bool]:
    chars, changed = list(word), False
    for i, ch in enumerate(chars):
        rep = mapping.get(ch)
        if rep and rep != ch:
            chars[i], changed = rep, True
    return "".join(chars), changed


def chaizi_word(word: str, table: dict[str, str]) -> tuple[str, bool]:
    chars, changed = [], False
    for ch in word:
        comp = table.get(ch)
        if comp and comp != ch:
            chars.append(comp)
            changed = True
        else:
            chars.append(ch)
    return "".join(chars), changed


def symbol_word(word: str, rng: random.Random) -> tuple[str, bool]:
    if len(word) < 2:
        return word, False
    chars = [word[0]]
    for ch in word[1:]:
        chars.extend((rng.choice(SYMBOLS), ch))
    return "".join(chars), True


def apply_transform(text: str, matches: list[tuple[int, int, str]], kind: str, *,
                    homophones: dict[str, str | None] | None = None,
                    chaizi: dict[str, str] | None = None,
                    rng: random.Random | None = None):
    """只替换匹配词，返回 (变体, 成功改动的词, [(原词, 新词)])。"""
    pieces: list[str] = []
    changed_words: list[str] = []
    changes: list[tuple[str, str]] = []
    last = 0
    for start, end, word in matches:
        if kind == "pinyin":
            new, ok = pinyin_word(word)
        elif kind == "homophone":
            new, ok = homophone_word(word, homophones or {})
        elif kind == "chaizi":
            new, ok = chaizi_word(word, chaizi or {})
        elif kind == "delete":
            new, ok = "", True
        elif kind == "symbol":
            if rng is None:
                raise ValueError("symbol 变换需要 rng")
            new, ok = symbol_word(word, rng)
        else:
            raise ValueError(f"未知变换 {kind!r}")
        pieces.append(text[last:start])
        pieces.append(new if ok else word)
        last = end
        if ok:
            changed_words.append(word)
            changes.append((word, new))
    pieces.append(text[last:])
    return "".join(pieces), changed_words, changes


def generate() -> dict[str, list[dict]]:
    """生成四份规避集，返回内存中的行，供抽查复用。"""
    words = load_lexicon()
    by_len, lengths = build_word_index(words)
    homophones = build_homophone_map()
    chaizi = load_chaizi()
    test = pd.read_csv(COLD_DIR / "test.csv")

    rows: dict[str, list[dict]] = {kind: [] for kind in KINDS + CONTROL_KINDS}
    rng = random.Random(SEED)
    for pair_id, row in test.iterrows():
        if int(row["label"]) != 1:
            continue
        text = str(row["TEXT"])
        matches = find_matches(text, by_len, lengths)
        if not matches:
            continue
        matched_words = "|".join(word for _, _, word in matches)
        for kind in KINDS + CONTROL_KINDS:
            edit, changed_words, changes = apply_transform(
                text, matches, kind, homophones=homophones,
                chaizi=chaizi, rng=rng,
            )
            if not changed_words:
                continue
            rows[kind].append({
                "pair_id": int(pair_id),
                "orig": text,
                "edit": edit,
                "label": 1,
                "kind": kind,
                "matched_words": matched_words,
                "_changes": changes,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["pair_id", "orig", "edit", "label", "kind", "matched_words"]
    for kind in KINDS + CONTROL_KINDS:
        frame = pd.DataFrame(
            [{key: row[key] for key in columns} for row in rows[kind]],
            columns=columns,
        )
        frame.to_csv(OUT_DIR / f"cloak_{kind}.csv", index=False, encoding="utf-8")
    return rows


def write_spotcheck(rows: dict[str, list[dict]], per_kind: int = 25) -> Path:
    """每种变换抽 per_kind 条，不足全取；UTF-8 带 BOM 方便 Excel 打开。"""
    rng = random.Random(SEED)
    picked: list[dict] = []
    for kind in KINDS:
        pool = rows[kind]
        take = pool if len(pool) <= per_kind else rng.sample(pool, per_kind)
        for row in take:
            changes = "；".join(f"{old}→{new}" for old, new in row["_changes"])
            picked.append({
                "变换": kind,
                "原句": row["orig"],
                "变体": row["edit"],
                "改动的词": changes,
                "意思是否不变（填 是/否）": "",
                "备注": "",
            })
    frame = pd.DataFrame(picked)
    frame.insert(0, "序号", range(1, len(frame) + 1))
    path = OUT_DIR / "抽查_规避.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    rows = generate()
    path = write_spotcheck(rows)
    for kind in KINDS:
        print(f"cloak_{kind}.csv: {len(rows[kind])} 条")
    print(f"抽查_规避.csv: {path}（{sum(min(len(rows[k]), 25) for k in KINDS)} 条）")


if __name__ == "__main__":
    main()
