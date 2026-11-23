"""生成 COLD 群体词互换集：按话题把评论中唯一出现的群体词换成同话题词。

region、race 每个原句换 3 个不同的词；gender 只有成对的那一个。
安全与攻击性评论都用，标签保留原评论的 label。
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLD_DIR = ROOT / "data" / "external" / "coldataset" / "raw" / "COLDataset"
OUT_DIR = ROOT / "data_selfmade"
SEED = 2026

GROUP_WORDS = {
    "region": (
        "北京人 上海人 天津人 重庆人 河北人 山西人 辽宁人 吉林人 黑龙江人 "
        "江苏人 浙江人 安徽人 福建人 江西人 山东人 河南人 湖北人 湖南人 "
        "广东人 海南人 四川人 贵州人 云南人 陕西人 甘肃人 青海人 台湾人 "
        "内蒙古人 广西人 西藏人 宁夏人 新疆人 香港人 澳门人 东北人 南方人 北方人"
    ).split(),
    "gender": "男人 女人 男性 女性 男生 女生 男的 女的 男孩 女孩".split(),
    "race": "黑人 白人 黄种人 亚洲人 非洲人 欧洲人 美国人 日本人 韩国人 印度人".split(),
}

GENDER_PAIRS = [
    ("男人", "女人"), ("男性", "女性"), ("男生", "女生"),
    ("男的", "女的"), ("男孩", "女孩"),
]
GENDER_SWAP = {a: b for a, b in GENDER_PAIRS} | {b: a for a, b in GENDER_PAIRS}


def build_index(words: list[str]) -> tuple[dict[int, set[str]], list[int]]:
    by_len: dict[int, set[str]] = {}
    for word in words:
        by_len.setdefault(len(word), set()).add(word)
    return by_len, sorted(by_len, reverse=True)


GROUP_INDEX = {topic: build_index(words) for topic, words in GROUP_WORDS.items()}


def find_matches(text: str, topic: str) -> list[tuple[int, int, str]]:
    """在某一话题的群体词表内做最长优先、不重叠匹配。"""
    if topic not in GROUP_INDEX:
        return []
    by_len, lengths = GROUP_INDEX[topic]
    out: list[tuple[int, int, str]] = []
    i, n = 0, len(text)
    while i < n:
        for length in lengths:
            if i + length <= n and text[i:i + length] in by_len[length]:
                out.append((i, i + length, text[i:i + length]))
                i += length
                break
        else:
            i += 1
    return out


def make_variants(text: str, topic: str, rng: random.Random) -> list[tuple[str, str, str]]:
    """群体词恰好出现一次时返回 [(变体, 原词, 新词)]，否则空表。"""
    matches = find_matches(text, topic)
    if len(matches) != 1:
        return []
    start, end, word = matches[0]
    if topic == "gender":
        targets = [GENDER_SWAP[word]]
    else:
        candidates = [w for w in GROUP_WORDS[topic] if w != word]
        targets = rng.sample(candidates, 3)
    return [(text[:start] + target + text[end:], word, target) for target in targets]


def generate() -> list[dict]:
    test = pd.read_csv(COLD_DIR / "test.csv")
    rows: list[dict] = []
    rng = random.Random(SEED)
    for src_row, row in test.iterrows():
        topic = str(row["topic"])
        text = str(row["TEXT"])
        for seq, (edit, from_word, to_word) in enumerate(make_variants(text, topic, rng), 1):
            rows.append({
                "pair_id": len(rows),
                "orig": text,
                "edit": edit,
                "label": int(row["label"]),
                "kind": "swap",
                "topic": topic,
                "from_word": from_word,
                "to_word": to_word,
                "src_row": int(src_row),
                "variant": seq,
            })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["pair_id", "orig", "edit", "label", "kind", "topic",
               "from_word", "to_word", "src_row", "variant"]
    pd.DataFrame(rows, columns=columns).to_csv(
        OUT_DIR / "swap.csv", index=False, encoding="utf-8")
    return rows


def write_spotcheck(rows: list[dict], n: int = 100) -> Path:
    """随机抽 n 条，UTF-8 带 BOM 方便 Excel 打开。"""
    rng = random.Random(SEED)
    take = rows if len(rows) <= n else rng.sample(rows, n)
    picked = [{
        "话题": row["topic"],
        "原句": row["orig"],
        "变体": row["edit"],
        "句子是否通顺且指代合理（填 是/否）": "",
        "备注": "",
    } for row in take]
    frame = pd.DataFrame(picked)
    frame.insert(0, "序号", range(1, len(frame) + 1))
    path = OUT_DIR / "抽查_互换.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    rows = generate()
    path = write_spotcheck(rows)
    counts = pd.Series([row["topic"] for row in rows]).value_counts()
    print(f"swap.csv: {len(rows)} 条")
    for topic in GROUP_WORDS:
        print(f"  {topic}: {int(counts.get(topic, 0))} 条")
    print(f"抽查_互换.csv: {path}（{min(len(rows), 100)} 条）")


if __name__ == "__main__":
    main()
