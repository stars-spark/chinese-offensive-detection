"""数据读取与清洗（流程第 1 步）。

本文件负责：
1. 从 parquet 读 IMDB / AG News；
2. 清洗文本；
3. 按标签分层抽 10% 官方训练集作验证集（种子固定 42，与模型种子无关）；
4. 正则分词；
5. subsample：按标签分层抽样，供阶段 2 的学习曲线使用。

后面 vocab.py 只接收这里的 token 列表，train.py 只接收这里的 DataFrame，
所以数据层面的改动都集中在本文件，便于检查。
"""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# 项目根目录（实验/），数据在它旁边的 data/ 下
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "external"
SELF_MADE_ROOT = ROOT / "data_selfmade"

# 每个数据集的默认超参数，train.py 从这里取默认
DATASETS = {
    "imdb": {
        "max_len": 256,
        "epochs": 6,
        "num_classes": 2,
        "class_names": ["负面", "正面"],
    },
    "ag_news": {
        "max_len": 64,
        "epochs": 4,
        "num_classes": 4,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "cold": {
        "max_len": 192,
        "epochs": 10,
        "num_classes": 2,
        "class_names": ["安全", "攻击性"],
    },
}

# 分词正则：先小写，再切成“字母数字词（可带 's）”或“单个非字母数字字符”
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
# 换行、制表、连续空格都算空白；清洗后统一成一个空格
_WS_RE = re.compile(r"\s+")
# COLD 去重要求去掉“全部”空白，包括词之间的空格，而不只是合并
_WS_STRIP_RE = re.compile(r"\s+")
# IMDB 的换行标记，大小写和多余空格都兼容
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", flags=re.IGNORECASE)


def _is_chinese_dataset(dataset: str) -> bool:
    """中文数据（COLD、HateCheck、ToxiCN、自制集）走中文清洗与 jieba。"""
    return (dataset in ("cold", "mhc_zh", "toxicn_test")
            or dataset.startswith("cloak_") or dataset.startswith("swap_"))


def clean_text(text: str, dataset: str) -> str:
    """清洗一条文本。

    顺序说明：先 html.unescape 再删 <br>。如果先删 <br>，像
    ``&lt;br /&gt;`` 这种被转义的标记会在 unescape 之后重新出现。
    AG News 的原文里有字面反斜杠（原数据换行残留），替换成空格，
    否则会被当成分词的一部分。

    中文数据只做 NFKC、空白合并和英文字母小写，不改写其他任何字符；
    英文数据的路径保持阶段 1 原样，逐位不变。
    """
    if _is_chinese_dataset(dataset):
        text = unicodedata.normalize("NFKC", str(text)).lower()
        return _WS_RE.sub(" ", text).strip()
    text = html.unescape(str(text))
    text = _BR_RE.sub(" ", text)
    if dataset == "ag_news":
        text = text.replace("\\", " ")
    # 连续空白合并，并去掉首尾空白
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """小写化后按正则切成词和标点。

    标点单独成词，是为了让 "great!" 和 "great" 共享 great，
    同时 "!" 本身也能作为一个（虽然很弱的）特征。
    """
    return _TOKEN_RE.findall(text.lower())


# 中文按字分词：一个汉字一个 token，连续字母数字串整体一个，其余非空白字符
# （标点、符号）各自一个。只匹配常用汉字区，扩展区字符会落到标点分支。
_CJK_CHAR = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
_CHAR_TOKEN_RE = re.compile(rf"{_CJK_CHAR}|[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\s]")


def char_tokenize(text: str) -> list[str]:
    """char 分词："你好abc，世界" → 你/好/abc/，/世/界。"""
    return _CHAR_TOKEN_RE.findall(text.lower())


def word_tokenize(text: str, dataset: str = "imdb") -> list[str]:
    """word 分词：中文用 jieba 精确模式，英文沿用阶段 1 的正则分词。"""
    if _is_chinese_dataset(dataset):
        import jieba

        return [tok for tok in jieba.lcut(text) if tok.strip()]
    return tokenize(text)


def tokenize_text(text: str, dataset: str = "imdb", tokenizer: str = "word") -> list[str]:
    """按 ``--tokenizer`` 选分词器；默认 word 与阶段 1 行为一致。"""
    if tokenizer == "char":
        return char_tokenize(text)
    if tokenizer == "word":
        return word_tokenize(text, dataset)
    raise ValueError(f"未知分词器 {tokenizer!r}，可选 char / word")


def _read_cold_csv(path: Path, split: str) -> pd.DataFrame:
    """读 COLD 的一份 csv，统一成 text/label，并保留测试集的细粒度标签。"""
    frame = pd.read_csv(path)
    out = pd.DataFrame({
        "text": frame["TEXT"].map(lambda t: clean_text(t, "cold")),
        "label": frame["label"].astype(np.int64),
    })
    if "topic" in frame.columns:
        out["topic"] = frame["topic"].astype(str)
    if split == "test":
        if "fine-grained-label" not in frame.columns:
            raise ValueError(f"{path} 缺少 fine-grained-label 列")
        out["fine-grained-label"] = frame["fine-grained-label"].astype(np.int64)
    return out


def _load_cold_splits() -> dict[str, pd.DataFrame]:
    """COLD 的官方 train/dev/test，不做跨划分去重（那个由 --decontaminate 控制）。"""
    raw = DATA_ROOT / "coldataset" / "raw" / "COLDataset"
    return {
        "train": _read_cold_csv(raw / "train.csv", "train").reset_index(drop=True),
        "val": _read_cold_csv(raw / "dev.csv", "dev").reset_index(drop=True),
        "test": _read_cold_csv(raw / "test.csv", "test").reset_index(drop=True),
    }


def load_splits(dataset: str) -> dict[str, pd.DataFrame]:
    """读取并清洗数据，返回 train / val / test 三个 DataFrame。

    只从官方训练集切验证集，官方测试集整份留到最后评估一次，
    不参与任何调参或选轮次。stratify 保证各标签比例一致。
    COLD 用官方 train/dev/test，验证集不重新切。
    """
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}，可选 {sorted(DATASETS)}")
    if dataset == "cold":
        return _load_cold_splits()
    raw_dir = DATA_ROOT / dataset / "raw"
    train = pd.read_parquet(raw_dir / "train-00000-of-00001.parquet")
    test = pd.read_parquet(raw_dir / "test-00000-of-00001.parquet")

    for frame in (train, test):
        frame["text"] = frame["text"].map(lambda t: clean_text(t, dataset))

    train_part, val_part = train_test_split(
        train,
        test_size=0.10,
        stratify=train["label"],
        random_state=42,  # 固定 42，所有模型共用同一份划分
    )
    return {
        "train": train_part.reset_index(drop=True),
        "val": val_part.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def subsample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """按标签分层抽 ``frac`` 比例，阶段 2 学习曲线用。

    逐类抽样而不是整体抽样，否则小数据量时某些类可能一条都抽不到。
    每类至少保留 1 条，保证分层抽样在 frac 很小时也不报错。
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac 必须在 (0, 1] 内")
    if frac == 1.0:
        return df.copy()
    parts = []
    for _, group in df.groupby("label", sort=True):
        n = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    out = pd.concat(parts)
    # 打乱顺序，避免训练时同类样本扎堆（DataLoader 还会再 shuffle 一次）
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_sample_n(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """按标签分层恰好抽 ``n`` 条训练样本（阶段 2 的 ``--train_n``）。

    和 ``subsample`` 的区别是这里指定的是绝对条数，不是比例。每类目标
    条数按原数据类别比例分配：先取整数部分，余下的名额按小数部分从大到小
    补齐（最大余数法），保证总数恰好为 n，且每类与理论比例的差不超过 1 条。

    抽样种子就是模型种子，所以不同种子既换样本也换初始化。验证集/测试集
    不在这里动，它们的划分始终由 ``load_splits`` 固定。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    total = len(df)
    if n > total:
        raise ValueError(f"train_n={n} 超过训练集大小 {total}")
    if n == total:
        return df.copy()

    counts = df["label"].value_counts().sort_index()
    labels = list(counts.index)
    sizes = counts.to_numpy(dtype=np.int64)
    raw = sizes * (n / total)
    target = np.floor(raw).astype(np.int64)
    # 最大余数法补足剩余名额；余数相同时按标签升序，保证可复现
    order = np.argsort(-(raw - target), kind="stable")
    remaining = n - int(target.sum())
    cursor = 0
    while remaining > 0:
        j = int(order[cursor % len(order)])
        if target[j] < sizes[j]:
            target[j] += 1
            remaining -= 1
        cursor += 1

    parts = []
    for label, size, k in zip(labels, sizes, target):
        group = df[df["label"] == label]
        parts.append(group.sample(n=int(min(k, size)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    # 打乱顺序，避免同类样本在训练前期扎堆
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def truncate_text(text: str, max_len: int | None,
                  tokenizer: str = "word", dataset: str = "imdb") -> str:
    """按神经网络同一套分词截断，再拼回空格分隔的字符串。

    TF-IDF 的 ``max_len`` 实验要让两种方法看到完全相同的词，因此不能按
    字符截断，必须先分词再取前 max_len 个 token。默认参数保持阶段 1 的
    ``tokenize`` 行为；COLD 传 ``tokenizer="char"/"word"`` 与训练一致。
    max_len 为 None 表示不截断，原样返回。
    """
    if max_len is None:
        return text
    if max_len <= 0:
        raise ValueError("max_len 必须为正整数或 None")
    return " ".join(tokenize_text(text, dataset, tokenizer)[:max_len])


# 阶段 3 的五份评测集。paired=True 的集有原始/改写两份，按 pair_id 一一对应。
EVAL_SETS = {
    "contrast_orig": {"family": "contrast", "file": "test_original.tsv", "paired": True},
    "contrast_edit": {"family": "contrast", "file": "test_contrast.tsv", "paired": True},
    "cad_orig": {"family": "cad", "file": "orig_test.tsv", "paired": True},
    "cad_edit": {"family": "cad", "file": "new_test.tsv", "paired": True},
    "yelp": {"family": "yelp", "file": "test-00000-of-00001.parquet", "paired": False},
    "mhc_zh": {"family": "mhc_zh", "file": "test.csv", "paired": False},
    "toxicn_test": {"family": "toxicn_test", "file": "test.json", "paired": False},
}
# 自制集：每个变换一份 CSV，含 orig/edit 两列，载入时拆成两份评测集。
SELF_MADE_KINDS = ("pinyin", "homophone", "chaizi", "symbol", "delete")
for _kind in SELF_MADE_KINDS:
    for _side in ("orig", "edit"):
        EVAL_SETS[f"cloak_{_kind}_{_side}"] = {
            "family": "selfmade", "file": f"cloak_{_kind}.csv",
            "paired": True, "side": _side, "kind": _kind,
        }
for _side in ("orig", "edit"):
    EVAL_SETS[f"swap_{_side}"] = {
        "family": "selfmade", "file": "swap.csv",
        "paired": True, "side": _side, "kind": "swap",
    }

# 污染检查用到的全部原始评论文件（测试和开发部分都算）
CONTAMINATION_FILES = {
    "contrast": ("test_original.tsv", "dev_original.tsv"),
    "cad": ("orig_test.tsv", "orig_dev.tsv"),
}

_SENTIMENT = {"positive": 1, "negative": 0}


def _read_sentiment_tsv(path: Path) -> pd.DataFrame:
    """读两列 TSV，并把列名两端的空格去掉。

    下载到的文件列名是 ``Sentiment``、``Text``，但可能带空格，
    所以先 strip 再检查，避免 ``df["Text"]`` 因为列名是 ``"Text "`` 报错。
    """
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame.columns = [c.strip() for c in frame.columns]
    missing = {"Sentiment", "Text"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列 {sorted(missing)}")
    frame["Text"] = frame["Text"].map(lambda t: clean_text(t, "imdb"))
    frame["label"] = frame["Sentiment"].map(
        lambda s: _SENTIMENT[s.strip().lower()]
    ).astype(np.int64)
    return frame


def _opposite_rowwise(labels_a: np.ndarray, labels_b: np.ndarray) -> bool:
    return len(labels_a) == len(labels_b) and bool(np.all(labels_a + labels_b == 1))


def _match_pairs(orig_texts: list[str], edit_texts: list[str]) -> np.ndarray:
    """给被打乱顺序的改写集找回配对。

    实测 CAD 的 ``orig_test.tsv`` 按标签排好序，``new_test.tsv`` 却是乱序的，
    直接按行号配对时 488 条里只有 264 条标签相反。两份文件的标签分布恰好
    互补（245/243 对 243/245），说明配对存在、只是顺序被打乱。这里用
    1~2 元 TF-IDF 余弦相似度做二分图最大权匹配（匈牙利算法）；改写通常
    保留原评论的大部分词，实测 488 对互为最近邻且标签全部相反。
    这是对“按行号一一对应”的必要偏离。
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(list(orig_texts) + list(edit_texts))
    n = len(orig_texts)
    sim = cosine_similarity(matrix[:n], matrix[n:])
    rows, cols = linear_sum_assignment(-sim)
    order = np.empty(n, dtype=np.int64)
    order[rows] = cols
    return order


def _load_paired(family: str, orig_file: str, edit_file: str):
    """读一份配对评测集，返回 (原始表, 改写表)，两份的行按 pair_id 对齐。"""
    raw = DATA_ROOT / f"imdb_{family}" / "raw"
    orig = _read_sentiment_tsv(raw / orig_file)
    edit = _read_sentiment_tsv(raw / edit_file)
    if len(orig) != len(edit):
        raise ValueError(f"{family} 原始/改写行数不同：{len(orig)} vs {len(edit)}")

    if not _opposite_rowwise(orig["label"].to_numpy(), edit["label"].to_numpy()):
        # CAD 的顺序被打乱：重排改写表，使第 i 行与原始表第 i 行是同一对
        order = _match_pairs(orig["Text"].tolist(), edit["Text"].tolist())
        edit = edit.iloc[order].reset_index(drop=True)
    if not _opposite_rowwise(orig["label"].to_numpy(), edit["label"].to_numpy()):
        n_bad = int((orig["label"].to_numpy() + edit["label"].to_numpy() != 1).sum())
        raise ValueError(f"{family} 配对后仍有 {n_bad} 条标签不相反，检查配对逻辑")
    return orig, edit


def load_eval_set(name: str) -> pd.DataFrame:
    """读一份评测集，返回统一的 text/label 表（配对集另有 pair_id）。

    中文评测集保留分析要用的列：HateCheck 保留 ``functionality``，
    ToxiCN 保留 ``platform``/``expression``，自制集保留 ``kind``/``topic``。
    旧英文评测集的行为保持阶段 3 原样。自制集文件不存在时抛
    ``FileNotFoundError``，由调用方跳过并记日志。
    """
    if name not in EVAL_SETS:
        raise ValueError(f"未知评测集 {name!r}，可选 {sorted(EVAL_SETS)}")
    spec = EVAL_SETS[name]
    if spec["family"] == "mhc_zh":
        raw = DATA_ROOT / "huggingface-Paul_hatecheck-mandarin" / "raw" / spec["file"]
        frame = pd.read_csv(raw)
        missing = {"test_case", "label_gold", "functionality"} - set(frame.columns)
        if missing:
            raise ValueError(f"{raw.name} 缺少列 {sorted(missing)}")
        out = pd.DataFrame({
            "text": frame["test_case"].map(lambda t: clean_text(t, "mhc_zh")),
            "label": (frame["label_gold"].str.strip().str.lower() == "hateful").astype(np.int64),
            "functionality": frame["functionality"].astype(str),
        })
        return out.reset_index(drop=True)

    if spec["family"] == "toxicn_test":
        raw = DATA_ROOT / "toxicn" / "raw" / "ToxiCN_ex" / "ToxiCN" / "data" / spec["file"]
        frame = pd.read_json(raw)
        missing = {"content", "toxic", "platform", "expression"} - set(frame.columns)
        if missing:
            raise ValueError(f"{raw.name} 缺少列 {sorted(missing)}")
        out = pd.DataFrame({
            "text": frame["content"].map(lambda t: clean_text(t, "toxicn_test")),
            "label": frame["toxic"].astype(np.int64),
            "platform": frame["platform"].astype(str),
            "expression": frame["expression"].astype(np.int64),
        })
        # 剔除与 COLD 任一划分归一化后相同的条目
        blocked = _cold_all_keys()
        mask = out["text"].map(normalize_for_cold_match).isin(blocked)
        out = out.loc[~mask].reset_index(drop=True)
        out.attrs["n_removed_cold"] = int(mask.sum())
        return out

    if spec["family"] == "selfmade":
        path = SELF_MADE_ROOT / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"自制集文件不存在: {path}")
        frame = pd.read_csv(path, keep_default_na=False)
        required = {"pair_id", "orig", "edit", "label", "kind"} - set(frame.columns)
        if required:
            raise ValueError(f"{path.name} 缺少列 {sorted(required)}")
        text_col = spec["side"]
        out = pd.DataFrame({
            "text": frame[text_col].map(lambda t: clean_text(t, name)),
            "label": frame["label"].astype(np.int64),
            "pair_id": frame["pair_id"].astype(np.int64),
            "kind": frame["kind"].astype(str),
        })
        if "topic" in frame.columns:
            out["topic"] = frame["topic"].astype(str)
        return out.reset_index(drop=True)

    if spec["family"] == "yelp":
        raw = DATA_ROOT / "yelp_polarity" / "raw" / spec["file"]
        frame = pd.read_parquet(raw)
        frame["text"] = frame["text"].map(lambda t: clean_text(t, "imdb"))
        frame["label"] = frame["label"].astype(np.int64)
        return frame[["text", "label"]].reset_index(drop=True)

    orig, edit = _load_paired(
        spec["family"],
        EVAL_SETS[f"{spec['family']}_orig"]["file"],
        EVAL_SETS[f"{spec['family']}_edit"]["file"],
    )
    frame = orig if name.endswith("_orig") else edit
    out = pd.DataFrame({
        "text": frame["Text"].to_numpy(),
        "label": frame["label"].to_numpy(),
        "pair_id": np.arange(len(frame), dtype=np.int64),
    })
    return out


def normalize_for_cold_match(text: str) -> str:
    """COLD 去重口径：NFKC、英文字母小写、去掉全部空白后比较。"""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _WS_STRIP_RE.sub("", text)


def deduplicate_cold_splits(
    splits: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """从 train 剔除 dev/test、从 dev 剔除 test 的重复评论。

    只跨划分去重，划分内部重复保留。返回新表和
    ``{"train": 剔除数, "val": 剔除数, "test": 0}``，不改传入的 DataFrame。
    """
    train = splits["train"].copy()
    val = splits["val"].copy()
    test = splits["test"].copy()
    train_keys = train["text"].map(normalize_for_cold_match)
    val_keys = val["text"].map(normalize_for_cold_match)
    test_keys = test["text"].map(normalize_for_cold_match)
    blocked_for_train = set(val_keys) | set(test_keys)
    blocked_for_val = set(test_keys)
    train_mask = train_keys.isin(blocked_for_train)
    val_mask = val_keys.isin(blocked_for_val)
    removed = {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": 0}
    return {
        "train": train.loc[~train_mask].reset_index(drop=True),
        "val": val.loc[~val_mask].reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }, removed


@lru_cache(maxsize=1)
def _cold_all_keys() -> frozenset[str]:
    """COLD 三个划分的全部归一化文本，供 ToxiCN 交叉去重。"""
    splits = load_splits("cold")
    keys = set()
    for frame in splits.values():
        keys.update(frame["text"].map(normalize_for_cold_match))
    return frozenset(keys)


def normalize_for_match(text: str) -> str:
    """污染比对用的归一化：反转义、去 <br>、小写、合并空白。

    ``clean_text`` 不做小写，而污染比对要求大小写不敏感，所以单独写。
    替换 <br> 用空格而不是空串，避免 ``a<br>b`` 被粘成一个词。
    """
    text = html.unescape(str(text))
    text = _BR_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip().lower()


@lru_cache(maxsize=1)
def _contamination_frozen() -> frozenset[str]:
    texts = set()
    for family, files in CONTAMINATION_FILES.items():
        raw = DATA_ROOT / f"imdb_{family}" / "raw"
        for file_name in files:
            frame = _read_sentiment_tsv(raw / file_name)
            texts.update(normalize_for_match(t) for t in frame["Text"])
    return frozenset(texts)


def contamination_texts() -> set[str]:
    """对比集与 CAD 全部原始评论（测试+开发）的归一化文本集合。"""
    return set(_contamination_frozen())


def decontaminate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """删掉归一化后落在污染集合里的行，返回 (新表, 剔除条数)。

    不修改传入的表；新表重置索引，方便后面按位置建词表和写预测。
    """
    blocked = contamination_texts()
    mask = df["text"].map(normalize_for_match).isin(blocked)
    removed = int(mask.sum())
    return df.loc[~mask].reset_index(drop=True), removed


def load_aug_csv(path: str | Path) -> pd.DataFrame:
    """读 D1 增强 CSV，只保留训练要用的 ``text`` 与 ``label``。

    相对路径按项目根目录解析。``--aug_csv`` 默认为空，调用方不传时完全
    不会走到这里，原有训练数据逐条不变。增强行里的文本已经是清洗后的
    训练文本的变体，这里不再做 ``clean_text``，避免空格类符号被合并。
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"增强 CSV 不存在: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列 {sorted(missing)}")
    out = pd.DataFrame({
        "text": frame["text"].astype(str),
        "label": frame["label"].astype(np.int64),
    })
    return out.reset_index(drop=True)


def append_aug(train_df: pd.DataFrame, aug_csv: str | Path | None
               ) -> tuple[pd.DataFrame, int]:
    """把 D1 增强行追加到训练集；``aug_csv`` 为空时原样返回并记 0 条。

    finetune_bert.py 与 baseline_tfidf.py 共用这一个函数，保证两个入口
    在默认空参数下都不会碰训练集，逐条与原来相同。
    """
    if not aug_csv:
        return train_df, 0
    aug = load_aug_csv(aug_csv)
    return pd.concat([train_df, aug], ignore_index=True), len(aug)


def parse_extra_eval(value: str | None) -> list[str]:
    """把 ``--extra_eval contrast_orig,cad_edit`` 解析成去重后的名字列表。"""
    if not value:
        return []
    names = []
    for part in value.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in EVAL_SETS:
            raise ValueError(f"未知评测集 {name!r}，可选 {sorted(EVAL_SETS)}")
        if name not in names:
            names.append(name)
    return names


def eval_set_available(name: str) -> bool:
    """自制集文件还没生成时返回 False；其他集默认可读。"""
    spec = EVAL_SETS.get(name)
    if spec is None:
        return False
    if spec["family"] == "selfmade":
        return (SELF_MADE_ROOT / spec["file"]).exists()
    return True
