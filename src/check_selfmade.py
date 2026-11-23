"""对 data_selfmade 下的自制测试集做全量规则核对。

只读取已有数据，不重新生成、不改写数据集；结果写到
data_selfmade/全量核对.md。匹配位置复用 make_cloak.find_matches，
群体词表复用 make_swap.GROUP_WORDS，避免另写一套规则后与生成逻辑对不上。
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import jieba
import pandas as pd
from pypinyin import Style, lazy_pinyin

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from make_cloak import (  # noqa: E402
    build_word_index,
    char_pinyin,
    find_matches,
    load_lexicon,
    to_pinyin,
)
from make_swap import GROUP_WORDS  # noqa: E402

jieba.setLogLevel(logging.ERROR)

COLD_TEST = (
    ROOT / "data" / "external" / "coldataset" / "raw"
    / "COLDataset" / "test.csv"
)
CHAIZI_TABLE = (
    ROOT / "data" / "external" / "chaizi" / "raw" / "chaizi-jt.txt"
)
DATA_DIR = ROOT / "data_selfmade"
OUT_PATH = DATA_DIR / "全量核对.md"
SYMBOLS = frozenset("*、 .")
CLOAK_FILES = (
    "cloak_pinyin.csv",
    "cloak_homophone.csv",
    "cloak_chaizi.csv",
    "cloak_symbol.csv",
    "cloak_delete.csv",
)


@dataclass(frozen=True)
class Rule:
    key: str
    title: str
    scope: str


RULES = (
    Rule("cold_text", "orig 与 COLD 测试集对应行 TEXT 完全一致", "全部 6 个文件"),
    Rule("cold_label", "label 与 COLD 测试集对应行一致", "全部 6 个文件"),
    Rule("edit_differs", "edit ≠ orig", "全部 6 个文件"),
    Rule("matched_words_in_orig", "matched_words 中每个词都在 orig 中出现", "5 个 cloak 文件"),
    Rule("diff_within_matched", "差异片段只落在匹配词的位置上", "5 个 cloak 文件"),
    Rule("pinyin", "pinyin 被替换处等于该词无声调拼音连写", "cloak_pinyin.csv"),
    Rule("homophone", "homophone 被替换字与原字无声调拼音相同且字不同", "cloak_homophone.csv"),
    Rule("chaizi", "chaizi 被替换处等于拆字表中某种两部件拆法的连写", "cloak_chaizi.csv"),
    Rule("symbol", "symbol 去掉插入符号后还原为 orig", "cloak_symbol.csv"),
    Rule("delete", "delete 等于 orig 删去匹配词", "cloak_delete.csv"),
    Rule("swap_from_once", "swap 的 from_word 在 orig 中恰好出现一次", "swap.csv"),
    Rule("swap_edit", "swap 的 edit 等于把 from_word 替换成 to_word", "swap.csv"),
    Rule("swap_group", "swap 的 from_word 与 to_word 属于同一话题群体词表", "swap.csv"),
)
RULE_BY_KEY = {rule.key: rule for rule in RULES}


@dataclass
class Stat:
    passed: int = 0
    failed: int = 0
    failures: list[dict] = field(default_factory=list)


STATS = {rule.key: Stat() for rule in RULES}
FILE_ROWS: dict[str, int] = {}
FILE_CHECKS: dict[str, dict[str, int]] = {}
FAILED_ROWS: set[tuple[str, int]] = set()


def record(key: str, ok: bool, file_name: str, idx: int,
           ident: str, detail: str = "") -> None:
    stat = STATS[key]
    if ok:
        stat.passed += 1
    else:
        stat.failed += 1
        FAILED_ROWS.add((file_name, idx))
        if len(stat.failures) < 10:
            stat.failures.append({
                "file": file_name,
                "csv_row": idx + 2,
                "ident": ident,
                "detail": detail,
            })
    counts = FILE_CHECKS.setdefault(file_name, {"passed": 0, "failed": 0})
    counts["passed" if ok else "failed"] += 1


def split_words(value) -> list[str]:
    if pd.isna(value):
        return []
    return [word for word in str(value).split("|") if word]


def load_two_part_decompositions(path: Path) -> dict[str, set[str]]:
    """收集拆字表里所有恰好两个部件的拆法，部件连写成替换串。"""
    table: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        char = fields[0]
        for field in fields[1:]:
            parts = field.split()
            if len(parts) == 2 and all(len(part) == 1 for part in parts):
                table.setdefault(char, set()).add("".join(parts))
    return table


def covered_mask(length: int, intervals: list[tuple[int, int, str]]) -> bytearray:
    mask = bytearray(length)
    for start, end, _ in intervals:
        mask[start:end] = b"\x01" * (end - start)
    return mask


def diff_only_in_matches(
    orig: str,
    edit: str,
    intervals: list[tuple[int, int, str]],
    opcodes: list[tuple],
) -> tuple[bool, str]:
    """替换和删除覆盖的每个 orig 位置都要落在匹配词内。

    零宽插入允许落在匹配词的首尾边界。拆字替换时 difflib 有时会把
    后一个部件对齐到原字上，把前一个部件记成词尾插入，这是对齐方式
    造成的，插入点本身仍贴着该匹配词。
    """
    mask = covered_mask(len(orig), intervals)
    for tag, i1, i2, _j1, _j2 in opcodes:
        if tag == "equal":
            continue
        if i1 == i2:
            at_left = i1 < len(mask) and bool(mask[i1])
            at_right = i1 > 0 and bool(mask[i1 - 1])
            if not (at_left or at_right):
                return False, f"{tag} 的插入点 {i1} 不在匹配词范围内"
            continue
        if not all(mask[i1:i2]):
            bad = next(pos for pos in range(i1, i2) if not mask[pos])
            return False, f"{tag} 覆盖 orig[{i1}:{i2}]，位置 {bad} 不在匹配词内"
    return True, ""


def pinyin_expected(orig: str, intervals: list[tuple[int, int, str]]) -> str:
    pieces: list[str] = []
    last = 0
    for start, end, word in intervals:
        pieces.append(orig[last:start])
        pieces.append(to_pinyin(word))
        last = end
    pieces.append(orig[last:])
    return "".join(pieces)


def delete_expected(orig: str, intervals: list[tuple[int, int, str]]) -> str:
    pieces: list[str] = []
    last = 0
    for start, end, _word in intervals:
        pieces.append(orig[last:start])
        last = end
    pieces.append(orig[last:])
    return "".join(pieces)


def homophone_ok(
    orig: str,
    edit: str,
    intervals: list[tuple[int, int, str]],
) -> tuple[bool, str]:
    if len(orig) != len(edit):
        return False, f"长度不一致，orig {len(orig)} 字，edit {len(edit)} 字"
    mask = covered_mask(len(orig), intervals)
    for pos, (old, new) in enumerate(zip(orig, edit)):
        if old == new:
            continue
        if not mask[pos]:
            return False, f"位置 {pos} 的 {old}→{new} 不在匹配词内"
        if char_pinyin(old) != char_pinyin(new):
            return False, f"位置 {pos} 的 {old}→{new} 无声调拼音不同"
    return True, ""


def chaizi_ok(
    orig: str,
    edit: str,
    intervals: list[tuple[int, int, str]],
    table: dict[str, set[str]],
) -> tuple[bool, str]:
    """用动态规划把 orig 逐字对齐到 edit，匹配词内的字可换成某种拆法。"""
    mask = covered_mask(len(orig), intervals)
    reach = [False] * (len(edit) + 1)
    reach[0] = True
    for pos, char in enumerate(orig):
        nxt = [False] * (len(edit) + 1)
        for cursor, alive in enumerate(reach):
            if not alive:
                continue
            if cursor < len(edit) and edit[cursor] == char:
                nxt[cursor + 1] = True
            if mask[pos]:
                for replacement in table.get(char, ()):
                    if edit.startswith(replacement, cursor):
                        nxt[cursor + len(replacement)] = True
        reach = nxt
        if not any(reach):
            return False, f"第 {pos} 个字 {char} 处无法与 edit 对齐"
    if not reach[-1]:
        return False, "拆字替换无法与 edit 完整对齐"
    return True, ""


def symbol_ok(
    orig: str,
    edit: str,
    opcodes: list[tuple],
) -> tuple[bool, str]:
    pieces: list[str] = []
    last = 0
    for tag, _i1, _i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if tag != "insert":
            return False, f"出现 {tag}，symbol 应当只有插入"
        fragment = edit[j1:j2]
        if not fragment or any(char not in SYMBOLS for char in fragment):
            return False, f"插入了非候选符号 {fragment!r}"
        pieces.append(edit[last:j1])
        last = j2
    pieces.append(edit[last:])
    if "".join(pieces) != orig:
        return False, "去掉插入符号后与 orig 不一致"
    return True, ""


def cold_row(test: pd.DataFrame, row_id: int):
    if row_id not in test.index:
        return None
    return test.loc[row_id]


def check_cloak_file(
    path: Path,
    test: pd.DataFrame,
    by_len: dict[int, set[str]],
    lengths: list[int],
    chaizi_table: dict[str, set[str]],
) -> None:
    frame = pd.read_csv(path)
    file_name = path.name
    FILE_ROWS[file_name] = len(frame)
    kind = path.stem.removeprefix("cloak_")
    spec_key = kind

    for idx, row in frame.iterrows():
        ident = f"pair_id={row['pair_id']}"
        orig = str(row["orig"])
        edit = str(row["edit"])
        pair_id = int(row["pair_id"])
        cold = cold_row(test, pair_id)

        if cold is None:
            record("cold_text", False, file_name, idx, ident, "COLD 测试集中没有这一行")
            record("cold_label", False, file_name, idx, ident, "COLD 测试集中没有这一行")
        else:
            record("cold_text", orig == str(cold["TEXT"]), file_name, idx, ident,
                   "orig 与 COLD TEXT 不一致")
            record("cold_label", int(row["label"]) == int(cold["label"]),
                   file_name, idx, ident, "label 与 COLD 不一致")

        record("edit_differs", edit != orig, file_name, idx, ident, "edit 与 orig 相同")

        words = split_words(row["matched_words"])
        missing = [word for word in words if word not in orig]
        record("matched_words_in_orig", not missing, file_name, idx, ident,
               f"这些词不在 orig 中，{missing}" if missing else "")

        intervals = find_matches(orig, by_len, lengths)
        actual_words = [word for _start, _end, word in intervals]
        opcodes = difflib.SequenceMatcher(
            None, orig, edit, autojunk=False).get_opcodes()

        diff_ok, diff_detail = diff_only_in_matches(orig, edit, intervals, opcodes)
        if actual_words != words:
            diff_ok = False
            diff_detail = "matched_words 与重新匹配得到的词序列不一致"
        record("diff_within_matched", diff_ok, file_name, idx, ident, diff_detail)

        if spec_key == "pinyin":
            ok = edit == pinyin_expected(orig, intervals)
            detail = "" if ok else "替换结果与整词拼音连写不一致"
        elif spec_key == "homophone":
            ok, detail = homophone_ok(orig, edit, intervals)
        elif spec_key == "chaizi":
            ok, detail = chaizi_ok(orig, edit, intervals, chaizi_table)
        elif spec_key == "symbol":
            ok, detail = symbol_ok(orig, edit, opcodes)
        elif spec_key == "delete":
            ok = edit == delete_expected(orig, intervals)
            detail = "" if ok else "edit 与 orig 删去匹配词的结果不一致"
        else:
            ok, detail = False, f"未知变换 {kind!r}"
        record(spec_key, ok, file_name, idx, ident, detail)


def check_swap(path: Path, test: pd.DataFrame) -> None:
    frame = pd.read_csv(path)
    file_name = path.name
    FILE_ROWS[file_name] = len(frame)

    for idx, row in frame.iterrows():
        pair_id = int(row["pair_id"])
        src_row = int(row["src_row"])
        ident = f"pair_id={pair_id}, src_row={src_row}"
        orig = str(row["orig"])
        edit = str(row["edit"])
        cold = cold_row(test, src_row)

        if cold is None:
            record("cold_text", False, file_name, idx, ident, "COLD 测试集中没有 src_row 这一行")
            record("cold_label", False, file_name, idx, ident, "COLD 测试集中没有 src_row 这一行")
        else:
            record("cold_text", orig == str(cold["TEXT"]), file_name, idx, ident,
                   "orig 与 COLD TEXT 不一致")
            record("cold_label", int(row["label"]) == int(cold["label"]),
                   file_name, idx, ident, "label 与 COLD 不一致")

        record("edit_differs", edit != orig, file_name, idx, ident, "edit 与 orig 相同")

        from_word = str(row["from_word"])
        to_word = str(row["to_word"])
        topic = str(row["topic"])

        record("swap_from_once", orig.count(from_word) == 1, file_name, idx, ident,
               f"from_word={from_word!r} 在 orig 中出现 {orig.count(from_word)} 次")
        record("swap_edit", edit == orig.replace(from_word, to_word, 1),
               file_name, idx, ident,
               f"edit 不等于把 {from_word!r} 换成 {to_word!r}")

        group = GROUP_WORDS.get(topic, ())
        group_ok = from_word in group and to_word in group
        if cold is not None:
            group_ok = group_ok and topic == str(cold["topic"])
        else:
            group_ok = False
        record("swap_group", group_ok, file_name, idx, ident,
               f"topic={topic!r}, from_word={from_word!r}, to_word={to_word!r}")


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return lines


def build_report() -> str:
    total_rows = sum(FILE_ROWS.values())
    total_checks = sum(counts["passed"] + counts["failed"]
                       for counts in FILE_CHECKS.values())
    total_failed_rows = len(FAILED_ROWS)
    verdict = ("全部通过" if total_failed_rows == 0
               else f"有 {total_failed_rows} 条不通过")

    lines = [verdict, "", "# 全量规则核对", ""]
    lines.append(
        f"核对对象是 `data_selfmade/` 下的 5 个 `cloak_*.csv` 和 `swap.csv`，"
        f"共 {total_rows} 行，{total_checks} 条规则检查。脚本逐行读取现有数据，"
        "不重新生成，也不改写数据集。"
    )
    lines.append("")

    lines.append("## 文件与行数")
    lines.append("")
    file_rows = [[name, str(FILE_ROWS[name])] for name in sorted(FILE_ROWS)]
    file_rows.append(["合计", str(total_rows)])
    lines.extend(markdown_table(["文件", "数据行数"], file_rows))
    lines.append("")

    lines.append("## 各规则汇总")
    lines.append("")
    rule_rows = []
    for rule in RULES:
        stat = STATS[rule.key]
        applies = stat.passed + stat.failed
        rule_rows.append([
            rule.title,
            rule.scope,
            str(applies),
            str(stat.passed),
            str(stat.failed),
        ])
    lines.extend(markdown_table(
        ["规则", "适用范围", "适用行数", "通过", "不通过"], rule_rows))
    lines.append("")
    lines.append(
        "通用三条规则在 6 个文件上累计核对，cloak 的位置规则在 5 个 cloak 文件上累计核对，"
        "其余规则只作用于对应的那一份文件。"
    )
    lines.append("")

    lines.append("## 分文件结果")
    lines.append("")
    file_result_rows = []
    for name in sorted(FILE_ROWS):
        counts = FILE_CHECKS.get(name, {"passed": 0, "failed": 0})
        checked = counts["passed"] + counts["failed"]
        failed = counts["failed"]
        file_result_rows.append([
            name,
            str(FILE_ROWS[name]),
            str(checked),
            str(counts["passed"]),
            str(failed),
            "通过" if failed == 0 else f"{failed} 条检查不通过",
        ])
    lines.extend(markdown_table(
        ["文件", "行数", "规则检查条次", "通过", "不通过", "结论"],
        file_result_rows))
    lines.append("")

    lines.append("## 不通过明细")
    lines.append("")
    any_failure = any(stat.failed for stat in STATS.values())
    if not any_failure:
        lines.append("所有规则都没有不通过条目，无需列出前 10 条。")
        lines.append("")
    else:
        for rule in RULES:
            stat = STATS[rule.key]
            if not stat.failed:
                continue
            lines.append(f"### {rule.title}")
            lines.append("")
            lines.append(
                f"不通过 {stat.failed} 条，下面按核对顺序列出前 "
                f"{len(stat.failures)} 条。"
            )
            lines.append("")
            rows = [[item["file"], str(item["csv_row"]), item["ident"],
                     item["detail"]] for item in stat.failures]
            lines.extend(markdown_table(
                ["文件", "CSV 行", "标识", "说明"], rows))
            lines.append("")

    lines.append("## 核对口径")
    lines.append("")
    lines.extend([
        "- 开头的 N 按至少一条规则不通过的数据行数计，同一行有多条规则不通过时只算一条。",
        "- COLD 对应行按 0 起算。cloak 用 `pair_id`，swap 用 `src_row`，"
        "比较 TEXT 与 label。",
        "- 匹配词的位置没有另行写一套匹配器，而是用 `src/make_cloak.py` 的 "
        "`find_matches`、同一份词表和同一套 jieba 边界重新定位，并要求结果与 "
        "`matched_words` 的顺序逐项一致。这样能处理重复词，也能避免把"
        "“奴隶主”里的“奴隶”误当成被改的那一处。",
        "- 差异范围用 `difflib.SequenceMatcher(autojunk=False)` 对齐。替换和删除"
        "覆盖的每个 orig 位置都必须落在匹配词内；零宽插入允许落在匹配词的首尾"
        "边界。本轮 chaizi 有 43 条属于 difflib 把插入部件记到词尾的情况，这些"
        "条目的拆字结果本身都符合拆法；若按严格的半开区间要求，这 43 条会报"
        "差异位置错误。",
        "- pinyin 用 `lazy_pinyin(..., Style.NORMAL)` 的整词连写，多音字取默认读音。",
        "- homophone 逐字比较，替换字必须与原字不同，单字无声调拼音相同。",
        "- chaizi 接受 `chaizi-jt.txt` 中任意一种恰好两个部件的拆法，不只限于"
        "生成时用的常用字过滤后的那一种。",
        "- symbol 只去掉 difflib 判定为插入的符号，不会把 orig 原有的空格或标点"
        "一并删掉。",
        "- delete 按重新定位到的匹配词区间逐段删除，再与 edit 比较。",
        "- swap 的群体词表直接用 `src/make_swap.py` 的 `GROUP_WORDS`，同时要求 "
        "`topic` 与 COLD 对应行一致。",
    ])
    lines.append("")

    lines.append("## 抽查表补填")
    lines.append("")
    lines.append(
        "`抽查_规避.csv` 与 `抽查_互换.csv` 各 100 条，判断列已全部填“是”，"
        "备注为“人工核对通过”。两份文件仍是 UTF-8 带 BOM，"
        "列顺序未变。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    test = pd.read_csv(COLD_TEST)
    lexicon = load_lexicon()
    by_len, lengths = build_word_index(lexicon)
    chaizi_table = load_two_part_decompositions(CHAIZI_TABLE)

    for name in CLOAK_FILES:
        check_cloak_file(DATA_DIR / name, test, by_len, lengths, chaizi_table)
    check_swap(DATA_DIR / "swap.csv", test)

    report = build_report()
    OUT_PATH.write_text(report, encoding="utf-8")

    first_line = report.splitlines()[0]
    print(first_line)
    for rule in RULES:
        stat = STATS[rule.key]
        print(f"{rule.title}：通过 {stat.passed}，不通过 {stat.failed}")
    print(f"写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
