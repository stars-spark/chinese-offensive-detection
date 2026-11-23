"""三级审核演示：阈值分层与空输入。"""

from demo_moderate import (
    assign_level,
    classify_one,
    format_confidence,
    summarize_results,
    truncate_display,
)


def _constant(pred, conf):
    return lambda text: (pred, conf)


def test_assign_level_boundaries():
    # τ1=0.9、τ2=0.7：等于阈值算采信，低于才交出
    assert assign_level(0.90, None, 0.90, 0.70) == 1
    assert assign_level(0.95, 0.10, 0.90, 0.70) == 1
    assert assign_level(0.89, 0.70, 0.90, 0.70) == 2
    assert assign_level(0.89, 0.69, 0.90, 0.70) == 3


def test_classify_one_skips_second_level_when_confident():
    def bert_should_not_run(text):
        raise AssertionError("第一级已采信，不该跑第二级")

    row = classify_one("今天天气不错", _constant(1, 0.95),
                       bert_should_not_run, tau1=0.9, tau2=0.7)
    assert row["level"] == 1
    assert row["pred"] == 1
    assert row["conf"] == 0.95


def test_classify_one_second_level_and_human_review():
    row = classify_one("测试", _constant(1, 0.50),
                       _constant(0, 0.80), tau1=0.9, tau2=0.7)
    assert (row["level"], row["pred"], row["conf"]) == (2, 0, 0.80)

    row = classify_one("测试", _constant(0, 0.50),
                       _constant(1, 0.60), tau1=0.9, tau2=0.7)
    assert row["level"] == 3
    assert row["pred"] is None
    assert row["conf"] == 0.60


def test_empty_string_does_not_crash():
    row = classify_one("", _constant(0, 0.40),
                       _constant(1, 0.90), tau1=0.9, tau2=0.7)
    assert row["level"] == 2
    assert row["text"] == ""
    assert truncate_display("") == ""
    assert format_confidence(None) == "—"


def test_summary_counts_and_per_1000():
    results = [
        {"level": 1, "elapsed_ms": 10.0},
        {"level": 2, "elapsed_ms": 20.0},
        {"level": 2, "elapsed_ms": 30.0},
        {"level": 3, "elapsed_ms": 40.0},
    ]
    stats = summarize_results(results)
    assert stats["counts"] == {1: 1, 2: 2, 3: 1}
    assert stats["ratios"][2] == 0.5
    assert stats["total_ms"] == 100.0
    assert stats["per_1000_sec"] == 25.0
