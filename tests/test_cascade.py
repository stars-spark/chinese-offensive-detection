"""阶段 4 级联：阈值选择、交出规则与稳健性口径。"""

import numpy as np
import pytest

from cascade import (
    cascade_predict,
    handover_fraction,
    select_tau,
    tau_for_fraction,
    visible_pair_ids,
)
from finetune_bert import chunk_spans


def _toy(n=10):
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    tfidf_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.int64)
    bert_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=np.int64)
    conf = np.linspace(0.51, 0.90, n)
    return labels, tfidf_pred, bert_pred, conf


def test_endpoints_equal_tfidf_and_bert():
    labels, tfidf_pred, bert_pred, conf = _toy()
    at_zero = cascade_predict(tfidf_pred, conf, bert_pred, tau=0.0)
    at_full = cascade_predict(tfidf_pred, conf, bert_pred, tau=float("inf"))
    np.testing.assert_array_equal(at_zero, tfidf_pred)
    np.testing.assert_array_equal(at_full, bert_pred)


def test_select_tau_breaks_tie_toward_smaller_fraction():
    # 10 条、置信度 0.51..0.60。TF-IDF 对最低置信度的 3 条判错，
    # DistilBERT 把这 3 条改对；第 4 条 DistilBERT 也对，所以交 30% 与
    # 40% 都是 100%，第 5 条 DistilBERT 判错，再往后降到 90%。
    labels = np.ones(10, dtype=np.int64)
    tfidf_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    bert_pred = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    conf = np.linspace(0.51, 0.60, 10)

    chosen = select_tau(tfidf_pred, conf, bert_pred, labels)

    assert chosen["fraction"] == pytest.approx(0.30)
    assert chosen["tau"] == pytest.approx(np.quantile(conf, 0.30))
    assert chosen["val_accuracy"] == pytest.approx(1.0)
    assert chosen["val_handover"] == pytest.approx(0.30)


def test_tau_not_fraction_applied_to_different_test_distribution():
    # 验证集选出的 τ 是 0.30 分位；测试集置信度整体更高，实际只交出 40%，
    # 而不是把“30%”这个比例直接搬到测试集。
    conf_val = np.linspace(0.51, 0.60, 10)
    tau = tau_for_fraction(conf_val, 0.30)
    conf_test = np.array([0.52, 0.53, 0.55, 0.56, 0.58])

    assert tau == pytest.approx(np.quantile(conf_val, 0.30))
    assert handover_fraction(conf_test, tau) == pytest.approx(0.40)


def test_visible_pairs_drop_no_diff_and_keep_visible_only():
    pairs = {
        "contrast": {
            "visible": [0, 1, 2, 3, 4],
            "excluded_no_diff": [2, 4],
            "n_total": 6,
        },
        "cad": {
            "visible": [0, 1, 2, 3],
            "excluded_no_diff": [3],
            "n_total": 8,
        },
    }
    assert visible_pair_ids(pairs, "contrast") == [0, 1, 3]
    assert visible_pair_ids(pairs, "cad") == [0, 1, 2]
    assert len(visible_pair_ids(pairs, "contrast")) == 3


def test_chunk_spans_known_segment_counts():
    # 510 子词一段、相邻段重叠 128，步长 382；空序列仍保留一个空段
    assert len(chunk_spans(0, 510, 128)) == 1
    assert len(chunk_spans(510, 510, 128)) == 1
    assert len(chunk_spans(511, 510, 128)) == 2
    assert len(chunk_spans(1000, 510, 128)) == 3
    assert chunk_spans(1000, 510, 128) == [(0, 510), (382, 892), (764, 1000)]
