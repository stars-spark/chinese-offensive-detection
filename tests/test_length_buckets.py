"""长度分组：边界归属与每组准确率。"""

import numpy as np

from analyze_length_buckets import BUCKET_LABELS, bucket_accuracy, length_bucket


def test_length_bucket_boundaries():
    assert length_bucket(1) == "≤128"
    assert length_bucket(128) == "≤128"
    assert length_bucket(129) == "129～256"
    assert length_bucket(256) == "129～256"
    assert length_bucket(257) == "257～512"
    assert length_bucket(512) == "257～512"
    assert length_bucket(513) == ">512"
    assert length_bucket(5000) == ">512"


def test_bucket_accuracy_known_values():
    labels = np.array([1, 1, 0, 0, 1, 0])
    preds = np.array([1, 0, 0, 1, 1, 1])
    # 前两条 ≤128，第三四两条 129～256，后两条 >512
    bucket_ids = np.array([0, 0, 1, 1, 3, 3])
    acc = bucket_accuracy(labels, preds, bucket_ids)
    assert acc["≤128"] == 0.5
    assert acc["129～256"] == 0.5
    assert acc["257～512"] is None
    assert acc[">512"] == 0.5
    assert set(acc) == set(BUCKET_LABELS)


def test_bucket_accuracy_all_correct_and_empty():
    labels = np.array([0, 1, 1])
    preds = np.array([0, 1, 1])
    acc = bucket_accuracy(labels, preds, np.array([0, 0, 0]))
    assert acc["≤128"] == 1.0
    assert acc["129～256"] is None
