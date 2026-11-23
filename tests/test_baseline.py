"""TF-IDF + 逻辑回归的测试：人造可分数据上应达到 100% 训练准确率。"""

import numpy as np

from baseline_tfidf import fit_and_select


def test_tfidf_lr_separable_data():
    good = ["good movie excellent"] * 10
    bad = ["bad movie awful"] * 10
    texts = good + bad
    labels = np.array([1] * 10 + [0] * 10)

    # 验证集用同一批人造数据，只检查流程和可分的上限
    vectorizer, best, info = fit_and_select(
        texts, labels, texts, labels, seed=0, log=lambda *a, **k: None
    )
    assert best["C"] in (0.25, 1.0, 4.0)
    assert best["val_acc"] == 1.0
    train_pred = best["model"].predict(vectorizer.transform(texts))
    assert (train_pred == labels).mean() == 1.0
    assert info["n_features"] > 0


def test_tfidf_lr_multiclass_one_vs_rest():
    # AG News 是 4 分类；sklearn 1.9 的 liblinear 需要显式 OneVsRest 包装
    texts = (
        ["alpha alpha alpha"] * 6
        + ["beta beta beta"] * 6
        + ["gamma gamma gamma"] * 6
    )
    labels = np.array([0] * 6 + [1] * 6 + [2] * 6)
    vectorizer, best, _ = fit_and_select(
        texts, labels, texts, labels, seed=0, log=lambda *a, **k: None
    )
    pred = best["model"].predict(vectorizer.transform(texts))
    assert (pred == labels).mean() == 1.0
