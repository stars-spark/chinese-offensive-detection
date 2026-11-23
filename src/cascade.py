"""级联规则的纯计算：置信度、阈值选择、交出与准确率。

规则：TF-IDF 的置信度 c = max(p, 1-p)。验证集上把
“交出比例”从 0% 到 100% 按 1% 一档扫，取验证准确率最高的一档，并列时
取比例最小的；再把这一档换算成验证集置信度的分位数 τ，原样用到测试集。
0% 的 τ 取 0（低于任何置信度），100% 的 τ 取 +inf（高于任何置信度），
保证两端严格等于 TF-IDF / DistilBERT。中间档位用 np.quantile 的线性分位数。
"""

from __future__ import annotations

import numpy as np

from significance import scope_pair_ids

FRACTIONS = tuple(i / 100.0 for i in range(101))


def confidence_from_prob_pos(prob_pos) -> np.ndarray:
    """二分类置信度 max(p, 1-p)，输入是正面概率。"""
    p = np.asarray(prob_pos, dtype=np.float64)
    return np.maximum(p, 1.0 - p)


def handover_mask(conf, tau: float) -> np.ndarray:
    """c < τ 的样本交给 DistilBERT。"""
    return np.asarray(conf, dtype=np.float64) < float(tau)


def handover_fraction(conf, tau: float) -> float:
    """实际交出比例；空输入返回 0。"""
    conf = np.asarray(conf, dtype=np.float64)
    return float(handover_mask(conf, tau).mean()) if len(conf) else 0.0


def cascade_predict(tfidf_pred, conf, bert_pred, tau: float) -> np.ndarray:
    """c < τ 用 DistilBERT 的预测，其余用 TF-IDF 的预测。"""
    tfidf_pred = np.asarray(tfidf_pred)
    bert_pred = np.asarray(bert_pred)
    if tfidf_pred.shape != bert_pred.shape:
        raise ValueError("TF-IDF 与 DistilBERT 预测条数不一致")
    return np.where(handover_mask(conf, tau), bert_pred, tfidf_pred)


def cascade_accuracy(tfidf_pred, conf, bert_pred, labels, tau: float) -> float:
    """级联在给定 τ 下的准确率。"""
    pred = cascade_predict(tfidf_pred, conf, bert_pred, tau)
    labels = np.asarray(labels)
    return float((pred == labels).mean()) if len(labels) else 0.0


def tau_for_fraction(conf, fraction: float) -> float:
    """把交出比例换算成置信度阈值；0% 与 100% 用边界值。"""
    conf = np.asarray(conf, dtype=np.float64).ravel()
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction 必须在 [0, 1] 内")
    if len(conf) == 0:
        return 0.0
    if fraction <= 0.0:
        return 0.0
    if fraction >= 1.0:
        return float("inf")
    return float(np.quantile(conf, fraction))


def select_tau(val_tfidf_pred, val_conf, val_bert_pred, val_labels) -> dict:
    """在验证集上按 1% 一档选 τ；准确率并列时取交出比例最小的档。

    n 小于 100 时，多个名义档位会落到同一个实际交出集合（例如 10 条里
    23% 与 30% 都只交 3 条）。这里按“实际交出比例”去重：准确率相同先取
    实际比例最小的；实际比例也相同，取名义档位离它最近的那个，这样返回
    的 τ 就是该实际比例对应的分位数。
    """
    best, best_key = None, None
    for fraction in FRACTIONS:
        tau = tau_for_fraction(val_conf, fraction)
        actual = handover_fraction(val_conf, tau)
        accuracy = cascade_accuracy(
            val_tfidf_pred, val_conf, val_bert_pred, val_labels, tau
        )
        # 准确率越高越好；并列时实际比例越小越好；再并列时名义档位越贴近实际越好
        key = (accuracy, -actual, -abs(fraction - actual))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "fraction": actual,
                "nominal": fraction,
                "tau": tau,
                "val_accuracy": accuracy,
                "val_handover": actual,
            }
    if best is None:
        raise ValueError("验证集为空，无法选择阈值")
    return best


def test_curve(tfidf_pred, conf, bert_pred, labels) -> dict:
    """测试集描述曲线：每个名义档位的实际交出比例与准确率。"""
    fractions, handovers, accuracies = [], [], []
    for fraction in FRACTIONS:
        tau = tau_for_fraction(conf, fraction)
        fractions.append(fraction)
        handovers.append(handover_fraction(conf, tau))
        accuracies.append(cascade_accuracy(
            tfidf_pred, conf, bert_pred, labels, tau
        ))
    return {
        "fraction": np.asarray(fractions),
        "handover": np.asarray(handovers),
        "accuracy": np.asarray(accuracies),
    }


def visible_pair_ids(pairs: dict, family: str) -> list[int]:
    """稳健性主口径的配对：只留可见配对，剔除 excluded_no_diff。"""
    return scope_pair_ids(pairs, family, "visible")
