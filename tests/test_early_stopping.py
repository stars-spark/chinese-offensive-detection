"""早停：连续 patience 轮不提升就停，且返回验证集最佳轮次。"""

import torch
import torch.nn as nn

import engine


def test_early_stopping_counter_sequence():
    stopper = engine.EarlyStopping(patience=3)
    # 第 2 轮最佳；之后 3、4、5 都没有提升，第 5 轮后应停
    seq = [0.50, 0.60, 0.60, 0.55, 0.58, 0.57, 0.59]
    for epoch, acc in enumerate(seq, start=1):
        stopper.step(acc, epoch)
        if stopper.should_stop:
            break
    assert stopper.should_stop
    assert stopper.best_epoch == 2
    assert stopper.best_val == 0.60
    assert epoch == 5


def test_fit_returns_best_epoch_and_stops(monkeypatch):
    model = nn.Linear(1, 1)
    seq = iter([0.50, 0.60, 0.60, 0.55, 0.58, 0.57, 0.59])

    def fake_train(*args, **kwargs):
        return 0.1, 0.01, 0.0

    def fake_eval(*args, **kwargs):
        return 0.1, next(seq), None, None

    monkeypatch.setattr(engine, "train_one_epoch", fake_train)
    monkeypatch.setattr(engine, "evaluate", fake_eval)

    history, best_epoch, best_val = engine.fit(
        model, [], [], epochs=20, lr=1e-3, device=torch.device("cpu"),
        patience=3, log=lambda *a, **k: None,
    )
    assert len(history) == 5
    assert best_epoch == 2
    assert best_val == 0.60


def test_no_patience_runs_all_epochs(monkeypatch):
    model = nn.Linear(1, 1)
    seq = iter([0.5] * 7)

    monkeypatch.setattr(engine, "train_one_epoch",
                        lambda *a, **k: (0.1, 0.01, 0.0))
    monkeypatch.setattr(engine, "evaluate",
                        lambda *a, **k: (0.1, next(seq), None, None))

    history, best_epoch, _ = engine.fit(
        model, [], [], epochs=7, lr=1e-3, device=torch.device("cpu"),
        patience=None, log=lambda *a, **k: None,
    )
    assert len(history) == 7
    assert best_epoch == 1  # 打平不更新，保留最早的最佳轮
