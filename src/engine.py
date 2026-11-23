"""神经网络训练引擎（被 train.py 调用）。

把“设种子、建 DataLoader、训练一轮、评估、按验证集选最佳轮”这些
与具体模型无关的部分集中在这里。这样 TextCNN / BiLSTM / Transformer
共用完全相同的训练流程，对比才公平。

关于选轮次：每轮在验证集上算准确率，保存验证准确率最高的那一轮。
不用最后一轮，是因为训练久了会过拟合，最后一轮未必最好；也不能用
测试集选轮，否则测试集就不再是“没见过的数据”。
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    """固定所有随机源，让同一种子的结果尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """可训练参数量（只数 requires_grad 的，Embedding 的 padding 行也算在内）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """按验证准确率的早停计数器。

    ``patience=None`` 表示不早停（阶段 1 行为）。判定用严格大于：
    验证准确率打平不算提升，与“选最早的最佳轮次”保持一致。
    连续 ``patience`` 轮没有提升就停；被停掉的那一轮仍算训练过，
    所以 ``stopped_epoch`` 是实际跑完的最后一轮，而不是最佳轮。
    """

    def __init__(self, patience: int | None = None):
        if patience is not None and patience < 1:
            raise ValueError("patience 必须 >= 1 或为 None")
        self.patience = patience
        self.best_val = -1.0
        self.best_epoch = 0
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, val_acc: float, epoch: int) -> bool:
        """喂入一轮验证准确率；返回这一轮是否刷新了最佳值。"""
        if val_acc > self.best_val:
            self.best_val = val_acc
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        if self.patience is not None and self.bad_epochs >= self.patience:
            self.should_stop = True
        return False


def make_loader(ids: np.ndarray, mask: np.ndarray, labels: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    """把 numpy 数组包成 DataLoader。ids/mask 是整数，标签是 long。"""
    dataset = TensorDataset(
        # torch.tensor 会复制一份，避免 pandas/numpy 的只读视图触发 PyTorch 警告
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             return_probs: bool = False):
    """在给定数据上算平均损失、准确率和预测结果（不更新参数）。

    ``return_probs=True`` 时多返回一个 ``(n, num_classes)`` 的 softmax
    概率矩阵，供 ``--save_preds`` 落盘；默认只返回四元组，保持阶段 1
    的调用方式不变。
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        logits = model(ids, mask)
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y.cpu())
        if return_probs:
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    if return_probs:
        probs = torch.cat(all_probs).numpy()
        return total_loss / max(n, 1), accuracy_score(labels, preds), preds, labels, probs
    return total_loss / max(n, 1), accuracy_score(labels, preds), preds, labels


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip: float = 1.0):
    """训练一轮，返回 (平均损失, 耗时秒, 本 epoch 显存峰值 MB)。"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # 只统计本轮，不含之前的数据搬运
    start = time.perf_counter()
    total_loss, n = 0.0, 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, mask)
        loss = criterion(logits, y)
        loss.backward()
        # 梯度裁剪：Transformer 偶尔会有很大的梯度，裁剪后训练更稳
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()  # 按 step 更新学习率（预热 + 线性衰减）
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    if on_cuda:
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_mb = 0.0
    return total_loss / max(n, 1), time.perf_counter() - start, peak_mb


def fit(model, train_loader, val_loader, *, epochs: int, lr: float,
        weight_decay: float = 0.01, warmup_frac: float = 0.05,
        grad_clip: float = 1.0, device, patience: int | None = None, log=print):
    """完整训练流程：AdamW + 预热/线性衰减 + 每轮验证 + 选最佳轮。

    早停只决定什么时候停，学习率调度仍按 ``epochs`` 的总步数预先算好，
    所以 ``patience=3`` 与不早停的前几轮学习率完全一致。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        # 前 warmup_steps 步从 0 线性升到 1，之后线性降到 0
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_state, best_val, best_epoch = None, -1.0, 0
    stopper = EarlyStopping(patience)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, sec, peak_mb = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "train_time_sec": sec,
            "gpu_peak_memory_mb": peak_mb if device.type == "cuda" else None,
        })
        log(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"time={sec:.1f}s  gpu_peak={peak_mb:.0f}MB")
        # 严格大于才更新：验证准确率打平时保留更早的轮次，偏向简单模型
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        stopper.step(val_acc, epoch)
        if stopper.should_stop:
            log(f"早停：连续 {patience} 轮验证准确率未提升，"
                f"最佳轮次={best_epoch}，实际训练到第 {epoch} 轮")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_val


def classification_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    """测试集指标：准确率、宏平均 F1、各类 F1 与正类精确率/召回率。

    正类固定是二分类的 ``class_names[1]``；中文实验的“攻击性类”就是它。
    原有字段与阶段 1 完全一致，只追加新字段。
    """
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class_f1": {name: float(report[name]["f1-score"]) for name in class_names},
        "per_class_precision": {
            name: float(report[name]["precision"]) for name in class_names
        },
        "per_class_recall": {
            name: float(report[name]["recall"]) for name in class_names
        },
    }
    if len(class_names) == 2:
        pos = class_names[1]
        out["pos_precision"] = float(report[pos]["precision"])
        out["pos_recall"] = float(report[pos]["recall"])
    return out
