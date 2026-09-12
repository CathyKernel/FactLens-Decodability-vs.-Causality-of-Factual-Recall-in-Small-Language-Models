"""Classification metrics implemented from scratch (numpy only).

Everything here is deliberately hand-rolled rather than imported from
sklearn: the probe evaluation is part of the scientific contribution, so
its numerics should be transparent and dependency-free.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean 0/1 accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer confusion matrix of shape ``(n_classes, n_classes)``.

    Rows are true labels, columns are predictions.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over the full label space.

    Classes absent from ``y_true`` count as F1 = 0, matching sklearn's
    ``f1_score(..., average="macro", labels=range(n_classes))`` — a strict
    choice that keeps relations with many rare classes comparable.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)
    return float(np.mean(f1s))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUC with exact tie handling (Mann-Whitney U statistic).

    Returns ``None`` when the slice is degenerate (single class present).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")  # stable: ties keep index order
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    rank = 1
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # 1-based ranks rank..rank+(j-i); ties receive their average.
        ranks[order[i : j + 1]] = rank + (j - i) / 2.0
        rank += j - i + 1
        i = j + 1

    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def one_vs_rest_auc(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float | None:
    """Macro-averaged one-vs-rest AUC from a softmax probability matrix.

    Args:
        probs: ``(n, n_classes)`` probabilities (e.g. from ``predict_proba``).
        y_true: ``(n,)`` integer labels.

    Returns:
        Macro mean over classes with both classes present in ``y_true``,
        or ``None`` if no class yields a valid AUC.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    aucs = []
    for c in range(n_classes):
        labels = (y_true == c).astype(np.int64)
        auc = _binary_auc(probs[:, c], labels)
        if auc is not None:
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else None


def selectivity(task_score: float, control_score: float) -> float:
    """Hewitt & Liang (2019) selectivity: task minus control performance."""
    return float(task_score - control_score)
