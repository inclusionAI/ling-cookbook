"""Binary guardrail metrics. Positive class = unsafe. Every table prints the trivial baselines."""

from __future__ import annotations

import math
import random


def confusion(labels, predictions):
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    return tp, fp, tn, fn


def binary_metrics(labels, predictions) -> dict:
    tp, fp, tn, fn = confusion(labels, predictions)
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": n,
        "acc": (tp + tn) / n if n else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_acc": (recall + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
        "precision": precision,
        "recall": recall,
        "miss_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_alarm": fp / (tn + fp) if tn + fp else 0.0,
    }


def trivial(labels) -> dict:
    n = len(labels)
    p = sum(labels) / n
    return {"always_unsafe_acc": p, "always_unsafe_f1": 2 * p / (1 + p), "always_safe_acc": 1 - p}


def group_bootstrap(labels, predictions, groups, stat="acc", rounds=1000, seed=7):
    """95% interval that resamples whole groups, because records share requests."""
    by_group: dict = {}
    for y, p, g in zip(labels, predictions, groups):
        by_group.setdefault(g, []).append((y, p))
    keys = list(by_group)
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        ys, ps = [], []
        for key in rng.choices(keys, k=len(keys)):
            for y, p in by_group[key]:
                ys.append(y)
                ps.append(p)
        values.append(binary_metrics(ys, ps)[stat])
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def ece(labels, probabilities, bins=10):
    """Expected calibration error of P(unsafe)."""
    total = len(labels)
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        items = [(y, p) for y, p in zip(labels, probabilities) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if items:
            confidence = sum(p for _, p in items) / len(items)
            frequency = sum(y for y, _ in items) / len(items)
            error += len(items) / total * abs(confidence - frequency)
    return error
