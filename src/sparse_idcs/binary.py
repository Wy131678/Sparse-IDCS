"""Binary-channel encoding shared by the sparse solver and synthetic example."""

import numpy as np


SLOWNESS_KARST = 1.0 / 0.06
SLOWNESS_NONKARST = 1.0 / 0.08


def ensure_binary(field):
    return (np.asarray(field) != 0).astype(np.uint8)


def slowness_from_binary(binary):
    return np.where(ensure_binary(binary) > 0, SLOWNESS_KARST, SLOWNESS_NONKARST).astype(float)


def binary_from_slowness(slowness):
    midpoint = 0.5 * (SLOWNESS_KARST + SLOWNESS_NONKARST)
    return (np.asarray(slowness) >= midpoint).astype(np.uint8)


def binary_metrics(pred, true):
    pred = ensure_binary(pred)
    true = ensure_binary(true)
    if pred.shape != true.shape:
        raise ValueError("Prediction and reference must have the same shape")
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "accuracy": (tp + tn) / pred.size,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "field_rmse": float(np.sqrt(np.mean((pred.astype(float) - true.astype(float)) ** 2))),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
