from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy-version-safe trapezoidal integration."""
    trapezoid_fn = getattr(np, "trapezoid", None)
    if trapezoid_fn is not None:
        return float(trapezoid_fn(y, x))

    trapz_fn = getattr(np, "trapz", None)
    if trapz_fn is not None:
        return float(trapz_fn(y, x))

    # Last-resort fallback if neither helper exists in runtime.
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if y_true.size == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    cc = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = cc["tp"], cc["tn"], cc["fp"], cc["fn"]
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else 2.0 * prec * rec / (prec + rec)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def roc_curve_auc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """Compute ROC curve and AUC.

    Returns dict with keys:
      - fpr: np.ndarray
      - tpr: np.ndarray
      - thresholds: np.ndarray
      - auc: float (nan if undefined)
      - defined: bool (True if both classes present)
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true.size == 0:
        return {"fpr": np.array([]), "tpr": np.array([]), "thresholds": np.array([]), "auc": float("nan"), "defined": False}

    classes = np.unique(y_true)
    if classes.size < 2:
        # ROC undefined when only one class is present
        return {"fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0]), "thresholds": np.array([np.inf, -np.inf]), "auc": float("nan"), "defined": False}

    # Sort by score descending
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    P = int(np.sum(y_true_sorted == 1))
    N = int(np.sum(y_true_sorted == 0))

    tps = 0
    fps = 0

    fpr = [0.0]
    tpr = [0.0]
    thresholds = [np.inf]

    # Iterate distinct thresholds
    uniq_scores, idx_start = np.unique(y_score_sorted, return_index=True)
    # uniq_scores are ascending because np.unique sorts; we want descending thresholds
    uniq_scores = uniq_scores[::-1]

    for thr in uniq_scores:
        # Add all points with score >= thr that haven't been counted yet.
        # We can compute via mask each time; for clarity vs speed.
        pred_pos = y_score_sorted >= thr
        y_pred_pos = y_true_sorted[pred_pos]
        tps = int(np.sum(y_pred_pos == 1))
        fps = int(np.sum(y_pred_pos == 0))

        fpr.append(fps / max(1, N))
        tpr.append(tps / max(1, P))
        thresholds.append(float(thr))

    fpr.append(1.0)
    tpr.append(1.0)
    thresholds.append(-np.inf)

    fpr_arr = np.asarray(fpr, dtype=float)
    tpr_arr = np.asarray(tpr, dtype=float)

    # AUC via trapezoidal rule on FPR axis
    auc = _trapezoid_integral(tpr_arr, fpr_arr)
    return {"fpr": fpr_arr, "tpr": tpr_arr, "thresholds": np.asarray(thresholds, dtype=float), "auc": auc, "defined": True}
