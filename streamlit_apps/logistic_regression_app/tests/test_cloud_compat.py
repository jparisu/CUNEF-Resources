from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from logistic_regression.metrics import roc_curve_auc  # noqa: E402


def test_roc_auc_computes_with_current_numpy() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=float)
    roc = roc_curve_auc(y_true, y_score)

    assert roc["defined"] is True
    assert np.isfinite(roc["auc"])
    assert 0.0 <= float(roc["auc"]) <= 1.0


def test_roc_auc_undefined_for_single_class() -> None:
    y_true = np.array([1, 1, 1], dtype=int)
    y_score = np.array([0.2, 0.9, 0.7], dtype=float)
    roc = roc_curve_auc(y_true, y_score)

    assert roc["defined"] is False
    assert np.isnan(roc["auc"])
