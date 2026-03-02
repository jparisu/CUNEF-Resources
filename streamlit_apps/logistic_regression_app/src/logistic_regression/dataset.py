from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd


def _as_float_list(x, n: int) -> list[float]:
    if isinstance(x, (list, tuple, np.ndarray)):
        x = list(x)
    else:
        x = [float(x)]
    if len(x) < n:
        x = x + [float(x[-1])] * (n - len(x))
    return [float(v) for v in x[:n]]


def sanitize_dataset_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sanitized copy of a dataset config dict.

    Expected keys:
      - n_samples: int
      - balance: float in [0,1] (proportion of class 1)
      - n_features: int (1 or 2)
      - means0, stds0: list[float] of length n_features
      - means1, stds1: list[float] of length n_features
      - train_split: float in [0, 0.9]
    """
    out = dict(cfg)
    n_samples = int(out.get("n_samples", 200))
    n_samples = max(1, n_samples)
    out["n_samples"] = n_samples

    n_features = int(out.get("n_features", 2))
    n_features = 1 if n_features <= 1 else 2
    out["n_features"] = n_features

    balance = float(out.get("balance", 0.5))
    balance = min(1.0, max(0.0, balance))
    out["balance"] = balance

    train_split = float(out.get("train_split", 0.7))
    train_split = min(0.9, max(0.0, train_split))
    out["train_split"] = train_split

    means0 = _as_float_list(out.get("means0", [0.0] * n_features), n_features)
    stds0 = _as_float_list(out.get("stds0", [1.0] * n_features), n_features)
    means1 = _as_float_list(out.get("means1", [2.0] * n_features), n_features)
    stds1 = _as_float_list(out.get("stds1", [1.0] * n_features), n_features)

    # Prevent non-positive stds
    stds0 = [max(1e-6, float(s)) for s in stds0]
    stds1 = [max(1e-6, float(s)) for s in stds1]

    out["means0"] = means0
    out["stds0"] = stds0
    out["means1"] = means1
    out["stds1"] = stds1
    return out


def generate_gaussian_dataset(
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic binary classification dataset using class-conditional Gaussians.

    Returns:
      X: (n_samples, n_features)
      y: (n_samples,)
      train_idx: indices into X/y
      test_idx: indices into X/y
    """
    cfg = sanitize_dataset_config(cfg)
    rng = np.random.default_rng(int(seed))

    n = int(cfg["n_samples"])
    d = int(cfg["n_features"])
    balance = float(cfg["balance"])
    n1 = int(np.round(n * balance))
    n1 = min(n, max(0, n1))
    n0 = n - n1

    means0 = np.asarray(cfg["means0"], dtype=float).reshape(1, d)
    stds0 = np.asarray(cfg["stds0"], dtype=float).reshape(1, d)
    means1 = np.asarray(cfg["means1"], dtype=float).reshape(1, d)
    stds1 = np.asarray(cfg["stds1"], dtype=float).reshape(1, d)

    X0 = rng.normal(loc=means0, scale=stds0, size=(n0, d)) if n0 > 0 else np.empty((0, d), dtype=float)
    y0 = np.zeros((n0,), dtype=int)
    X1 = rng.normal(loc=means1, scale=stds1, size=(n1, d)) if n1 > 0 else np.empty((0, d), dtype=float)
    y1 = np.ones((n1,), dtype=int)

    X = np.vstack([X0, X1]) if n > 0 else np.empty((0, d), dtype=float)
    y = np.concatenate([y0, y1]) if n > 0 else np.empty((0,), dtype=int)

    perm = rng.permutation(len(y)) if len(y) > 0 else np.array([], dtype=int)
    X = X[perm]
    y = y[perm]

    p_train = float(cfg["train_split"])
    n_train = int(np.round(len(y) * p_train))
    n_train = min(len(y), max(0, n_train))

    train_idx = np.arange(0, n_train, dtype=int)
    test_idx = np.arange(n_train, len(y), dtype=int)
    return X, y, train_idx, test_idx


def to_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    d = int(X.shape[1]) if X.ndim == 2 and X.size else 1
    data: Dict[str, Any] = {}
    data["class"] = y.astype(int)
    data["class_str"] = y.astype(int).astype(str)

    split = np.array(["train"] * len(y), dtype=object)
    split[test_idx] = "test"
    data["split"] = split

    if d >= 1:
        data["x1"] = X[:, 0]
    if d >= 2:
        data["x2"] = X[:, 1]

    return pd.DataFrame(data)
