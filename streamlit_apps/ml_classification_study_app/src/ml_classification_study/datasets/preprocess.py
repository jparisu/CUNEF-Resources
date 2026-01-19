from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class PreprocessConfig:
    scaling: str  # "none" | "standard" | "minmax"
    outlier_std: float  # 0 disables
    test_size: float
    stratify: bool = True


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=strat,
    )
    return X_train, X_test, y_train, y_test


def remove_outliers_by_std(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Remove outliers using mean/std estimated from the training set.

    A point is removed if any feature is outside mean ± n_std * std.
    """

    if n_std <= 0:
        return X_train, y_train, X_test, y_test, {"removed_train": 0, "removed_test": 0}

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-12

    lo = mu - n_std * sigma
    hi = mu + n_std * sigma

    train_keep = np.all((X_train >= lo) & (X_train <= hi), axis=1)
    test_keep = np.all((X_test >= lo) & (X_test <= hi), axis=1)

    meta = {
        "outlier_mu": mu,
        "outlier_sigma": sigma,
        "outlier_lo": lo,
        "outlier_hi": hi,
        "removed_train": int((~train_keep).sum()),
        "removed_test": int((~test_keep).sum()),
    }

    return (
        X_train[train_keep],
        y_train[train_keep],
        X_test[test_keep],
        y_test[test_keep],
        meta,
    )


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    scaling: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    scaling = (scaling or "none").lower()

    if scaling == "none":
        return X_train, X_test, {"scaler": None}

    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    meta = {
        "scaler": scaler,
        "scaler_type": scaling,
    }
    return X_train_s, X_test_s, meta
