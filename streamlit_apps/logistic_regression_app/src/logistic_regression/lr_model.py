from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Any

import numpy as np


RegType = Literal["None", "L1", "L2"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def cross_entropy_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


@dataclass
class LogisticRegressionScratch:
    n_features: int
    w: np.ndarray
    b: float

    @classmethod
    def init_zeros(cls, n_features: int) -> "LogisticRegressionScratch":
        return cls(n_features=int(n_features), w=np.zeros((int(n_features),), dtype=float), b=0.0)

    def set_params(self, w: np.ndarray, b: float) -> None:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.shape[0] != self.n_features:
            # Resize safely
            nw = np.zeros((self.n_features,), dtype=float)
            k = min(self.n_features, w.shape[0])
            nw[:k] = w[:k]
            w = nw
        self.w = w
        self.b = float(b)

    def logits(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= float(threshold)).astype(int)

    def loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reg_type: RegType = "None",
        reg_strength: float = 0.0,
    ) -> float:
        p = self.predict_proba(X)
        base = cross_entropy_loss(y, p)
        lam = float(reg_strength)
        if reg_type == "L2":
            # (lam / (2m)) * ||w||^2 added to mean loss
            m = max(1, len(y))
            base += (lam / (2.0 * m)) * float(np.sum(self.w ** 2))
        elif reg_type == "L1":
            m = max(1, len(y))
            base += (lam / m) * float(np.sum(np.abs(self.w)))
        return float(base)

    def gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reg_type: RegType = "None",
        reg_strength: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        m = max(1, X.shape[0])
        p = self.predict_proba(X)
        err = (p - y).reshape(-1, 1)  # (m,1)
        dw = (X.T @ err).reshape(-1) / m  # (d,)
        db = float(np.sum(p - y) / m)

        lam = float(reg_strength)
        if reg_type == "L2":
            dw = dw + (lam / m) * self.w
        elif reg_type == "L1":
            dw = dw + (lam / m) * np.sign(self.w)

        return dw, db

    def step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        reg_type: RegType = "None",
        reg_strength: float = 0.0,
    ) -> Dict[str, Any]:
        """One gradient descent step on the provided batch."""
        dw, db = self.gradients(X, y, reg_type=reg_type, reg_strength=reg_strength)
        loss_before = self.loss(X, y, reg_type=reg_type, reg_strength=reg_strength)

        self.w = self.w - float(lr) * dw
        self.b = float(self.b - float(lr) * db)

        loss_after = self.loss(X, y, reg_type=reg_type, reg_strength=reg_strength)
        return {
            "dw": dw,
            "db": db,
            "loss_before": float(loss_before),
            "loss_after": float(loss_after),
        }
