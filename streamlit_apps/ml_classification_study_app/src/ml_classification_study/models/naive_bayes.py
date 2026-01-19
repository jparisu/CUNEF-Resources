from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.naive_bayes import GaussianNB

from ml_classification_study.types import ArgSpec, ModelCapabilities, TrainStepResult
from .base import BaseClassifier


class GaussianNaiveBayesClassifier(BaseClassifier):
    id = "gaussian_nb"
    name = "Gaussian Naive Bayes"
    description = "Probabilistic classifier assuming independent Gaussian features per class."
    capabilities = ModelCapabilities(supports_step_training=True, supports_predict_proba=True, supports_params_view=True)

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(key="var_smoothing", label="var_smoothing", kind="float", default=1e-9, min_value=1e-12, max_value=1e-3, step=1e-10, help="Portion of the largest variance added to variances for stability."),
            ArgSpec(
                key="step_fraction",
                label="Step fraction",
                kind="float",
                default=0.2,
                min_value=0.05,
                max_value=1.0,
                step=0.05,
                help="In step-by-step training, each step updates on this fraction of the training points.",
            ),
        ]

    def _reset_impl(self) -> None:
        self._model: Optional[GaussianNB] = None
        self._X_full: Optional[np.ndarray] = None
        self._y_full: Optional[np.ndarray] = None
        self._cursor: int = 0
        self._order: Optional[np.ndarray] = None

    def _build_model(self) -> GaussianNB:
        return GaussianNB(var_smoothing=float(self.hyperparams.get("var_smoothing", 1e-9)))

    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X_full = np.asarray(X)
        self._y_full = np.asarray(y)
        self._order = np.arange(len(y))
        self._cursor = len(y)

        self._model = self._build_model()
        self._model.fit(self._X_full, self._y_full)

    def train_one_step(self, X: np.ndarray, y: np.ndarray) -> TrainStepResult:
        self._step_index += 1
        X = np.asarray(X)
        y = np.asarray(y)

        if self._X_full is None or self._y_full is None:
            self._X_full = X
            self._y_full = y
            # Deterministic order for stability
            rng = np.random.default_rng(int(self.hyperparams.get("_seed", 0)))
            self._order = rng.permutation(len(y))
            self._cursor = 0

        if self._model is None:
            self._model = self._build_model()

        frac = float(self.hyperparams.get("step_fraction", 0.2))
        frac = min(max(frac, 0.01), 1.0)
        n_total = len(self._y_full)
        batch = max(1, int(round(n_total * frac)))

        start = self._cursor
        end = min(n_total, start + batch)
        idx = self._order[start:end] if self._order is not None else np.arange(start, end)

        if len(idx) == 0:
            # Already consumed all; keep model as-is
            return TrainStepResult(step_index=self._step_index, message=f"Step {self._step_index}: no remaining data (already trained on all points)")

        Xb = self._X_full[idx]
        yb = self._y_full[idx]

        # partial_fit needs known classes at first call
        if start == 0:
            self._model.partial_fit(Xb, yb, classes=np.array([0, 1]))
        else:
            self._model.partial_fit(Xb, yb)

        self._cursor = end
        self._is_trained = True

        return TrainStepResult(
            step_index=self._step_index,
            message=f"Step {self._step_index}: updated on {end}/{n_total} points",
            payload={"seen": end, "total": n_total},
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        return self._model.predict_proba(X)

    def trained_parameters(self) -> Dict[str, Any]:
        if self._model is None or not hasattr(self._model, "theta_"):
            return {"hyperparams": dict(self.hyperparams)}

        return {
            "hyperparams": dict(self.hyperparams),
            "class_prior": self._model.class_prior_.tolist(),
            "means_theta": self._model.theta_.tolist(),
            "variances": self._model.var_.tolist(),
            "n_features": int(getattr(self._model, "n_features_in_", 2)),
        }
