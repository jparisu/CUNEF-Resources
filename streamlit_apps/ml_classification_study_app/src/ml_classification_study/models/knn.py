from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ml_classification_study.types import ArgSpec, ModelCapabilities, TrainStepResult
from .base import BaseClassifier


class KNNClassifier(BaseClassifier):
    id = "knn"
    name = "K-Nearest Neighbors"
    description = "Non-parametric method that classifies based on majority vote among k nearest points."
    capabilities = ModelCapabilities(supports_step_training=True, supports_predict_proba=True, supports_params_view=True)

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(key="n_neighbors", label="k (neighbors)", kind="int", default=5, min_value=1, max_value=50, step=1),
            ArgSpec(
                key="weights",
                label="weights",
                kind="choice",
                default="uniform",
                choices=[("uniform", "uniform"), ("distance", "distance")],
            ),
            ArgSpec(key="p", label="Minkowski p", kind="int", default=2, min_value=1, max_value=5, step=1, help="p=2 is Euclidean; p=1 is Manhattan."),
            ArgSpec(
                key="step_fraction",
                label="Step fraction",
                kind="float",
                default=0.2,
                min_value=0.05,
                max_value=1.0,
                step=0.05,
                help="In step-by-step training, each step adds this fraction of training points (refits).",
            ),
        ]

    def _reset_impl(self) -> None:
        self._model: Optional[KNeighborsClassifier] = None
        self._X_full: Optional[np.ndarray] = None
        self._y_full: Optional[np.ndarray] = None

    def _build_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            n_neighbors=int(self.hyperparams.get("n_neighbors", 5)),
            weights=self.hyperparams.get("weights", "uniform"),
            p=int(self.hyperparams.get("p", 2)),
        )

    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X_full = np.asarray(X)
        self._y_full = np.asarray(y)
        self._model = self._build_model()
        self._model.fit(self._X_full, self._y_full)

    def train_one_step(self, X: np.ndarray, y: np.ndarray) -> TrainStepResult:
        self._step_index += 1
        if self._X_full is None or self._y_full is None:
            self._X_full = np.asarray(X)
            self._y_full = np.asarray(y)

        frac = float(self.hyperparams.get("step_fraction", 0.2))
        frac = min(max(frac, 0.01), 1.0)
        n_total = len(self._y_full)
        n_use = min(n_total, max(1, int(round(n_total * min(1.0, frac * self._step_index)))))

        self._model = self._build_model()
        self._model.fit(self._X_full[:n_use], self._y_full[:n_use])
        self._is_trained = True

        return TrainStepResult(
            step_index=self._step_index,
            message=f"Step {self._step_index}: fit on {n_use}/{n_total} points",
            payload={"n_used": n_use, "n_total": n_total},
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def trained_parameters(self) -> Dict[str, Any]:
        params = {
            "hyperparams": dict(self.hyperparams),
            "n_stored": int(getattr(self._model, "_fit_X", np.empty((0, 2))).shape[0]) if self._model is not None else 0,
        }
        return params
