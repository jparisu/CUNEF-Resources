from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml_classification_study.types import ArgSpec, ModelCapabilities
from .base import BaseClassifier


class RandomForestStudyClassifier(BaseClassifier):
    id = "random_forest"
    name = "Random Forest"
    description = "Ensemble of decision trees (scikit-learn RandomForestClassifier)."
    capabilities = ModelCapabilities(supports_step_training=False, supports_predict_proba=True, supports_params_view=True)

    def __init__(self, **hyperparams: Any) -> None:
        super().__init__(**hyperparams)
        self.model: Optional[RandomForestClassifier] = None
        self._reset_impl()

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="n_estimators",
                label="Number of trees",
                kind="int",
                default=200,
                min_value=10,
                max_value=2000,
                step=10,
            ),
            ArgSpec(
                key="max_depth",
                label="Max depth (0 = None)",
                kind="int",
                default=0,
                min_value=0,
                max_value=50,
                step=1,
                help="Use 0 for unlimited depth.",
            ),
            ArgSpec(
                key="min_samples_split",
                label="Min samples split",
                kind="int",
                default=2,
                min_value=2,
                max_value=100,
                step=1,
            ),
            ArgSpec(
                key="random_state",
                label="Random seed",
                kind="int",
                default=0,
                min_value=0,
                max_value=10_000,
                step=1,
            ),
        ]

    def _reset_impl(self) -> None:
        self.model = None

    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        n_estimators = int(self.hyperparams.get("n_estimators", 200))
        max_depth_raw = int(self.hyperparams.get("max_depth", 0))
        max_depth = None if max_depth_raw == 0 else max_depth_raw
        min_samples_split = int(self.hyperparams.get("min_samples_split", 2))
        random_state = int(self.hyperparams.get("random_state", 0))

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            return self.model.predict_proba(X)
        except Exception:
            return None

    def trained_parameters(self) -> Dict[str, Any]:
        base = super().trained_parameters()
        if self.model is None:
            return base

        out: Dict[str, Any] = {
            **base,
            "n_estimators": getattr(self.model, "n_estimators", None),
            "max_depth": getattr(self.model, "max_depth", None),
            "feature_importances": getattr(self.model, "feature_importances_", None).tolist() if getattr(self.model, "feature_importances_", None) is not None else None,
        }
        return out
