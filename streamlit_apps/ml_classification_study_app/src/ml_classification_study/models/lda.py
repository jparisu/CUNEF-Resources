from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ml_classification_study.types import ArgSpec, ModelCapabilities
from .base import BaseClassifier


class LDAClassifier(BaseClassifier):
    id = "lda"
    name = "LDA"
    description = "Linear Discriminant Analysis (scikit-learn LinearDiscriminantAnalysis)."
    capabilities = ModelCapabilities(supports_step_training=False, supports_predict_proba=True, supports_params_view=True)

    def __init__(self, **hyperparams: Any) -> None:
        super().__init__(**hyperparams)
        self.model: Optional[LinearDiscriminantAnalysis] = None
        self._reset_impl()

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="solver",
                label="Solver",
                kind="choice",
                default="svd",
                choices=[("svd", "svd"), ("lsqr", "lsqr"), ("eigen", "eigen")],
                help="svd is the default and does not support shrinkage.",
            ),
            ArgSpec(
                key="shrinkage",
                label="Shrinkage",
                kind="choice",
                default="none",
                choices=[("None", "none"), ("auto", "auto"), ("0.1", 0.1), ("0.3", 0.3), ("0.5", 0.5), ("0.7", 0.7), ("0.9", 0.9)],
                help="Only used when solver is lsqr or eigen.",
            ),
        ]

    def _reset_impl(self) -> None:
        self.model = None

    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        solver = str(self.hyperparams.get("solver", "svd"))
        shrink = self.hyperparams.get("shrinkage", "none")

        shrinkage: Any
        if solver == "svd":
            shrinkage = None
        else:
            if shrink == "none":
                shrinkage = None
            else:
                shrinkage = shrink  # "auto" or float

        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
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

        out: Dict[str, Any] = {**base}
        for attr in ["priors_", "means_", "coef_", "intercept_", "scalings_", "explained_variance_ratio_"]:
            val = getattr(self.model, attr, None)
            if val is None:
                continue
            out[attr.rstrip("_")] = np.asarray(val).tolist()
        out["classes"] = getattr(self.model, "classes_", None).tolist() if getattr(self.model, "classes_", None) is not None else None
        return out
