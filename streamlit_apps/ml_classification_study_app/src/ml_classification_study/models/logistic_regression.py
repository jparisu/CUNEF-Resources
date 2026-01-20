from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml_classification_study.types import ArgSpec, ModelCapabilities
from .base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    id = "log_reg"
    name = "Logistic Regression"
    description = "Linear decision boundary trained with logistic loss (scikit-learn LogisticRegression)."
    capabilities = ModelCapabilities(supports_step_training=False, supports_predict_proba=True, supports_params_view=True)

    def __init__(self, **hyperparams: Any) -> None:
        super().__init__(**hyperparams)
        self.model: Optional[LogisticRegression] = None
        self._reset_impl()

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="C",
                label="Inverse regularization (C)",
                kind="float",
                default=1.0,
                min_value=0.001,
                max_value=1000.0,
                step=0.1,
                help="Smaller C => stronger regularization.",
            ),
            ArgSpec(
                key="max_iter",
                label="Max iterations",
                kind="int",
                default=500,
                min_value=50,
                max_value=5000,
                step=50,
            ),
            ArgSpec(
                key="solver",
                label="Solver",
                kind="choice",
                default="lbfgs",
                choices=[("lbfgs", "lbfgs"), ("liblinear", "liblinear"), ("saga", "saga")],
                help="Most stable choices for small 2D problems: lbfgs/liblinear.",
            ),
            ArgSpec(
                key="fit_intercept",
                label="Fit intercept",
                kind="bool",
                default=True,
            ),
        ]

    def _reset_impl(self) -> None:
        self.model = None

    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        C = float(self.hyperparams.get("C", 1.0))
        max_iter = int(self.hyperparams.get("max_iter", 500))
        solver = str(self.hyperparams.get("solver", "lbfgs"))
        fit_intercept = bool(self.hyperparams.get("fit_intercept", True))

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            fit_intercept=fit_intercept,
            penalty="l2",
            n_jobs=None,
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

        coef = getattr(self.model, "coef_", None)
        intercept = getattr(self.model, "intercept_", None)
        out: Dict[str, Any] = {
            **base,
            "coef": None if coef is None else np.asarray(coef).tolist(),
            "intercept": None if intercept is None else np.asarray(intercept).tolist(),
            "classes": getattr(self.model, "classes_", None).tolist() if getattr(self.model, "classes_", None) is not None else None,
        }
        return out
