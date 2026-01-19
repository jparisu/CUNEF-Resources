from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from ml_classification_study.types import ArgSpec, ModelCapabilities, TrainStepResult
from .base import BaseClassifier


def _depth_choices(max_depth_limit: int = 20):
    choices = [("None", None)]
    for d in range(1, max_depth_limit + 1):
        choices.append((str(d), d))
    return choices


class DecisionTreeStudyClassifier(BaseClassifier):
    id = "decision_tree"
    name = "Decision Tree"
    description = "CART decision tree classifier. In step mode, it refits with increasing max_depth."
    capabilities = ModelCapabilities(supports_step_training=True, supports_predict_proba=True, supports_params_view=True)

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="criterion",
                label="criterion",
                kind="choice",
                default="gini",
                choices=[("gini", "gini"), ("entropy", "entropy"), ("log_loss", "log_loss")],
            ),
            ArgSpec(
                key="max_depth",
                label="max_depth",
                kind="choice",
                default=None,
                choices=_depth_choices(20),
                help="Limit tree depth (None means unlimited).",
            ),
            ArgSpec(key="min_samples_split", label="min_samples_split",
                kind="int", default=2, min_value=2, max_value=50, step=1),
            ArgSpec(key="min_samples_leaf", label="min_samples_leaf",
                kind="int", default=1, min_value=1, max_value=50, step=1),
            # ArgSpec(
            #     key="step_depth_cap",
            #     label="step depth cap",
            #     kind="int",
            #     default=10,
            #     min_value=1,
            #     max_value=30,
            #     step=1,
            #     help="Used only when max_depth=None and step training is enabled.",
            # ),
        ]

    def _reset_impl(self) -> None:
        self._model: Optional[DecisionTreeClassifier] = None
        self._current_depth: int = 0
        self._X_full: Optional[np.ndarray] = None
        self._y_full: Optional[np.ndarray] = None

    def _build_model(self, *, override_depth: Optional[int] = None) -> DecisionTreeClassifier:
        max_depth = override_depth if override_depth is not None else self.hyperparams.get("max_depth", None)

        return DecisionTreeClassifier(
            criterion=self.hyperparams.get("criterion", "gini"),
            max_depth=max_depth,
            min_samples_split=int(self.hyperparams.get("min_samples_split", 2)),
            min_samples_leaf=int(self.hyperparams.get("min_samples_leaf", 1)),
            random_state=int(self.hyperparams.get("random_state", 0)),
        )

    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X_full = np.asarray(X)
        self._y_full = np.asarray(y)
        self._model = self._build_model(override_depth=None)
        self._model.fit(self._X_full, self._y_full)

    def train_one_step(self, X: np.ndarray, y: np.ndarray) -> TrainStepResult:
        self._step_index += 1
        self._X_full = np.asarray(X)
        self._y_full = np.asarray(y)

        target = self.hyperparams.get("max_depth", None)
        if target is None:
            cap = int(self.hyperparams.get("step_depth_cap", 10))
            target_depth = cap
        else:
            target_depth = int(target)

        if self._current_depth >= target_depth:
            # No further refinement
            if self._model is None:
                self._model = self._build_model(override_depth=target_depth)
                self._model.fit(self._X_full, self._y_full)
                self._is_trained = True
            return TrainStepResult(step_index=self._step_index, message=f"Step {self._step_index}: already at max depth {target_depth}")

        self._current_depth += 1
        self._model = self._build_model(override_depth=self._current_depth)
        self._model.fit(self._X_full, self._y_full)
        self._is_trained = True

        return TrainStepResult(step_index=self._step_index, message=f"Step {self._step_index}: fit with max_depth={self._current_depth}", payload={"current_depth": self._current_depth, "target_depth": target_depth})

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        return self._model.predict_proba(X)

    def trained_parameters(self) -> Dict[str, Any]:
        if self._model is None:
            return {"hyperparams": dict(self.hyperparams)}

        try:
            txt = export_text(self._model, feature_names=["x1", "x2"])
        except Exception:
            txt = "(tree export failed)"

        return {
            "hyperparams": dict(self.hyperparams),
            "node_count": int(self._model.tree_.node_count),
            "max_depth_trained": int(self._model.get_depth()),
            "n_leaves": int(self._model.get_n_leaves()),
            "feature_importances": self._model.feature_importances_.tolist(),
            "tree_text": txt,
        }
