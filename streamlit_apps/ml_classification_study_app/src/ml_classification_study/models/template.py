from __future__ import annotations

"""Template for adding a new classifier model.

How to use
----------
1) Copy this file and rename the class.
2) Implement `arg_spec()` to declare hyperparameters for the UI.
3) Implement `_reset_impl()`, `_train_impl()`, and `predict()`.
4) Optionally implement:
   - `predict_proba()` to enable ROC curve support.
   - `train_one_step()` to enable step-by-step training.
   - `trained_parameters()` to show parameters in the UI.
5) Register your model in `ml_classification_study/models/registry.py`.

Notes
-----
- Keep `id` stable once the model is used in saved configs/session state.
- `BaseClassifier` provides generic metrics (accuracy, f1, confusion matrix, ROC)
  built on top of `predict()` and `predict_proba()`.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ml_classification_study.types import ArgSpec, ModelCapabilities, TrainStepResult

from .base import BaseClassifier


class TemplateClassifier(BaseClassifier):
    """A stub classifier.

    Replace this class with your own implementation.
    """

    id = "template_model"
    name = "Template Model (TODO)"
    description = "Skeleton model: fill in required methods."

    # Update these when you implement optional features.
    capabilities = ModelCapabilities(can_step_train=False, has_proba=False, has_parameters_view=False)

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        """Return hyperparameter specs.

        Example:
            return [
                ArgSpec(key="learning_rate", label="Learning rate", kind="float", default=0.1, min_value=1e-4, max_value=10.0, step=0.01),
            ]
        """

        raise NotImplementedError("Implement arg_spec() for your model.")

    def _reset_impl(self) -> None:
        """Reset model-specific state."""

        raise NotImplementedError("Implement _reset_impl() to clear model state.")

    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Full training implementation."""

        raise NotImplementedError("Implement _train_impl() to train the model.")

    def train_one_step(self, X: np.ndarray, y: np.ndarray) -> TrainStepResult:
        """Optional: one-step training.

        If you support this, also set:
            capabilities.can_step_train = True
        """

        raise NotImplementedError("Optional: implement train_one_step() for step-by-step training.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard predictions as integers in {0,1}."""

        raise NotImplementedError("Implement predict() returning labels in {0,1}.")

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Optional: return probability estimates with shape (n,2).

        If you support this, also set:
            capabilities.has_proba = True
        """

        raise NotImplementedError("Optional: implement predict_proba() for ROC curve support.")

    def trained_parameters(self) -> Dict[str, Any]:
        """Optional: return a UI-friendly dict of trained parameters."""

        raise NotImplementedError("Optional: implement trained_parameters() for UI parameter display.")
