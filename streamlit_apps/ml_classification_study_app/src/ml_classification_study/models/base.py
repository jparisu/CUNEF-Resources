from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc as sk_auc,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    roc_curve as sk_roc_curve,
)

from ml_classification_study.types import ArgSpec, ModelCapabilities, RocCurveData, TrainStepResult


class BaseClassifier(ABC):
    """Abstract model interface used by the UI."""

    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str] = ""
    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities()

    def __init__(self, **hyperparams: Any) -> None:
        self.hyperparams: Dict[str, Any] = dict(hyperparams)
        self._is_trained: bool = False
        self._step_index: int = 0

    @classmethod
    @abstractmethod
    def arg_spec(cls) -> List[ArgSpec]:
        """Hyperparameter specifications."""

    def arguments(self) -> List[ArgSpec]:
        return self.arg_spec()

    def reset(self) -> None:
        self._is_trained = False
        self._step_index = 0
        self._reset_impl()

    @abstractmethod
    def _reset_impl(self) -> None:
        """Model-specific reset."""

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._train_impl(X, y)
        self._is_trained = True

    @abstractmethod
    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Model-specific full training."""

    def train_one_step(self, X: np.ndarray, y: np.ndarray) -> TrainStepResult:
        """Perform one training step.

        Default behavior: call full training once.
        """

        self._step_index += 1
        self.train(X, y)
        return TrainStepResult(step_index=self._step_index, message="Trained (single-step)")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions {0,1}."""

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Return probability estimates with shape (n,2) or None if unavailable."""
        return None

    def acc(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))

    def f1score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Binary classification, positive label = 1
        try:
            return float(f1_score(y, self.predict(X), pos_label=1))
        except Exception:
            return float("nan")

    def confussion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Keep misspelling for API compatibility with the prompt
        return sk_confusion_matrix(y, self.predict(X), labels=[0, 1])

    def roc_curve(self, X: np.ndarray, y: np.ndarray) -> Optional[RocCurveData]:
        proba = self.predict_proba(X)
        if proba is None:
            return None

        # Use probability of positive class
        scores = proba[:, 1]
        try:
            fpr, tpr, thresholds = sk_roc_curve(y, scores, pos_label=1)
            return RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds, auc=float(sk_auc(fpr, tpr)))
        except Exception:
            return None

    def trained_parameters(self) -> Dict[str, Any]:
        """Return trained parameters in a UI-friendly format."""
        return {"hyperparams": dict(self.hyperparams)}
