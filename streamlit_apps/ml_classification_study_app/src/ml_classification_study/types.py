from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Literal, Optional, Sequence, Tuple, TypeVar

import numpy as np


ArgKind = Literal["int", "float", "bool", "choice", "text"]


@dataclass(frozen=True)
class ArgSpec:
    """Declarative UI/config specification.

    The Streamlit UI renders widgets from these specs so datasets/models can declare
    their own hyperparameters in a generic way.
    """

    key: str
    label: str
    kind: ArgKind
    default: Any

    # Numeric constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None

    # Choices for kind == "choice". Each element is (display_label, value).
    choices: Optional[Sequence[Tuple[str, Any]]] = None

    help: str = ""


@dataclass
class DatasetBundle:
    """Holds a generated dataset and its train/test split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    # Original full dataset (after preprocessing/outlier removal)
    X: np.ndarray
    y: np.ndarray

    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    supports_step_training: bool = False
    supports_predict_proba: bool = True
    supports_params_view: bool = True


@dataclass
class RocCurveData:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


T = TypeVar("T")


@dataclass
class TrainStepResult:
    """Returned from train_one_step to support UI feedback."""

    step_index: int
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
