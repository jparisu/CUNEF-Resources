from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, List

import numpy as np

from ml_classification_study.types import ArgSpec


class DatasetGenerator(ABC):
    """Base interface for synthetic dataset generators."""

    # Short stable id used for keys/session state
    id: ClassVar[str]
    # Human-readable name
    name: ClassVar[str]
    description: ClassVar[str] = ""

    @classmethod
    def common_arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="n_samples",
                label="Number of points",
                kind="int",
                default=300,
                min_value=50,
                max_value=5000,
                step=50,
                help="Total number of points (both classes combined).",
            ),
            ArgSpec(
                key="random_state",
                label="Random seed",
                kind="int",
                default=0,
                min_value=0,
                max_value=10_000,
                step=1,
                help="Controls reproducibility.",
            ),
        ]

    @classmethod
    @abstractmethod
    def arg_spec(cls) -> List[ArgSpec]:
        """Generator-specific argument specs."""

    @classmethod
    def full_arg_spec(cls) -> List[ArgSpec]:
        return cls.common_arg_spec() + cls.arg_spec()

    @abstractmethod
    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return X (n,2) and y (n,) with classes {0,1}."""

