from __future__ import annotations

from typing import Dict, List, Type

from .base import DatasetGenerator
from .generators import (
    Circular2D,
    Gaussian2Class,
    NormalVsExponential,
    RandomLinearSeparable,
    Spiral2D,
    Xor2D,
)


DATASET_REGISTRY: Dict[str, Type[DatasetGenerator]] = {
    Gaussian2Class.id: Gaussian2Class,
    Xor2D.id: Xor2D,
    NormalVsExponential.id: NormalVsExponential,
    RandomLinearSeparable.id: RandomLinearSeparable,
    Circular2D.id: Circular2D,
    Spiral2D.id: Spiral2D,
}


def list_datasets() -> List[Type[DatasetGenerator]]:
    return list(DATASET_REGISTRY.values())


def get_dataset_cls(dataset_id: str) -> Type[DatasetGenerator]:
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset id: {dataset_id}")
    return DATASET_REGISTRY[dataset_id]
