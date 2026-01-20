from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseClassifier
from .decision_tree import DecisionTreeStudyClassifier
from .knn import KNNClassifier
from .naive_bayes import GaussianNaiveBayesClassifier
from .logistic_regression import LogisticRegressionClassifier
from .lda import LDAClassifier
from .random_forest import RandomForestStudyClassifier


MODEL_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    KNNClassifier.id: KNNClassifier,
    GaussianNaiveBayesClassifier.id: GaussianNaiveBayesClassifier,
    DecisionTreeStudyClassifier.id: DecisionTreeStudyClassifier,
    LogisticRegressionClassifier.id: LogisticRegressionClassifier,
    LDAClassifier.id: LDAClassifier,
    RandomForestStudyClassifier.id: RandomForestStudyClassifier,
}


def list_models() -> List[Type[BaseClassifier]]:
    return list(MODEL_REGISTRY.values())


def get_model_cls(model_id: str) -> Type[BaseClassifier]:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model id: {model_id}")
    return MODEL_REGISTRY[model_id]
