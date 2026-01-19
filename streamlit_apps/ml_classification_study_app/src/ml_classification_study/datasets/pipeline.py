from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ml_classification_study.types import DatasetBundle

from .preprocess import PreprocessConfig, remove_outliers_by_std, scale_features, split_train_test
from .registry import get_dataset_cls


def make_dataset_bundle(
    dataset_id: str,
    *,
    gen_args: Dict[str, Any],
    preprocess: PreprocessConfig,
) -> DatasetBundle:
    """Generate a dataset and apply preprocessing (split, outlier removal, scaling)."""

    ds_cls = get_dataset_cls(dataset_id)
    generator = ds_cls()

    X, y = generator.generate(**gen_args)

    # Split
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=preprocess.test_size,
        random_state=int(gen_args.get("random_state", 0)),
        stratify=bool(preprocess.stratify),
    )

    # Outlier removal
    X_train, y_train, X_test, y_test, out_meta = remove_outliers_by_std(
        X_train,
        y_train,
        X_test,
        y_test,
        n_std=float(preprocess.outlier_std),
    )

    # Scaling
    X_train, X_test, scale_meta = scale_features(X_train, X_test, scaling=preprocess.scaling)

    # Reconstruct combined X/y AFTER preprocessing (useful for plotting)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    meta = {
        "dataset_id": dataset_id,
        "dataset_name": ds_cls.name,
        "generator_args": dict(gen_args),
        "preprocess": preprocess,
        **out_meta,
        **scale_meta,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X=X_all,
        y=y_all,
        meta=meta,
    )
