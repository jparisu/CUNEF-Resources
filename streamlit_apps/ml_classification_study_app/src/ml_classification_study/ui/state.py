from __future__ import annotations

import json
from typing import Any, Dict


# Session state keys
KEY_DATASET_ID = "dataset_id"
KEY_DATASET_GEN_ARGS = "dataset_gen_args"
KEY_PREPROCESS = "preprocess"
KEY_DATASET_BUNDLE = "dataset_bundle"

KEY_MODEL_ID = "model_id"
KEY_MODEL_HYPERPARAMS = "model_hyperparams"
KEY_MODEL_INSTANCE = "model_instance"
KEY_MODEL_FINGERPRINT = "model_fingerprint"
KEY_TRAIN_STEP_MESSAGE = "train_step_message"


def fingerprint(obj: Any) -> str:
    """Stable fingerprint for caching/invalidation.

    Uses JSON serialization with sorting; falls back to `str` for non-serializable values.
    """

    def _default(x: Any):
        try:
            return x.__dict__
        except Exception:
            return str(x)

    return json.dumps(obj, sort_keys=True, default=_default)
