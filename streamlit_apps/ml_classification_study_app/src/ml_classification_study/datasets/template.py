from __future__ import annotations

"""Template for adding a new synthetic dataset generator.

How to use
----------
1) Copy this file and rename the class.
2) Implement `arg_spec()` to declare your generator-specific parameters.
3) Implement `generate()` to return X (n,2) and y (n,) with classes {0,1}.
4) Register your dataset in `ml_classification_study/datasets/registry.py`.

Notes
-----
- Common parameters are already provided by `DatasetGenerator.common_arg_spec()`:
  `n_samples` and `random_state`.
- Keep `id` stable once the dataset is used in saved configs/session state.
"""

from typing import Any, List

import numpy as np

from ml_classification_study.types import ArgSpec

from .base import DatasetGenerator


class TemplateDatasetGenerator(DatasetGenerator):
    """A stub dataset generator.

    Replace this class with your own implementation.
    """

    id = "template_dataset"
    name = "Template Dataset (TODO)"
    description = "Skeleton generator: fill in arg_spec() and generate()."

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        """Return generator-specific argument specs.

        Example:
            return [
                ArgSpec(key="noise", label="Noise", kind="float", default=0.1, min_value=0.0, max_value=2.0, step=0.05),
            ]
        """

        raise NotImplementedError("Implement arg_spec() for your dataset generator.")

    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Generate and return X and y.

        Required:
        - X must be shape (n, 2)
        - y must be shape (n,) with integer labels {0,1}

        The UI will pass both common args (n_samples, random_state) and your
        generator-specific args into this method.
        """

        raise NotImplementedError("Implement generate() to create X (n,2) and y (n,) with classes {0,1}.")
