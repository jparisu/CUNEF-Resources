from __future__ import annotations

from typing import TYPE_CHECKING

from optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    from core.config import AppConfig
    from core.function_handler import FunctionHandler
    from core.solutions_handler import SolutionsHandler


class RandomOptimizer(BaseOptimizer):
    """Suggests points sampled uniformly from the configured domain."""

    @property
    def name(self) -> str:
        return "Random"

    @property
    def description(self) -> str:
        return "Sample a new random point inside the configured x range."

    def suggest_next(
        self,
        function_handler: FunctionHandler,  # noqa: ARG002
        solutions_handler: SolutionsHandler,  # noqa: ARG002
        config: AppConfig,
    ) -> tuple[float, float] | None:
        return config.sample_uniform_point()
