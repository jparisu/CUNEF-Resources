from __future__ import annotations

from typing import TYPE_CHECKING

from optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    from core.config import AppConfig
    from core.function_handler import FunctionHandler
    from core.solutions_handler import SolutionsHandler


class GradientAscentOptimizer(BaseOptimizer):
    """Optimizer that moves from a current point in the gradient direction."""

    def __init__(self, step_size: float = 0.1) -> None:
        self.step_size: float = step_size
        self._current_point: tuple[float, float] | None = None
        self._iteration: int = 0

    @property
    def name(self) -> str:
        return "Gradient"

    @property
    def description(self) -> str:
        return "Use the derivative to step in the gradient direction."

    @property
    def requires_derivative(self) -> bool:
        return True

    @property
    def current_point(self) -> tuple[float, float] | None:
        return self._current_point

    def set_current_point(self, x1: float, x2: float) -> None:
        self._current_point = (x1, x2)

    def set_learning_rate(self, learning_rate: float) -> None:
        self.step_size = max(1e-6, float(learning_rate))

    def suggest_random_initial_point(self, config: AppConfig) -> tuple[float, float]:
        self._current_point = config.sample_uniform_point()
        return self._current_point

    def suggest_next_with_gradient(
        self,
        function_handler: FunctionHandler,
        config: AppConfig,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        if self._current_point is None or not function_handler.has_derivative():
            return None
        x1, x2 = self._current_point
        g1, g2 = function_handler.evaluate_gradient(x1, x2)
        next_point = config.clamp(x1 + self.step_size * g1, x2 + self.step_size * g2)
        return (self._current_point, next_point, (g1, g2))

    def suggest_next(
        self,
        function_handler: FunctionHandler,
        solutions_handler: SolutionsHandler,  # noqa: ARG002
        config: AppConfig,
    ) -> tuple[float, float] | None:
        move = self.suggest_next_with_gradient(function_handler, config)
        if move is None:
            return None
        current, next_point, _gradient = move
        current_value = function_handler.evaluate(current[0], current[1])
        next_value = function_handler.evaluate(next_point[0], next_point[1])
        if next_value <= current_value:
            return None
        self._current_point = next_point
        self._iteration += 1
        return next_point

    def reset(self) -> None:
        """Reset the iteration counter and current point."""
        self._iteration = 0
        self._current_point = None
