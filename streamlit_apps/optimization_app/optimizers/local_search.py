from __future__ import annotations

from math import cos, pi, sin
from typing import TYPE_CHECKING

from optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    from core.config import AppConfig
    from core.function_handler import FunctionHandler
    from core.solutions_handler import SolutionsHandler


class LocalSearchOptimizer(BaseOptimizer):
    """Local search driven by a user-configurable locality function."""

    PRESETS = ("Manhattan (4 neighbors)", "Euclidean (6 neighbors)")

    def __init__(self) -> None:
        self._current_point: tuple[float, float] | None = None
        self._preset: str = self.PRESETS[0]
        self._size: float = 1.0

    @property
    def name(self) -> str:
        return "Local search"

    @property
    def description(self) -> str:
        return "Iteratively evaluate neighbors from a locality(x1, x2) function."

    @property
    def preset(self) -> str:
        return self._preset

    @property
    def size(self) -> float:
        return self._size

    @property
    def current_point(self) -> tuple[float, float] | None:
        return self._current_point

    def set_initial_point(self, x1: float, x2: float) -> None:
        self._current_point = (x1, x2)

    def configure_locality(self, preset: str, size: float) -> None:
        if preset not in self.PRESETS:
            raise ValueError(f"Unsupported locality preset: {preset}")
        self._preset = preset
        self._size = max(1e-6, float(size))

    def suggest_locality_points(self, config: AppConfig) -> list[tuple[float, float]]:
        if self._current_point is None:
            return []
        x1, x2 = self._current_point

        if self._preset == "Manhattan (4 neighbors)":
            offsets = [
                (self._size, 0.0),
                (-self._size, 0.0),
                (0.0, self._size),
                (0.0, -self._size),
            ]
        else:
            offsets = [
                (self._size * cos(angle), self._size * sin(angle))
                for angle in (index * 2 * pi / 6 for index in range(6))
            ]

        return [config.clamp(x1 + dx, x2 + dy) for dx, dy in offsets]

    def suggest_next(
        self,
        function_handler: FunctionHandler,
        solutions_handler: SolutionsHandler,  # noqa: ARG002
        config: AppConfig,
    ) -> tuple[float, float] | None:
        if self._current_point is None:
            return None
        candidates = self.suggest_locality_points(config)
        if not candidates:
            return None
        best = max(candidates, key=lambda p: function_handler.evaluate(p[0], p[1]))
        current_value = function_handler.evaluate(self._current_point[0], self._current_point[1])
        best_value = function_handler.evaluate(best[0], best[1])
        if best_value <= current_value:
            return None
        self._current_point = best
        return best

    def suggest_random_initial_point(self, config: AppConfig) -> tuple[float, float]:
        self._current_point = config.sample_uniform_point()
        return self._current_point

    def reset(self) -> None:
        self._current_point = None
