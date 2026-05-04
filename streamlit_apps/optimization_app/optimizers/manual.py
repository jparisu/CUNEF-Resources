from __future__ import annotations

from typing import TYPE_CHECKING

from optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    from core.config import AppConfig
    from core.function_handler import FunctionHandler
    from core.solutions_handler import SolutionsHandler


class ManualOptimizer(BaseOptimizer):
    """
    Optimizer where the user explicitly chooses the next point.

    The point is set via numeric sliders in the sidebar (or a future plot-click).
    suggest_next() returns the pending point and clears it.
    """

    def __init__(self) -> None:
        self._pending: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        return "Manual"

    @property
    def description(self) -> str:
        return "Pick the next point manually using the x₁ and x₂ sliders."

    def set_point(self, x1: float, x2: float) -> None:
        """Register the next point to be returned by suggest_next()."""
        self._pending = (x1, x2)

    def has_pending(self) -> bool:
        """Return True if a point is waiting to be evaluated."""
        return self._pending is not None

    def suggest_next(
        self,
        function_handler: FunctionHandler,  # noqa: ARG002
        solutions_handler: SolutionsHandler,  # noqa: ARG002
        config: AppConfig,  # noqa: ARG002
    ) -> tuple[float, float] | None:
        """Return the pending point and clear it, or None if none is set."""
        point = self._pending
        self._pending = None
        return point

    def reset(self) -> None:
        """Clear the pending point."""
        self._pending = None
