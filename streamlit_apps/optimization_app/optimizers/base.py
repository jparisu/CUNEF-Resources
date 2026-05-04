from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import AppConfig
    from core.function_handler import FunctionHandler
    from core.solutions_handler import SolutionsHandler


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.

    Subclasses implement suggest_next() to propose the next point to evaluate.
    The UI layer is responsible for evaluating F at the suggested point and
    adding it to the SolutionsHandler.

    The optimizer is stateless by default; subclasses may maintain internal
    state (e.g., step count, temperature) that must be reset via reset().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in the optimizer selector dropdown."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description of the algorithm shown in the UI."""
        ...

    @property
    def requires_derivative(self) -> bool:
        """
        Return True if this optimizer needs gradient information from FunctionHandler.

        The UI should warn the user if the selected optimizer requires a derivative
        but none is available in the current FunctionHandler.
        """
        return False

    @abstractmethod
    def suggest_next(
        self,
        function_handler: FunctionHandler,
        solutions_handler: SolutionsHandler,
        config: AppConfig,
    ) -> tuple[float, float] | None:
        """
        Suggest the next (x1, x2) point to evaluate.

        Args:
            function_handler: Provides F and its gradient.
            solutions_handler: Provides all previously evaluated points.
            config: Domain bounds and other global settings.

        Returns:
            A (x1, x2) tuple within the domain, or None if no suggestion
            can be produced (e.g., the optimizer needs prior points but
            none exist yet).
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal optimizer state to its initial values.

        Called when the user resets solutions or switches optimizer.
        Base implementation is a no-op.
        """
        return None
