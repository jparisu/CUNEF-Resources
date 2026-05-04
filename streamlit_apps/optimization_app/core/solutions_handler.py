from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Solution:
    """A single evaluated point in the optimization space."""

    x1: float
    x2: float
    f_value: float
    highlighted: bool = False
    semi_highlighted: bool = False
    label: str | None = None


class SolutionsHandler:
    """
    Manages the ordered collection of evaluated solutions.

    Solutions are stored in insertion order, which defines the optimization
    path visualized in the plot.
    """

    def __init__(self) -> None:
        self._solutions: list[Solution] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_point(self, x1: float, x2: float, f_value: float) -> Solution:
        """
        Append a new evaluated point.

        The newly added point is highlighted; all previous points lose
        their highlight so the 'current' position is always visible.
        """
        for s in self._solutions:
            s.highlighted = False
        solution = Solution(x1=x1, x2=x2, f_value=f_value, highlighted=True)
        self._solutions.append(solution)
        return solution

    def reset(self) -> None:
        """Remove all stored solutions."""
        self._solutions.clear()

    def remove_last(self) -> Solution | None:
        """Remove and return the most recently added solution, or None if empty."""
        if not self._solutions:
            return None
        removed = self._solutions.pop()
        if self._solutions:
            self._solutions[-1].highlighted = True
        return removed

    def toggle_highlight(self, index: int) -> None:
        """Toggle the highlighted flag for the solution at the given index."""
        self._solutions[index].highlighted = not self._solutions[index].highlighted

    def set_semi_highlighted(self, index: int, value: bool = True) -> None:
        """Set the semi_highlighted flag for the solution at the given index."""
        self._solutions[index].semi_highlighted = value

    def set_label(self, index: int, label: str) -> None:
        """Attach a custom display label to the solution at the given index."""
        self._solutions[index].label = label

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all(self) -> list[Solution]:
        """Return all solutions in insertion order (shallow copy)."""
        return list(self._solutions)

    def get_best(self) -> Solution | None:
        """Return the solution with the highest f_value, or None if empty."""
        if not self._solutions:
            return None
        return max(self._solutions, key=lambda s: s.f_value)

    def get_best_all(self) -> list[Solution]:
        """Return all solutions tied for the highest f_value, or empty list if empty."""
        if not self._solutions:
            return []
        best_val = max(s.f_value for s in self._solutions)
        return [s for s in self._solutions if s.f_value == best_val]

    def get_statistics(self) -> dict:
        """Return mean, median, std, min, max of f_values, or empty dict if no solutions."""
        if not self._solutions:
            return {}
        import numpy as np

        vals = [s.f_value for s in self._solutions]
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    def get_latest(self) -> Solution | None:
        """Return the most recently added solution, or None if empty."""
        return self._solutions[-1] if self._solutions else None

    def count(self) -> int:
        """Return the total number of stored solutions."""
        return len(self._solutions)

    def is_empty(self) -> bool:
        """Return True if no solutions have been stored yet."""
        return len(self._solutions) == 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return solutions as a DataFrame with columns: #, x1, x2, F(x1,x2), highlighted, semi_highlighted."""
        if self.is_empty():
            return pd.DataFrame(
                columns=["#", "x1", "x2", "F(x1,x2)", "highlighted", "semi_highlighted"]
            )
        rows = [
            {
                "#": i + 1,
                "x1": round(s.x1, 4),
                "x2": round(s.x2, 4),
                "F(x1,x2)": round(s.f_value, 6),
                "highlighted": s.highlighted,
                "semi_highlighted": s.semi_highlighted,
            }
            for i, s in enumerate(self._solutions)
        ]
        return pd.DataFrame(rows)
