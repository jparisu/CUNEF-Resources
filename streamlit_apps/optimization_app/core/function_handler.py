from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class FunctionSource(Enum):
    PREDEFINED = "predefined"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Predefined function registry
# ---------------------------------------------------------------------------


@dataclass
class PredefinedFunctionDef:
    """Everything needed to display and evaluate a built-in function."""

    display_name: str
    description: str
    func: Callable[[float, float], float]
    latex: str
    derivative_func: Callable[[float, float], tuple[float, float]] | None = None
    derivative_latex: str | None = None


# Predefined callables — no float() cast so they accept numpy arrays in evaluate_grid.


def _sphere_f(x1, x2):
    return -(x1**2 + x2**2)


def _sincos_f(x1, x2):
    return np.sin(x1) * np.cos(x2)


def _gaussian_f(x1, x2):
    return np.exp(-(x1**2 + x2**2) / 4.0)


def _twin_peaks_f(x1, x2):
    return np.exp(-((x1 - 3) ** 2 + x2**2) / 4.0) + 0.7 * np.exp(
        -((x1 + 3) ** 2 + (x2 - 2) ** 2) / 4.0
    )


def _saddle_f(x1, x2):
    return x1 * x2 * np.exp(-(x1**2 + x2**2) / 8.0)


def _triple_peaks_f(x1, x2):
    return (
        np.exp(-((x1 - 3) ** 2 + (x2 + 1) ** 2) / 4.0)
        + 0.85 * np.exp(-((x1 + 3) ** 2 + (x2 - 2) ** 2) / 5.0)
        + 0.65 * np.exp(-((x1 - 1) ** 2 + (x2 - 4) ** 2) / 3.0)
    )


def _bilinear_f(x1, x2):
    return x1 * x2


def _sphere_d(x1: float, x2: float) -> tuple[float, float]:
    return (-2.0 * x1, -2.0 * x2)


def _sincos_d(x1: float, x2: float) -> tuple[float, float]:
    return (float(np.cos(x1) * np.cos(x2)), float(-np.sin(x1) * np.sin(x2)))


def _gaussian_d(x1: float, x2: float) -> tuple[float, float]:
    g = float(np.exp(-(x1**2 + x2**2) / 4.0))
    return (-x1 / 2.0 * g, -x2 / 2.0 * g)


def _twin_peaks_d(x1: float, x2: float) -> tuple[float, float]:
    a = np.exp(-((x1 - 3) ** 2 + x2**2) / 4.0)
    b = np.exp(-((x1 + 3) ** 2 + (x2 - 2) ** 2) / 4.0)
    return (
        float(a * (-(x1 - 3) / 2.0) + 0.7 * b * (-(x1 + 3) / 2.0)),
        float(a * (-x2 / 2.0) + 0.7 * b * (-(x2 - 2) / 2.0)),
    )


def _saddle_d(x1: float, x2: float) -> tuple[float, float]:
    g = np.exp(-(x1**2 + x2**2) / 8.0)
    return (float(x2 * (1.0 - x1**2 / 4.0) * g), float(x1 * (1.0 - x2**2 / 4.0) * g))


def _triple_peaks_d(x1: float, x2: float) -> tuple[float, float]:
    a = np.exp(-((x1 - 3) ** 2 + (x2 + 1) ** 2) / 4.0)
    b = np.exp(-((x1 + 3) ** 2 + (x2 - 2) ** 2) / 5.0)
    c = np.exp(-((x1 - 1) ** 2 + (x2 - 4) ** 2) / 3.0)
    return (
        float(a * (-(x1 - 3) / 2.0) + 0.85 * b * (-2.0 * (x1 + 3) / 5.0) + 0.65 * c * (-2.0 * (x1 - 1) / 3.0)),
        float(a * (-(x2 + 1) / 2.0) + 0.85 * b * (-2.0 * (x2 - 2) / 5.0) + 0.65 * c * (-2.0 * (x2 - 4) / 3.0)),
    )


def _bilinear_d(x1: float, x2: float) -> tuple[float, float]:
    return (x2, x1)


PREDEFINED_FUNCTIONS: dict[str, PredefinedFunctionDef] = {
    "Inverted Sphere": PredefinedFunctionDef(
        display_name="Inverted Sphere",
        description="Simple paraboloid with a single global maximum at the origin.",
        func=_sphere_f,
        latex=r"-(x_1^2 + x_2^2)",
        derivative_func=_sphere_d,
        derivative_latex=r"\nabla F = (-2x_1,\;-2x_2)",
    ),
    "Sine × Cosine": PredefinedFunctionDef(
        display_name="Sine × Cosine",
        description="Periodic function with multiple local optima — good for testing global search.",
        func=_sincos_f,
        latex=r"\sin(x_1)\cdot\cos(x_2)",
        derivative_func=_sincos_d,
        derivative_latex=r"\nabla F = (\cos x_1\cos x_2,\;-\sin x_1\sin x_2)",
    ),
    "Gaussian Bell": PredefinedFunctionDef(
        display_name="Gaussian Bell",
        description="Smooth bell curve — unimodal and easy to optimise.",
        func=_gaussian_f,
        latex=r"e^{-(x_1^2+x_2^2)/4}",
        derivative_func=_gaussian_d,
        derivative_latex=r"\nabla F = e^{-(x_1^2+x_2^2)/4}\!\left(-\tfrac{x_1}{2},\,-\tfrac{x_2}{2}\right)",
    ),
    "Twin Peaks": PredefinedFunctionDef(
        display_name="Twin Peaks",
        description="Two asymmetric Gaussian peaks — highlights competing local optima.",
        func=_twin_peaks_f,
        latex=r"e^{-\frac{(x_1-3)^2+x_2^2}{4}}+0.7\,e^{-\frac{(x_1+3)^2+(x_2-2)^2}{4}}",
        derivative_func=_twin_peaks_d,
        derivative_latex=(
            r"\nabla F_1 = -\tfrac{x_1-3}{2}\,A - \tfrac{0.7(x_1+3)}{2}\,B"
            "\n"
            r"\nabla F_2 = -\tfrac{x_2}{2}\,A - \tfrac{0.7(x_2-2)}{2}\,B"
        ),
    ),
    "Triple Peaks": PredefinedFunctionDef(
        display_name="Triple Peaks",
        description="Three offset Gaussian peaks — useful for testing richer multi-modal search behaviour.",
        func=_triple_peaks_f,
        latex=(
            r"e^{-\frac{(x_1-3)^2+(x_2+1)^2}{4}}"
            r"+0.85\,e^{-\frac{(x_1+3)^2+(x_2-2)^2}{5}}"
            r"+0.65\,e^{-\frac{(x_1-1)^2+(x_2-4)^2}{3}}"
        ),
        derivative_func=_triple_peaks_d,
        derivative_latex=(
            r"\nabla F_1 = -\tfrac{x_1-3}{2}\,A - \tfrac{1.7(x_1+3)}{5}\,B - \tfrac{1.3(x_1-1)}{3}\,C"
            "\n"
            r"\nabla F_2 = -\tfrac{x_2+1}{2}\,A - \tfrac{1.7(x_2-2)}{5}\,B - \tfrac{1.3(x_2-4)}{3}\,C"
        ),
    ),
    "Bilinear Saddle": PredefinedFunctionDef(
        display_name="Bilinear Saddle",
        description="Simple bilinear surface F(x1, x2) = x1 x2 with straight-line level sets and a saddle at the origin.",
        func=_bilinear_f,
        latex=r"x_1 x_2",
        derivative_func=_bilinear_d,
        derivative_latex=r"\nabla F = (x_2,\;x_1)",
    ),
    "Mount Chair (Saddle)": PredefinedFunctionDef(
        display_name="Mount Chair (Saddle)",
        description="Saddle-shaped function — useful for studying saddle points and directional search.",
        func=_saddle_f,
        latex=r"x_1 x_2\,e^{-(x_1^2+x_2^2)/8}",
        derivative_func=_saddle_d,
        derivative_latex=(
            r"\nabla F = e^{-(x_1^2+x_2^2)/8}"
            r"\!\left(x_2\!\left(1-\tfrac{x_1^2}{4}\right),\;"
            r"x_1\!\left(1-\tfrac{x_2^2}{4}\right)\right)"
        ),
    ),
}


# ---------------------------------------------------------------------------
# FunctionHandler
# ---------------------------------------------------------------------------


class FunctionHandler:
    """
    Manages the objective function F(x1, x2) to be maximized.

    Functions come from two sources: PREDEFINED or RANDOM.
    Only one source is active at a time; the relevant bookkeeping fields
    are set on activation and cleared on switch.
    """

    def __init__(self) -> None:
        self.source: FunctionSource | None = None
        self._func: Callable[[float, float], float] | None = None
        self._gradient_func: Callable[[float, float], tuple[float, float]] | None = None
        self._latex: str = ""
        self._derivative_latex: str = ""

        self.predefined_name: str | None = None
        self.random_seed: int | None = None
        self.random_nf: int | None = None

    def _reset_source_fields(self) -> None:
        self.predefined_name = None
        self.random_seed = None
        self.random_nf = None

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_predefined(self, name: str) -> None:
        """Activate a built-in function by name. Raises KeyError if unknown."""
        defn = PREDEFINED_FUNCTIONS[name]
        self._reset_source_fields()
        self._func = defn.func
        self._gradient_func = defn.derivative_func
        self._latex = defn.latex
        self._derivative_latex = defn.derivative_latex or ""
        self.source = FunctionSource.PREDEFINED
        self.predefined_name = name

    def set_random(self, seed: int, nf: int) -> None:
        """Generate and activate a reproducible random symbolic function."""
        from core import random_function

        _expr, latex_str, safe_callable = random_function.generate(seed, nf)

        self._reset_source_fields()
        self._func = safe_callable
        self._gradient_func = None
        self._latex = latex_str
        self._derivative_latex = ""
        self.source = FunctionSource.RANDOM
        self.random_seed = seed
        self.random_nf = nf

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x1: float, x2: float) -> float:
        """Evaluate F(x1, x2). Raises RuntimeError if no function is set."""
        if self._func is None:
            raise RuntimeError("No function has been set.")
        return float(self._func(x1, x2))

    def evaluate_gradient(self, x1: float, x2: float) -> tuple[float, float]:
        """Evaluate ∇F at (x1, x2). Raises RuntimeError if no derivative is available."""
        if self._gradient_func is None:
            raise RuntimeError("No derivative is available.")
        result = self._gradient_func(x1, x2)
        return (float(result[0]), float(result[1]))

    def evaluate_grid(
        self, x_min: float, x_max: float, resolution: int = 100
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate F on a regular grid over [x_min, x_max]^2.

        Returns (X1, X2, Z) as 2-D arrays of shape (resolution, resolution).
        Predefined functions use direct numpy broadcasting (~100× faster than
        np.vectorize); custom/random functions fall back to np.vectorize.
        """
        if self._func is None:
            raise RuntimeError("No function has been set.")
        x1_vals = np.linspace(x_min, x_max, resolution)
        x2_vals = np.linspace(x_min, x_max, resolution)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        if self.source == FunctionSource.PREDEFINED:
            Z = self._func(X1, X2)
        else:
            Z = np.vectorize(self._func)(X1, X2)
        return X1, X2, Z

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._func is not None

    def has_derivative(self) -> bool:
        return self._gradient_func is not None

    def get_latex(self) -> str:
        return self._latex

    def get_derivative_latex(self) -> str:
        return self._derivative_latex

    def get_display_expr(self) -> str:
        if self.predefined_name:
            return PREDEFINED_FUNCTIONS[self.predefined_name].display_name
        if self.source == FunctionSource.RANDOM:
            return "Random function"
        return self._latex

    @property
    def cache_key(self) -> str:
        """Stable string key for cache invalidation when the function changes."""
        if self.source == FunctionSource.PREDEFINED:
            return f"pre:{self.predefined_name}"
        if self.source == FunctionSource.RANDOM:
            return f"rng:{self.random_seed}:{self.random_nf}"
        return "none"
