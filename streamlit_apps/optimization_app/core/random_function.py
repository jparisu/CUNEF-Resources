"""Random symbolic function generator using sympy."""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import sympy as sp

_x1, _x2 = sp.symbols("x1 x2")

_TERM_BUILDERS = [
    lambda a, b: sp.sin(a * _x1 + b * _x2),
    lambda a, b: sp.cos(a * _x1 + b * _x2),
    lambda a, b: sp.sin(a * _x1) * sp.cos(b * _x2),
    lambda a, b: sp.cos(a * _x1) * sp.sin(b * _x2),
    lambda a, b: sp.exp(-(a * _x1**2 + b * _x2**2) / 4),
    lambda a, b: _x1 * sp.exp(-(a * _x1**2) / 4),
    lambda a, b: _x2 * sp.exp(-(b * _x2**2) / 4),
    lambda a, b: _x1 * sp.sin(a * _x2),
    lambda a, b: _x2 * sp.cos(b * _x1),
    lambda a, b: sp.sin(a * _x1**2 - b * _x2**2),
]

_COEFF_CHOICES = [sp.Integer(1), sp.Integer(-1), sp.Rational(1, 2), sp.Integer(2)]
_A_CHOICES = [sp.Integer(1), sp.Rational(1, 2), sp.Integer(2), sp.Rational(3, 2)]


@functools.lru_cache(maxsize=32)
def generate(seed: int, nf: int) -> tuple[sp.Expr, str, Callable[[float, float], float]]:
    """
    Build a random symbolic function from ``nf`` composed terms.

    Cached by ``(seed, nf)``: repeated calls with the same arguments are free.
    The callable handles non-finite values by returning 0.0.

    Args:
        seed: Integer seed — same seed always produces the same expression.
        nf:   Number of term compositions (clamped to 1–20).

    Returns:
        ``(sympy_expr, latex_str, callable)``
    """
    nf = max(1, min(nf, 20))
    rng = np.random.default_rng(seed)

    n_templates = len(_TERM_BUILDERS)
    n_coeffs = len(_COEFF_CHOICES)
    n_a = len(_A_CHOICES)

    expr = sp.Integer(0)
    for _ in range(nf):
        builder = _TERM_BUILDERS[int(rng.integers(n_templates))]
        outer = _COEFF_CHOICES[int(rng.integers(n_coeffs))]
        a = _A_CHOICES[int(rng.integers(n_a))]
        b = _A_CHOICES[int(rng.integers(n_a))]
        expr = expr + outer * builder(a, b)

    latex_str = sp.latex(expr)
    raw_fn = sp.lambdify([_x1, _x2], expr, modules=["numpy"])

    def safe_callable(x1: float, x2: float) -> float:
        try:
            val = float(raw_fn(x1, x2))
            return val if np.isfinite(val) else 0.0
        except Exception:
            return 0.0

    return expr, latex_str, safe_callable
