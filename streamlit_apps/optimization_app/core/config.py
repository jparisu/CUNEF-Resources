from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AppConfig:
    """Global configuration shared across all components of the optimization app."""

    x_min: float = -10.0
    x_max: float = 10.0
    seed: int = 0
    integer_mode: bool = False
    grid_resolution: int = 100
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.apply_seed()

    def validate(self) -> bool:
        """Return True if the configuration is internally consistent."""
        return self.x_min < self.x_max and self.grid_resolution > 0

    def clamp(self, x1: float, x2: float) -> tuple[float, float]:
        """Clamp (x1, x2) to [x_min, x_max]^2, rounding to int if integer_mode is set."""
        if self.integer_mode:
            x1, x2 = round(x1), round(x2)
        x1 = max(self.x_min, min(self.x_max, x1))
        x2 = max(self.x_min, min(self.x_max, x2))
        return x1, x2

    def is_in_domain(self, x1: float, x2: float) -> bool:
        """Return True if the point lies within [x_min, x_max]^2."""
        return self.x_min <= x1 <= self.x_max and self.x_min <= x2 <= self.x_max

    def get_range(self) -> tuple[float, float]:
        """Return (x_min, x_max)."""
        return (self.x_min, self.x_max)

    def apply_seed(self) -> None:
        """Reset the shared app-level random generator to the configured seed."""
        self._rng = np.random.default_rng(self.seed)

    @property
    def rng(self) -> np.random.Generator:
        """Return the shared app-level random generator."""
        return self._rng

    def clone_rng(self) -> np.random.Generator:
        """Return a generator with the same state as the shared RNG."""
        clone = np.random.default_rng()
        clone.bit_generator.state = copy.deepcopy(self._rng.bit_generator.state)
        return clone

    def draw_seed(self) -> int:
        """Draw a deterministic per-operation seed from the shared RNG."""
        return int(self._rng.integers(0, 2**32, dtype=np.uint32))

    def peek_seed(self) -> int:
        """Preview the next derived seed without consuming shared RNG state."""
        return int(self.clone_rng().integers(0, 2**32, dtype=np.uint32))

    def sample_uniform_point(self) -> tuple[float, float]:
        """Sample one point uniformly from the configured domain."""
        x1 = float(self._rng.uniform(self.x_min, self.x_max))
        x2 = float(self._rng.uniform(self.x_min, self.x_max))
        return self.clamp(x1, x2)
