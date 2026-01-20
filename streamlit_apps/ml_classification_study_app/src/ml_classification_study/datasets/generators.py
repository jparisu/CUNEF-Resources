from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ml_classification_study.types import ArgSpec
from .base import DatasetGenerator


class Gaussian2Class(DatasetGenerator):
    id = "gaussian"
    name = "Two Gaussians"
    description = "Two normally-distributed classes with configurable means and standard deviations."

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="class_ratio",
                label="Class 0 ratio",
                kind="float",
                default=0.5,
                min_value=0.05,
                max_value=0.95,
                step=0.05,
                help="Proportion of points in class 0.",
            ),
            ArgSpec(key="mu0_x", label="Class 0 mean X", kind="float", default=-1.0, min_value=-5.0, max_value=5.0, step=0.1),
            ArgSpec(key="mu0_y", label="Class 0 mean Y", kind="float", default=-1.0, min_value=-5.0, max_value=5.0, step=0.1),
            ArgSpec(key="sigma0", label="Class 0 std", kind="float", default=1.0, min_value=0.05, max_value=5.0, step=0.05),
            ArgSpec(key="mu1_x", label="Class 1 mean X", kind="float", default=1.0, min_value=-5.0, max_value=5.0, step=0.1),
            ArgSpec(key="mu1_y", label="Class 1 mean Y", kind="float", default=1.0, min_value=-5.0, max_value=5.0, step=0.1),
            ArgSpec(key="sigma1", label="Class 1 std", kind="float", default=1.0, min_value=0.05, max_value=5.0, step=0.05),
        ]

    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(kwargs["n_samples"])
        random_state = int(kwargs["random_state"])
        ratio = float(kwargs["class_ratio"])

        n0 = max(1, int(round(n_samples * ratio)))
        n1 = max(1, n_samples - n0)

        rng = np.random.default_rng(random_state)

        mu0 = np.array([float(kwargs["mu0_x"]), float(kwargs["mu0_y"])])
        mu1 = np.array([float(kwargs["mu1_x"]), float(kwargs["mu1_y"])])
        s0 = float(kwargs["sigma0"])
        s1 = float(kwargs["sigma1"])

        X0 = rng.normal(loc=mu0, scale=s0, size=(n0, 2))
        X1 = rng.normal(loc=mu1, scale=s1*2, size=(n1, 2))

        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

        # Shuffle to remove any ordering
        idx = rng.permutation(len(y))

        return X[idx], y[idx]


class Xor2D(DatasetGenerator):
    id = "xor"
    name = "XOR"
    description = "Classic XOR pattern (non-linearly separable), optionally with Gaussian noise."

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(key="range", label="Coordinate range", kind="float", default=2.0, min_value=0.5, max_value=10.0, step=0.25, help="Uniform sampling range [-r, r]."),
            ArgSpec(key="noise", label="Noise (std)", kind="float", default=0.25, min_value=0.0, max_value=2.0, step=0.05),
        ]

    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(kwargs["n_samples"])
        random_state = int(kwargs["random_state"])
        r = float(kwargs["range"])
        noise = float(kwargs["noise"])

        rng = np.random.default_rng(random_state)
        X = rng.uniform(low=-r, high=r, size=(n_samples, 2))

        # XOR label based on quadrant signs
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

        if noise > 0:
            X = X + rng.normal(loc=0.0, scale=noise, size=X.shape)

        return X, y


class NormalVsExponential(DatasetGenerator):
    id = "normal_vs_exp"
    name = "Normal vs Exponential"
    description = "Class 0 from a 2D Normal distribution; Class 1 from a shifted 2D Exponential distribution."

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(key="class_ratio", label="Class 0 ratio", kind="float", default=0.5, min_value=0.05, max_value=0.95, step=0.05),
            ArgSpec(key="normal_mean", label="Normal mean", kind="float", default=0.0, min_value=-5.0, max_value=5.0, step=0.1),
            ArgSpec(key="normal_std", label="Normal std", kind="float", default=1.0, min_value=0.05, max_value=5.0, step=0.05),
            ArgSpec(key="exp_scale", label="Exponential scale", kind="float", default=1.0, min_value=0.05, max_value=5.0, step=0.05),
            ArgSpec(key="exp_shift", label="Exponential shift", kind="float", default=-1.0, min_value=-5.0, max_value=5.0, step=0.1, help="Adds a constant shift to exponential samples (can move them into negative space)."),
        ]

    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(kwargs["n_samples"])
        random_state = int(kwargs["random_state"])
        ratio = float(kwargs["class_ratio"])

        n0 = max(1, int(round(n_samples * ratio)))
        n1 = max(1, n_samples - n0)

        rng = np.random.default_rng(random_state)

        m = float(kwargs["normal_mean"])
        s = float(kwargs["normal_std"])
        scale = float(kwargs["exp_scale"])
        shift = float(kwargs["exp_shift"])

        X0 = rng.normal(loc=m, scale=s, size=(n0, 2))
        X1 = rng.exponential(scale=scale, size=(n1, 2)) + shift

        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
        idx = rng.permutation(len(y))
        return X[idx], y[idx]


class RandomLinearSeparable(DatasetGenerator):
    id = "linear_sep"
    name = "Random Linear Separator"
    description = "Uniform points labeled by a random line; optional label noise and margin."

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(key="range", label="Coordinate range", kind="float", default=2.5, min_value=0.5, max_value=20.0, step=0.5),
            ArgSpec(key="margin", label="Margin", kind="float", default=0.2, min_value=0.0, max_value=3.0, step=0.05, help="Points near the boundary within this margin are re-sampled (increases separability)."),
            ArgSpec(key="label_noise", label="Label noise", kind="float", default=0.0, min_value=0.0, max_value=0.5, step=0.01, help="Probability of flipping a label."),
        ]

    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(kwargs["n_samples"])
        random_state = int(kwargs["random_state"])
        r = float(kwargs["range"])
        margin = float(kwargs["margin"])
        label_noise = float(kwargs["label_noise"])

        rng = np.random.default_rng(random_state)

        # Random line: w.x + b = 0
        w = rng.normal(size=(2,))
        w = w / (np.linalg.norm(w) + 1e-12)
        b = float(rng.uniform(-0.5, 0.5))

        X = np.empty((0, 2), dtype=float)
        y = np.empty((0,), dtype=int)

        # Rejection sampling to enforce margin
        attempts = 0
        while len(y) < n_samples and attempts < n_samples * 50:
            batch = rng.uniform(low=-r, high=r, size=(min(512, n_samples - len(y)), 2))
            scores = batch @ w + b
            keep = np.abs(scores) >= margin
            batch = batch[keep]
            scores = scores[keep]

            labels = (scores > 0).astype(int)

            if label_noise > 0:
                flip = rng.uniform(size=len(labels)) < label_noise
                labels = np.where(flip, 1 - labels, labels)

            X = np.vstack([X, batch]) if len(batch) else X
            y = np.concatenate([y, labels]) if len(batch) else y
            attempts += 1

        if len(y) < n_samples:
            # Fallback: fill without margin constraint
            remaining = n_samples - len(y)
            batch = rng.uniform(low=-r, high=r, size=(remaining, 2))
            scores = batch @ w + b
            labels = (scores > 0).astype(int)
            if label_noise > 0:
                flip = rng.uniform(size=len(labels)) < label_noise
                labels = np.where(flip, 1 - labels, labels)
            X = np.vstack([X, batch])
            y = np.concatenate([y, labels])

        # Augment X values second column x10
        X[:, 1] = X[:, 1] * 10.0

        # Shuffle
        idx = rng.permutation(len(y))
        meta = {"w": w, "b": b}

        # Return only X,y (meta handled elsewhere)
        return X[idx], y[idx]



class Circular2D(DatasetGenerator):
    id = "circular"
    name = "Circles (Inner vs Ring)"
    description = "Class 0 is an inner disk; Class 1 is an outer ring. Non-linearly separable."

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="class_ratio",
                label="Class 0 ratio",
                kind="float",
                default=0.5,
                min_value=0.05,
                max_value=0.95,
                step=0.05,
                help="Proportion of points in the inner disk (class 0).",
            ),
            ArgSpec(
                key="inner_radius",
                label="Inner radius (class 0)",
                kind="float",
                default=1.0,
                min_value=0.1,
                max_value=10.0,
                step=0.1,
            ),
            ArgSpec(
                key="ring_radius",
                label="Ring start radius (class 1)",
                kind="float",
                default=2.0,
                min_value=0.2,
                max_value=20.0,
                step=0.1,
                help="Inner radius of the outer ring.",
            ),
            ArgSpec(
                key="ring_width",
                label="Ring width (class 1)",
                kind="float",
                default=0.6,
                min_value=0.05,
                max_value=10.0,
                step=0.05,
                help="Thickness of the outer ring.",
            ),
            ArgSpec(
                key="noise",
                label="Noise (std)",
                kind="float",
                default=0.05,
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                help="Gaussian noise added to coordinates.",
            ),
        ]

    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(kwargs["n_samples"])
        random_state = int(kwargs["random_state"])
        ratio = float(kwargs["class_ratio"])
        inner_radius = float(kwargs["inner_radius"])
        ring_radius = float(kwargs["ring_radius"])
        ring_width = float(kwargs["ring_width"])
        noise = float(kwargs["noise"])

        # Ensure geometry is sane
        ring_radius = max(ring_radius, inner_radius + 1e-6)
        ring_width = max(ring_width, 1e-6)

        n0 = max(1, int(round(n_samples * ratio)))
        n1 = max(1, n_samples - n0)

        rng = np.random.default_rng(random_state)

        # Class 0: uniform in disk => radius ~ sqrt(U) * R
        a0 = rng.uniform(0.0, 2.0 * np.pi, size=n0)
        r0 = np.sqrt(rng.uniform(0.0, 1.0, size=n0)) * inner_radius
        X0 = np.column_stack([r0 * np.cos(a0), r0 * np.sin(a0)])

        # Class 1: uniform in ring area => radius ~ sqrt(U*(R2^2-R1^2)+R1^2)
        a1 = rng.uniform(0.0, 2.0 * np.pi, size=n1)
        r1_inner = ring_radius
        r1_outer = ring_radius + ring_width
        r1 = np.sqrt(rng.uniform(r1_inner**2, r1_outer**2, size=n1))
        X1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])

        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

        if noise > 0:
            X = X + rng.normal(0.0, noise, size=X.shape)

        idx = rng.permutation(len(y))
        return X[idx], y[idx]


class Spiral2D(DatasetGenerator):
    id = "spiral"
    name = "Spirals (Two Arms)"
    description = "Two interleaved spiral arms. Strongly non-linear decision boundary."

    @classmethod
    def arg_spec(cls) -> List[ArgSpec]:
        return [
            ArgSpec(
                key="turns",
                label="Turns",
                kind="float",
                default=2.5,
                min_value=0.5,
                max_value=10.0,
                step=0.25,
                help="Number of spiral turns.",
            ),
            ArgSpec(
                key="radius",
                label="Max radius",
                kind="float",
                default=3.0,
                min_value=0.5,
                max_value=20.0,
                step=0.25,
                help="Overall scale of the spiral.",
            ),
            ArgSpec(
                key="arm_separation",
                label="Arm separation (phase)",
                kind="float",
                default=np.pi,
                min_value=0.0,
                max_value=2.0 * np.pi,
                step=0.1,
                help="Phase shift between the two spiral arms (radians).",
            ),
            ArgSpec(
                key="noise",
                label="Noise (std)",
                kind="float",
                default=0.08,
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                help="Gaussian noise added to coordinates.",
            ),
        ]

    def generate(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(kwargs["n_samples"])
        random_state = int(kwargs["random_state"])
        turns = float(kwargs["turns"])
        radius = float(kwargs["radius"])
        arm_sep = float(kwargs["arm_separation"])
        noise = float(kwargs["noise"])

        rng = np.random.default_rng(random_state)

        n0 = n_samples // 2
        n1 = n_samples - n0

        # Parameter t controls angle and radius
        t0 = rng.uniform(0.0, 2.0 * np.pi * turns, size=n0)
        t1 = rng.uniform(0.0, 2.0 * np.pi * turns, size=n1)

        # Make radius grow with t, normalized to [0, radius]
        r0 = (t0 / (2.0 * np.pi * turns)) * radius
        r1 = (t1 / (2.0 * np.pi * turns)) * radius

        X0 = np.column_stack([r0 * np.cos(t0), r0 * np.sin(t0)])
        X1 = np.column_stack([r1 * np.cos(t1 + arm_sep), r1 * np.sin(t1 + arm_sep)])

        X = np.vstack([X0, X1])
        y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

        if noise > 0:
            X = X + rng.normal(0.0, noise, size=X.shape)

        idx = rng.permutation(len(y))
        return X[idx], y[idx]
