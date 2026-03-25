from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLES = 200
OUTPUT_DIR = Path(__file__).resolve().parent


def _make_dataframe(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": np.asarray(x1, dtype=float),
            "x2": np.asarray(x2, dtype=float),
            "y": np.asarray(y, dtype=int),
        }
    )


def _balanced_counts(n_samples: int) -> tuple[int, int]:
    first = n_samples // 2
    second = n_samples - first
    return first, second


def linear_easy(seed: int, n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_class_0, n_class_1 = _balanced_counts(n_samples)

    class_0 = rng.normal(loc=(-2.0, -1.8), scale=(0.55, 0.55), size=(n_class_0, 2))
    class_1 = rng.normal(loc=(2.0, 1.8), scale=(0.55, 0.55), size=(n_class_1, 2))

    data = np.vstack([class_0, class_1])
    labels = np.concatenate([np.zeros(n_class_0, dtype=int), np.ones(n_class_1, dtype=int)])
    order = rng.permutation(n_samples)

    return _make_dataframe(data[order, 0], data[order, 1], labels[order])


def linear_tilted(seed: int, n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights = np.array([0.9, -1.25])
    bias = 0.15
    normal = weights / np.linalg.norm(weights)
    tangent = np.array([normal[1], -normal[0]])
    boundary_anchor = -(bias / np.dot(weights, weights)) * weights
    n_class_0, n_class_1 = _balanced_counts(n_samples)

    def make_side(count: int, signed_distance: float) -> np.ndarray:
        tangent_component = rng.uniform(-3.0, 3.0, size=count)
        normal_component = rng.normal(loc=signed_distance, scale=0.35, size=count)
        points = (
            boundary_anchor
            + tangent_component[:, None] * tangent
            + normal_component[:, None] * normal
            + rng.normal(scale=0.08, size=(count, 2))
        )
        return points

    class_0 = make_side(n_class_0, signed_distance=-1.15)
    class_1 = make_side(n_class_1, signed_distance=1.15)

    data = np.vstack([class_0, class_1])
    labels = np.concatenate([np.zeros(n_class_0, dtype=int), np.ones(n_class_1, dtype=int)])
    order = rng.permutation(n_samples)

    return _make_dataframe(data[order, 0], data[order, 1], labels[order])


def xor(seed: int, n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    quadrant_centers = np.array(
        [
            (-1.5, -1.5),
            (-1.5, 1.5),
            (1.5, -1.5),
            (1.5, 1.5),
        ]
    )
    quadrant_labels = np.array([0, 1, 1, 0], dtype=int)
    counts = np.full(4, n_samples // 4, dtype=int)
    counts[: n_samples % 4] += 1

    blocks = []
    labels = []
    for center, label, count in zip(quadrant_centers, quadrant_labels, counts):
        points = rng.normal(loc=center, scale=(0.45, 0.45), size=(count, 2))
        blocks.append(points)
        labels.append(np.full(count, label, dtype=int))

    data = np.vstack(blocks)
    target = np.concatenate(labels)
    order = rng.permutation(n_samples)

    return _make_dataframe(data[order, 0], data[order, 1], target[order])


def circles_centered(seed: int, n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_inner, n_outer = _balanced_counts(n_samples)

    inner_angles = rng.uniform(0.0, 2.0 * np.pi, size=n_inner)
    inner_radii = 1.15 * np.sqrt(rng.uniform(0.0, 1.0, size=n_inner))
    inner = np.column_stack([inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles)])

    outer_angles = rng.uniform(0.0, 2.0 * np.pi, size=n_outer)
    outer_radii = np.clip(rng.normal(loc=2.4, scale=0.12, size=n_outer), 2.0, 2.8)
    outer = np.column_stack([outer_radii * np.cos(outer_angles), outer_radii * np.sin(outer_angles)])

    data = np.vstack([inner, outer])
    labels = np.concatenate([np.zeros(n_inner, dtype=int), np.ones(n_outer, dtype=int)])
    order = rng.permutation(n_samples)

    return _make_dataframe(data[order, 0], data[order, 1], labels[order])


DATASET_BUILDERS = {
    "linear_easy": linear_easy,
    "linear_tilted": linear_tilted,
    "xor": xor,
    "circles_centered": circles_centered,
}


def save_dataset(
    name: str,
    dataframe: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    directory = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"{name}.csv"
    dataframe.to_csv(output_path, index=False)
    return output_path


def generate_all_datasets(
    seed: int = 42,
    n_samples: int = DEFAULT_SAMPLES,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(DATASET_BUILDERS))

    saved_paths: dict[str, Path] = {}
    for (name, builder), child_sequence in zip(DATASET_BUILDERS.items(), child_sequences):
        child_seed = int(child_sequence.generate_state(1)[0])
        dataframe = builder(seed=child_seed, n_samples=n_samples)
        saved_paths[name] = save_dataset(name, dataframe, output_dir=output_dir)

    return saved_paths


if __name__ == "__main__":
    created_files = generate_all_datasets()
    for dataset_name, path in created_files.items():
        print(f"{dataset_name}: {path}")
