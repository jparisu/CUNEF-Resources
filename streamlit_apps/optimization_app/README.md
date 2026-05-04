# Optimization Playground

Interactive Streamlit app for exploring optimization algorithms on 2D objective functions.

## Features

- Predefined benchmark functions with optional analytical gradients
- Random symbolic function generation driven by a shared app-level RNG
- Manual, random, local-search, gradient, and evolutive optimization modes
- Plotly visualization for contours, trajectories, highlighted points, and population previews

## Function Sources

- `Predefined`: curated functions with descriptions and, when available, derivatives
- `Random`: symbolic functions generated from a reproducible seed stream and a configurable number of compositions

Custom Python code input is intentionally disabled.

## Local Search

Local search offers two fixed locality presets:

- `Manhattan (4 neighbors)`: axis-aligned neighbors
- `Euclidean (6 neighbors)`: six evenly spaced points on a circle

The `Locality size` parameter controls the step length or radius.

## Randomness

The project uses a single shared random generator stored in `AppConfig`.

- Resetting the configured seed resets the full app RNG stream
- Random functions draw a per-function seed from that shared stream
- Random points, local-search initial points, gradient initial points, and evolutive search all consume the same RNG stream

## Installation

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e .[dev]
```

## Run

```bash
streamlit run app.py
```

## Tests

```bash
python -m pytest
```

## Project Layout

- [app.py](/home/jparisu/projects/cunef/repositories/CUNEF-Resources/streamlit_apps/optimization_app/app.py)
- [core](/home/jparisu/projects/cunef/repositories/CUNEF-Resources/streamlit_apps/optimization_app/core)
- [optimizers](/home/jparisu/projects/cunef/repositories/CUNEF-Resources/streamlit_apps/optimization_app/optimizers)
- [ui](/home/jparisu/projects/cunef/repositories/CUNEF-Resources/streamlit_apps/optimization_app/ui)
