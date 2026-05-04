import pytest

from core.config import AppConfig
from core.function_handler import PREDEFINED_FUNCTIONS, FunctionHandler, FunctionSource
from optimizers.local_search import LocalSearchOptimizer


def test_function_handler_only_uses_predefined_and_random_sources() -> None:
    handler = FunctionHandler()

    handler.set_predefined("Inverted Sphere")
    assert handler.source == FunctionSource.PREDEFINED

    handler.set_random(seed=10, nf=4)
    assert handler.source == FunctionSource.RANDOM
    assert handler.random_seed == 10
    assert handler.random_nf == 4


def test_new_predefined_functions_are_available_and_evaluable() -> None:
    assert "Triple Peaks" in PREDEFINED_FUNCTIONS
    assert "Bilinear Saddle" in PREDEFINED_FUNCTIONS

    handler = FunctionHandler()

    handler.set_predefined("Triple Peaks")
    assert handler.has_derivative()
    assert handler.evaluate(3.0, -1.0) > 0.0

    handler.set_predefined("Bilinear Saddle")
    assert handler.evaluate(2.0, 3.0) == pytest.approx(6.0)
    assert handler.evaluate_gradient(2.0, 3.0) == pytest.approx((3.0, 2.0))


def test_local_search_manhattan_size_controls_neighbors() -> None:
    config = AppConfig()
    optimizer = LocalSearchOptimizer()
    optimizer.set_initial_point(1.0, 2.0)
    optimizer.configure_locality("Manhattan (4 neighbors)", 2.5)

    assert optimizer.suggest_locality_points(config) == [
        (3.5, 2.0),
        (-1.5, 2.0),
        (1.0, 4.5),
        (1.0, -0.5),
    ]


def test_local_search_euclidean_size_controls_radius() -> None:
    config = AppConfig()
    optimizer = LocalSearchOptimizer()
    optimizer.set_initial_point(0.0, 0.0)
    optimizer.configure_locality("Euclidean (6 neighbors)", 2.0)

    points = optimizer.suggest_locality_points(config)

    assert len(points) == 6
    assert points[0] == (2.0, 0.0)
    assert points[3][0] == pytest.approx(-2.0)
    assert points[3][1] == pytest.approx(0.0)
