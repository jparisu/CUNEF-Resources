from core.config import AppConfig


def test_shared_rng_is_reproducible_after_reset() -> None:
    config = AppConfig(seed=123)

    first_seed = config.draw_seed()
    first_point = config.sample_uniform_point()

    config.apply_seed()

    assert config.draw_seed() == first_seed
    assert config.sample_uniform_point() == first_point


def test_peek_seed_does_not_consume_rng_state() -> None:
    config = AppConfig(seed=77)

    preview = config.peek_seed()

    assert config.draw_seed() == preview
