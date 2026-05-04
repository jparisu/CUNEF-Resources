from __future__ import annotations

import streamlit as st

from core.config import AppConfig
from core.function_handler import FunctionHandler
from core.solutions_handler import SolutionsHandler
from optimizers.evolutive_search import EvolutiveSearchOptimizer
from optimizers.gradient_ascent import GradientAscentOptimizer
from optimizers.local_search import LocalSearchOptimizer
from optimizers.manual import ManualOptimizer
from optimizers.random import RandomOptimizer


def render_optimizer_sidebar(
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
) -> None:
    with st.sidebar:
        st.title("Optimizer")

        if not function_handler.is_ready():
            st.warning("Select a function first.")
            return

        manual: ManualOptimizer = st.session_state["optimizer"]
        random_opt: RandomOptimizer = st.session_state["random_optimizer"]
        local_search: LocalSearchOptimizer = st.session_state["local_search_optimizer"]
        gradient_opt: GradientAscentOptimizer = st.session_state["gradient_optimizer"]
        evolutive_opt: EvolutiveSearchOptimizer = st.session_state["evolutive_optimizer"]

        selected_optimizer = st.radio(
            "Optimizer",
            options=["Manual", "Random", "Local search", "Gradient", "Evolutive search"],
            horizontal=True,
            label_visibility="collapsed",
            key="selected_optimizer",
        )
        st.session_state["active_optimizer_tab"] = selected_optimizer

        if selected_optimizer == "Manual":
            st.caption("Pick exact x1/x2 manually using number inputs and sliders.")
            _render_manual_controls(manual, function_handler, solutions_handler, config)
            st.session_state["local_overlay"] = None
            st.session_state["gradient_overlay"] = None
            st.session_state["evolutive_overlay"] = None
        elif selected_optimizer == "Random":
            st.caption("Generate random points in-range using the shared app RNG.")
            _render_random_controls(random_opt, function_handler, solutions_handler, config)
            st.session_state["local_overlay"] = None
            st.session_state["gradient_overlay"] = None
            st.session_state["evolutive_overlay"] = None
        elif selected_optimizer == "Local search":
            st.caption("Search with fixed locality presets around a current point.")
            _render_local_search_controls(local_search, function_handler, solutions_handler, config)
            st.session_state["gradient_overlay"] = None
            st.session_state["evolutive_overlay"] = None
        elif selected_optimizer == "Gradient":
            st.caption("Use the derivative to move in gradient direction.")
            _render_gradient_controls(gradient_opt, function_handler, solutions_handler, config)
            st.session_state["local_overlay"] = None
            st.session_state["evolutive_overlay"] = None
        else:
            st.caption("Population search with mutation, crossover, and selection.")
            _render_evolutive_controls(evolutive_opt, function_handler, solutions_handler, config)
            st.session_state["local_overlay"] = None
            st.session_state["gradient_overlay"] = None

        st.divider()
        _render_stats(solutions_handler)


def _sync_point_inputs(x_key: str, y_key: str, point: tuple[float, float]) -> None:
    st.session_state[x_key] = float(point[0])
    st.session_state[y_key] = float(point[1])


def _queue_point_inputs(prefix: str, point: tuple[float, float]) -> None:
    st.session_state[f"{prefix}_pending_point"] = (float(point[0]), float(point[1]))


def _apply_queued_point_inputs(prefix: str) -> None:
    pending_key = f"{prefix}_pending_point"
    pending = st.session_state.pop(pending_key, None)
    if pending is None:
        return
    _sync_point_inputs(f"{prefix}_init_x1", f"{prefix}_init_x2", pending)


def _set_status_message(key: str, message: str | None) -> None:
    if message:
        st.session_state[key] = message
    else:
        st.session_state.pop(key, None)


def _clear_evolutive_preview() -> None:
    st.session_state["evo_show_preview"] = False


def _render_evolutive_controls(
    optimizer: EvolutiveSearchOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
) -> None:
    n = st.number_input(
        "Population size (N)",
        min_value=2,
        max_value=200,
        value=int(optimizer.population_size),
        step=1,
        key="evo_n",
    )
    sigma = st.number_input(
        "Mutation sigma",
        min_value=0.000001,
        max_value=100.0,
        value=float(optimizer.mutation_sigma),
        step=0.05,
        key="evo_sigma",
    )
    temp = st.number_input(
        "Crossover temperature",
        min_value=0.000001,
        max_value=100.0,
        value=float(optimizer.crossover_temperature),
        step=0.05,
        key="evo_temp",
    )
    method = st.selectbox(
        "Selection method",
        ["Best", "Roulette", "Tournament"],
        index=["Best", "Roulette", "Tournament"].index(optimizer.selection_method),
        key="evo_method",
    )
    selection_temp = st.number_input(
        "Selection temperature",
        min_value=0.000001,
        max_value=100.0,
        value=float(optimizer.selection_temperature),
        step=0.05,
        key="evo_selection_temp",
        help="Lower values make Roulette/Tournament greedier; higher values make them more random.",
    )
    optimizer.configure(int(n), float(sigma), float(temp), float(selection_temp), method)

    c1, c2, c3 = st.columns(3)
    if c1.button("First population", type="primary", use_container_width=True, key="evo_first"):
        _clear_evolutive_preview()
        population = optimizer.initialize_population(function_handler, config)
        for individual in population:
            solutions_handler.add_point(individual.x1, individual.x2, individual.f_value)
        st.rerun()

    has_population = len(optimizer.population) > 0
    if c2.button(
        "Next generation", use_container_width=True, key="evo_next", disabled=not has_population
    ):
        _clear_evolutive_preview()
        _run_evolutive_generations(optimizer, function_handler, solutions_handler, config, 1)
    if c3.button(
        "5 generations", use_container_width=True, key="evo_5", disabled=not has_population
    ):
        _clear_evolutive_preview()
        _run_evolutive_generations(optimizer, function_handler, solutions_handler, config, 5)
    if st.button(
        "10 generations", use_container_width=True, key="evo_10", disabled=not has_population
    ):
        _clear_evolutive_preview()
        _run_evolutive_generations(optimizer, function_handler, solutions_handler, config, 10)

    if has_population:
        if st.button(
            "Show next generation",
            use_container_width=True,
            key="evo_preview",
        ):
            st.session_state["evo_show_preview"] = True
        preview = None
        if st.session_state.get("evo_show_preview", False):
            preview = optimizer.preview_next_generation(function_handler, config)
        st.session_state["evolutive_overlay"] = {
            "population": [(p.slot, p.x1, p.x2, p.f_value) for p in optimizer.population],
            "preview": preview,
            "history": optimizer.history,
            "selection_method": optimizer.selection_method,
        }
        st.caption(f"Current generation: {optimizer.iteration}")
    else:
        st.session_state["evolutive_overlay"] = None
        _clear_evolutive_preview()


def _run_evolutive_generations(
    optimizer: EvolutiveSearchOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
    steps: int,
) -> None:
    for _ in range(steps):
        population = optimizer.advance_generation(function_handler, config)
        if population is None:
            break
        for individual in population:
            solutions_handler.add_point(individual.x1, individual.x2, individual.f_value)
    st.rerun()


def _render_manual_controls(
    optimizer: ManualOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
) -> None:
    x_min, x_max = config.get_range()
    step = 1.0 if config.integer_mode else 0.01

    default_x1 = float((x_min + x_max) / 2)
    default_x2 = float((x_min + x_max) / 2)

    x1 = st.number_input(
        "x1 value",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(st.session_state.get("manual_x1", default_x1)),
        step=step,
    )
    x1 = st.slider(
        "x1 slider",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(x1),
        step=step,
        key="manual_x1",
    )

    x2 = st.number_input(
        "x2 value",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(st.session_state.get("manual_x2", default_x2)),
        step=step,
    )
    x2 = st.slider(
        "x2 slider",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(x2),
        step=step,
        key="manual_x2",
    )

    optimizer.set_point(x1, x2)
    f_val = function_handler.evaluate(x1, x2)
    st.metric("F(x1, x2)", f"{f_val:.6f}")

    col_add, col_undo = st.columns(2)
    with col_add:
        if st.button("Add point", type="primary", use_container_width=True, key="manual_add"):
            point = optimizer.suggest_next(function_handler, solutions_handler, config)
            if point is not None:
                x1c, x2c = config.clamp(point[0], point[1])
                solutions_handler.add_point(x1c, x2c, function_handler.evaluate(x1c, x2c))
                st.rerun()
    with col_undo:
        if st.button(
            "Undo",
            use_container_width=True,
            disabled=solutions_handler.is_empty(),
            key="manual_undo",
        ):
            solutions_handler.remove_last()
            st.rerun()


def _render_random_controls(
    optimizer: RandomOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
) -> None:
    if st.button("New random point", type="primary", use_container_width=True, key="random_new"):
        point = optimizer.suggest_next(function_handler, solutions_handler, config)
        if point is not None:
            solutions_handler.add_point(
                point[0], point[1], function_handler.evaluate(point[0], point[1])
            )
            st.rerun()


def _render_local_search_controls(
    optimizer: LocalSearchOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
) -> None:
    _apply_queued_point_inputs("local")
    x_min, x_max = config.get_range()
    step = 1.0 if config.integer_mode else 0.01
    if message := st.session_state.get("local_status_message"):
        st.info(message)

    current = optimizer.current_point
    if current is None:
        current = optimizer.suggest_random_initial_point(config)
        _sync_point_inputs("local_init_x1", "local_init_x2", current)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Random initial", use_container_width=True, key="local_random_initial"):
            current = optimizer.suggest_random_initial_point(config)
            _sync_point_inputs("local_init_x1", "local_init_x2", current)
            _set_status_message("local_status_message", None)
    with c2:
        if st.button(
            "Use latest point",
            use_container_width=True,
            key="local_latest",
            disabled=solutions_handler.is_empty(),
        ):
            latest = solutions_handler.get_latest()
            if latest is not None:
                optimizer.set_initial_point(latest.x1, latest.x2)
                current = (latest.x1, latest.x2)
                _sync_point_inputs("local_init_x1", "local_init_x2", current)
                _set_status_message("local_status_message", None)

    x1 = st.number_input(
        "Initial x1",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(st.session_state.get("local_init_x1", current[0])),
        step=step,
        key="local_init_x1",
    )
    x2 = st.number_input(
        "Initial x2",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(st.session_state.get("local_init_x2", current[1])),
        step=step,
        key="local_init_x2",
    )
    optimizer.set_initial_point(x1, x2)
    _set_status_message("local_status_message", None)

    preset_options = list(LocalSearchOptimizer.PRESETS)
    default_preset = optimizer.preset if optimizer.preset in preset_options else preset_options[0]
    preset = st.selectbox(
        "Locality preset",
        preset_options,
        index=preset_options.index(default_preset),
        key="local_preset",
    )
    size = st.number_input(
        "Locality size",
        min_value=0.000001,
        max_value=100.0,
        value=float(optimizer.size),
        step=0.1,
        key="local_size",
        help="Radius for Euclidean locality and step size for Manhattan locality.",
    )
    optimizer.configure_locality(preset, size)

    candidates = optimizer.suggest_locality_points(config)
    st.caption(f"Candidate points this step: {len(candidates)}")

    current_point = optimizer.current_point
    current_with_f = None
    if current_point is not None:
        current_with_f = (
            current_point[0],
            current_point[1],
            function_handler.evaluate(current_point[0], current_point[1]),
        )
    candidates_with_f = [
        (point[0], point[1], function_handler.evaluate(point[0], point[1])) for point in candidates
    ]

    st.session_state["local_overlay"] = {
        "current": current_with_f,
        "candidates": candidates_with_f,
    }

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("1 step", use_container_width=True, key="local_1"):
        _run_local_steps(optimizer, function_handler, solutions_handler, config, 1)
    if col2.button("5 steps", use_container_width=True, key="local_5"):
        _run_local_steps(optimizer, function_handler, solutions_handler, config, 5)
    if col3.button("20 steps", use_container_width=True, key="local_20"):
        _run_local_steps(optimizer, function_handler, solutions_handler, config, 20)
    if col4.button("100 steps", use_container_width=True, key="local_100"):
        _run_local_steps(optimizer, function_handler, solutions_handler, config, 100)


def _render_gradient_controls(
    optimizer: GradientAscentOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
) -> None:
    _apply_queued_point_inputs("grad")
    x_min, x_max = config.get_range()
    point_step = 1.0 if config.integer_mode else 0.01
    if message := st.session_state.get("gradient_status_message"):
        st.info(message)

    current = optimizer.current_point
    if current is None:
        current = optimizer.suggest_random_initial_point(config)
        _sync_point_inputs("grad_init_x1", "grad_init_x2", current)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Random initial", use_container_width=True, key="grad_random_initial"):
            current = optimizer.suggest_random_initial_point(config)
            _sync_point_inputs("grad_init_x1", "grad_init_x2", current)
            _set_status_message("gradient_status_message", None)
    with c2:
        if st.button(
            "Use latest point",
            use_container_width=True,
            key="grad_latest",
            disabled=solutions_handler.is_empty(),
        ):
            latest = solutions_handler.get_latest()
            if latest is not None:
                optimizer.set_current_point(latest.x1, latest.x2)
                current = (latest.x1, latest.x2)
                _sync_point_inputs("grad_init_x1", "grad_init_x2", current)
                _set_status_message("gradient_status_message", None)

    x1 = st.number_input(
        "Current x1",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(st.session_state.get("grad_init_x1", current[0])),
        step=point_step,
        key="grad_init_x1",
    )
    x2 = st.number_input(
        "Current x2",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(st.session_state.get("grad_init_x2", current[1])),
        step=point_step,
        key="grad_init_x2",
    )
    optimizer.set_current_point(x1, x2)
    _set_status_message("gradient_status_message", None)

    lr = st.number_input(
        "Learning rate",
        min_value=0.000001,
        max_value=100.0,
        value=float(optimizer.step_size),
        step=0.01,
        key="grad_lr",
    )
    optimizer.set_learning_rate(lr)

    has_derivative = function_handler.has_derivative()
    if not has_derivative:
        st.warning("The selected function has no derivative, so Gradient is unavailable.")
        st.session_state["gradient_overlay"] = None
    else:
        move = optimizer.suggest_next_with_gradient(function_handler, config)
        if move is not None:
            current_point, next_point, gradient = move
            current_f = function_handler.evaluate(current_point[0], current_point[1])
            next_f = function_handler.evaluate(next_point[0], next_point[1])
            st.session_state["gradient_overlay"] = {
                "current": (current_point[0], current_point[1], current_f),
                "next": (next_point[0], next_point[1], next_f),
                "gradient": gradient,
            }
            st.caption(f"Suggested next point F: {next_f:.6f}")

    col1, col2, col3, col4 = st.columns(4)
    disabled = not has_derivative
    if col1.button("1 step", use_container_width=True, key="grad_1", disabled=disabled):
        _run_gradient_steps(optimizer, function_handler, solutions_handler, config, 1)
    if col2.button("5 steps", use_container_width=True, key="grad_5", disabled=disabled):
        _run_gradient_steps(optimizer, function_handler, solutions_handler, config, 5)
    if col3.button("20 steps", use_container_width=True, key="grad_20", disabled=disabled):
        _run_gradient_steps(optimizer, function_handler, solutions_handler, config, 20)
    if col4.button("100 steps", use_container_width=True, key="grad_100", disabled=disabled):
        _run_gradient_steps(optimizer, function_handler, solutions_handler, config, 100)


def _run_local_steps(
    optimizer: LocalSearchOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
    n_steps: int,
) -> None:
    steps_taken = 0
    for _ in range(n_steps):
        point = optimizer.suggest_next(function_handler, solutions_handler, config)
        if point is None:
            break
        solutions_handler.add_point(
            point[0], point[1], function_handler.evaluate(point[0], point[1])
        )
        steps_taken += 1
    current = optimizer.current_point
    if current is not None:
        _queue_point_inputs("local", current)
    if steps_taken < n_steps and current is not None:
        _set_status_message(
            "local_status_message",
            f"Local maxima found in ({current[0]:.4f}, {current[1]:.4f}) in {steps_taken} steps",
        )
    else:
        _set_status_message("local_status_message", None)
    st.rerun()


def _run_gradient_steps(
    optimizer: GradientAscentOptimizer,
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
    n_steps: int,
) -> None:
    steps_taken = 0
    for _ in range(n_steps):
        point = optimizer.suggest_next(function_handler, solutions_handler, config)
        if point is None:
            break
        solutions_handler.add_point(
            point[0], point[1], function_handler.evaluate(point[0], point[1])
        )
        steps_taken += 1
    current = optimizer.current_point
    if current is not None:
        _queue_point_inputs("grad", current)
    if steps_taken < n_steps and current is not None:
        _set_status_message(
            "gradient_status_message",
            f"Local maxima found in ({current[0]:.4f}, {current[1]:.4f}) in {steps_taken} steps",
        )
    else:
        _set_status_message("gradient_status_message", None)
    st.rerun()


def _render_stats(solutions_handler: SolutionsHandler) -> None:
    if solutions_handler.is_empty():
        st.caption("No points added yet.")
        return

    best = solutions_handler.get_best()
    latest = solutions_handler.get_latest()

    st.markdown("**Progress**")
    st.metric("Points added", solutions_handler.count())
    if best:
        st.metric("Best F", f"{best.f_value:.6f}", help=f"x1={best.x1:.4f}, x2={best.x2:.4f}")
    if latest and latest is not best:
        st.metric(
            "Latest F", f"{latest.f_value:.6f}", help=f"x1={latest.x1:.4f}, x2={latest.x2:.4f}"
        )
