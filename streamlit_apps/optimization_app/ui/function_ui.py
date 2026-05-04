"""Function Handler UI — top panel with view/edit modes."""

from __future__ import annotations

import streamlit as st

from core.config import AppConfig
from core.function_handler import PREDEFINED_FUNCTIONS, FunctionHandler, FunctionSource

_EDITING_KEY = "fn_editing"
_KEY_SOURCE = "fn_source_radio"
_KEY_PRE_SELECT = "pre_select"
_KEY_RNG_NF = "rng_nf"
_KEY_XMIN = "cfg_xmin"
_KEY_XMAX = "cfg_xmax"
_KEY_SEED = "cfg_seed"
_KEY_INT = "cfg_int"

_SOURCE_LABELS = ["Predefined", "Random"]
_SOURCE_TO_ENUM = {
    "Predefined": FunctionSource.PREDEFINED,
    "Random": FunctionSource.RANDOM,
}
_ENUM_TO_SOURCE = {value: key for key, value in _SOURCE_TO_ENUM.items()}


def render_function_handler(function_handler: FunctionHandler, config: AppConfig) -> None:
    """Render the Function & Configuration panel."""
    if _EDITING_KEY not in st.session_state:
        st.session_state[_EDITING_KEY] = not function_handler.is_ready()

    if (
        st.session_state[_EDITING_KEY]
        and _KEY_SOURCE not in st.session_state
        and function_handler.is_ready()
        and function_handler.source in _ENUM_TO_SOURCE
    ):
        st.session_state[_KEY_SOURCE] = _ENUM_TO_SOURCE[function_handler.source]

    if function_handler.is_ready() and not st.session_state[_EDITING_KEY]:
        _render_view_mode(function_handler)
    else:
        _render_edit_mode(function_handler, config)


def _render_view_mode(function_handler: FunctionHandler) -> None:
    with st.container(border=True):
        col_info, col_btn = st.columns([10, 1])
        with col_info:
            source_label = _ENUM_TO_SOURCE.get(function_handler.source, "")
            if function_handler.source == FunctionSource.RANDOM:
                source_label = (
                    f"Random  (draw seed={function_handler.random_seed}, "
                    f"N={function_handler.random_nf})"
                )
            st.caption(f"Function · {source_label}")
            st.latex(r"F(x_1,x_2)=" + function_handler.get_latex())
            if function_handler.get_derivative_latex():
                st.latex(function_handler.get_derivative_latex())
        with col_btn:
            st.write("")
            if st.button("✏️", key="fn_edit_btn", help="Edit function & configuration"):
                st.session_state[_EDITING_KEY] = True
                st.rerun()


def _render_edit_mode(function_handler: FunctionHandler, config: AppConfig) -> None:
    with st.container(border=True):
        st.markdown("#### ⚙️ Function & Configuration")
        source = st.radio(
            "Source",
            options=_SOURCE_LABELS,
            horizontal=True,
            key=_KEY_SOURCE,
            label_visibility="collapsed",
        )

        st.divider()

        if source == "Predefined":
            _render_predefined_section(function_handler)
        else:
            _render_random_section(function_handler, config)

        st.divider()
        _render_config_section(config)
        st.divider()

        col_apply, col_cancel = st.columns(2)
        with col_apply:
            if st.button("✅ Apply", type="primary", use_container_width=True, key="fn_apply"):
                _apply(function_handler, config)
        with col_cancel:
            if function_handler.is_ready() and st.button(
                "✖ Cancel",
                use_container_width=True,
                key="fn_cancel",
            ):
                st.session_state[_EDITING_KEY] = False
                st.rerun()


def _render_predefined_section(function_handler: FunctionHandler) -> None:
    names = list(PREDEFINED_FUNCTIONS.keys())
    current = (
        function_handler.predefined_name
        if function_handler.source == FunctionSource.PREDEFINED
        else names[0]
    )
    idx = names.index(current) if current in names else 0

    selected = st.selectbox("Function", options=names, index=idx, key=_KEY_PRE_SELECT)
    if selected:
        definition = PREDEFINED_FUNCTIONS[selected]
        st.caption(definition.description)
        with st.expander("Preview", expanded=True):
            st.latex(r"F(x_1,x_2)=" + definition.latex)
            if definition.derivative_latex:
                st.latex(definition.derivative_latex)


def _render_random_section(function_handler: FunctionHandler, config: AppConfig) -> None:
    st.markdown("Generate a random symbolic function from compositions of `sin`, `cos`, and `exp`.")

    current_nf = (
        function_handler.random_nf
        if function_handler.source == FunctionSource.RANDOM and function_handler.random_nf
        else 5
    )
    nf = st.slider("N (compositions)", min_value=1, max_value=20, value=current_nf, key=_KEY_RNG_NF)

    preview_seed = config.peek_seed()
    st.caption(f"Next generated function draw seed: `{preview_seed}`")
    _show_random_preview(preview_seed, nf)

    if function_handler.source == FunctionSource.RANDOM and function_handler.get_latex():
        st.caption("Current active random function")
        st.latex(r"F(x_1,x_2)=" + function_handler.get_latex())


def _show_random_preview(seed: int, nf: int) -> None:
    """Show the LaTeX preview for the next derived random-function seed."""
    from core import random_function

    try:
        _expr, latex_str, _fn = random_function.generate(seed, nf)
        st.latex(r"F(x_1,x_2)=" + latex_str)
    except Exception as exc:
        st.error(f"Generation failed: {exc}")


def _render_config_section(config: AppConfig) -> None:
    st.markdown("**Configuration**")
    col_min, col_max, col_seed, col_int = st.columns([2, 2, 2, 1])
    with col_min:
        st.number_input("MIN", value=config.x_min, step=1.0, format="%.1f", key=_KEY_XMIN)
    with col_max:
        st.number_input("MAX", value=config.x_max, step=1.0, format="%.1f", key=_KEY_XMAX)
    with col_seed:
        st.number_input("SEED", value=config.seed, step=1, min_value=0, key=_KEY_SEED)
    with col_int:
        st.checkbox(
            "INT",
            value=config.integer_mode,
            key=_KEY_INT,
            help="Restrict x₁ and x₂ to integer values",
        )

    if st.button(
        "🔄 Reset RNG",
        key="cfg_reset_seed",
        help="Reset the shared random generator to the configured seed",
    ):
        config.seed = int(st.session_state.get(_KEY_SEED, config.seed))
        config.apply_seed()
        st.success(f"Shared random generator reset to seed {config.seed}.")


def _apply(function_handler: FunctionHandler, config: AppConfig) -> None:
    """Apply function/config changes and reset any stale optimization state."""
    new_min = float(st.session_state.get(_KEY_XMIN, config.x_min))
    new_max = float(st.session_state.get(_KEY_XMAX, config.x_max))
    new_seed = int(st.session_state.get(_KEY_SEED, config.seed))
    new_int = bool(st.session_state.get(_KEY_INT, config.integer_mode))
    source = st.session_state.get(_KEY_SOURCE, "Predefined")

    if new_min >= new_max:
        st.error("MIN must be less than MAX.")
        return

    if source == "Predefined":
        selected_name = st.session_state.get(_KEY_PRE_SELECT)
        if not selected_name:
            st.error("Select a predefined function.")
            return
        if selected_name not in PREDEFINED_FUNCTIONS:
            st.error(f"Unknown predefined function: {selected_name}")
            return

    config.x_min = new_min
    config.x_max = new_max
    config.integer_mode = new_int
    if new_seed != config.seed:
        config.seed = new_seed
        config.apply_seed()

    if source == "Predefined":
        function_handler.set_predefined(st.session_state[_KEY_PRE_SELECT])
    elif source == "Random":
        nf = int(st.session_state.get(_KEY_RNG_NF, 5))
        function_handler.set_random(config.draw_seed(), nf)
    else:
        st.error(f"Unsupported function source: {source}")
        return

    _reset_exploration_state()
    st.session_state[_EDITING_KEY] = False
    st.rerun()


def _reset_exploration_state() -> None:
    solutions_handler = st.session_state.get("solutions_handler")
    if solutions_handler is not None:
        solutions_handler.reset()

    for key in [
        "optimizer",
        "random_optimizer",
        "local_search_optimizer",
        "gradient_optimizer",
        "evolutive_optimizer",
    ]:
        optimizer = st.session_state.get(key)
        if optimizer is not None and hasattr(optimizer, "reset"):
            optimizer.reset()

    for key in [
        "local_overlay",
        "gradient_overlay",
        "evolutive_overlay",
        "_last_clicked_sol",
        "manual_x1",
        "manual_x2",
        "local_init_x1",
        "local_init_x2",
        "grad_init_x1",
        "grad_init_x2",
    ]:
        st.session_state.pop(key, None)
