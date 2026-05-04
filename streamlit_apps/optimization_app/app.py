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
from ui.function_ui import render_function_handler
from ui.optimizer_ui import render_optimizer_sidebar
from ui.plot_ui import render_solutions_plot


def _init_session_state() -> None:
    """Initialize all session-state objects on the first run."""
    if "config" not in st.session_state:
        st.session_state["config"] = AppConfig()

    if "function_handler" not in st.session_state:
        st.session_state["function_handler"] = FunctionHandler()

    if "solutions_handler" not in st.session_state:
        st.session_state["solutions_handler"] = SolutionsHandler()

    if "optimizer" not in st.session_state:
        st.session_state["optimizer"] = ManualOptimizer()
    if "random_optimizer" not in st.session_state:
        st.session_state["random_optimizer"] = RandomOptimizer()
    if "local_search_optimizer" not in st.session_state:
        st.session_state["local_search_optimizer"] = LocalSearchOptimizer()
    if "gradient_optimizer" not in st.session_state:
        st.session_state["gradient_optimizer"] = GradientAscentOptimizer()
    if "evolutive_optimizer" not in st.session_state:
        st.session_state["evolutive_optimizer"] = EvolutiveSearchOptimizer()


def main() -> None:
    st.set_page_config(
        page_title="Optimization Playground",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    config: AppConfig = st.session_state["config"]
    function_handler: FunctionHandler = st.session_state["function_handler"]
    solutions_handler: SolutionsHandler = st.session_state["solutions_handler"]

    render_function_handler(function_handler, config)
    render_optimizer_sidebar(function_handler, solutions_handler, config)
    render_solutions_plot(function_handler, solutions_handler, config)


if __name__ == "__main__":
    main()
