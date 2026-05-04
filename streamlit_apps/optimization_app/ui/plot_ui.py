from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.config import AppConfig
from core.function_handler import FunctionHandler
from core.solutions_handler import Solution, SolutionsHandler


@st.cache_data(show_spinner=False)
def _cached_grid(
    cache_key: str,  # drives cache invalidation; _handler is not hashed
    x_min: float,
    x_max: float,
    resolution: int,
    _handler: FunctionHandler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _handler.evaluate_grid(x_min, x_max, resolution)


# Colors for special-point overlays
_COLOR_BEST = "#FFD700"
_COLOR_HIGHLIGHTED = "#FF4B4B"
_COLOR_SEMI = "#FF8C00"
_SIZE_NORMAL = 10
_SIZE_SPECIAL = 16
_SOLUTIONS_COLORSCALE = "Viridis"


def render_solutions_plot(
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
) -> None:
    """
    Render the main Plotly chart, display controls, and the information section.
    """
    _render_plot_controls(solutions_handler)

    show_heatmap: bool = st.session_state.get("show_heatmap", False)
    show_contours: bool = st.session_state.get("show_contours", False)
    show_colorbar: bool = st.session_state.get("show_colorbar", True)
    show_path: bool = st.session_state.get("show_path", True)
    show_special: bool = st.session_state.get("show_special", True)

    fig = _build_figure(
        function_handler,
        solutions_handler,
        config,
        show_heatmap,
        show_contours,
        show_colorbar,
        show_path,
        show_special,
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="main_plot",
        on_select="rerun",
        selection_mode="points",
    )

    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        sol_idx = pt.get("customdata")
        last_key = "_last_clicked_sol"
        if sol_idx is not None and st.session_state.get(last_key) != sol_idx:
            st.session_state[last_key] = sol_idx
            solutions_handler.toggle_highlight(int(sol_idx))
            st.rerun()

    if not solutions_handler.is_empty():
        _render_info_section(solutions_handler)
        _render_solutions_table(solutions_handler)
    if st.session_state.get("active_optimizer_tab") == "Evolutive search":
        _render_evolutive_evolution()

    return None


# ---------------------------------------------------------------------------
# Controls row
# ---------------------------------------------------------------------------


def _render_plot_controls(solutions_handler: SolutionsHandler) -> None:
    """Render the five visibility checkboxes and the reset button."""
    cols = st.columns([1, 1, 1, 1, 1, 1])
    with cols[0]:
        st.checkbox("Contour", value=False, key="show_heatmap")
    with cols[1]:
        st.checkbox("Height curves", value=False, key="show_contours")
    with cols[2]:
        st.checkbox("Colorbar", value=True, key="show_colorbar")
    with cols[3]:
        st.checkbox("Path", value=True, key="show_path")
    with cols[4]:
        st.checkbox("Special points", value=True, key="show_special")
    with cols[5]:
        if st.button(
            "🗑 Reset points",
            use_container_width=True,
            disabled=solutions_handler.is_empty(),
        ):
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
            st.session_state["local_overlay"] = None
            st.session_state["gradient_overlay"] = None
            st.session_state["evolutive_overlay"] = None
            st.session_state.pop("_last_clicked_sol", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def _build_figure(
    function_handler: FunctionHandler,
    solutions_handler: SolutionsHandler,
    config: AppConfig,
    show_heatmap: bool,
    show_contours: bool,
    show_colorbar: bool,
    show_path: bool,
    show_special: bool,
) -> go.Figure:
    fig = go.Figure()
    is_evo_active = st.session_state.get("active_optimizer_tab") == "Evolutive search"

    if function_handler.is_ready():
        X1, X2, Z = _cached_grid(
            function_handler.cache_key,
            config.x_min,
            config.x_max,
            config.grid_resolution,
            function_handler,
        )
        x1_vals = X1[0]
        x2_vals = X2[:, 0]

        if show_heatmap:
            fig.add_trace(_build_heatmap_trace(x1_vals, x2_vals, Z))

        if show_contours:
            fig.add_trace(_build_contour_trace(x1_vals, x2_vals, Z))

    if not solutions_handler.is_empty():
        solutions = solutions_handler.get_all()
        f_vals = [s.f_value for s in solutions]
        f_min, f_max = min(f_vals), max(f_vals)

        if show_path and not is_evo_active:
            fig.add_trace(_build_path_trace(solutions))

        fig.add_trace(_build_solutions_scatter(solutions, f_min, f_max, show_colorbar))

        if show_special:
            bests = solutions_handler.get_best_all()
            highlighted = [s for s in solutions if s.highlighted]
            semi = [s for s in solutions if s.semi_highlighted]

            if bests:
                fig.add_trace(_build_best_trace(bests))
            if highlighted:
                fig.add_trace(_build_highlighted_trace(highlighted))
            if semi:
                fig.add_trace(_build_semi_highlighted_trace(semi))

    overlay = st.session_state.get("local_overlay")
    is_local_active = st.session_state.get("active_optimizer_tab") == "Local search"
    if overlay and is_local_active:
        current = overlay.get("current")
        candidates = overlay.get("candidates", [])
        if current:
            fig.add_trace(
                go.Scatter(
                    x=[current[0]],
                    y=[current[1]],
                    mode="markers",
                    marker=dict(
                        color=[current[2]],
                        colorscale=_SOLUTIONS_COLORSCALE,
                        size=20,
                        symbol="circle",
                        line=dict(color="white", width=2),
                        showscale=False,
                    ),
                    text=[f"{current[2]:.4f}"],
                    name="Local current",
                    hovertemplate="x1: %{x:.3f}<br>x2: %{y:.3f}<br>F: %{text}<extra>Local current</extra>",
                )
            )
        if candidates:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in candidates],
                    y=[p[1] for p in candidates],
                    mode="markers",
                    marker=dict(
                        color=[p[2] for p in candidates],
                        colorscale=_SOLUTIONS_COLORSCALE,
                        size=12,
                        symbol="diamond-open",
                        line=dict(color="white", width=1),
                        showscale=False,
                    ),
                    text=[f"{p[2]:.4f}" for p in candidates],
                    name="Locality points",
                    hovertemplate="x1: %{x:.3f}<br>x2: %{y:.3f}<br>F: %{text}<extra>Locality</extra>",
                )
            )
        if current and candidates:
            for candidate in candidates:
                fig.add_trace(
                    go.Scatter(
                        x=[current[0], candidate[0]],
                        y=[current[1], candidate[1]],
                        mode="lines",
                        line=dict(color="rgba(255,165,0,0.45)", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    grad_overlay = st.session_state.get("gradient_overlay")
    is_grad_active = st.session_state.get("active_optimizer_tab") == "Gradient"
    if grad_overlay and is_grad_active:
        current = grad_overlay.get("current")
        next_point = grad_overlay.get("next")
        arrow_tip = None
        if current and next_point:
            dx = next_point[0] - current[0]
            dy = next_point[1] - current[1]
            scale = 2.8
            arrow_tip = (current[0] + scale * dx, current[1] + scale * dy)
            arrow_tip = (
                max(config.x_min, min(config.x_max, arrow_tip[0])),
                max(config.x_min, min(config.x_max, arrow_tip[1])),
            )
        if current:
            fig.add_trace(
                go.Scatter(
                    x=[current[0]],
                    y=[current[1]],
                    mode="markers",
                    marker=dict(
                        color=[current[2]],
                        colorscale=_SOLUTIONS_COLORSCALE,
                        size=20,
                        symbol="circle",
                        line=dict(color="white", width=2),
                        showscale=False,
                    ),
                    text=[f"{current[2]:.4f}"],
                    name="Gradient current",
                    hovertemplate="x1: %{x:.3f}<br>x2: %{y:.3f}<br>F: %{text}<extra>Gradient current</extra>",
                )
            )
        if current and arrow_tip:
            fig.add_trace(
                go.Scatter(
                    x=[current[0], arrow_tip[0]],
                    y=[current[1], arrow_tip[1]],
                    mode="lines",
                    line=dict(color="rgba(0,229,255,0.38)", width=6),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        if current and arrow_tip:
            fig.add_annotation(
                x=arrow_tip[0],
                y=arrow_tip[1],
                ax=current[0],
                ay=current[1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.6,
                arrowwidth=3.0,
                arrowcolor="rgba(0,229,255,0.55)",
            )
        if next_point:
            fig.add_trace(
                go.Scatter(
                    x=[next_point[0]],
                    y=[next_point[1]],
                    mode="markers",
                    marker=dict(
                        color=[next_point[2]],
                        colorscale=_SOLUTIONS_COLORSCALE,
                        size=14,
                        symbol="triangle-up",
                        showscale=False,
                    ),
                    text=[f"{next_point[2]:.4f}"],
                    name="Gradient next",
                    hovertemplate="x1: %{x:.3f}<br>x2: %{y:.3f}<br>F: %{text}<extra>Gradient next</extra>",
                )
            )

    evo_overlay = st.session_state.get("evolutive_overlay")
    if evo_overlay and is_evo_active:
        population = evo_overlay.get("population", [])
        preview = evo_overlay.get("preview") or {}
        palette = _population_palette(max([p[0] for p in population], default=-1) + 1)
        by_slot = {slot: color for slot, color in enumerate(palette)}

        if population:
            fig.add_trace(
                go.Scatter(
                    x=[p[1] for p in population],
                    y=[p[2] for p in population],
                    mode="markers",
                    marker=dict(
                        size=15,
                        symbol="square",
                        color=[by_slot[p[0]] for p in population],
                        line=dict(color="white", width=1),
                    ),
                    customdata=[p[0] for p in population],
                    text=[f"{p[3]:.4f}" for p in population],
                    hovertemplate="slot=%{customdata}<br>x1:%{x:.3f}<br>x2:%{y:.3f}<br>F:%{text}<extra>Population</extra>",
                    name="Population",
                )
            )
        if preview:
            for idx in preview.get("mutated_indices", []):
                if idx < len(population):
                    p = population[idx]
                    fig.add_shape(
                        type="circle",
                        xref="x",
                        yref="y",
                        x0=p[1] - preview.get("sigma", 0.0),
                        x1=p[1] + preview.get("sigma", 0.0),
                        y0=p[2] - preview.get("sigma", 0.0),
                        y1=p[2] + preview.get("sigma", 0.0),
                        line=dict(color="rgba(0,200,255,0.45)", width=1),
                        fillcolor="rgba(0,200,255,0.07)",
                    )

            mutated_points = preview.get("mutated_points", [])
            if mutated_points:
                fig.add_trace(
                    go.Scatter(
                        x=[p.x1 for p in mutated_points],
                        y=[p.x2 for p in mutated_points],
                        mode="markers",
                        marker=dict(
                            color="rgba(0,200,255,0.85)",
                            size=12,
                            symbol="circle-open",
                            line=dict(width=2),
                        ),
                        text=[f"{p.f_value:.4f}" for p in mutated_points],
                        hovertemplate="x1:%{x:.3f}<br>x2:%{y:.3f}<br>F:%{text}<extra>Mutation</extra>",
                        name="Mutation",
                    )
                )

            crossover_pairs = preview.get("crossover_pairs", [])
            crossover_points = preview.get("crossover_points", [])
            for pair in crossover_pairs:
                i, j = pair
                if i < len(population) and j < len(population):
                    a = population[i]
                    b = population[j]
                    fig.add_trace(
                        go.Scatter(
                            x=[a[1], b[1]],
                            y=[a[2], b[2]],
                            mode="lines",
                            line=dict(color="rgba(255,255,255,0.25)", width=1),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
            if crossover_points:
                fig.add_trace(
                    go.Scatter(
                        x=[p.x1 for p in crossover_points],
                        y=[p.x2 for p in crossover_points],
                        mode="markers",
                        marker=dict(color="rgba(150,255,150,0.85)", size=12, symbol="x"),
                        text=[f"{p.f_value:.4f}" for p in crossover_points],
                        hovertemplate="x1:%{x:.3f}<br>x2:%{y:.3f}<br>F:%{text}<extra>Crossover</extra>",
                        name="Crossover",
                    )
                )

            selection = preview.get("selection", {})
            selected_points = preview.get("selected_points", [])
            selected_labels = ["" for _ in selected_points]
            selected_hover = ["Selected next generation" for _ in selected_points]
            if selection.get("mode") == "roulette":
                probs = selection.get("candidate_probabilities") or []
                selected_indices = selection.get("selected_indices") or []
                selected_labels = [
                    f"p={100 * probs[idx]:.1f}%" if idx < len(probs) else ""
                    for idx in selected_indices
                ]
                selected_hover = [
                    f"Roulette probability: {100 * probs[idx]:.2f}%"
                    if idx < len(probs)
                    else "Selected next generation"
                    for idx in selected_indices
                ]
            if selected_points:
                fig.add_trace(
                    go.Scatter(
                        x=[p.x1 for p in selected_points],
                        y=[p.x2 for p in selected_points],
                        mode="markers+text" if any(selected_labels) else "markers",
                        marker=dict(color="rgba(255,220,80,0.92)", size=12, symbol="star"),
                        text=selected_labels,
                        textposition="top center",
                        customdata=selected_hover,
                        hovertemplate="x1:%{x:.3f}<br>x2:%{y:.3f}<br>%{customdata}<extra>Selected</extra>",
                        name="Selected next gen",
                    )
                )

    fig.update_layout(
        xaxis_title="x₁",
        yaxis_title="x₂",
        xaxis=dict(range=[config.x_min, config.x_max], constrain="domain"),
        yaxis=dict(range=[config.x_min, config.x_max], scaleanchor="x"),
        height=560,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.4)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------


def _build_heatmap_trace(x1_vals: np.ndarray, x2_vals: np.ndarray, Z: np.ndarray) -> go.Heatmap:
    return go.Heatmap(
        x=x1_vals,
        y=x2_vals,
        z=Z,
        colorscale="RdBu",
        reversescale=True,
        showscale=True,
        colorbar=dict(title="F(grid)", thickness=12, len=0.6, x=1.0),
        opacity=0.85,
        name="F(x₁,x₂)",
        showlegend=True,
        hovertemplate="x₁: %{x:.3f}<br>x₂: %{y:.3f}<br>F: %{z:.4f}<extra>Contour</extra>",
    )


def _population_palette(n: int) -> list[str]:
    base = [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
        "#1f78b4",
        "#33a02c",
        "#e31a1c",
        "#ff7f00",
        "#6a3d9a",
        "#b15928",
        "#17becf",
        "#bcbd22",
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def _build_contour_trace(x1_vals: np.ndarray, x2_vals: np.ndarray, Z: np.ndarray) -> go.Contour:
    return go.Contour(
        x=x1_vals,
        y=x2_vals,
        z=Z,
        showscale=False,
        colorscale="Greys",
        contours=dict(showlabels=True, labelfont=dict(size=9, color="white")),
        line=dict(width=1),
        opacity=0.6,
        name="Height curves",
        showlegend=True,
        hoverinfo="skip",
    )


def _build_solutions_scatter(
    solutions: list[Solution],
    f_min: float,
    f_max: float,
    show_colorbar: bool,
) -> go.Scatter:
    """All evaluated points colored continuously by f_value."""
    return go.Scatter(
        x=[s.x1 for s in solutions],
        y=[s.x2 for s in solutions],
        mode="markers",
        marker=dict(
            color=[s.f_value for s in solutions],
            colorscale=_SOLUTIONS_COLORSCALE,
            cmin=f_min,
            cmax=f_max,
            size=_SIZE_NORMAL,
            symbol="circle",
            line=dict(color="rgba(255,255,255,0.5)", width=1),
            showscale=show_colorbar,
            colorbar=dict(title="F(pts)", thickness=12, len=0.4, x=1.13, y=0.25),
        ),
        customdata=list(range(len(solutions))),
        text=[f"{s.f_value:.4f}" for s in solutions],
        hovertemplate=(
            "<b>#%{customdata}</b><br>"
            "x₁: %{x:.3f}<br>"
            "x₂: %{y:.3f}<br>"
            "F: %{text}<br>"
            "<i>click to toggle highlight</i>"
            "<extra>Solutions</extra>"
        ),
        name="Solutions",
    )


def _build_best_trace(bests: list[Solution]) -> go.Scatter:
    return go.Scatter(
        x=[s.x1 for s in bests],
        y=[s.x2 for s in bests],
        mode="markers",
        marker=dict(
            color=_COLOR_BEST,
            size=_SIZE_SPECIAL + 4,
            symbol="star",
            line=dict(color="white", width=1.5),
        ),
        text=[f"{s.f_value:.4f}" for s in bests],
        hovertemplate="x₁: %{x:.3f}<br>x₂: %{y:.3f}<br>F: %{text}<extra>Best</extra>",
        name="Best",
    )


def _build_highlighted_trace(highlighted: list[Solution]) -> go.Scatter:
    return go.Scatter(
        x=[s.x1 for s in highlighted],
        y=[s.x2 for s in highlighted],
        mode="markers",
        marker=dict(
            color=_COLOR_HIGHLIGHTED,
            size=_SIZE_SPECIAL,
            symbol="circle",
            line=dict(color="white", width=2),
        ),
        text=[f"{s.f_value:.4f}" for s in highlighted],
        hovertemplate="x₁: %{x:.3f}<br>x₂: %{y:.3f}<br>F: %{text}<extra>Highlighted</extra>",
        name="Highlighted",
    )


def _build_semi_highlighted_trace(semi: list[Solution]) -> go.Scatter:
    return go.Scatter(
        x=[s.x1 for s in semi],
        y=[s.x2 for s in semi],
        mode="markers",
        marker=dict(
            color=_COLOR_SEMI,
            size=_SIZE_SPECIAL - 2,
            symbol="diamond",
            line=dict(color="white", width=1),
        ),
        text=[f"{s.f_value:.4f}" for s in semi],
        hovertemplate="x₁: %{x:.3f}<br>x₂: %{y:.3f}<br>F: %{text}<extra>Semi-highlighted</extra>",
        name="Semi-highlighted",
    )


def _build_path_trace(solutions: list[Solution]) -> go.Scatter:
    return go.Scatter(
        x=[s.x1 for s in solutions],
        y=[s.x2 for s in solutions],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dot"),
        hoverinfo="skip",
        showlegend=True,
        name="Path",
    )


# ---------------------------------------------------------------------------
# Information section
# ---------------------------------------------------------------------------


def _render_info_section(solutions_handler: SolutionsHandler) -> None:
    """Render best point(s), point count, and f_value statistics below the chart."""
    stats = solutions_handler.get_statistics()
    bests = solutions_handler.get_best_all()

    st.divider()
    col_best, col_stats = st.columns([2, 3])

    with col_best:
        if len(bests) == 1:
            b = bests[0]
            st.markdown("**Best point**")
            st.markdown(f"x₁ = `{b.x1:.4f}` · x₂ = `{b.x2:.4f}` · F = `{b.f_value:.6f}`")
        else:
            st.markdown(f"**Best points ({len(bests)} tied)**")
            for b in bests:
                st.markdown(f"- x₁ = `{b.x1:.4f}` · x₂ = `{b.x2:.4f}` · F = `{b.f_value:.6f}`")

    with col_stats:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Points", solutions_handler.count())
        with c2:
            st.metric("Mean F", f"{stats['mean']:.4f}")
        with c3:
            st.metric("Median F", f"{stats['median']:.4f}")
        with c4:
            st.metric("Std F", f"{stats['std']:.4f}")


# ---------------------------------------------------------------------------
# Solutions table
# ---------------------------------------------------------------------------


def _render_solutions_table(solutions_handler: SolutionsHandler) -> None:
    with st.expander(f"Solutions table ({solutions_handler.count()})", expanded=False):
        df = solutions_handler.to_dataframe()
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "x1": st.column_config.NumberColumn(format="%.4f"),
                "x2": st.column_config.NumberColumn(format="%.4f"),
                "F(x1,x2)": st.column_config.NumberColumn(format="%.6f"),
                "highlighted": st.column_config.CheckboxColumn(width="small"),
                "semi_highlighted": st.column_config.CheckboxColumn(width="small"),
            },
        )


def _render_evolutive_evolution() -> None:
    evo_overlay = st.session_state.get("evolutive_overlay")
    if not evo_overlay:
        return
    history = evo_overlay.get("history", [])
    if not history:
        return

    fig = go.Figure()
    palette = _population_palette(200)
    for row in history:
        it = row["iteration"]
        for slot, _x1, _x2, f in row["population"]:
            fig.add_trace(
                go.Scatter(
                    x=[it],
                    y=[f],
                    mode="markers",
                    marker=dict(size=8, color=palette[slot % len(palette)]),
                    showlegend=False,
                    hovertemplate=f"iter={it}<br>slot={slot}<br>F=%{{y:.4f}}<extra>Population</extra>",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=[row["iteration"] for row in history],
            y=[row["best"] for row in history],
            mode="lines+markers",
            line=dict(color="#FFD700", width=2),
            name="Best",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[row["iteration"] for row in history],
            y=[row["mean"] for row in history],
            mode="lines+markers",
            line=dict(color="#7FDBFF", width=2),
            name="Mean",
        )
    )
    fig.update_layout(
        title="Population Evolution",
        xaxis_title="Iteration",
        yaxis_title="Utility",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="evo_evolution_chart")
