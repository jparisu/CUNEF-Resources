from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fig_dataset_1d(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="x1",
        y="class",
        color="class_str",
        symbol="split",
        hover_data={"class_str": False},
        labels={"x1": "Feature x1", "class": "Class (0/1)", "class_str": "Class"},
        title="Dataset (1D)",
    )
    fig.update_yaxes(tickmode="array", tickvals=[0, 1], range=[-0.5, 1.5])
    return fig


def fig_hist_1d(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="x1",
        color="class_str",
        barmode="overlay",
        opacity=0.6,
        labels={"x1": "Feature x1", "class_str": "Class"},
        title="Histogram of x1",
    )
    return fig


def fig_dataset_2d(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="x1",
        y="x2",
        color="class_str",
        symbol="split",
        labels={"x1": "Feature x1", "x2": "Feature x2", "class_str": "Class"},
        title="Dataset (2D)",
    )
    fig.update_layout(legend_title_text="Legend")
    return fig


def _add_scatter_px(fig: go.Figure, df: pd.DataFrame, d: int) -> None:
    """Overlay a scatter plot where color encodes class and marker symbol encodes train/test."""
    if d == 1:
        sfig = px.scatter(
            df,
            x="x1",
            y="class",
            color="class_str",
            symbol="split",
            hover_data={"class_str": False},
            labels={"x1": "Feature x1", "class": "Class (0/1)", "class_str": "Class"},
        )
        sfig.update_yaxes(tickmode="array", tickvals=[0, 1], range=[-0.5, 1.5])
    else:
        sfig = px.scatter(
            df,
            x="x1",
            y="x2",
            color="class_str",
            symbol="split",
            labels={"x1": "Feature x1", "x2": "Feature x2", "class_str": "Class"},
        )

    for tr in sfig.data:
        fig.add_trace(tr)


def fig_decision_boundary_1d(
    df: pd.DataFrame,
    x_grid: np.ndarray,
    p_grid: np.ndarray,
    boundary_x: Optional[float],
) -> go.Figure:
    y_vals = np.linspace(-0.5, 1.5, 50)
    z = np.tile(p_grid.reshape(1, -1), (len(y_vals), 1))

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_grid,
            y=y_vals,
            z=z,
            hovertemplate="x=%{x:.3f}<br>P(class=1)=%{z:.3f}<extra></extra>",
            name="P(class=1)",
            opacity=0.35,
            colorbar=dict(title="P(class=1)"),
        )
    )

    _add_scatter_px(fig, df, d=1)

    if boundary_x is not None and np.isfinite(boundary_x):
        fig.add_shape(
            type="line",
            x0=float(boundary_x),
            x1=float(boundary_x),
            y0=-0.5,
            y1=1.5,
            line=dict(dash="dash"),
        )

    fig.update_layout(
        title="Decision boundary (1D) and probability field",
        xaxis_title="Feature x1",
        yaxis_title="Class (0/1)",
    )
    fig.update_yaxes(tickmode="array", tickvals=[0, 1], range=[-0.5, 1.5])
    return fig


def fig_decision_boundary_2d(
    df: pd.DataFrame,
    x1_vals: np.ndarray,
    x2_vals: np.ndarray,
    p_grid: np.ndarray,
    boundary: Dict[str, Any],
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=x1_vals,
            y=x2_vals,
            z=p_grid,
            contours=dict(start=0.0, end=1.0, size=0.05),
            opacity=0.6,
            colorbar=dict(title="P(class=1)"),
            hovertemplate="x1=%{x:.3f}<br>x2=%{y:.3f}<br>P(class=1)=%{z:.3f}<extra></extra>",
            name="P(class=1)",
        )
    )

    _add_scatter_px(fig, df, d=2)

    # Add explicit decision boundary line where possible
    btype = boundary.get("type")
    if btype == "line":
        x1_line = boundary["x1_line"]
        x2_line = boundary["x2_line"]
        fig.add_trace(
            go.Scatter(
                x=x1_line,
                y=x2_line,
                mode="lines",
                name="Decision boundary (p=0.5)",
                line=dict(dash="dash"),
            )
        )
    elif btype == "vertical":
        x0 = float(boundary["x"])
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x0,
            y0=float(x2_vals.min()),
            y1=float(x2_vals.max()),
            line=dict(dash="dash"),
        )

    fig.update_layout(
        title="Decision boundary (2D) and probability field",
        xaxis_title="Feature x1",
        yaxis_title="Feature x2",
    )
    return fig


def fig_params_history(
    history: list[Dict[str, Any]],
    n_features: int,
    current_iter: int,
    w: np.ndarray,
    b: float,
    next_step: Optional[Dict[str, Any]] = None,
    arrow_scale: float = 1.0,
) -> go.Figure:
    fig = go.Figure()

    if history:
        it = [h["iter"] for h in history]
        for j in range(n_features):
            fig.add_trace(go.Scatter(x=it, y=[h["w"][j] for h in history], mode="lines", name=f"w{j}"))
        fig.add_trace(go.Scatter(x=it, y=[h["b"] for h in history], mode="lines", name="b"))

    # Current point overlay
    x0 = [current_iter]
    for j in range(n_features):
        fig.add_trace(go.Scatter(x=x0, y=[float(w[j])], mode="markers", name=f"w{j} (current)"))
    fig.add_trace(go.Scatter(x=x0, y=[float(b)], mode="markers", name="b (current)"))

    # Predicted one-step movement arrows (scaled for visibility).
    if next_step:
        it_from = float(next_step.get("iter_from", current_iter))
        current_vals = dict(next_step.get("current", {}))
        next_vals = dict(next_step.get("next", {}))
        scale = max(1.0, float(arrow_scale))

        params = [f"w{j}" for j in range(n_features)] + ["b"]
        for p in params:
            if p not in current_vals or p not in next_vals:
                continue
            y0 = float(current_vals[p])
            dy = float(next_vals[p]) - y0
            y1 = y0 + scale * dy

            fig.add_trace(
                go.Scatter(
                    x=[it_from, it_from],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(dash="dot", width=2),
                    name=f"{p} next-step dir",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=it_from,
                y=y1,
                ax=it_from,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=1.8,
                opacity=0.8,
            )

    fig.update_layout(
        title="Parameter evolution",
        xaxis_title="Iteration",
        yaxis_title="Value",
    )
    return fig


def fig_loss_surface_3d(
    w_vals: np.ndarray,
    b_vals: np.ndarray,
    j_grid: np.ndarray,
    current_point: Dict[str, float],
    history_points: Optional[Dict[str, Any]] = None,
    gradient_arrow: Optional[Dict[str, float]] = None,
    w_label: str = "w0",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=w_vals,
            y=b_vals,
            z=j_grid,
            colorscale="Viridis",
            opacity=0.85,
            name="J(w,b)",
            showscale=True,
            colorbar=dict(title="J"),
        )
    )

    if history_points and len(history_points.get("w", [])) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=history_points["w"],
                y=history_points["b"],
                z=history_points["j"],
                mode="lines+markers",
                name="Training path",
                marker=dict(size=3),
                line=dict(width=4),
                text=history_points.get("text"),
                hovertemplate=f"{w_label}=%{{x:.4f}}<br>b=%{{y:.4f}}<br>J=%{{z:.6f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[current_point["w"]],
            y=[current_point["b"]],
            z=[current_point["j"]],
            mode="markers",
            name="Current",
            marker=dict(size=8, symbol="diamond"),
            hovertemplate=f"{w_label}=%{{x:.4f}}<br>b=%{{y:.4f}}<br>J=%{{z:.6f}}<extra></extra>",
        )
    )

    if gradient_arrow is not None:
        x0 = float(gradient_arrow["w0"])
        y0 = float(gradient_arrow["b0"])
        z0 = float(gradient_arrow["j0"])
        x1 = float(gradient_arrow["w1"])
        y1 = float(gradient_arrow["b1"])
        z1 = float(gradient_arrow["j1"])
        u = x1 - x0
        v = y1 - y0
        w = z1 - z0

        fig.add_trace(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                name="Gradient direction",
                line=dict(width=6, dash="dash"),
                hovertemplate=f"{w_label}=%{{x:.4f}}<br>b=%{{y:.4f}}<br>J=%{{z:.6f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Cone(
                x=[x1],
                y=[y1],
                z=[z1],
                u=[u],
                v=[v],
                w=[w],
                anchor="tip",
                sizemode="absolute",
                sizeref=0.22,
                showscale=False,
                name="Gradient arrow",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Loss surface J({w_label}, b)",
        scene=dict(
            xaxis_title=w_label,
            yaxis_title="b",
            zaxis_title="J",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def fig_metrics_history(history: list[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(title="Metrics evolution (no training history yet)")
        return fig

    it = [h["iter"] for h in history]
    for split in ["train", "test"]:
        for m in ["accuracy", "precision", "recall", "f1"]:
            fig.add_trace(
                go.Scatter(
                    x=it,
                    y=[h[split][m] for h in history],
                    mode="lines",
                    name=f"{split} {m}",
                )
            )

    fig.update_layout(
        title="Metrics evolution (train vs test)",
        xaxis_title="Iteration",
        yaxis_title="Metric value",
        yaxis=dict(range=[0.0, 1.0]),
        legend=dict(orientation="h"),
    )
    return fig


def fig_roc(train_roc: Optional[Dict[str, Any]], test_roc: Optional[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))

    if train_roc is not None:
        fig.add_trace(
            go.Scatter(
                x=train_roc["fpr"],
                y=train_roc["tpr"],
                mode="lines",
                name=f"Train ROC (AUC={train_roc.get('auc', float('nan')):.3f})" if train_roc.get("defined", False) else "Train ROC (AUC=undefined)",
            )
        )

    if test_roc is not None:
        fig.add_trace(
            go.Scatter(
                x=test_roc["fpr"],
                y=test_roc["tpr"],
                mode="lines",
                name=f"Test ROC (AUC={test_roc.get('auc', float('nan')):.3f})" if test_roc.get("defined", False) else "Test ROC (AUC=undefined)",
            )
        )

    fig.update_layout(
        title="ROC curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.0]),
    )
    return fig
