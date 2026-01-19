from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from ml_classification_study.types import DatasetBundle, RocCurveData
from ml_classification_study.models.base import BaseClassifier


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
# Dataset colors:
#   - class 0: dark + light blue
#   - class 1: dark + light green
# Misclassification highlight colors:
#   - true class 0 misclassified: red
#   - true class 1 misclassified: orange

TEST_CLASS_COLORS = {
    0: "#1f77b4",  # blue
    1: "#2ca02c",  # green
}

TRAIN_CLASS_COLORS = {
    0: "#0a0c73",  # dark blue
    1: "#075e07",  # dark green
}

MISCLASS_COLORS = {
    0: "#f53387",  # red
    1: "#f53333",  # orange
}


def plot_dataset_scatter(bundle: DatasetBundle) -> go.Figure:
    """Scatter plot with class colors and train/test symbols."""

    fig = go.Figure()

    Xtr, ytr = bundle.X_train, bundle.y_train
    Xte, yte = bundle.X_test, bundle.y_test

    # Train traces
    for cls in [0, 1]:
        mask = ytr == cls
        fig.add_trace(
            go.Scatter(
                x=Xtr[mask, 0],
                y=Xtr[mask, 1],
                mode="markers",
                name=f"Train - class {cls}",
                opacity=0.8,
                marker=dict(
                    symbol="circle",
                    size=10,
                    color=TRAIN_CLASS_COLORS[cls],
                    line=dict(width=1, color="rgba(0,0,0,0.45)"),
                ),
            )
        )

    # Test traces
    for cls in [0, 1]:
        mask = yte == cls
        fig.add_trace(
            go.Scatter(
                x=Xte[mask, 0],
                y=Xte[mask, 1],
                mode="markers",
                name=f"Test - class {cls}",
                opacity=0.8,
                marker=dict(
                    symbol="x",
                    size=10,
                    color=TEST_CLASS_COLORS[cls],
                    line=dict(width=1),
                ),
            )
        )

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x1",
        yaxis_title="x2",
        legend_title="Split / class",
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _grid_bounds(X: np.ndarray, *, margin: float = 0.5) -> Tuple[float, float, float, float]:
    x_min, x_max = float(X[:, 0].min()), float(X[:, 0].max())
    y_min, y_max = float(X[:, 1].min()), float(X[:, 1].max())
    dx = x_max - x_min
    dy = y_max - y_min
    if dx <= 0:
        dx = 1.0
    if dy <= 0:
        dy = 1.0
    return (
        x_min - margin * dx,
        x_max + margin * dx,
        y_min - margin * dy,
        y_max + margin * dy,
    )


def plot_decision_boundary(
    model: BaseClassifier,
    bundle: DatasetBundle,
    *,
    grid_res: int = 220,
    margin: float = 0.4,
) -> go.Figure:
    """Decision boundary plot: contour + training/test points."""

    X_all = bundle.X
    x0, x1, y0, y1 = _grid_bounds(X_all, margin=margin)

    xs = np.linspace(x0, x1, grid_res)
    ys = np.linspace(y0, y1, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    proba = model.predict_proba(grid)
    if proba is not None:
        zz = proba[:, 1].reshape(xx.shape)
        contour = go.Contour(
            x=xs,
            y=ys,
            z=zz,
            name="P(class=1)",
            showlegend=False,
            showscale=False,
            colorscale=[
                [0.0, TRAIN_CLASS_COLORS[0]],
                [0.5, "#17becf"],
                [1.0, TRAIN_CLASS_COLORS[1]],
            ],
            zmin=0.0,
            zmax=1.0,
            # colorbar=dict(title="P(class=1)"),
            contours=dict(showlines=False),
            opacity=0.55,
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>P(class=1)=%{z:.3f}<extra></extra>",
        )
    else:
        pred = model.predict(grid).reshape(xx.shape)
        contour = go.Contour(
            x=xs,
            y=ys,
            z=pred,
            name="Predicted class",
            showlegend=False,
            showscale=False,
            colorscale=[
                [0.0, TEST_CLASS_COLORS[0]],
                [1.0, TEST_CLASS_COLORS[1]],
            ],
            zmin=0.0,
            zmax=1.0,
            contours=dict(showlines=False),
            opacity=0.5,
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>pred=%{z:.0f}<extra></extra>",
        )

    fig = go.Figure(data=[contour])

    # Overlay points (train/test) with different symbols and explicit colors.
    # Misclassified points are highlighted with distinct colors.
    Xtr, ytr = bundle.X_train, bundle.y_train
    Xte, yte = bundle.X_test, bundle.y_test

    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)

    tr_correct = ytr_pred == ytr
    te_correct = yte_pred == yte

    def _add_scatter(
        X: np.ndarray,
        mask: np.ndarray,
        *,
        name: str,
        symbol: str,
        color: str,
        size: int,
    ) -> None:
        if mask is None or mask.sum() == 0:
            return
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=name,
                marker=dict(
                    symbol=symbol,
                    size=size,
                    color=color,
                    line=dict(width=0.6, color="rgba(0,0,0,0.55)"),
                ),
            )
        )

    for cls in [0, 1]:
        # Train
        _add_scatter(
            Xtr,
            (ytr == cls) & tr_correct,
            name=f"Train - class {cls} (correct)",
            symbol="circle",
            color=TRAIN_CLASS_COLORS[cls],
            size=7,
        )
        _add_scatter(
            Xtr,
            (ytr == cls) & (~tr_correct),
            name=f"Train - class {cls} (wrong)",
            symbol="circle",
            color=MISCLASS_COLORS[cls],
            size=8,
        )

        # Test
        _add_scatter(
            Xte,
            (yte == cls) & te_correct,
            name=f"Test - class {cls} (correct)",
            symbol="x",
            color=TEST_CLASS_COLORS[cls],
            size=9,
        )
        _add_scatter(
            Xte,
            (yte == cls) & (~te_correct),
            name=f"Test - class {cls} (wrong)",
            symbol="x",
            color=MISCLASS_COLORS[cls],
            size=10,
        )

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x1",
        yaxis_title="x2",
        legend_title="Split / class",
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """2x2 confusion matrix heatmap."""

    x_labels = ["Pred 0", "Pred 1"]
    y_labels = ["True 0", "True 1"]

    # Defensive defaults
    max_v = int(cm.max()) if cm.size else 0
    zmax = max(1, max_v)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=cm,
                x=x_labels,
                y=y_labels,
                zmin=0,
                zmax=zmax,
                colorscale="Cividis",
                hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
            )
        ]
    )

    # Add value labels with contrasting colors
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            # Cividis gets brighter as values increase -> use black text on bright cells
            if max_v > 0 and val >= 0.6 * max_v:
                font_color = "black"
            else:
                font_color = "white"
            fig.add_annotation(
                x=x_labels[j],
                y=y_labels[i],
                text=str(val),
                showarrow=False,
                font=dict(color=font_color, size=16),
            )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Predicted",
        yaxis_title="True",
        template="plotly_white",
    )
    return fig


def plot_roc_curve(roc: RocCurveData) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc.fpr, y=roc.tpr, mode="lines", name=f"ROC (AUC={roc.auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    return fig
