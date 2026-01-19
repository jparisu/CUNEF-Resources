from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from ml_classification_study.types import DatasetBundle, RocCurveData
from ml_classification_study.models.base import BaseClassifier


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
                marker=dict(symbol="circle", size=7),
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
                marker=dict(symbol="x", size=9),
            )
        )

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x1",
        yaxis_title="x2",
        legend_title="Split / class",
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
            showscale=True,
            contours=dict(showlines=False),
            opacity=0.65,
        )
    else:
        pred = model.predict(grid).reshape(xx.shape)
        contour = go.Contour(
            x=xs,
            y=ys,
            z=pred,
            name="Predicted class",
            showscale=False,
            contours=dict(showlines=False),
            opacity=0.55,
        )

    fig = go.Figure(data=[contour])

    # Overlay points (train/test) with different symbols
    Xtr, ytr = bundle.X_train, bundle.y_train
    Xte, yte = bundle.X_test, bundle.y_test

    for cls in [0, 1]:
        m = ytr == cls
        fig.add_trace(
            go.Scatter(
                x=Xtr[m, 0],
                y=Xtr[m, 1],
                mode="markers",
                name=f"Train class {cls}",
                marker=dict(symbol="circle", size=7),
            )
        )

    for cls in [0, 1]:
        m = yte == cls
        fig.add_trace(
            go.Scatter(
                x=Xte[m, 0],
                y=Xte[m, 1],
                mode="markers",
                name=f"Test class {cls}",
                marker=dict(symbol="x", size=9),
            )
        )

    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x1",
        yaxis_title="x2",
        legend_title="Split / class",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """2x2 confusion matrix heatmap."""

    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=cm,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
            )
        ]
    )

    # Add annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(x=j, y=i, text=str(int(cm[i, j])), showarrow=False)

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Predicted",
        yaxis_title="True",
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
    )
    return fig
