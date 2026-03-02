from __future__ import annotations

from pathlib import Path
import sys

import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import streamlit as st

# Allow running `streamlit run app.py` without installing the package.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logistic_regression.dataset import generate_gaussian_dataset, sanitize_dataset_config, to_dataframe
from logistic_regression.lr_model import LogisticRegressionScratch
from logistic_regression.metrics import classification_metrics, roc_curve_auc
from logistic_regression.viz import (
    fig_dataset_1d,
    fig_hist_1d,
    fig_dataset_2d,
    fig_decision_boundary_1d,
    fig_decision_boundary_2d,
    fig_params_history,
    fig_metrics_history,
    fig_roc,
    fig_loss_surface_3d,
)


# -----------------------------
# Defaults
# -----------------------------
def default_dataset_config() -> Dict[str, Any]:
    return {
        "n_samples": 200,
        "balance": 0.5,
        "n_features": 2,
        "means0": [0.0, 0.0],
        "stds0": [1.0, 1.0],
        "means1": [2.0, 2.0],
        "stds1": [1.0, 1.0],
        "train_split": 0.7,
    }


def default_model_config() -> Dict[str, Any]:
    return {
        "lr": 0.1,
        "n_iters": 200,
        "reg_type": "None",
        "reg_strength": 0.0,
        "batch_size": 32,
    }


# -----------------------------
# Session state utilities
# -----------------------------
def _ensure_key(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _init_widget_state() -> None:
    _ensure_key("seed", 42)

    # Dataset widgets (pending config)
    _ensure_key("ds_dynamic", True)
    _ensure_key("ds_n_samples", 200)
    _ensure_key("ds_balance", 0.5)
    _ensure_key("ds_n_features", 2)
    _ensure_key("ds_train_split", 0.7)

    # Means/stds widgets for up to 2 features
    for c in [0, 1]:
        for f in [0, 1]:
            if c == 0:
                _ensure_key(f"ds_mean_c{c}_f{f}", 0.0 if f == 0 else 0.0)
                _ensure_key(f"ds_std_c{c}_f{f}", 1.0)
            else:
                _ensure_key(f"ds_mean_c{c}_f{f}", 2.0 if f == 0 else 2.0)
                _ensure_key(f"ds_std_c{c}_f{f}", 1.0)

    # Model widgets
    _ensure_key("lr", 0.1)
    _ensure_key("n_iters", 200)
    _ensure_key("reg_type", "None")
    _ensure_key("reg_strength", 0.0)
    _ensure_key("batch_size", 32)
    _ensure_key("dynamic_fps", 8)
    _ensure_key("param_arrow_scale", 8.0)
    _ensure_key("surface_arrow_scale", 8.0)
    _ensure_key("surface_grid_n", 45)
    _ensure_key("surface_w_idx", 0)
    _ensure_key("surface_live_during_dynamic", False)

    # Parameter widgets (current model parameters)
    _ensure_key("w0", 0.0)
    _ensure_key("w1", 0.0)
    _ensure_key("b", 0.0)


def _pending_dataset_config_from_widgets() -> Dict[str, Any]:
    d = int(st.session_state.ds_n_features)
    cfg = {
        "n_samples": int(st.session_state.ds_n_samples),
        "balance": float(st.session_state.ds_balance),
        "n_features": d,
        "train_split": float(st.session_state.ds_train_split),
        "means0": [float(st.session_state[f"ds_mean_c0_f{j}"]) for j in range(d)],
        "stds0": [float(st.session_state[f"ds_std_c0_f{j}"]) for j in range(d)],
        "means1": [float(st.session_state[f"ds_mean_c1_f{j}"]) for j in range(d)],
        "stds1": [float(st.session_state[f"ds_std_c1_f{j}"]) for j in range(d)],
    }
    return sanitize_dataset_config(cfg)


def _current_model_config_from_widgets() -> Dict[str, Any]:
    return {
        "lr": float(st.session_state.lr),
        "n_iters": int(st.session_state.n_iters),
        "reg_type": str(st.session_state.reg_type),
        "reg_strength": float(st.session_state.reg_strength),
        "batch_size": int(st.session_state.batch_size),
    }


def _model_params_from_widgets(n_features: int) -> Tuple[np.ndarray, float]:
    if n_features == 1:
        w = np.array([float(st.session_state.w0)], dtype=float)
    else:
        w = np.array([float(st.session_state.w0), float(st.session_state.w1)], dtype=float)
    b = float(st.session_state.b)
    return w, b


def _write_model_params_to_widgets(w: np.ndarray, b: float) -> None:
    w = np.asarray(w, dtype=float).reshape(-1)
    st.session_state.w0 = float(w[0]) if w.size >= 1 else 0.0
    st.session_state.w1 = float(w[1]) if w.size >= 2 else float(st.session_state.get("w1", 0.0))
    st.session_state.b = float(b)


def _randomize_model_params_from_seed(seed: int, n_features: int) -> None:
    # Deterministic random initialization from seed.
    rng = np.random.default_rng(int(seed) + 10_007)
    w = rng.uniform(-2.0, 2.0, size=(int(n_features),))
    b = float(rng.uniform(-2.0, 2.0))
    _write_model_params_to_widgets(w=w, b=b)


def _reset_training_state(keep_params: bool = True) -> None:
    if not keep_params:
        st.session_state.w0 = 0.0
        st.session_state.w1 = 0.0
        st.session_state.b = 0.0

    st.session_state.iter = 0
    st.session_state.history_params = []
    st.session_state.history_metrics = []
    st.session_state.last_step = None
    st.session_state.next_batch_idx = None


def _regenerate_dataset(applied_cfg: Dict[str, Any]) -> None:
    X, y, train_idx, test_idx = generate_gaussian_dataset(applied_cfg, seed=int(st.session_state.seed))
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.train_idx = train_idx
    st.session_state.test_idx = test_idx
    st.session_state.df = to_dataframe(X, y, train_idx, test_idx)
    st.session_state.ds_cfg_applied = dict(applied_cfg)
    # New dataset => clear training history (but keep current params)
    _reset_training_state(keep_params=True)
    st.session_state.dynamic_train = False


def _ensure_core_state() -> None:
    _init_widget_state()

    _ensure_key("seed_prev", int(st.session_state.seed))
    _ensure_key("ds_dynamic_prev", bool(st.session_state.ds_dynamic))

    _ensure_key("ds_cfg_applied", default_dataset_config())
    _ensure_key("X", np.empty((0, 2), dtype=float))
    _ensure_key("y", np.empty((0,), dtype=int))
    _ensure_key("train_idx", np.array([], dtype=int))
    _ensure_key("test_idx", np.array([], dtype=int))
    _ensure_key("df", None)

    _ensure_key("iter", 0)
    _ensure_key("history_params", [])
    _ensure_key("history_metrics", [])
    _ensure_key("last_step", None)
    _ensure_key("next_batch_idx", None)

    _ensure_key("train_rng", np.random.default_rng(int(st.session_state.seed)))

    _ensure_key("dynamic_train", False)
    _ensure_key("params_initialized", False)

    if not bool(st.session_state.params_initialized):
        d = int(sanitize_dataset_config(st.session_state.ds_cfg_applied)["n_features"])
        _randomize_model_params_from_seed(seed=int(st.session_state.seed), n_features=d)
        st.session_state.params_initialized = True


def _maybe_apply_dataset_updates() -> None:
    seed = int(st.session_state.seed)
    seed_prev = int(st.session_state.seed_prev)

    ds_dynamic = bool(st.session_state.ds_dynamic)
    ds_dynamic_prev = bool(st.session_state.ds_dynamic_prev)

    pending = _pending_dataset_config_from_widgets()
    applied = sanitize_dataset_config(st.session_state.ds_cfg_applied)

    seed_changed = seed != seed_prev
    toggled_on = (not ds_dynamic_prev) and ds_dynamic

    if seed_changed:
        # Seed affects reproducibility: reset training RNG always.
        st.session_state.train_rng = np.random.default_rng(seed)

    should_apply = False
    if ds_dynamic:
        # Live update while dynamic is ON
        should_apply = pending != applied
    else:
        # Dynamic OFF => no live changes
        should_apply = False

    if toggled_on:
        # When re-enabling dynamic mode, apply pending changes once.
        should_apply = True

    # Seed change always regenerates dataset (using applied unless dynamic is ON).
    if seed_changed:
        should_apply = True

    if should_apply:
        cfg_to_apply = pending if ds_dynamic else applied
        _regenerate_dataset(cfg_to_apply)
        if seed_changed:
            d = int(sanitize_dataset_config(st.session_state.ds_cfg_applied)["n_features"])
            _randomize_model_params_from_seed(seed=seed, n_features=d)
            _reset_training_state(keep_params=True)

    st.session_state.seed_prev = seed
    st.session_state.ds_dynamic_prev = ds_dynamic


def _ensure_param_dim_matches_dataset() -> None:
    d = int(sanitize_dataset_config(st.session_state.ds_cfg_applied)["n_features"])
    # Ensure widget parameters exist
    if d == 1:
        # keep w1 around, but it won't be used
        pass
    else:
        # ensure w1 exists
        _ensure_key("w1", float(st.session_state.get("w1", 0.0)))


def _get_model() -> LogisticRegressionScratch:
    d = int(sanitize_dataset_config(st.session_state.ds_cfg_applied)["n_features"])
    w, b = _model_params_from_widgets(d)
    return LogisticRegressionScratch(n_features=d, w=w.copy(), b=float(b))


def _get_train_test() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = st.session_state.X
    y = st.session_state.y
    train_idx = st.session_state.train_idx
    test_idx = st.session_state.test_idx
    Xtr = X[train_idx] if train_idx.size else np.empty((0, X.shape[1]), dtype=float)
    ytr = y[train_idx] if train_idx.size else np.empty((0,), dtype=int)
    Xte = X[test_idx] if test_idx.size else np.empty((0, X.shape[1]), dtype=float)
    yte = y[test_idx] if test_idx.size else np.empty((0,), dtype=int)
    return Xtr, ytr, Xte, yte


def _sample_next_batch_indices() -> Optional[np.ndarray]:
    train_idx = st.session_state.train_idx
    if train_idx.size == 0:
        return None

    batch_size = int(st.session_state.batch_size)
    if batch_size >= train_idx.size:
        return train_idx.copy()

    rng = st.session_state.train_rng
    return rng.choice(train_idx, size=batch_size, replace=False)


def _ensure_next_batch_indices() -> Optional[np.ndarray]:
    # Reuse the same batch for "preview gradient" and "next step" to stay consistent.
    batch_idx = st.session_state.next_batch_idx
    if batch_idx is None:
        batch_idx = _sample_next_batch_indices()
        st.session_state.next_batch_idx = batch_idx
    else:
        batch_idx = np.asarray(batch_idx, dtype=int)
    return batch_idx


def _record_history(model: LogisticRegressionScratch) -> None:
    d = model.n_features
    it = int(st.session_state.iter)

    # Params history
    st.session_state.history_params.append({"iter": it, "w": model.w.tolist(), "b": float(model.b)})

    # Metrics history
    Xtr, ytr, Xte, yte = _get_train_test()
    yhat_tr = model.predict(Xtr) if Xtr.size else np.array([], dtype=int)
    yhat_te = model.predict(Xte) if Xte.size else np.array([], dtype=int)
    m_tr = classification_metrics(ytr, yhat_tr)
    m_te = classification_metrics(yte, yhat_te)

    st.session_state.history_metrics.append({"iter": it, "train": m_tr, "test": m_te})


def _next_step_projection(
    model: LogisticRegressionScratch,
    batch_idx: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if batch_idx is None or len(batch_idx) == 0:
        return None

    Xb = st.session_state.X[batch_idx]
    yb = st.session_state.y[batch_idx]
    dw, db = model.gradients(Xb, yb, reg_type=str(cfg["reg_type"]), reg_strength=float(cfg["reg_strength"]))
    lr = float(cfg["lr"])

    current = {f"w{j}": float(model.w[j]) for j in range(model.n_features)}
    current["b"] = float(model.b)

    nxt = {f"w{j}": float(model.w[j] - lr * dw[j]) for j in range(model.n_features)}
    nxt["b"] = float(model.b - lr * db)

    return {
        "dw": dw,
        "db": db,
        "current": current,
        "next": nxt,
        "iter_from": int(st.session_state.iter),
        "iter_to": int(st.session_state.iter) + 1,
    }


def training_step() -> bool:
    cfg = _current_model_config_from_widgets()
    model = _get_model()

    X = st.session_state.X
    y = st.session_state.y
    train_idx = st.session_state.train_idx

    if train_idx.size == 0:
        st.session_state.last_step = {"error": "Train set is empty. Increase train split or number of samples."}
        return False

    batch_idx = _ensure_next_batch_indices()
    if batch_idx is None or len(batch_idx) == 0:
        st.session_state.last_step = {"error": "Unable to sample a batch (train set empty)."}
        return False

    Xb = X[batch_idx]
    yb = y[batch_idx]

    step_info = model.step(
        Xb,
        yb,
        lr=float(cfg["lr"]),
        reg_type=str(cfg["reg_type"]),
        reg_strength=float(cfg["reg_strength"]),
    )

    st.session_state.iter = int(st.session_state.iter) + 1
    _write_model_params_to_widgets(model.w, model.b)

    # Consumed batch
    st.session_state.next_batch_idx = None

    st.session_state.last_step = {
        **step_info,
        "batch_size": int(len(batch_idx)),
        "iter": int(st.session_state.iter),
    }

    _record_history(model)
    return True


def train_all() -> None:
    target = int(st.session_state.n_iters)
    # Train until iter reaches target
    while int(st.session_state.iter) < target:
        ok = training_step()
        if not ok:
            # Avoid infinite loops when training cannot proceed
            st.session_state.dynamic_train = False
            break


def reset_all() -> None:
    # Reset dataset using applied config + current seed, reset parameters
    applied = sanitize_dataset_config(st.session_state.ds_cfg_applied)
    _regenerate_dataset(applied)
    d = int(applied["n_features"])
    _randomize_model_params_from_seed(seed=int(st.session_state.seed), n_features=d)
    _reset_training_state(keep_params=True)
    st.session_state.train_rng = np.random.default_rng(int(st.session_state.seed))
    st.session_state.dynamic_train = False


def start_dynamic_train() -> None:
    st.session_state.dynamic_train = True


def stop_dynamic_train() -> None:
    st.session_state.dynamic_train = False



def _rerun() -> None:
    # Compatibility with older Streamlit versions
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Logistic Regression Trainer (Teaching App)", layout="wide")

_ensure_core_state()
_maybe_apply_dataset_updates()
_ensure_param_dim_matches_dataset()

# If batch size changes, invalidate the cached preview batch
if "batch_size_prev" not in st.session_state:
    st.session_state.batch_size_prev = int(st.session_state.batch_size)
if int(st.session_state.batch_size_prev) != int(st.session_state.batch_size):
    st.session_state.next_batch_idx = None
    st.session_state.batch_size_prev = int(st.session_state.batch_size)

# Dynamic training: do one step per run; schedule the next run at the end of the script.
if bool(st.session_state.dynamic_train):
    target = int(st.session_state.n_iters)
    if int(st.session_state.iter) < target:
        ok = training_step()
        if not ok:
            st.session_state.dynamic_train = False
    else:
        st.session_state.dynamic_train = False


# Layout: left configuration column, right visualization column
left, main = st.columns([1.1, 2.4], gap="large")


# -----------------------------
# Left column - Configuration
# -----------------------------
with left:
    st.markdown("## Configuration")

    with st.expander("General", expanded=True):
        st.number_input("Seed", min_value=0, max_value=10000, step=1, key="seed")

    with st.expander("Dataset", expanded=True):
        st.checkbox("Dynamic dataset modification", key="ds_dynamic")
        if not bool(st.session_state.ds_dynamic):
            st.caption("Dynamic mode is OFF: changing dataset sliders will not regenerate the dataset until you re-enable it.")

        st.slider("Number of samples", min_value=10, max_value=1000, step=10, key="ds_n_samples")
        st.slider("Balanced (proportion of class 1)", min_value=0.0, max_value=1.0, step=0.01, key="ds_balance")
        st.radio("Number of features", options=[1, 2], horizontal=True, key="ds_n_features")

        # Means/stds
        d = int(st.session_state.ds_n_features)
        for c in [0, 1]:
            st.markdown(f"**Class {c}**")
            for f in range(d):
                cols = st.columns(2)
                with cols[0]:
                    st.number_input(f"Mean (feature {f+1})", key=f"ds_mean_c{c}_f{f}", step=0.1, format="%.3f")
                with cols[1]:
                    st.number_input(f"Std (feature {f+1})", key=f"ds_std_c{c}_f{f}", step=0.1, min_value=1e-6, format="%.3f")

        st.slider("Train/Test split (train proportion)", min_value=0.0, max_value=0.9, step=0.01, key="ds_train_split")

        # Show currently applied config (may differ when dynamic is OFF)
        applied = sanitize_dataset_config(st.session_state.ds_cfg_applied)
        pending = _pending_dataset_config_from_widgets()
        st.markdown("**Applied vs Pending**")
        st.write(
            {
                "applied": {
                    "n_samples": applied["n_samples"],
                    "balance": applied["balance"],
                    "n_features": applied["n_features"],
                    "train_split": applied["train_split"],
                    "means0": applied["means0"],
                    "stds0": applied["stds0"],
                    "means1": applied["means1"],
                    "stds1": applied["stds1"],
                },
                "pending": {
                    "n_samples": pending["n_samples"],
                    "balance": pending["balance"],
                    "n_features": pending["n_features"],
                    "train_split": pending["train_split"],
                    "means0": pending["means0"],
                    "stds0": pending["stds0"],
                    "means1": pending["means1"],
                    "stds1": pending["stds1"],
                },
            }
        )

    with st.expander("Model", expanded=True):
        st.slider("Learning rate", min_value=0.0001, max_value=10.0, value=float(st.session_state.lr), step=0.0001, key="lr")
        st.slider("Number of iterations (target)", min_value=1, max_value=10000, step=1, key="n_iters")
        st.selectbox("Regularization", options=["None", "L1", "L2"], key="reg_type")
        st.slider("Regularization strength (λ)", min_value=0.0001, max_value=10.0, step=0.0001, key="reg_strength")
        st.slider("Batch size", min_value=1, max_value=1000, step=1, key="batch_size")

    with st.expander("Parameters", expanded=True):
        d_applied = int(sanitize_dataset_config(st.session_state.ds_cfg_applied)["n_features"])
        st.slider("w0", min_value=-10.0, max_value=10.0, step=0.01, key="w0")
        if d_applied == 2:
            st.slider("w1", min_value=-10.0, max_value=10.0, step=0.01, key="w1")
        st.slider("b (bias)", min_value=-10.0, max_value=10.0, step=0.01, key="b")

        w_now, b_now = _model_params_from_widgets(d_applied)
        st.code(f"Current parameters:\nw = {w_now}\nb = {b_now:.6f}")

    with st.expander("Training", expanded=True):
        st.button("Train All", on_click=train_all)
        st.button("Step", on_click=training_step)
        st.button("Reset", on_click=reset_all)
        st.slider("Arrow amplification", min_value=1.0, max_value=40.0, step=0.5, key="param_arrow_scale")
        st.slider("3D gradient arrow amplification", min_value=1.0, max_value=40.0, step=0.5, key="surface_arrow_scale")

        st.markdown(f"**Iteration:** {int(st.session_state.iter)} / {int(st.session_state.n_iters)}")

    with st.expander("Dynamic training playback", expanded=False):
        if not bool(st.session_state.dynamic_train):
            st.button("Start dynamic train", on_click=start_dynamic_train)
        else:
            st.button("Pause dynamic train", on_click=stop_dynamic_train)
        st.slider("Frame rate (FPS)", min_value=1, max_value=30, step=1, key="dynamic_fps")


# -----------------------------
# Central panel - Visualization
# -----------------------------
with main:
    st.markdown("## Visualization")

    # Build model and get data
    applied_cfg = sanitize_dataset_config(st.session_state.ds_cfg_applied)
    d_applied = int(applied_cfg["n_features"])
    X = st.session_state.X
    y = st.session_state.y
    df = st.session_state.df
    model = _get_model()

    Xtr, ytr, Xte, yte = _get_train_test()

    # Current evaluation metrics
    yhat_tr = model.predict(Xtr) if Xtr.size else np.array([], dtype=int)
    yhat_te = model.predict(Xte) if Xte.size else np.array([], dtype=int)
    m_tr = classification_metrics(ytr, yhat_tr)
    m_te = classification_metrics(yte, yhat_te)

    with st.expander("Dataset visualization", expanded=True):
        if df is None or len(df) == 0:
            st.warning("Dataset is empty.")
        else:
            if d_applied == 1:
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_dataset_1d(df), use_container_width=True)
                with c2:
                    st.plotly_chart(fig_hist_1d(df), use_container_width=True)
            else:
                st.plotly_chart(fig_dataset_2d(df), use_container_width=True)

    with st.expander("Decision boundary", expanded=True):
        if df is None or len(df) == 0:
            st.warning("Dataset is empty.")
        else:
            if d_applied == 1:
                x1 = df["x1"].to_numpy(dtype=float)
                x_min, x_max = float(np.min(x1)), float(np.max(x1))
                pad = 0.2 * max(1e-6, (x_max - x_min))
                x_grid = np.linspace(x_min - pad, x_max + pad, 300)
                p_grid = model.predict_proba(x_grid.reshape(-1, 1))

                w = model.w[0]
                boundary_x = None
                if abs(w) > 1e-12:
                    boundary_x = -model.b / w

                st.plotly_chart(
                    fig_decision_boundary_1d(df, x_grid=x_grid, p_grid=p_grid, boundary_x=boundary_x),
                    use_container_width=True,
                )
            else:
                x1 = df["x1"].to_numpy(dtype=float)
                x2 = df["x2"].to_numpy(dtype=float)
                x1_min, x1_max = float(np.min(x1)), float(np.max(x1))
                x2_min, x2_max = float(np.min(x2)), float(np.max(x2))
                pad1 = 0.15 * max(1e-6, (x1_max - x1_min))
                pad2 = 0.15 * max(1e-6, (x2_max - x2_min))

                x1_vals = np.linspace(x1_min - pad1, x1_max + pad1, 120)
                x2_vals = np.linspace(x2_min - pad2, x2_max + pad2, 120)
                xx, yy = np.meshgrid(x1_vals, x2_vals)
                grid = np.c_[xx.ravel(), yy.ravel()]
                pp = model.predict_proba(grid).reshape(xx.shape)

                w0, w1 = float(model.w[0]), float(model.w[1])
                b = float(model.b)
                boundary: Dict[str, Any] = {"type": "none"}
                if abs(w1) > 1e-12:
                    x1_line = np.array([x1_vals.min(), x1_vals.max()])
                    x2_line = -(w0 * x1_line + b) / w1
                    boundary = {"type": "line", "x1_line": x1_line, "x2_line": x2_line}
                elif abs(w0) > 1e-12:
                    boundary = {"type": "vertical", "x": -b / w0}

                st.plotly_chart(
                    fig_decision_boundary_2d(df, x1_vals=x1_vals, x2_vals=x2_vals, p_grid=pp, boundary=boundary),
                    use_container_width=True,
                )

    with st.expander("Parameters evolution", expanded=True):
        batch_idx = _ensure_next_batch_indices()
        cfg = _current_model_config_from_widgets()
        next_step = _next_step_projection(model, batch_idx, cfg)

        # Plot history
        st.plotly_chart(
            fig_params_history(
                history=st.session_state.history_params,
                n_features=d_applied,
                current_iter=int(st.session_state.iter),
                w=model.w,
                b=model.b,
                next_step=next_step,
                arrow_scale=float(st.session_state.param_arrow_scale),
            ),
            use_container_width=True,
        )

        # Preview gradient on the *next* batch (consistent with Step)
        if batch_idx is None or len(batch_idx) == 0:
            st.info("No train batch available (train set empty).")
        else:
            Xb = st.session_state.X[batch_idx]
            yb = st.session_state.y[batch_idx]
            dw = next_step["dw"]
            db = next_step["db"]

            st.markdown("### Gradient descent derivation (used by this app)")
            st.latex(r"z = Xw + b")
            st.latex(r"\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}")
            st.latex(r"J(w,b) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right] + \lambda\,R(w)")
            st.latex(r"\frac{\partial J}{\partial w} = \frac{1}{m}X^T(\hat{y}-y) + \lambda\,\frac{\partial R}{\partial w}")
            st.latex(r"\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^m (\hat{y}^{(i)}-y^{(i)})")
            st.latex(r"w \leftarrow w - \alpha\,\frac{\partial J}{\partial w},\quad b \leftarrow b - \alpha\,\frac{\partial J}{\partial b}")

            lr = float(cfg["lr"])
            w_next = model.w - lr * dw
            b_next = model.b - lr * db

            rows = []
            for j in range(d_applied):
                rows.append(
                    {
                        "parameter": f"w{j}",
                        "current": float(model.w[j]),
                        "dJ/dw": float(dw[j]),
                        "next (after 1 step)": float(w_next[j]),
                    }
                )
            rows.append({"parameter": "b", "current": float(model.b), "dJ/db": float(db), "next (after 1 step)": float(b_next)})

            st.markdown("### Derivatives on the next batch")
            st.dataframe(rows, use_container_width=True)

            st.caption(f"Batch used for preview: {len(batch_idx)} samples (respects batch size setting).")

        if st.session_state.last_step is not None:
            st.markdown("### Last training step (executed)")
            st.write(st.session_state.last_step)

    with st.expander("Loss surface 3D (J vs weight and b)", expanded=True):
        if Xtr.size == 0 or ytr.size == 0:
            st.info("Train set is empty, so the loss surface cannot be computed.")
        else:
            st.checkbox(
                "Update surface during dynamic training (slower)",
                key="surface_live_during_dynamic",
            )
            pause_surface = bool(st.session_state.dynamic_train) and not bool(st.session_state.surface_live_during_dynamic)
            if pause_surface:
                st.info(
                    "Surface updates are paused during dynamic training to keep playback fluid. "
                    "Enable the checkbox above to force live surface updates."
                )
            else:
                if d_applied > 1:
                    st.radio(
                        "Weight axis for surface",
                        options=[0, 1],
                        format_func=lambda idx: f"w{idx}",
                        horizontal=True,
                        key="surface_w_idx",
                    )
                w_idx = max(0, min(d_applied - 1, int(st.session_state.surface_w_idx)))
                st.slider("Surface resolution", min_value=21, max_value=101, step=2, key="surface_grid_n")

                cfg = _current_model_config_from_widgets()
                reg_type = str(cfg["reg_type"])
                reg_strength = float(cfg["reg_strength"])

                model_tmp = LogisticRegressionScratch(n_features=d_applied, w=model.w.copy(), b=float(model.b))
                current_w = float(model.w[w_idx])
                current_b = float(model.b)

                history = list(st.session_state.history_params)
                hist_w = [float(h["w"][w_idx]) for h in history if len(h.get("w", [])) > w_idx]
                hist_b = [float(h["b"]) for h in history]
                all_w = hist_w + [current_w]
                all_b = hist_b + [current_b]

                w_min, w_max = min(all_w), max(all_w)
                b_min, b_max = min(all_b), max(all_b)
                span_w = max(0.8, w_max - w_min)
                span_b = max(0.8, b_max - b_min)

                w_vals = np.linspace(w_min - 0.5 * span_w, w_max + 0.5 * span_w, int(st.session_state.surface_grid_n))
                b_vals = np.linspace(b_min - 0.5 * span_b, b_max + 0.5 * span_b, int(st.session_state.surface_grid_n))
                j_grid = np.zeros((len(b_vals), len(w_vals)), dtype=float)

                for i, b_val in enumerate(b_vals):
                    for j, w_val in enumerate(w_vals):
                        w_vec = model.w.copy()
                        w_vec[w_idx] = float(w_val)
                        model_tmp.set_params(w_vec, float(b_val))
                        j_grid[i, j] = model_tmp.loss(Xtr, ytr, reg_type=reg_type, reg_strength=reg_strength)

                model_tmp.set_params(model.w.copy(), model.b)
                current_j = model_tmp.loss(Xtr, ytr, reg_type=reg_type, reg_strength=reg_strength)

                history_points = {"w": [], "b": [], "j": [], "text": []}
                for h in history:
                    w_hist = np.asarray(h.get("w", []), dtype=float).reshape(-1)
                    b_hist = float(h.get("b", 0.0))
                    if w_hist.size == 0:
                        continue
                    model_tmp.set_params(w_hist, b_hist)
                    j_hist = model_tmp.loss(Xtr, ytr, reg_type=reg_type, reg_strength=reg_strength)
                    history_points["w"].append(float(w_hist[w_idx] if w_hist.size > w_idx else w_hist[0]))
                    history_points["b"].append(b_hist)
                    history_points["j"].append(float(j_hist))
                    history_points["text"].append(f"iter={h.get('iter', '?')}")

                gradient_arrow = None
                next_step = _next_step_projection(model, batch_idx, cfg)
                if next_step is not None:
                    arrow_scale = float(st.session_state.surface_arrow_scale)
                    w1 = current_w + arrow_scale * (float(next_step["next"][f"w{w_idx}"]) - current_w)
                    b1 = current_b + arrow_scale * (float(next_step["next"]["b"]) - current_b)
                    w_vec = model.w.copy()
                    w_vec[w_idx] = w1
                    model_tmp.set_params(w_vec, b1)
                    j1 = model_tmp.loss(Xtr, ytr, reg_type=reg_type, reg_strength=reg_strength)
                    gradient_arrow = {"w0": current_w, "b0": current_b, "j0": float(current_j), "w1": w1, "b1": b1, "j1": float(j1)}

                st.plotly_chart(
                    fig_loss_surface_3d(
                        w_vals=w_vals,
                        b_vals=b_vals,
                        j_grid=j_grid,
                        current_point={"w": current_w, "b": current_b, "j": float(current_j)},
                        history_points=history_points,
                        gradient_arrow=gradient_arrow,
                        w_label=f"w{w_idx}",
                    ),
                    use_container_width=True,
                )

    with st.expander("Evaluation", expanded=True):
        # Current metrics
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Current (train)")
            st.metric("Accuracy", f"{m_tr['accuracy']:.3f}" if np.isfinite(m_tr["accuracy"]) else "nan")
            st.metric("Precision", f"{m_tr['precision']:.3f}" if np.isfinite(m_tr["precision"]) else "nan")
            st.metric("Recall", f"{m_tr['recall']:.3f}" if np.isfinite(m_tr["recall"]) else "nan")
            st.metric("F1", f"{m_tr['f1']:.3f}" if np.isfinite(m_tr["f1"]) else "nan")
        with c2:
            st.markdown("### Current (test)")
            st.metric("Accuracy", f"{m_te['accuracy']:.3f}" if np.isfinite(m_te["accuracy"]) else "nan")
            st.metric("Precision", f"{m_te['precision']:.3f}" if np.isfinite(m_te["precision"]) else "nan")
            st.metric("Recall", f"{m_te['recall']:.3f}" if np.isfinite(m_te["recall"]) else "nan")
            st.metric("F1", f"{m_te['f1']:.3f}" if np.isfinite(m_te["f1"]) else "nan")

        st.plotly_chart(fig_metrics_history(st.session_state.history_metrics), use_container_width=True)

        # ROC
        # Use probabilities on train/test
        train_roc = None
        test_roc = None
        if Xtr.size and ytr.size:
            p_tr = model.predict_proba(Xtr)
            train_roc = roc_curve_auc(ytr, p_tr)
        if Xte.size and yte.size:
            p_te = model.predict_proba(Xte)
            test_roc = roc_curve_auc(yte, p_te)

        st.plotly_chart(fig_roc(train_roc, test_roc), use_container_width=True)

        if train_roc is not None:
            st.write({"train_auc": train_roc.get("auc"), "train_auc_defined": train_roc.get("defined", False)})
        if test_roc is not None:
            st.write({"test_auc": test_roc.get("auc"), "test_auc_defined": test_roc.get("defined", False)})


# Schedule next frame for dynamic training (after rendering).
if bool(st.session_state.dynamic_train) and int(st.session_state.iter) < int(st.session_state.n_iters):
    fps = max(1, int(st.session_state.dynamic_fps))
    frame_interval_s = 1.0 / float(fps)
    time.sleep(frame_interval_s)
    _rerun()
