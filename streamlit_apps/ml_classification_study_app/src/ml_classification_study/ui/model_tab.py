from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from ml_classification_study.models.registry import get_model_cls, list_models
from ml_classification_study.ui.components import button_stretch, plotly_chart_stretch, render_arg_specs
from ml_classification_study.ui.state import (
    KEY_DATASET_BUNDLE,
    KEY_DATASET_GEN_ARGS,
    KEY_DATASET_ID,
    KEY_MODEL_FINGERPRINT,
    KEY_MODEL_HYPERPARAMS,
    KEY_MODEL_ID,
    KEY_MODEL_INSTANCE,
    KEY_PREPROCESS,
    KEY_TRAIN_STEP_MESSAGE,
    fingerprint,
)
from ml_classification_study.viz import plot_confusion_matrix, plot_decision_boundary, plot_roc_curve


def _render_params_view(params: Dict[str, Any]) -> None:
    """Best-effort generic parameter view."""

    if not params:
        st.write("(no parameters)")
        return

    # Special-case a tree text if present
    tree_text = params.pop("tree_text", None)
    if tree_text is not None:
        st.markdown("#### Tree structure")
        st.code(tree_text, language="text")

    # Remaining parameters
    st.markdown("#### Parameters")
    st.write(params)


def render() -> None:
    bundle = st.session_state.get(KEY_DATASET_BUNDLE, None)
    if bundle is None:
        st.info("Generate a dataset in the 'Dataset Selection' tab first.")
        return

    col_cfg, col_main = st.columns([1, 2], gap="large")

    model_classes = list_models()
    model_ids = [cls.id for cls in model_classes]

    if not model_ids:
        st.error("No models registered.")
        return

    default_model_id = st.session_state.get(KEY_MODEL_ID, model_ids[0])
    if default_model_id not in model_ids:
        default_model_id = model_ids[0]

    with col_cfg:
        st.subheader("Model configuration")

        model_id = st.selectbox(
            "Model",
            options=model_ids,
            index=model_ids.index(default_model_id),
            format_func=lambda _id: get_model_cls(_id).name,
            key=f"{KEY_MODEL_ID}_widget",
        )
        model_cls = get_model_cls(model_id)
        if getattr(model_cls, "description", ""):
            st.caption(model_cls.description)

        st.markdown("### Hyperparameters")
        # Use dataset-id in prefix to avoid collisions when switching datasets/models
        hyperparams = render_arg_specs(model_cls.arg_spec(), key_prefix=f"model:{model_id}")

        # Store hyperparams/model selection
        st.session_state[KEY_MODEL_ID] = model_id
        st.session_state[KEY_MODEL_HYPERPARAMS] = hyperparams

        st.markdown("### Training")
        eval_split = st.selectbox("Evaluate on", options=["test", "train"], index=0, key="eval_split")

        btn_cols = st.columns(3)
        reset_clicked = button_stretch(btn_cols[0], "Reset")
        train_clicked = button_stretch(btn_cols[1], "Train")
        step_clicked = button_stretch(btn_cols[2], "Train step", disabled=True)

        # Fingerprint model + dataset config for invalidation
        ds_id = st.session_state.get(KEY_DATASET_ID, "")
        ds_args = st.session_state.get(KEY_DATASET_GEN_ARGS, {})
        ds_pre = st.session_state.get(KEY_PREPROCESS, {})

        fp = fingerprint(
            {
                "dataset_id": ds_id,
                "dataset_args": ds_args,
                "dataset_pre": ds_pre,
                "model_id": model_id,
                "hyperparams": hyperparams,
            }
        )

        current_fp = st.session_state.get(KEY_MODEL_FINGERPRINT, None)

        if reset_clicked or (current_fp is not None and current_fp != fp):
            st.session_state[KEY_MODEL_INSTANCE] = model_cls(**hyperparams)
            st.session_state[KEY_MODEL_FINGERPRINT] = fp
            st.session_state[KEY_TRAIN_STEP_MESSAGE] = "Model reset."

        # Ensure model exists
        if st.session_state.get(KEY_MODEL_INSTANCE, None) is None:
            st.session_state[KEY_MODEL_INSTANCE] = model_cls(**hyperparams)
            st.session_state[KEY_MODEL_FINGERPRINT] = fp

        model = st.session_state[KEY_MODEL_INSTANCE]

        # Train controls
        if train_clicked:
            model.reset()
            model.train(bundle.X_train, bundle.y_train)
            st.session_state[KEY_TRAIN_STEP_MESSAGE] = "Trained (full)."

        if step_clicked:
            res = model.train_one_step(bundle.X_train, bundle.y_train)
            st.session_state[KEY_TRAIN_STEP_MESSAGE] = res.message

        msg = st.session_state.get(KEY_TRAIN_STEP_MESSAGE, "")
        if msg:
            st.info(msg)

    with col_main:
        st.subheader("Model visualization")

        model = st.session_state.get(KEY_MODEL_INSTANCE, None)
        if model is None or not getattr(model, "is_trained", False):
            st.info("Train the model to see decision boundaries and metrics.")
            return

        # Decision boundary plot
        try:
            fig = plot_decision_boundary(model, bundle)
            plotly_chart_stretch(st, fig)
        except Exception as e:
            st.error(f"Decision boundary plotting failed: {e}")

        # Metrics
        if eval_split == "test":
            Xe, ye = bundle.X_test, bundle.y_test
            split_name = "Test"
        else:
            Xe, ye = bundle.X_train, bundle.y_train
            split_name = "Train"

        st.markdown(f"### Metrics ({split_name})")
        try:
            acc = model.acc(Xe, ye)
            f1 = model.f1score(Xe, ye)
        except Exception as e:
            st.error(f"Metric computation failed: {e}")
            return

        met_cols = st.columns(2)
        met_cols[0].metric("Accuracy", f"{acc:.3f}")
        met_cols[1].metric("F1 score", f"{f1:.3f}")

        # Confusion matrix
        try:
            cm = model.confussion_matrix(Xe, ye)
            plotly_chart_stretch(st, plot_confusion_matrix(cm))
        except Exception as e:
            st.error(f"Confusion matrix failed: {e}")

        # ROC curve
        roc = None
        try:
            roc = model.roc_curve(Xe, ye)
        except Exception:
            roc = None

        if roc is None:
            st.info("ROC curve not available (needs predict_proba and both classes in the evaluation split).")
        else:
            plotly_chart_stretch(st, plot_roc_curve(roc))

        # Trained parameters
        st.markdown("### Trained parameters")
        try:
            params = model.trained_parameters()
        except Exception as e:
            st.error(f"Could not read trained parameters: {e}")
            return

        _render_params_view(dict(params))
