from __future__ import annotations

import json

import streamlit as st

from ml_classification_study.datasets import PreprocessConfig, make_dataset_bundle
from ml_classification_study.datasets.registry import get_dataset_cls, list_datasets
from ml_classification_study.types import DatasetBundle
from ml_classification_study.ui.components import plotly_chart_stretch, render_arg_specs
from ml_classification_study.ui.state import (
    KEY_DATASET_BUNDLE,
    KEY_DATASET_GEN_ARGS,
    KEY_DATASET_ID,
    KEY_PREPROCESS,
    fingerprint,
)
from ml_classification_study.viz import plot_dataset_scatter


@st.cache_data(show_spinner=False)
def _cached_bundle(dataset_id: str, gen_args_json: str, preprocess_json: str) -> DatasetBundle:
    gen_args = json.loads(gen_args_json)
    preprocess_dict = json.loads(preprocess_json)
    cfg = PreprocessConfig(**preprocess_dict)
    return make_dataset_bundle(dataset_id, gen_args=gen_args, preprocess=cfg)


def render() -> None:
    col_cfg, col_main = st.columns([1, 2], gap="large")

    dataset_classes = list_datasets()
    dataset_ids = [cls.id for cls in dataset_classes]

    if not dataset_ids:
        st.error("No datasets registered.")
        return

    default_id = st.session_state.get(KEY_DATASET_ID, dataset_ids[0])
    if default_id not in dataset_ids:
        default_id = dataset_ids[0]

    with col_cfg:
        st.subheader("Dataset configuration")

        dataset_id = st.selectbox(
            "Dataset",
            options=dataset_ids,
            index=dataset_ids.index(default_id),
            format_func=lambda _id: get_dataset_cls(_id).name,
            key=f"{KEY_DATASET_ID}_widget",
        )

        ds_cls = get_dataset_cls(dataset_id)
        if getattr(ds_cls, "description", ""):
            st.caption(ds_cls.description)

        st.markdown("### Generator parameters")
        gen_args = render_arg_specs(ds_cls.full_arg_spec(), key_prefix=f"ds:{dataset_id}")

        st.markdown("### Preprocessing")
        scaling = st.selectbox(
            "Scaling",
            options=["none", "standard", "minmax"],
            index=0,
            help="Applied after split and outlier removal. Fit on train, applied to test.",
            key="pre:scaling",
        )
        outlier_std = st.slider(
            "Outlier removal (std)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            help="0 disables. Points outside mean \u00b1 k*std (per-feature) are removed (stats from train).",
            key="pre:outlier_std",
        )
        test_size = st.slider(
            "Test split fraction",
            min_value=0.05,
            max_value=0.5,
            value=0.25,
            step=0.05,
            key="pre:test_size",
        )
        stratify = st.checkbox("Stratify split", value=True, key="pre:stratify")

        preprocess_dict = {
            "scaling": scaling,
            "outlier_std": float(outlier_std),
            "test_size": float(test_size),
            "stratify": bool(stratify),
        }

        # Persist configs for other tabs
        st.session_state[KEY_DATASET_ID] = dataset_id
        st.session_state[KEY_DATASET_GEN_ARGS] = gen_args
        st.session_state[KEY_PREPROCESS] = preprocess_dict

    with col_main:
        st.subheader("Dataset preview")

        try:
            bundle = _cached_bundle(dataset_id, fingerprint(gen_args), fingerprint(preprocess_dict))
        except Exception as e:
            st.error(f"Dataset generation failed: {e}")
            return

        st.session_state[KEY_DATASET_BUNDLE] = bundle

        fig = plot_dataset_scatter(bundle)
        plotly_chart_stretch(st, fig)

        meta = bundle.meta or {}
        removed_train = meta.get("removed_train", 0)
        removed_test = meta.get("removed_test", 0)

        c0_train = int((bundle.y_train == 0).sum())
        c1_train = int((bundle.y_train == 1).sum())
        c0_test = int((bundle.y_test == 0).sum())
        c1_test = int((bundle.y_test == 1).sum())

        st.markdown("### Summary")
        st.write(
            {
                "train_size": int(meta.get("n_train", len(bundle.y_train))),
                "test_size": int(meta.get("n_test", len(bundle.y_test))),
                "train_class_counts": {"0": c0_train, "1": c1_train},
                "test_class_counts": {"0": c0_test, "1": c1_test},
                "outliers_removed": {"train": int(removed_train), "test": int(removed_test)},
                "scaling": meta.get("scaler_type", "none") if meta.get("scaler") is not None else "none",
            }
        )

        if len(set(bundle.y_train.tolist())) < 2 or len(set(bundle.y_test.tolist())) < 2:
            st.warning(
                "One of the splits contains a single class after outlier removal. Some metrics (ROC/AUC) may be undefined."
            )
