from pathlib import Path
import sys

import streamlit as st


# Allow running `streamlit run app.py` without installing the package.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_classification_study.ui.dataset_tab import render as render_dataset_tab
from ml_classification_study.ui.model_tab import render as render_model_tab


st.set_page_config(
    page_title="ML Classification Study",
    layout="wide",
)

st.title("ML Classification Study")

tabs = st.tabs(["Dataset Selection", "Model Visualization"])

with tabs[0]:
    render_dataset_tab()

with tabs[1]:
    render_model_tab()
