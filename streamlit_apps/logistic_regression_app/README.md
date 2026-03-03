# Streamlit app: Logistic Regression Training Visualizer

## Run locally

```bash
pip install streamlit plotly numpy pandas
streamlit run app.py

## Cloud compatibility check (local)

Run this before pushing to catch dependency/runtime issues similar to Streamlit Cloud:

```bash
bash scripts/cloud_compat_check.sh
```
```

## Files

- `app.py`: Streamlit UI + session state + training buttons
- `dataset.py`: Synthetic Gaussian dataset generation
- `lr_model.py`: Logistic regression implemented from scratch (sigmoid, cross-entropy, gradients, GD step)
- `metrics.py`: Accuracy/precision/recall/F1 + ROC/AUC (no sklearn dependency)
- `viz.py`: Plotly figures for dataset, decision boundary, parameter/metric evolution, ROC
