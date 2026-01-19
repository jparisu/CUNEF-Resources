# ML Classification Study App (Streamlit)

A modular Streamlit application to help students explore **binary classification** problems end-to-end:

- Generate synthetic **2-class datasets**
- Apply common **preprocessing** steps
- Train and visualize **classification models**
- Inspect **decision boundaries**, **metrics**, and **trained parameters**
- Designed for **scalability**: add new datasets/models via registries and declarative argument specs

---

## Features

### Dataset generation (2-class)
Built-in generators (extendable):
- **Gaussian (2-class)**: class-specific mean/std
- **XOR (2D)**
- **Normal vs Exponential**
- **Random linearly separable** (random separating line)

All generators output `X ∈ R^(N×2)` and labels `y ∈ {0,1}`.

### Preprocessing pipeline
Applied consistently across datasets:
- **Scaling / Normalization**
  - `none`
  - `standard` (z-score)
  - `minmax`
  - `normalize` (row-wise L2)
- **Outlier removal**
  - removes samples beyond a configurable number of standard deviations
- **Train/test split**
  - configurable proportion
  - optional stratification (recommended)

### Models (shared interface)
Initial models:
- **KNN**
- **Gaussian Naive Bayes**
- **Decision Tree**

Each model exposes the same interface so the UI can treat them generically:
- `train(...)` (fit all at once)
- `train_one_step(...)` (optional; may be disabled)
- `predict(...)`
- Metrics: accuracy, F1, confusion matrix, ROC curve (if supported)
- `trained_parameters()` for model introspection

### Visualization
- Dataset view: Plotly scatter with train/test styling and class-consistent colors
- Model view:
  - decision boundary over the dataset
  - metrics (accuracy, F1)
  - confusion matrix heatmap
  - ROC curve (if the model provides probabilities)
  - trained parameter display (e.g., tree text for Decision Tree)

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run the app

```bash
streamlit run app.py
```

---

## Adding new datasets or models

Use the provided templates:

- `datasets/template.py`
- `models/template.py`

Register them in the corresponding `registry.py` file and they will appear automatically in the UI.

---

## License
Educational / internal use.
