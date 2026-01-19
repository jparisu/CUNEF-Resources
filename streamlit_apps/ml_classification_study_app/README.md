# ML Classification Study (Streamlit)

Interactive Streamlit app to explore 2D synthetic datasets and classic classification models (KNN, Naive Bayes, Decision Tree).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)

pip install -r requirements.txt
streamlit run app.py
```

## Project layout

- `app.py`: Streamlit entrypoint (tabs: Dataset Selection, Model Visualization)
- `src/ml_classification_study/`: library code
  - `datasets/`: dataset generators + preprocessing
  - `models/`: model interface + implementations
  - `viz/`: plotly visualizations
  - `ui/`: streamlit UI components

## Notes

- All datasets are 2D to support decision-boundary visualization.
- Step-by-step training is model-dependent:
  - KNN: gradually increases the number of stored training points
  - Naive Bayes: incremental updates over mini-batches
  - Decision Tree: refits with increasing `max_depth`
