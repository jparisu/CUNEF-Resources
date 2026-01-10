"""Model benchmark script (text-only output).

Requirements satisfied:
1) Uses an in-situ generated dataset (no external CSV required).
2) Prints results in text (no output files generated).

Models evaluated:
0) Prior model: always predicts most frequent class in train
1) Basic models with minimal preprocessing: KNN, Logistic Regression, Decision Tree
2) Divided model: two binary models (cost_pos and sales_pos) + recombination
3) Mechanism-aware model: engineered features approximating hidden mechanism (without constants)

Run:
  python model_benchmarks.py --N 20000 --seed 42 --outlier_prob 0.01
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from enterprise_risk_generator import ClassProportions, GeneratorConstants, generate_dataset


CLASS_ORDER = ["healthy", "cost_risk", "sales_risk", "both_risks"]


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


class MarketFocusJoiner(BaseEstimator, TransformerMixin):
    """Turns list[str] into a single string 'tag1|tag2' for one-hot encoding."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for v in np.asarray(X).ravel():
            if isinstance(v, list):
                out.append("|".join(map(str, v)) if len(v) else "EMPTY")
            elif isinstance(v, str):
                out.append(v)
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                out.append("MISSING")
            else:
                out.append("INVALID")
        return np.array(out, dtype=object).reshape(-1, 1)


def basic_preprocess_pipeline(model) -> Pipeline:
    """Minimal preprocessing: impute + scale numeric; join market_focus and one-hot it."""
    numeric_features = ["revenue", "costs", "audited", "ceo_tenure_years"]
    cat_features = ["market_focus"]

    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])

    cat_pipe = Pipeline(steps=[
        ("join", MarketFocusJoiner()),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return Pipeline(steps=[("pre", pre), ("model", model)])


class MechanismFeatures(BaseEstimator, TransformerMixin):
    """Mechanism-aware feature engineering without using generator constants.

    Creates intermediate features resembling the hidden mechanism:
      - log(revenue) standardized (fit on train)
      - profit_margin = (revenue - costs) / revenue
      - p_star based on train median/IQR of profit_margin
      - nonlinearities: max(0, p_star)^2 and r_star^2
      - market multi-hot for the known tags
    """

    def __init__(self, known_tags: Optional[List[str]] = None):
        self.known_tags = known_tags or ["Industrial", "Ads", "Contract", "Scale"]
        self.logrev_mean_ = None
        self.logrev_std_ = None
        self.pm_median_ = None
        self.pm_iqr_ = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        rev = safe_numeric(df["revenue"]).clip(lower=1.0)
        logrev = np.log(rev)
        self.logrev_mean_ = float(np.nanmean(logrev))
        self.logrev_std_ = float(np.nanstd(logrev) + 1e-9)

        costs = safe_numeric(df["costs"])
        pm = (rev - costs) / rev
        pm = pm.replace([np.inf, -np.inf], np.nan)

        self.pm_median_ = float(np.nanmedian(pm))
        q1, q3 = np.nanquantile(pm, [0.25, 0.75])
        self.pm_iqr_ = float((q3 - q1) + 1e-9)
        return self

    def _normalize_market(self, v):
        if isinstance(v, list):
            return [str(t) for t in v]
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                inner = v.strip("[]").strip()
                if not inner:
                    return []
                parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
                return [p for p in parts if p]
            return [v]
        return []

    def transform(self, X):
        df = X.copy()

        rev = safe_numeric(df["revenue"]).clip(lower=1.0)
        costs = safe_numeric(df["costs"])
        audited = safe_numeric(df["audited"])
        tenure = safe_numeric(df["ceo_tenure_years"])

        logrev = np.log(rev)
        r_star = (logrev - self.logrev_mean_) / self.logrev_std_
        r2 = r_star ** 2

        pm = (rev - costs) / rev
        pm = pm.replace([np.inf, -np.inf], np.nan)

        p_star = (self.pm_median_ - pm) / self.pm_iqr_
        p_quad = np.maximum(0.0, p_star) ** 2

        # market multi-hot
        mf = df["market_focus"].apply(self._normalize_market)
        mf_cols = {}
        for t in self.known_tags:
            mf_cols[f"mf_{t}"] = mf.apply(lambda lst: int(t in lst)).to_numpy(dtype=float)

        feats = pd.DataFrame({
            "r_star": r_star,
            "r2": r2,
            "profit_margin": pm,
            "p_star": p_star,
            "p_quad": p_quad,
            "audited": audited,
            "tenure": tenure,
        })
        for k, v in mf_cols.items():
            feats[k] = v

        feats = feats.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        feats = feats.fillna(feats.median(numeric_only=True))

        return feats.to_numpy()


def mechanism_aware_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("feat", MechanismFeatures()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=800, class_weight="balanced")),
    ])


def print_confusion(y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    df_cm = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_ORDER], columns=[f"pred_{c}" for c in CLASS_ORDER])
    print("\n" + "=" * 90)
    print(title)
    print("-" * 90)
    print(df_cm.to_string())
    print("=" * 90)


def metrics_row(y_true, y_pred) -> Dict[str, float]:
    rep = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    out = {"accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"], "weighted_f1": rep["weighted avg"]["f1-score"]}
    for cls in CLASS_ORDER:
        out[f"f1_{cls}"] = rep.get(cls, {}).get("f1-score", 0.0)
    return out


def make_divided_predictions(cost_model, sales_model, X_test: pd.DataFrame) -> np.ndarray:
    cost_pos = cost_model.predict(X_test).astype(int)
    sales_pos = sales_model.predict(X_test).astype(int)
    return np.where(
        (cost_pos == 1) & (sales_pos == 1), "both_risks",
        np.where(cost_pos == 1, "cost_risk", np.where(sales_pos == 1, "sales_risk", "healthy"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outlier_prob", type=float, default=0.01)
    parser.add_argument("--test_size", type=float, default=0.25)
    args = parser.parse_args()

    constants = GeneratorConstants()

    # In-situ generation; df_all is optional but useful for inspection.
    df_students, df_all = generate_dataset(
        N=args.N,
        seed=args.seed,
        outlier_prob=args.outlier_prob,
        return_all_columns=True,
        constants=constants,
    )
    assert df_all is not None

    print("\nDataset summary")
    print("-" * 90)
    print(f"N={len(df_students)}  seed={args.seed}  outlier_prob={args.outlier_prob}")
    print("Class distribution (overall):")
    print(df_students["risk"].value_counts(normalize=True).reindex(CLASS_ORDER).fillna(0.0).to_string())
    print("\nRevenue-costs correlation (numeric, ignoring NaNs):")
    rc = df_students[["revenue", "costs"]].apply(pd.to_numeric, errors="coerce").corr().iloc[0, 1]
    print(f"corr(revenue, costs) = {rc:.4f}")
    print("-" * 90)

    X = df_students.drop(columns=["risk"])
    y = df_students["risk"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    results = []

    # 0) Prior model
    prior_class = y_train.value_counts().idxmax()
    y_pred0 = np.array([prior_class] * len(y_test))
    print_confusion(y_test, y_pred0, f"0) Prior model (always predicts '{prior_class}')")
    print(classification_report(y_test, y_pred0, labels=CLASS_ORDER, zero_division=0))
    results.append({"model": "0_prior", **metrics_row(y_test, y_pred0)})

    # 1) Basic models
    basic_models = {
        "1_knn": KNeighborsClassifier(n_neighbors=25),
        "1_logreg": LogisticRegression(max_iter=600),
        "1_tree": DecisionTreeClassifier(max_depth=8, random_state=args.seed),
    }

    for name, mdl in basic_models.items():
        pipe = basic_preprocess_pipeline(mdl)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print_confusion(y_test, y_pred, f"{name}")
        print(classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0))
        results.append({"model": name, **metrics_row(y_test, y_pred)})

    # 2) Divided model
    y_cost_train = y_train.isin(["cost_risk", "both_risks"]).astype(int)
    y_sales_train = y_train.isin(["sales_risk", "both_risks"]).astype(int)

    cost_pipe = basic_preprocess_pipeline(LogisticRegression(max_iter=600))
    sales_pipe = basic_preprocess_pipeline(LogisticRegression(max_iter=600))

    cost_pipe.fit(X_train, y_cost_train)
    sales_pipe.fit(X_train, y_sales_train)

    y_pred2 = make_divided_predictions(cost_pipe, sales_pipe, X_test)
    print_confusion(y_test, y_pred2, "2) Divided model: two binaries (cost_pos + sales_pos)")
    print(classification_report(y_test, y_pred2, labels=CLASS_ORDER, zero_division=0))
    results.append({"model": "2_divided_cost+sales", **metrics_row(y_test, y_pred2)})

    # 3) Mechanism-aware model
    mech = mechanism_aware_pipeline()
    mech.fit(X_train, y_train)
    y_pred3 = mech.predict(X_test)
    print_confusion(y_test, y_pred3, "3) Mechanism-aware features + multinomial logistic regression")
    print(classification_report(y_test, y_pred3, labels=CLASS_ORDER, zero_division=0))
    results.append({"model": "3_mechanism_aware", **metrics_row(y_test, y_pred3)})

    # Summary table
    res_df = pd.DataFrame(results).set_index("model").sort_index()
    print("\nSummary metrics (test set)")
    print("-" * 90)
    with pd.option_context("display.max_columns", 200, "display.width", 150):
        print(res_df.round(4).to_string())
    print("-" * 90)


if __name__ == "__main__":
    main()
