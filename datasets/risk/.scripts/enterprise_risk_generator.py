"""Synthetic enterprise risk dataset generator.

This module generates a multi-class classification dataset with intentionally tricky
properties for ML competitions (scaling, interactions, nonlinearities, multi-tag feature,
confounding, and optional corrupted/outlier cells).

Student-facing columns:
    - revenue (numeric, large scale)
    - costs (numeric, highly correlated with revenue; can be slightly > revenue)
    - audited (binary, but may be corrupted by outliers)
    - market_focus (list[str] with 1-2 tags; may be corrupted by outliers)
    - ceo_tenure_years (random numeric)
    - risk (target): healthy | cost_risk | sales_risk | both_risks

Hidden columns (available in return_all_columns=True):
    - profit_margin
    - cost_risk_factor
    - sales_risk_factor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassProportions:
    """Desired marginal proportions.

    Calibration interprets:
        P(cost_risk_factor > X) ~= cost_risk
        P(sales_risk_factor > Y) ~= sales_risk

    both_risks is not specified; it emerges from overlap.

    Note: 'healthy' is kept for API completeness but is not directly enforced unless
    you implement a more advanced calibration strategy.
    """
    healthy: float = 0.50
    cost_risk: float = 0.30
    sales_risk: float = 0.20


@dataclass
class GeneratorConstants:
    """All tunable constants for the generator."""

    # Revenue: log(revenue) ~ N(mu, sigma^2)
    mu_log_revenue: float = 16.0
    sigma_log_revenue: float = 1.0

    # Costs:
    #   costs = revenue * (base + span*sigmoid(z)) * lognormal_noise
    #
    # IMPORTANT: cost_ratio_span is set so costs can be slightly above revenue
    # (slightly negative profits) for a realistic fraction of companies.
    cost_ratio_base: float = 0.60
    cost_ratio_span: float = 0.50  # max ratio ~ 1.10
    cost_ratio_noise_sigma: float = 0.03  # multiplicative noise

    # Audit probability: sigmoid(alpha + beta*(log(revenue)-mu_log_revenue))
    audit_alpha: float = 0.30
    audit_beta: float = 1.00

    # Market focus: multi-tag (1-2 tags)
    market_tags: Tuple[str, str, str, str] = ("Industrial", "Ads", "Contract", "Scale")
    market_base_probs: Tuple[float, float, float, float] = (0.35, 0.30, 0.20, 0.15)
    p_one_tag: float = 0.70
    p_two_tags: float = 0.30

    # Optional conditional probs for second tag (realism)
    market_second_tag_conditional: Optional[Dict[str, Dict[str, float]]] = field(
        default_factory=lambda: {
            "Industrial": {"Ads": 0.10, "Contract": 0.70, "Scale": 0.20},
            "Ads": {"Industrial": 0.15, "Contract": 0.15, "Scale": 0.70},
            "Contract": {"Industrial": 0.55, "Ads": 0.10, "Scale": 0.35},
            "Scale": {"Industrial": 0.20, "Ads": 0.60, "Contract": 0.20},
        }
    )

    # CEO tenure ~ Gamma(k, theta) clipped
    ceo_tenure_k: float = 2.0
    ceo_tenure_theta: float = 3.0
    ceo_tenure_clip_max: float = 25.0

    # Profit margin scaling for cost risk
    profit_margin_center: float = 0.18
    profit_margin_scale: float = 0.08

    # Market weights (additive per tag) for cost and sales risk factors
    w_cost: Dict[str, float] = field(
        default_factory=lambda: {
            "Industrial": 0.70,
            "Ads": 0.00,
            "Contract": -0.50,
            "Scale": 0.45,
        }
    )
    w_sales: Dict[str, float] = field(
        default_factory=lambda: {
            "Industrial": 0.00,
            "Ads": 0.80,
            "Contract": -0.50,
            "Scale": 0.45,
        }
    )

    # Audit effect (reduces both)
    audit_effect: float = -0.55

    # Risk factor coefficients and noise
    cost_coef_linear: float = 1.10
    cost_coef_quad: float = 0.45
    cost_risk_noise_sigma: float = 0.40

    sales_coef_u2: float = 0.90
    sales_coef_linear: float = 0.30
    sales_risk_noise_sigma: float = 0.45

    # Default thresholds (if calibration off)
    threshold_cost_X: float = 1.10
    threshold_sales_Y: float = 1.35

    # Calibration toggle
    calibrate_thresholds: bool = True

    # Outlier injection parameters
    outlier_revenue_multiplier_range: Tuple[float, float] = (5.0, 50.0)
    outlier_costs_multiplier_range: Tuple[float, float] = (5.0, 80.0)
    outlier_negative_costs_prob: float = 0.10
    outlier_nan_prob: float = 0.15
    outlier_audited_invalid_prob: float = 0.20
    outlier_market_focus_invalid_prob: float = 0.10
    outlier_tenure_multiplier_range: Tuple[float, float] = (2.0, 8.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _validate_constants(c: GeneratorConstants) -> None:
    probs = np.array(c.market_base_probs, dtype=float)
    if len(c.market_tags) != 4 or len(probs) != 4:
        raise ValueError("market_tags and market_base_probs must have length 4")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("market_base_probs must sum to 1")
    if not np.isclose(c.p_one_tag + c.p_two_tags, 1.0):
        raise ValueError("p_one_tag + p_two_tags must equal 1")


# -----------------------------
# Column generators (visible)
# -----------------------------

def make_revenue(df: pd.DataFrame, c: GeneratorConstants, rng: np.random.Generator) -> pd.DataFrame:
    log_rev = rng.normal(c.mu_log_revenue, c.sigma_log_revenue, size=len(df))
    df["revenue"] = np.exp(log_rev)
    return df


def make_costs(df: pd.DataFrame, c: GeneratorConstants, rng: np.random.Generator) -> pd.DataFrame:
    z = rng.normal(0.0, 1.0, size=len(df))
    cost_ratio = c.cost_ratio_base + c.cost_ratio_span * _sigmoid(z)
    eps = rng.lognormal(mean=0.0, sigma=c.cost_ratio_noise_sigma, size=len(df))
    df["costs"] = df["revenue"].to_numpy() * cost_ratio * eps
    return df


def make_audited(df: pd.DataFrame, c: GeneratorConstants, rng: np.random.Generator) -> pd.DataFrame:
    log_rev = np.log(df["revenue"].to_numpy())
    p = _sigmoid(c.audit_alpha + c.audit_beta * (log_rev - c.mu_log_revenue))
    df["audited"] = rng.binomial(1, p, size=len(df)).astype(int)
    return df


def make_market_focus(df: pd.DataFrame, c: GeneratorConstants, rng: np.random.Generator) -> pd.DataFrame:
    tags = list(c.market_tags)
    base_probs = np.array(c.market_base_probs, dtype=float)

    k = rng.choice([1, 2], size=len(df), p=[c.p_one_tag, c.p_two_tags])

    market_focus: List[List[str]] = []
    for ki in k:
        first = rng.choice(tags, p=base_probs)
        if ki == 1:
            market_focus.append(str(first))
            continue

        # Second tag
        if c.market_second_tag_conditional is not None and first in c.market_second_tag_conditional:
            cond = c.market_second_tag_conditional[first]
            second_tags = [t for t in tags if t != first]
            cond_probs = np.array([cond[t] for t in second_tags], dtype=float)
            cond_probs = cond_probs / cond_probs.sum()
            second = rng.choice(second_tags, p=cond_probs)
        else:
            second_tags = [t for t in tags if t != first]
            second_probs = np.array([base_probs[tags.index(t)] for t in second_tags], dtype=float)
            second_probs = second_probs / second_probs.sum()
            second = rng.choice(second_tags, p=second_probs)

        market_focus.append(','.join([str(first), str(second)]))

    df["market_focus"] = market_focus
    return df


def make_ceo_tenure_years(df: pd.DataFrame, c: GeneratorConstants, rng: np.random.Generator) -> pd.DataFrame:
    tenure = rng.gamma(shape=c.ceo_tenure_k, scale=c.ceo_tenure_theta, size=len(df))
    tenure = np.clip(tenure, 0.0, c.ceo_tenure_clip_max)
    df["ceo_tenure_years"] = tenure
    return df


# -----------------------------
# Hidden feature generators
# -----------------------------

def make_profit_margin(df: pd.DataFrame, c: GeneratorConstants) -> pd.DataFrame:
    revenue = df["revenue"].to_numpy()
    costs = df["costs"].to_numpy()
    df["profit_margin"] = (revenue - costs) / np.maximum(revenue, 1e-12)
    return df


def _market_shift(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    shifts = np.zeros(len(df), dtype=float)
    for i, tags in enumerate(df["market_focus"].tolist()):
        s = 0.0
        if isinstance(tags, list):
            for t in tags:
                s += weights.get(t, 0.0)
        shifts[i] = s
    return shifts


def make_cost_risk_factor(df: pd.DataFrame, c: GeneratorConstants, rng: np.random.Generator) -> pd.DataFrame:
    pm = df["profit_margin"].to_numpy()
    p_star = (c.profit_margin_center - pm) / c.profit_margin_scale

    market_shift = _market_shift(df, c.w_cost)
    audit_shift = c.audit_effect * df["audited"].to_numpy()

    quad = np.maximum(0.0, p_star) ** 2
    noise = rng.normal(0.0, c.cost_risk_noise_sigma, size=len(df))

    df["cost_risk_factor"] = (
        c.cost_coef_linear * p_star
        + c.cost_coef_quad * quad
        + market_shift
        + audit_shift
        + noise
    )
    return df


def make_sales_risk_factor(df: pd.DataFrame, c: GeneratorConstants, rng: np.random.Generator) -> pd.DataFrame:
    log_rev = np.log(df["revenue"].to_numpy())
    r_star = (log_rev - c.mu_log_revenue) / c.sigma_log_revenue
    u2 = r_star ** 2

    market_shift = _market_shift(df, c.w_sales)
    audit_shift = c.audit_effect * df["audited"].to_numpy()

    noise = rng.normal(0.0, c.sales_risk_noise_sigma, size=len(df))

    df["sales_risk_factor"] = (
        c.sales_coef_u2 * u2
        + c.sales_coef_linear * r_star
        + market_shift
        + audit_shift
        + noise
    )
    return df


# -----------------------------
# Threshold calibration & target
# -----------------------------

def calibrate_thresholds(df: pd.DataFrame, props: ClassProportions) -> Tuple[float, float]:
    """Calibrate thresholds to match marginal proportions."""
    if not (0.0 < props.cost_risk < 1.0 and 0.0 < props.sales_risk < 1.0):
        raise ValueError("cost_risk and sales_risk must be in (0,1)")

    x = float(np.quantile(df["cost_risk_factor"].to_numpy(), 1.0 - props.cost_risk))
    y = float(np.quantile(df["sales_risk_factor"].to_numpy(), 1.0 - props.sales_risk))
    return x, y


def make_target(df: pd.DataFrame, X: float, Y: float) -> pd.DataFrame:
    cost_pos = df["cost_risk_factor"].to_numpy() > X
    sales_pos = df["sales_risk_factor"].to_numpy() > Y

    df["risk"] = np.where(
        cost_pos & sales_pos,
        "both_risks",
        np.where(cost_pos, "cost_risk", np.where(sales_pos, "sales_risk", "healthy")),
    )
    return df


# -----------------------------
# Outlier / corruption injector
# -----------------------------

def inject_outliers(
    df: pd.DataFrame,
    c: GeneratorConstants,
    rng: np.random.Generator,
    outlier_prob: float,
    columns: Sequence[str] = ("revenue", "costs", "audited", "market_focus", "ceo_tenure_years"),
) -> pd.DataFrame:
    """Corrupt one random cell per selected row."""
    if outlier_prob <= 0:
        return df
    if not (0.0 <= outlier_prob <= 1.0):
        raise ValueError("outlier_prob must be in [0,1]")

    n = len(df)
    mask = rng.random(n) < outlier_prob
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return df

    cols = list(columns)

    for i in idxs:
        col = rng.choice(cols)

        # optional NaN corruption for non-list columns
        if rng.random() < c.outlier_nan_prob and col != "market_focus":
            df.at[i, col] = np.nan
            continue

        if col == "audited":
            if rng.random() < c.outlier_audited_invalid_prob:
                df.at[i, col] = int(rng.choice([2, -1, 3]))
            else:
                try:
                    df.at[i, col] = int(1 - int(df.at[i, col]))
                except Exception:
                    df.at[i, col] = 2

        elif col == "revenue":
            mult = rng.uniform(*c.outlier_revenue_multiplier_range)
            df.at[i, col] = float(df.at[i, col]) * mult

        elif col == "costs":
            if rng.random() < c.outlier_negative_costs_prob:
                df.at[i, col] = -abs(float(df.at[i, col])) * rng.uniform(0.5, 3.0)
            else:
                mult = rng.uniform(*c.outlier_costs_multiplier_range)
                df.at[i, col] = float(df.at[i, col]) * mult

        elif col == "market_focus":
            if rng.random() < c.outlier_market_focus_invalid_prob:
                df.at[i, col] = ["Unknown"]
            else:
                df.at[i, col] = "Industrial"  # wrong type: str instead of list

        elif col == "ceo_tenure_years":
            v = df.at[i, col]
            if pd.isna(v):
                continue
            mult = rng.uniform(*c.outlier_tenure_multiplier_range)
            df.at[i, col] = float(v) * mult

        else:
            df.at[i, col] = np.nan

    return df


# -----------------------------
# Wrapper
# -----------------------------

def generate_dataset(
    N: int,
    seed: Optional[int] = None,
    outlier_prob: float = 0.0,
    return_all_columns: bool = False,
    constants: Optional[GeneratorConstants] = None,
    class_proportions: Optional[ClassProportions] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Generate student and (optionally) full datasets."""
    if N <= 0:
        raise ValueError("N must be positive")

    c = constants or GeneratorConstants()
    props = class_proportions or ClassProportions()
    _validate_constants(c)
    rng = _rng(seed)

    df = pd.DataFrame(index=np.arange(N))

    # visible
    df = make_revenue(df, c, rng)
    df = make_costs(df, c, rng)
    df = make_audited(df, c, rng)
    df = make_market_focus(df, c, rng)
    df = make_ceo_tenure_years(df, c, rng)

    # hidden
    df = make_profit_margin(df, c)
    df = make_cost_risk_factor(df, c, rng)
    df = make_sales_risk_factor(df, c, rng)

    # thresholds
    if c.calibrate_thresholds:
        X, Y = calibrate_thresholds(df, props)
    else:
        X, Y = c.threshold_cost_X, c.threshold_sales_Y

    df = make_target(df, X, Y)

    # outliers injected after target assignment to create inconsistencies
    df = inject_outliers(df, c, rng, outlier_prob=outlier_prob)

    student_cols = ["revenue", "costs", "audited", "market_focus", "ceo_tenure_years", "risk"]
    hidden_cols = ["profit_margin", "cost_risk_factor", "sales_risk_factor"]

    df_students = df[student_cols].copy()

    if return_all_columns:
        df_all = df[student_cols + hidden_cols].copy()
        df_all.attrs["threshold_cost_X"] = X
        df_all.attrs["threshold_sales_Y"] = Y
        df_all.attrs["target_cost_marginal"] = props.cost_risk
        df_all.attrs["target_sales_marginal"] = props.sales_risk
        df_all.attrs["achieved_cost_marginal"] = float((df_all["cost_risk_factor"] > X).mean())
        df_all.attrs["achieved_sales_marginal"] = float((df_all["sales_risk_factor"] > Y).mean())
        df_all.attrs["achieved_class_distribution"] = df_all["risk"].value_counts(normalize=True).to_dict()
        return df_students, df_all

    return df_students, None
