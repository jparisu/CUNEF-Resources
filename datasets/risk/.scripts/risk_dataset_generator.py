"""company_risk_dataset.py

Synthetic company risk dataset generator.

Design goals
- Generate realistic-ish company features with correlations.
- Compute hidden characteristics and risk indexes.
- Create 4-class label: healthy / sales risk / cost risk / both.
- Inject outliers via per-column outlier modifiers.

Usage
-----
from company_risk_dataset import generate_company_risk_dataset

df_students = generate_company_risk_dataset(n=5000, seed=123, outlier_prob=0.03, return_all_columns=False)

df_full = generate_company_risk_dataset(n=5000, seed=123, outlier_prob=0.03, return_all_columns=True)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _require_columns(df: pd.DataFrame, cols: Sequence[str], fn_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{fn_name}: missing required columns: {missing}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(a, lo), hi)


def _scalate(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a_min = a.min()
    a_max = a.max()
    if a_max - a_min < 1e-12:
        return np.full_like(a, (lo + hi) / 2.0)
    scaled = (a - a_min) / (a_max - a_min)
    return scaled * (hi - lo) + lo


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------
# Configuration
# ----------------------------

MARKET_TAGS: Tuple[str, ...] = (
    "B2B_contracts",
    "consumer_retail",
    "subscription",
    "ads_marketplace",
    "regulated",
    "manufacturing",
)

COST_EXPOSURE_WEIGHTS = {
    "manufacturing": +0.45,
    "consumer_retail": +0.25,
    "B2B_contracts": +0.20,
    "regulated": +0.15,
    "ads_marketplace": +0.10,
    "subscription": -0.25,
}


@dataclass(frozen=True)
class Thresholds:
    t_sales: float = 0.90
    t_cost: float = 1.50


@dataclass(frozen=True)
class Noise:
    # noise for risk indexes
    eps_sales_sd: float = 0.15
    eps_cost_sd: float = 0.15
    # noise for costs multiplicative
    costs_lognormal_sd: float = 0.05
    # optional noise for cost_exposure
    cost_exposure_sd: float = 0.05
    # external score noise
    external_score_sd: float = 15.0


# ----------------------------
# Step 1: Base feature generators
# Each function: (df, rng, ...) -> df
# ----------------------------


def add_annual_revenue_eur(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add annual_revenue_eur ~ log-uniform 50k..50M.

    Requires: none
    Produces: annual_revenue_eur
    """
    if "annual_revenue_eur" in df.columns:
        return df

    # log10(rev) ~ U(4.7, 7.7)
    log10_rev = rng.uniform(4.7, 7.7, size=len(df))
    rev = np.power(10.0, log10_rev)
    df["annual_revenue_eur"] = rev.astype(float)
    return df


def outliers_annual_revenue_eur(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject outliers into annual_revenue_eur: extremely large values."""
    _require_columns(df, ["annual_revenue_eur"], "outliers_annual_revenue_eur")
    # Multiply by 20..200
    factors = rng.uniform(20.0, 200.0, size=idx.sum())
    df.loc[idx, "annual_revenue_eur"] = df.loc[idx, "annual_revenue_eur"].to_numpy() * factors
    return df


def add_size_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add size_score = clip((log10(rev)-5)/3, 0, 1)."""
    if "size_score" in df.columns:
        return df
    _require_columns(df, ["annual_revenue_eur"], "add_size_score")

    rev = df["annual_revenue_eur"].to_numpy(dtype=float)
    size = _clip((np.log10(rev) - 5.0) / 3.0, 0.0, 1.0)
    df["size_score"] = size
    return df


def add_profit_margin(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add hidden profit margin for cost generation.

    profit_margin ~ clipped Normal(mean=0.10, sd=0.12, [-0.25, 0.35])

    Produces: _profit_margin_raw (hidden helper)
    """
    if "_profit_margin_raw" in df.columns:
        return df

    pm = rng.normal(0.10, 0.12, size=len(df))
    pm = _clip(pm, -0.25, 0.35)
    df["_profit_margin_raw"] = pm
    return df


def add_annual_costs_eur(df: pd.DataFrame, rng: np.random.Generator, noise: Noise = Noise()) -> pd.DataFrame:
    """Add annual_costs_eur correlated with revenue via profit margin.

    costs = rev * (1 - pm) * LogNormal(mean=0, sd=0.05)

    Requires: annual_revenue_eur
    Produces: annual_costs_eur
    """
    if "annual_costs_eur" in df.columns:
        return df

    _require_columns(df, ["annual_revenue_eur"], "add_annual_costs_eur")
    df = add_profit_margin(df, rng)

    rev = df["annual_revenue_eur"].to_numpy(dtype=float)
    pm = df["_profit_margin_raw"].to_numpy(dtype=float)

    mult = rng.lognormal(mean=0.0, sigma=noise.costs_lognormal_sd, size=len(df))
    costs = rev * (1.0 - pm) * mult

    # Clip to plausible bounds but allow losses
    # costs = _scalate(costs, 30_000.0, 60_000_000.0)

    df["annual_costs_eur"] = costs.astype(float)
    return df


def outliers_annual_costs_eur(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject outliers into annual_costs_eur: break correlation with revenue."""
    _require_columns(df, ["annual_revenue_eur", "annual_costs_eur"], "outliers_annual_costs_eur")

    rev = df.loc[idx, "annual_revenue_eur"].to_numpy(dtype=float)

    # Two modes: too high (big losses) or too low (unrealistically low costs)
    mode = rng.random(size=idx.sum())
    high = mode < 0.5

    costs = np.empty(idx.sum(), dtype=float)
    # high-loss costs: 1.2..3.0 of revenue
    costs[high] = rev[high] * rng.uniform(1.2, 3.0, size=high.sum())
    # too-low costs: 0.05..0.25 of revenue
    costs[~high] = rev[~high] * rng.uniform(0.05, 0.25, size=(~high).sum())

    df.loc[idx, "annual_costs_eur"] = _clip(costs, 10_000.0, 200_000_000.0)
    return df


def add_audited(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add audited binary; probability increases with size."""
    if "audited" in df.columns:
        return df

    df = add_size_score(df)
    size = df["size_score"].to_numpy(dtype=float)

    p = _sigmoid(-1.2 + 1.0 * size)
    audited = rng.binomial(1, p, size=len(df))
    df["audited"] = audited.astype(int)
    return df


def outliers_audited(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject audited outliers: NaNs / invalid values."""
    _require_columns(df, ["audited"], "outliers_audited")

    # Half NaN, half invalid (2)
    mode = rng.random(size=idx.sum())
    nan_mask = mode < 0.6

    audited_vals = df.loc[idx, "audited"].astype(float).to_numpy()
    audited_vals[nan_mask] = np.nan
    audited_vals[~nan_mask] = 2.0

    df.loc[idx, "audited"] = audited_vals
    return df


def add_clients(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add clients correlated with revenue but noisy."""
    if "clients" in df.columns:
        return df

    _require_columns(df, ["annual_revenue_eur"], "add_clients")

    rev = df["annual_revenue_eur"].to_numpy(dtype=float)
    # log(clients) = Normal(-2.5 + 0.75*log(rev), 0.7)
    mu = -2.5 + 0.75 * np.log(rev)
    log_clients = rng.normal(mu, 0.7)
    clients = np.exp(log_clients)
    clients = np.round(_scalate(clients, 5.0, 10_000.0)).astype(int)

    df["clients"] = clients
    return df


def outliers_clients(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject clients outliers: absurdly low/high or negative."""
    _require_columns(df, ["clients"], "outliers_clients")

    mode = rng.random(size=idx.sum())
    vals = df.loc[idx, "clients"].to_numpy(dtype=float)

    # 40% negative, 30% extremely large, 30% extremely small (1-2)
    neg = mode < 0.4
    big = (mode >= 0.4) & (mode < 0.7)
    small = mode >= 0.7

    vals[neg] = -rng.integers(1, 100, size=neg.sum())
    vals[big] = rng.integers(50_000, 2_000_000, size=big.sum())
    vals[small] = rng.integers(1, 3, size=small.sum())

    df.loc[idx, "clients"] = vals
    return df


def add_top_client_share(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add top_client_share depending on clients."""
    if "top_client_share" in df.columns:
        return df

    _require_columns(df, ["clients"], "add_top_client_share")

    clients = df["clients"].to_numpy(dtype=float)

    base = 0.85 / np.sqrt(np.maximum(clients, 1.0))
    beta = rng.beta(2.0, 8.0, size=len(df))
    top_share = base + beta

    # If few clients (<30) shift upward
    top_share = np.where(clients < 30, top_share + 0.10, top_share)

    top_share = _clip(top_share, 0.05, 0.95)
    df["top_client_share"] = top_share.astype(float)
    return df


def outliers_top_client_share(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject top_client_share outliers: invalid shares (<0 or >1)."""
    _require_columns(df, ["top_client_share"], "outliers_top_client_share")

    mode = rng.random(size=idx.sum())
    vals = df.loc[idx, "top_client_share"].to_numpy(dtype=float)

    # Some >1, some <0
    above = mode < 0.5
    vals[above] = rng.uniform(1.05, 2.0, size=above.sum())
    vals[~above] = rng.uniform(-1.0, -0.05, size=(~above).sum())

    df.loc[idx, "top_client_share"] = vals
    return df


def add_market_focus(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add market_focus as list[str] (1-3 tags) with simple conditional correlations."""
    if "market_focus" in df.columns:
        return df

    # Primary tag probabilities (sum to 1)
    primary_tags = np.array(MARKET_TAGS)
    primary_p = np.array([0.22, 0.18, 0.24, 0.16, 0.12, 0.08], dtype=float)
    primary_p = primary_p / primary_p.sum()

    primary = rng.choice(primary_tags, size=len(df), replace=True, p=primary_p)

    focus_lists: List[List[str]] = []
    for ptag in primary:
        tags = [ptag]

        # Determine how many secondary tags
        k = rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15])

        candidates = [t for t in MARKET_TAGS if t != ptag]

        # Conditional adjustments
        weights = np.ones(len(candidates), dtype=float)
        cand_idx = {t: i for i, t in enumerate(candidates)}

        if ptag == "manufacturing":
            if "regulated" in cand_idx:
                weights[cand_idx["regulated"]] *= 2.0
        if ptag == "ads_marketplace":
            if "regulated" in cand_idx:
                weights[cand_idx["regulated"]] *= 0.6
        if ptag == "consumer_retail":
            if "ads_marketplace" in cand_idx:
                weights[cand_idx["ads_marketplace"]] *= 1.5

        # Draw k secondary without replacement
        if k > 0:
            weights = weights / weights.sum()
            secondaries = rng.choice(np.array(candidates), size=k, replace=False, p=weights)
            tags.extend(list(secondaries))

        focus_lists.append(tags)

    df["market_focus"] = focus_lists
    return df


def outliers_market_focus(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject market_focus outliers: empty list / invalid tags / NaN."""
    _require_columns(df, ["market_focus"], "outliers_market_focus")

    # For selected idx, apply one of: None, empty, invalid tag
    indices = df.index[idx]
    modes = rng.random(size=len(indices))

    for i, m in zip(indices, modes):
        if m < 0.4:
            df.at[i, "market_focus"] = None
        elif m < 0.7:
            df.at[i, "market_focus"] = []
        else:
            df.at[i, "market_focus"] = ["unknown_focus"]

    return df


def add_employees(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add employees correlated with revenue."""
    if "employees" in df.columns:
        return df

    _require_columns(df, ["annual_revenue_eur"], "add_employees")

    rev = df["annual_revenue_eur"].to_numpy(dtype=float)
    mu = -6.0 + 0.55 * np.log(rev)
    log_emp = rng.normal(mu, 0.6)
    emp = np.exp(log_emp)
    emp = np.round(_clip(emp, 3.0, 2000.0)).astype(int)

    df["employees"] = emp
    return df


def outliers_employees(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject employees outliers: zeros/negatives or extremely large."""
    _require_columns(df, ["employees"], "outliers_employees")

    mode = rng.random(size=idx.sum())
    vals = df.loc[idx, "employees"].to_numpy(dtype=float)

    neg = mode < 0.4
    big = (mode >= 0.4) & (mode < 0.8)

    vals[neg] = -rng.integers(1, 50, size=neg.sum())
    vals[big] = rng.integers(10_000, 500_000, size=big.sum())
    vals[~(neg | big)] = 0

    df.loc[idx, "employees"] = vals
    return df


def add_ceo_tenure_years(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add random CEO tenure (trap feature)."""
    if "ceo_tenure_years" in df.columns:
        return df

    tenure = rng.uniform(0.0, 25.0, size=len(df))
    df["ceo_tenure_years"] = tenure.astype(float)
    return df


def outliers_ceo_tenure_years(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject CEO tenure outliers: negative or absurdly large."""
    _require_columns(df, ["ceo_tenure_years"], "outliers_ceo_tenure_years")

    mode = rng.random(size=idx.sum())
    vals = df.loc[idx, "ceo_tenure_years"].to_numpy(dtype=float)

    neg = mode < 0.5
    vals[neg] = -rng.uniform(0.1, 10.0, size=neg.sum())
    vals[~neg] = rng.uniform(50.0, 200.0, size=(~neg).sum())

    df.loc[idx, "ceo_tenure_years"] = vals
    return df


# ----------------------------
# Step 2: Hidden characteristics
# ----------------------------


def add_cost_exposure(df: pd.DataFrame, rng: np.random.Generator, noise: Noise = Noise()) -> pd.DataFrame:
    """Compute hidden cost_exposure in [0,1] from market_focus tags."""
    if "cost_exposure" in df.columns:
        return df

    _require_columns(df, ["market_focus"], "add_cost_exposure")

    def score_tags(tags) -> float:
        if not isinstance(tags, list) or len(tags) == 0:
            return 0.5  # neutral default for malformed entries
        z = 0.0
        for t in tags:
            z += COST_EXPOSURE_WEIGHTS.get(t, 0.0)
        # Convert to [0,1]
        val = 0.5 + z
        return float(np.clip(val, 0.0, 1.0))

    ce = df["market_focus"].apply(score_tags).to_numpy(dtype=float)
    if noise.cost_exposure_sd > 0:
        ce = _clip(ce + rng.normal(0.0, noise.cost_exposure_sd, size=len(df)), 0.0, 1.0)

    df["cost_exposure"] = ce
    return df


def add_customer_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hidden customer_concentration from clients and top_client_share."""
    if "customer_concentration" in df.columns:
        return df

    _require_columns(df, ["clients", "top_client_share"], "add_customer_concentration")

    clients = df["clients"].to_numpy(dtype=float)
    top = df["top_client_share"].to_numpy(dtype=float)

    conc = (top ** 2) * np.exp(-0.0004 * (clients - 1.0))
    df["customer_concentration"] = conc.astype(float)
    return df


def add_profit_margin_from_rev_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Compute profit_margin from annual_revenue_eur and annual_costs_eur."""
    if "profit_margin" in df.columns:
        return df

    _require_columns(df, ["annual_revenue_eur", "annual_costs_eur"], "add_profit_margin_from_rev_cost")

    rev = df["annual_revenue_eur"].to_numpy(dtype=float)
    costs = df["annual_costs_eur"].to_numpy(dtype=float)

    pm = (rev - costs) / np.maximum(rev, 1.0)
    df["profit_margin"] = pm.astype(float)
    return df


def add_sales_risk(df: pd.DataFrame, rng: np.random.Generator, noise: Noise = Noise()) -> pd.DataFrame:
    """Compute hidden sales_risk index."""
    if "sales_risk" in df.columns:
        return df

    df = add_size_score(df)
    df = add_customer_concentration(df)
    _require_columns(df, ["audited", "market_focus"], "add_sales_risk")

    size = df["size_score"].to_numpy(dtype=float)
    conc = df["customer_concentration"].to_numpy(dtype=float)

    audited = pd.to_numeric(df["audited"], errors="coerce").to_numpy(dtype=float)
    audited = np.where(np.isfinite(audited), audited, 0.0)
    audited = np.where((audited == 0.0) | (audited == 1.0), audited, 0.0)

    sub_flag = df["market_focus"].apply(lambda tags: 1 if isinstance(tags, list) and ("subscription" in tags) else 0).to_numpy(dtype=float)

    eps = rng.normal(0.0, noise.eps_sales_sd, size=len(df))
    sales_risk = (
        1.1 * conc
        + 0.5 * (1.0 - size)
        + 0.3 * (1.0 - audited)
        + 0.2 * (1.0 - sub_flag)
        + eps
    )

    df["sales_risk"] = sales_risk.astype(float)
    return df


def add_cost_risk(df: pd.DataFrame, rng: np.random.Generator, noise: Noise = Noise()) -> pd.DataFrame:
    """Compute hidden cost_risk index."""
    if "cost_risk" in df.columns:
        return df

    df = add_size_score(df)
    df = add_profit_margin_from_rev_cost(df)
    df = add_cost_exposure(df, rng, noise)

    _require_columns(df, ["audited"], "add_cost_risk")

    size = df["size_score"].to_numpy(dtype=float)
    pm = df["profit_margin"].to_numpy(dtype=float)
    cost_exposure = df["cost_exposure"].to_numpy(dtype=float)

    audited = pd.to_numeric(df["audited"], errors="coerce").to_numpy(dtype=float)
    audited = np.where(np.isfinite(audited), audited, 0.0)
    audited = np.where((audited == 0.0) | (audited == 1.0), audited, 0.0)

    eps = rng.normal(0.0, noise.eps_cost_sd, size=len(df))

    margin_gap = np.maximum(0.18 - pm, 0.0)

    cost_risk = (
        1.2 * margin_gap
        + 0.9 * cost_exposure
        + 0.6 * (1.0 - size)
        + 0.35 * (1.0 - audited)
        + eps
    )

    df["cost_risk"] = cost_risk.astype(float)
    return df


def add_risk_flags(df: pd.DataFrame, thresholds: Thresholds = Thresholds()) -> pd.DataFrame:
    """Compute sales_flag and cost_flag from risk indexes."""
    if "sales_flag" in df.columns and "cost_flag" in df.columns:
        return df

    _require_columns(df, ["sales_risk", "cost_risk"], "add_risk_flags")

    df["sales_flag"] = (df["sales_risk"].to_numpy(dtype=float) >= thresholds.t_sales).astype(int)
    df["cost_flag"] = (df["cost_risk"].to_numpy(dtype=float) >= thresholds.t_cost).astype(int)
    return df


def add_target_y(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4-class target y from sales_flag and cost_flag.

    y mapping:
      0: healthy
      1: sales risk
      2: cost risk
      3: both
    """
    if "y" in df.columns:
        return df

    _require_columns(df, ["sales_flag", "cost_flag"], "add_target_y")

    y = df["sales_flag"].to_numpy(dtype=int) + 2 * df["cost_flag"].to_numpy(dtype=int)
    df["y"] = y.astype(int)

    # Convert number to categorical:
    # - healthy
    # - sales risk
    # - cost risk
    # - both risks

    labels = {0: "healthy", 1: "sales_risk", 2: "cost_risk", 3: "both_risks"}
    df["risk [target]"] = df["y"].map(labels)
    return df


def add_risk_score_external(df: pd.DataFrame, rng: np.random.Generator, noise: Noise = Noise(), max_score: float = 100.0) -> pd.DataFrame:
    """Add risk_score_external correlated with risk/no-risk but weak on class type."""
    if "risk_score_external" in df.columns:
        return df

    _require_columns(df, ["sales_risk", "cost_risk"], "add_risk_score_external")

    base = 70.0 * (df["sales_risk"].to_numpy(dtype=float) + df["cost_risk"].to_numpy(dtype=float))
    score = base + rng.normal(0.0, noise.external_score_sd, size=len(df))

    # Normalize to 0..max_score
    score = (score - score.min()) / (score.max() - score.min()) * max_score

    df["risk_score_external"] = score.astype(float)
    return df


def outliers_risk_score_external(df: pd.DataFrame, rng: np.random.Generator, idx: np.ndarray) -> pd.DataFrame:
    """Inject risk_score_external outliers: missing, negative, or huge."""
    _require_columns(df, ["risk_score_external"], "outliers_risk_score_external")

    mode = rng.random(size=idx.sum())
    vals = df.loc[idx, "risk_score_external"].to_numpy(dtype=float)

    nan = mode < 0.4
    neg = (mode >= 0.4) & (mode < 0.7)
    huge = mode >= 0.7

    vals[nan] = np.nan
    vals[neg] = -rng.uniform(1.0, 100.0, size=neg.sum())
    vals[huge] = rng.uniform(500.0, 5000.0, size=huge.sum())

    df.loc[idx, "risk_score_external"] = vals
    return df


# ----------------------------
# Outlier orchestration
# ----------------------------


def _apply_outliers(df: pd.DataFrame, rng: np.random.Generator, outlier_prob: float) -> pd.DataFrame:
    """Apply outliers across multiple columns.

    outlier_prob behavior:
    - If <= 1: treat as probability per row.
    - If > 1: force N = floor(outlier_prob) outliers, then remaining fractional probability.

    We apply a *global* outlier row set and then per-column modifiers on subsets
    to diversify the corruption.
    """

    n = len(df)
    if n == 0:
        return df

    forced = int(np.floor(outlier_prob)) if outlier_prob > 1 else 0
    frac = outlier_prob - forced if outlier_prob > 1 else outlier_prob

    forced = min(forced, n)

    forced_idx = np.zeros(n, dtype=bool)
    if forced > 0:
        forced_rows = rng.choice(n, size=forced, replace=False)
        forced_idx[forced_rows] = True

    rand_idx = rng.random(n) < frac
    outlier_rows = forced_idx | rand_idx

    if not outlier_rows.any():
        return df

    # Split outlier rows into disjoint groups per feature to keep variety
    outlier_indices = np.flatnonzero(outlier_rows)
    rng.shuffle(outlier_indices)

    # allocate roughly equally to each outlier function
    k = 10
    groups = np.array_split(outlier_indices, k)

    def idx_mask(g):
        m = np.zeros(n, dtype=bool)
        m[g] = True
        return m

    # Ensure columns exist before applying corresponding outlier function
    if "annual_revenue_eur" in df.columns:
        df = outliers_annual_revenue_eur(df, rng, idx_mask(groups[0]))
    if "annual_costs_eur" in df.columns:
        df = outliers_annual_costs_eur(df, rng, idx_mask(groups[1]))
    if "audited" in df.columns:
        df = outliers_audited(df, rng, idx_mask(groups[2]))
    if "clients" in df.columns:
        df = outliers_clients(df, rng, idx_mask(groups[3]))
    if "top_client_share" in df.columns:
        df = outliers_top_client_share(df, rng, idx_mask(groups[4]))
    if "market_focus" in df.columns:
        df = outliers_market_focus(df, rng, idx_mask(groups[5]))
    if "employees" in df.columns:
        df = outliers_employees(df, rng, idx_mask(groups[6]))
    if "ceo_tenure_years" in df.columns:
        df = outliers_ceo_tenure_years(df, rng, idx_mask(groups[7]))
    if "risk_score_external" in df.columns:
        df = outliers_risk_score_external(df, rng, idx_mask(groups[8]))

    # group[9] intentionally unused for future extensions
    return df


# ----------------------------
# Main generator
# ----------------------------


def generate_company_risk_dataset(
    n: int,
    seed: int = 123,
    outlier_prob: float = 0.0,
    return_all_columns: bool = False,
    thresholds: Thresholds = Thresholds(),
    noise: Noise = Noise(),
    max_external_score: float = 100.0,
) -> pd.DataFrame:
    """Generate the synthetic dataset.

    Parameters
    ----------
    n:
        Number of rows.
    seed:
        Random seed.
    outlier_prob:
        Outlier control.
        - If <= 1: probability per row of becoming an outlier row.
        - If > 1: force floor(outlier_prob) outliers, then apply remaining
          fractional probability.
    return_all_columns:
        If True, returns visible + hidden columns.
        If False, returns only student-visible columns + target y.
    thresholds:
        Thresholds for sales/cost flags.
    noise:
        Noise settings.
    max_external_score:
        Max clipping for risk_score_external.

    Returns
    -------
    pd.DataFrame
        Generated dataset.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    rng = _rng(seed)
    df = pd.DataFrame(index=np.arange(n))

    # Visible base columns
    df = add_annual_revenue_eur(df, rng)
    df = add_annual_costs_eur(df, rng, noise)
    df = add_audited(df, rng)
    df = add_clients(df, rng)
    df = add_top_client_share(df, rng)
    df = add_market_focus(df, rng)
    df = add_employees(df, rng)
    df = add_ceo_tenure_years(df, rng)

    # Hidden characteristics + risks
    df = add_cost_exposure(df, rng, noise)
    df = add_customer_concentration(df)
    df = add_profit_margin_from_rev_cost(df)
    df = add_sales_risk(df, rng, noise)
    df = add_cost_risk(df, rng, noise)
    df = add_risk_flags(df, thresholds)
    df = add_target_y(df)
    df = add_risk_score_external(df, rng, noise, max_external_score)

    # Apply outliers AFTER all columns exist, so they can corrupt anything
    if outlier_prob and outlier_prob > 0:
        df = _apply_outliers(df, rng, float(outlier_prob))

    if return_all_columns:
        return df

    # Student-visible subset
    student_cols = [
        "annual_revenue_eur",
        "annual_costs_eur",
        "audited",
        "clients",
        "top_client_share",
        "market_focus",
        "employees",
        "ceo_tenure_years",
        "risk_score_external",
        "risk [target]",
    ]

    return df[student_cols].copy()
