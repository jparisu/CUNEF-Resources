# Enterprise Risk Dataset Generation

This directory contains the code used to generate the synthetic enterprise risk
classification dataset stored one level above this folder.

The main generator is
[`enterprise_risk_generator.py`](enterprise_risk_generator.py). The helper script
[`generate_datasets.py`](generate_datasets.py) calls that generator with fixed
sizes and seeds, then writes the public CSV files.

## Generated Files

Running `generate_datasets.py` creates:

- `../risk_train_dataset.csv`
- `../risk_test_dataset_1.csv`
- `../risk_test_dataset_24.csv`
- `../risk_test_dataset_42.csv`
- `../risk_test_dataset_20260326.csv`

The training file contains 5,000 rows. Each test file contains 500 rows.

## How To Regenerate

From this directory:

```bash
python generate_datasets.py
```

The script writes the CSV files to the parent dataset directory. Existing files
with the same names are overwritten.

## Dataset Columns

The student-facing dataset contains:

- `revenue`: annual company revenue, generated from a log-normal distribution.
- `costs`: annual operating costs, strongly correlated with revenue.
- `audited`: binary indicator for whether the company is audited.
- `market_focus`: one or two comma-separated strategic focus tags.
- `ceo_tenure_years`: CEO tenure in years.
- `risk`: target class.

The possible `risk` classes are:

- `healthy`
- `cost_risk`
- `sales_risk`
- `both_risks`

When `generate_dataset(..., return_all_columns=True)` is used directly, the
generator can also return hidden columns used to construct the target:

- `profit_margin`
- `cost_risk_factor`
- `sales_risk_factor`

These hidden columns are not written by `generate_datasets.py`.

## Generation Process

`generate_dataset()` builds the data in several stages.

1. It creates an empty dataframe with `N` rows and initializes a NumPy random
   generator from the provided `seed`.
2. It generates visible company features:
   - `revenue` is sampled as `exp(N(mu_log_revenue, sigma_log_revenue))`.
   - `costs` are revenue multiplied by a noisy cost ratio. The default ratio can
     slightly exceed 1, so some companies can have negative profit.
   - `audited` is sampled from a Bernoulli distribution whose probability rises
     with log revenue.
   - `market_focus` is sampled from the tags `Industrial`, `Ads`, `Contract`,
     and `Scale`. Most rows receive one tag; some receive two, with conditional
     probabilities for realistic pairings.
   - `ceo_tenure_years` is sampled from a clipped gamma distribution.
3. It computes hidden features:
   - `profit_margin = (revenue - costs) / revenue`.
   - `cost_risk_factor`, based on low profit margin, market-focus effects,
     audit effects, nonlinear terms, and random noise.
   - `sales_risk_factor`, based on normalized log revenue, market-focus effects,
     audit effects, nonlinear terms, and random noise.
4. It calibrates thresholds for the hidden risk factors. By default, the
   thresholds are chosen from quantiles so that approximately 30% of rows exceed
   the cost-risk threshold and approximately 20% exceed the sales-risk threshold.
5. It assigns the target:
   - rows above neither threshold become `healthy`;
   - rows above only the cost threshold become `cost_risk`;
   - rows above only the sales threshold become `sales_risk`;
   - rows above both thresholds become `both_risks`.
6. It optionally injects corrupted cells after the target is assigned. This can
   create realistic inconsistencies between visible features and the target.

## Outliers And Corruption

The generator can corrupt one random visible cell per selected row. The selected
rows are controlled by `outlier_prob`.

The default generated files use:

- training data: `outlier_prob=0.005`
- test data: `outlier_prob=0.002`

Possible corruptions include:

- unusually large `revenue` or `costs`
- negative `costs`
- missing values in non-list columns
- invalid `audited` values such as `-1`, `2`, or `3`
- malformed `market_focus` values
- unusually large `ceo_tenure_years`

Because corruption happens after target assignment, these outliers are intended
to make data cleaning and modelling more realistic.

## Formulas

This section gives the exact mathematical formulas implemented by the generator.
All default constants come from `GeneratorConstants`.

### Visible features

**Revenue**

```
log(revenue) ~ N(μ=16, σ=1)
revenue = exp(log(revenue))
```

**Costs**

```
z ~ N(0, 1)
cost_ratio = 0.60 + 0.50 · σ(z)          # σ = sigmoid; range ≈ (0.60, 1.10)
ε ~ LogNormal(0, σ=0.03)
costs = revenue · cost_ratio · ε
```

**Audited**

```
p = σ(0.30 + 1.00 · (log(revenue) − 16))
audited ~ Bernoulli(p)
```

**Market focus**

```
tag₁ ~ Categorical({Industrial: 0.35, Ads: 0.30, Contract: 0.20, Scale: 0.15})

k ~ Bernoulli(p_two=0.30)   # 1 tag with prob 0.70, 2 tags with prob 0.30
tag₂ | tag₁ ~ Categorical(conditional table)   # see market_second_tag_conditional

market_focus = tag₁              if k = 1
             = "tag₁,tag₂"       if k = 2   (comma-separated string)
```

**CEO tenure**

```
ceo_tenure_years ~ Gamma(k=2, θ=3), clipped to [0, 25]
```

### Hidden features

**Profit margin**

```
profit_margin = (revenue − costs) / revenue
```

**Market shift** *(see implementation note below)*

```
market_shift(w) = Σ w[t]  for each tag t in market_focus
```

Where the tag weights are:

| Tag        | w_cost | w_sales |
|------------|--------|---------|
| Industrial |  0.70  |  0.00   |
| Ads        |  0.00  |  0.80   |
| Contract   | −0.50  | −0.50   |
| Scale      |  0.45  |  0.45   |

> **Implementation note:** `_market_shift` only accumulates tag weights when the
> `market_focus` value is a Python `list`. Because the generator stores
> `market_focus` as a comma-separated string (e.g. `"Industrial,Ads"`), the
> type check never triggers for normal rows, and `market_shift` is effectively
> **0** for all non-corrupted rows. The market weights defined above are
> therefore not active in the current default generation.

**Cost risk factor**

```
p* = (0.18 − profit_margin) / 0.08
ε  ~ N(0, σ=0.40)

cost_risk_factor = 1.10 · p*
                 + 0.45 · max(0, p*)²
                 + market_shift(w_cost)          # = 0 in practice (see note)
                 − 0.55 · audited
                 + ε
```

**Sales risk factor**

```
r* = (log(revenue) − 16) / 1
ε  ~ N(0, σ=0.45)

sales_risk_factor = 0.90 · r*²
                  + 0.30 · r*
                  + market_shift(w_sales)        # = 0 in practice (see note)
                  − 0.55 · audited
                  + ε
```

### Threshold calibration and target assignment

Thresholds X and Y are set from quantiles of the generated risk factors so that
approximately 30 % of rows exceed X and approximately 20 % exceed Y:

```
X = quantile(cost_risk_factor,  0.70)   # 70th percentile
Y = quantile(sales_risk_factor, 0.80)   # 80th percentile
```

The target label is then assigned deterministically:

```
risk = "both_risks"   if cost_risk_factor > X  AND  sales_risk_factor > Y
     = "cost_risk"    if cost_risk_factor > X  AND  sales_risk_factor ≤ Y
     = "sales_risk"   if cost_risk_factor ≤ X  AND  sales_risk_factor > Y
     = "healthy"      otherwise
```

## Feature Influence on Classification

This section traces, in plain language, how each observable column feeds into
the final `risk` label.

**TL;DR**

- `revenue` — U-shaped effect on **sales risk**: very low and very high revenue both increase it; mid-range is safest.
- `costs` — drives **cost risk** via profit margin; falling below an 18 % margin triggers a nonlinear penalty.
- `audited` — reduces **both** risk factors by a fixed amount; the strongest protective signal in the dataset.
- `market_focus` — intended to shift risk by tag, but has **no effect** in practice due to an implementation bug; treat it as noise.
- `ceo_tenure_years` — **no effect** on the target; pure noise feature.

**revenue**

Revenue has no direct effect on cost risk. Its influence runs entirely through
the sales risk channel. The sales risk factor is a quadratic function of
standardized log-revenue (`r*`), which means both very small and very large
companies score higher on sales risk than a mid-sized company does. The quadratic
term dominates, so the relationship is U-shaped around a revenue of roughly
`exp(15.83) ≈ 7.5 M`. A small positive linear tilt makes the right tail
(very high revenue) slightly riskier than the left tail (very low revenue) at
the same distance from the centre. Revenue also raises the probability of being
audited, which in turn dampens both risk factors (see `audited` below), so the
net effect of very high revenue is partially self-correcting.

**costs**

Costs drive cost risk exclusively through profit margin. The higher the cost
relative to revenue, the lower the profit margin and the higher `p*`. The cost
risk factor has two terms that both push in the same direction when margin falls
below 18 %: a linear term and an additional one-sided quadratic penalty that
only activates below that threshold. This creates a nonlinear acceleration —
a company whose margin drops from 20 % to 10 % suffers a much larger increase
in cost risk than one whose margin drops from 30 % to 20 %. Companies with
margins above 18 % experience only the linear benefit with no quadratic
correction, so the downside is steeper than the upside. Costs have no direct
effect on sales risk.

**audited**

Being audited subtracts a fixed 0.55 from both the cost risk factor and the
sales risk factor simultaneously. This is the only feature that influences both
risk dimensions at once, and it does so unconditionally: the reduction is the
same regardless of revenue, costs, or market focus. Because the standard
deviation of the noise in each risk factor is around 0.40–0.45, a 0.55 shift
is roughly 1.2–1.4 standard deviations, which is large enough to meaningfully
push borderline companies out of any risk category. Audited companies are
therefore disproportionately likely to be classified as healthy.

**market_focus**

The generator defines market weights intended to shift risk factors by tag
(for example, Industrial was meant to raise cost risk, Ads to raise sales risk,
and Contract to reduce both). However, due to an implementation detail described
in the Formulas section, those weights are never applied to normal rows. In
practice, `market_focus` carries no predictive signal for the target label in
the default generated datasets. It is effectively a decoy feature and
should not improve model accuracy when used honestly.

**ceo_tenure_years**

CEO tenure does not appear anywhere in the risk factor formulas. It has no
causal path to the target label and carries zero predictive signal by design.
It is a pure noise feature intended to test whether students add irrelevant
columns to their models.

## Reproducibility

The generated datasets are deterministic for a fixed set of arguments:

- `N`
- `seed`
- `outlier_prob`
- `GeneratorConstants`
- `ClassProportions`

The checked-in CSV files are produced with these calls:

```python
train_df, _ = generate_dataset(N=5000, seed=0, outlier_prob=0.005)

for seed in [1, 24, 42, 20260326]:
    test_df, _ = generate_dataset(N=500, seed=seed, outlier_prob=0.002)
```

Changing any generator constants, class proportions, seeds, row counts, or
outlier probabilities will produce different files.
