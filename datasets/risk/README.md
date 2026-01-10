# Enterprise Risk Dataset

This dataset contains synthetic company information for a **multi-class classification** task:
predict the company's risk category.

### Target (`risk`)

Four classes:

- `healthy`
- `cost_risk`
- `sales_risk`
- `both_risks`


### `revenue` (numeric)
Annual revenue. Large scale with a wide range (small to very large companies).

### `costs` (numeric)
Annual operating costs. Strongly related to revenue.

### `audited` (binary, sometimes corrupted)
Indicator for whether the company is audited.

Expected clean values:
- `0` = not audited
- `1` = audited

### `market_focus` (multi-tag categorical)
A list of **1 or 2** strategic focuses for the company.

Possible tags:
- `Industrial`
- `Ads`
- `Contract`
- `Scale`

### `ceo_tenure_years` (numeric)
CEO tenure in years.
