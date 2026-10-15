# Model card

Version 1.0.0. Trained 2026-10-15T12:23:52+00:00.

Advisory only. A recommendation is an invitation to investigate, not authorization to change a factory assignment.

## Target

Primary target: `LeadTimeDays = Ship Date − Order Date`, in calendar days.
Dates are parsed as day-month-year (%d-%m-%Y).
Ship date, lead time, order identifiers, customer id, postal code, and the order-id year are not inputs.

A second target, the within-cohort operational residual, subtracts the training-set median lead time of the Order ID year. That identifier is used only to decompose the anomalous target. It is not a feature. The residual model is a sensitivity analysis, not a replacement ETA.

## Split

- Strategy: chronological holdout by order date
- Cutoff: 2025-10-09 (test starts on this date)
- Train rows: 8149 (2024-01-02 to 2025-10-08)
- Test rows: 2045 (2025-10-09 to 2025-12-31)
- Train median lead time: 1273.0 days
- Test median lead time: 1637.0 days

Rows that share an order date stay in the same partition because the cutoff is a date boundary. Order lines in this extract share one order date.

Model choice uses chronological cross-validation MAE on the training period. The holdout is reported once and is not used to pick the model.

## Raw lead-time results

Selected model: `group_median_baseline`.

Selected group_median_baseline by mean chronological cross-validation MAE on the training period. Models within 1 day of the best CV MAE are treated as tied, and the simpler model is kept. The test set was not used for selection.

| Model | CV MAE | Test MAE | Test RMSE | Test R² | Test rows |
|---|---:|---:|---:|---:|---:|
| group_median_baseline | 205.59 | 206.75 | 274.90 | -1.296 | 2045 |
| linear_regression | 311.93 | 273.36 | 332.42 | -2.357 | 2045 |
| random_forest | 368.62 | 364.46 | 420.96 | -4.383 | 2045 |
| gradient_boosting | 367.59 | 357.34 | 409.52 | -4.094 | 2045 |
| global_median | n/a | 206.91 | 274.76 | -1.293 | 2045 |

## Within-cohort residual results

Selected model: `group_median_baseline`.

Selected group_median_baseline by mean chronological cross-validation MAE on the training period. Models within 1 day of the best CV MAE are treated as tied, and the simpler model is kept. The test set was not used for selection.

| Model | CV MAE | Test MAE | Test RMSE | Test R² | Test rows |
|---|---:|---:|---:|---:|---:|
| group_median_baseline | 0.81 | 0.79 | 1.00 | 0.686 | 2045 |
| linear_regression | 0.86 | 0.82 | 1.01 | 0.682 | 2045 |
| random_forest | 0.84 | 0.81 | 1.00 | 0.687 | 2045 |
| gradient_boosting | 0.85 | 0.81 | 0.99 | 0.690 | 2045 |
| global_median | n/a | 1.37 | 1.79 | -0.000 | 2045 |

## Leakage check

Passed: True. Forbidden features found: none.
Features: ship_mode, region, division, product_name, state, country, units, order_month, order_dow, order_quarter.
Factory effect learned from history: False.

## Outlier treatment

Lead time greater than 60 calendar days is unusually long for a ship-date target.
Flagged rows: 10194. Rows retained if excluded: 0. Rows deleted: 0.
No rows were deleted. Excluding unusually long lead times does not leave enough rows to train when the date anomaly affects the whole extract. A parallel within-cohort residual target is reported as a documented sensitivity analysis, not as a replacement for the raw target.

## Limitations

- The raw target is dominated by a multi-year ship-date gap. Metrics describe that gap. They are not a carrier ETA.
- Factory origin is absent unless a future extract supplies it. Recommendations do not invent a factory effect.
- Customer coordinates, freight, capacity, and factory cost are absent. Profit impact stays unquantified unless a user labels a sensitivity assumption.
- A lower scenario lead time is a reason to investigate, not proof that a transfer will improve delivery or profit.

## Data-quality snapshot

Rows: 10194. Source columns: 18.
Training allowed: True. Target credibility: low.
Critical findings: 3. Warning-level findings: 11.
