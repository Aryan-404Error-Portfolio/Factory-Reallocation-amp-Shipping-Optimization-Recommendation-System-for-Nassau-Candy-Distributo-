# Factory Reallocation Desk

Decision-support application for Nassau Candy Distributor. It loads the supplied order extract, checks data quality, trains a lead-time model against a simple baseline, and lets an analyst compare factory assignments without pretending missing data is known.

This is not an autonomous allocation system. Recommendations are advisory and require a person to review them before any assignment changes.

The product specification is in [docs/PRD.md](docs/PRD.md). This repository does not contain a research paper.

## What the desk does

- Validates the extract and keeps extreme lead times instead of deleting them.
- Calculates calendar lead time as ship date minus order date.
- Trains a ship-mode median baseline, linear regression, random forest, and gradient boosting on a chronological holdout.
- Compares those models on the raw lead-time target and on a documented within-cohort residual.
- Simulates the current factory against alternatives using the same product, region, ship mode, and order size.
- Ranks lanes with an explicit speed, profit, and risk score.
- Exports the filtered recommendation table, scenario assumptions, validation report, and model card.

## What it will not claim

The supplied extract has no origin factory, no customer coordinates, and no freight or factory-cost fields. Ship dates parse as day-month-year and then sit 904 to 1,642 days after the order date. That gap tracks the year embedded in Order ID, which is an identifier and is not used as a model feature.

Because of that:

- Swapping a factory label does not change the historical prediction. The model never saw a factory.
- Profit impact is **Not quantified** unless you type a labeled sensitivity assumption.
- Straight-line distance is never described as road distance.
- A lower scenario lead time is a reason to investigate, not proof of a better assignment.

On the supplied file, the selected raw-target model is the ship-mode median baseline. More complex models do not beat it on the chronological test set. The within-cohort residual is a few days and is mostly ship mode. Details are on the Model and data quality page and in `reports/model_card.md`.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The supplied extract is already at `data/Nassau_Candy_Distributor.csv`. Reference tables are in `data/`:

- `factory_reference.csv` — supplied factory coordinates
- `product_factory_mapping.csv` — supplied current assignment
- `factory_eligibility.csv` — current factory confirmed; alternatives require validation
- `state_centroids.csv` — approximate centroids for an opt-in straight-line scenario only

Trained artifacts are in `models/`. Rebuild them with:

```bash
python scripts/train_models.py
```

## Run

From the repository root:

```bash
streamlit run app.py
```

The app binds to `0.0.0.0:8501` so it can be opened through a preview proxy. Cross-origin and XSRF checks are disabled in `.streamlit/config.toml` for that preview. Turn them back on for a hardened local deployment if you do not need iframe embedding.

Optional uploads, from the sidebar:

- A replacement orders CSV with the same columns.
- A coordinate file with `City`, `State/Province`, `Latitude`, and `Longitude`. A header-only template is in `data/templates/customer_geocodes.csv`. Coordinates are not used until you enable the distance scenario.

## Tests

```bash
pytest
```

Tests cover date parsing, lead-time math, leakage exclusions, chronological splitting, score renormalization when profit is missing, the ban on invented factory effects, negative scenario outcomes, and a small end-to-end training run.

## Project layout

```text
app.py                      Streamlit entry point
src/data_loader.py          Load, validate, reference tables
src/features.py             Target, leakage audit, chronological split
src/train.py                Baseline and model comparison
src/simulator.py            Factory scenario comparison
src/recommendations.py      Scores, evidence, risk flags
src/clustering.py           Descriptive route clusters
src/visualizations.py       Charts
src/pages/                  Overview, simulator, what-if, recommendations, risk, model
tests/                      pytest suite
models/                     Versioned metrics and fitted models
reports/                    Model card and data-quality JSON
data/                       Extract and reference tables
docs/PRD.md                 Product specification
```

## How to read a recommendation

1. Start on Overview. The date anomaly is the first finding, not a footnote.
2. Open Factory simulator. The current factory is the baseline. Alternatives use the same destination and ship mode.
3. Leave distance and cost assumptions off if you want the historical model only. Days saved will be zero.
4. Turn on a labeled assumption only to explore a what-if. The financial column will say the result is not validated.
5. Use Recommendations to filter, sort, and export. Actionable count stays at zero until origin factory is actually in the training data and eligibility is confirmed.
6. Use Model and data quality to see MAE, RMSE, R², the baseline, the split, and the leakage check.

The optimization priority slider always shows requested speed and profit weights. If no cost assumption is entered, the profit weight is not applied and the ranking does not invent a dollar impact. Load the example sensitivity on the assumption panel to see the slider reorder a labeled tradeoff. Those example numbers are not company costs.

## Assumptions that affect every scenario

- Dates are day-month-year. Month-first parsing fails on dates such as 31-05-2024.
- Lead time is calendar days.
- Current factory comes from the supplied mapping, not from the order row.
- Model features are ship mode, region, division, product, state, country, units, order month, weekday, and quarter.
- Order ID, customer ID, postal code, ship date, lead time, sales, cost, and gross profit are not predictors.
- Eligibility defaults to requires-validation for every factory that is not the mapped current factory.
- Reference timing for a scenario is June, Wednesday, unless you change it.
- Bulk distance scenarios use an approximate state centroid or an uploaded coordinate file only after you opt in.

## Next data, before any real reassignment

Actual origin factory, confirmed product-factory eligibility and capacity, customer coordinates or a carrier route distance, freight, factory-specific cost, and a source-owner explanation of the ship date.
