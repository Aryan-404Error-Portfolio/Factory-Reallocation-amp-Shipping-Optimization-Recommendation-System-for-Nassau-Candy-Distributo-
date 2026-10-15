# Product Requirements Document (PRD)
## Factory Reallocation & Shipping Optimization Recommendation System
**Client:** Nassau Candy Distributor  
**Product type:** Data science decision-support dashboard  
**Primary interface:** Streamlit web application  
**Version:** 1.0  
**Status:** Proposed MVP specification

---

## 1. Executive Summary

Nassau Candy Distributor needs a practical way to evaluate whether assigning a product to a different factory could improve shipping speed without creating unacceptable financial or operational risk. The proposed system will combine historical order data, a shipping lead-time prediction model, factory-to-product assignment rules, and a scenario simulator.

The application will let an analyst select a product, destination region, and shipping mode; compare the current factory with eligible alternatives; and review predicted lead time, expected improvement, estimated profit impact, risk indicators, and the reasoning behind each recommendation. It will also provide a ranked recommendation table for reviewing many products at once.

This is a decision-support tool, not an autonomous factory-allocation system. Recommendations must be reviewed by an authorized business user before any real assignment is changed.

## 2. Business Problem

The current factory assignments are treated as static rules. That can lead to unnecessarily long routes, slower delivery for some destination regions, and logistics costs that reduce the margin available on an order. The business needs to test alternatives before changing its operating process.

The system should answer five questions:

1. How long is an order likely to take to ship under a given product, factory, destination region, and ship mode?
2. Which alternative factory appears to offer the best improvement for a product?
3. How much lead-time improvement is predicted, in days and percentage terms?
4. Could the reassignment reduce profitability or introduce operational risk?
5. How confident is the system in its recommendation, and what evidence supports it?

## 3. Product Goals and Success Measures

### Goals
- Estimate shipping lead time using historical order and shipment information.
- Simulate alternative factory assignments for eligible products.
- Rank recommendations using configurable priorities for speed and profitability.
- Make the trade-offs visible through clear tables, charts, and risk flags.
- Support repeatable analysis with transparent assumptions and data-quality warnings.

### Proposed MVP success measures
These are evaluation targets, not claims of achieved performance.

- Model metrics (MAE, RMSE, and R²) are reported on a held-out test set and compared with a simple baseline.
- Every recommendation shows its baseline, alternative scenario, predicted change, and confidence or evidence level.
- Users can filter and export recommendation results.
- No recommendation is described as financially validated when incremental freight or factory operating costs are unavailable.
- Data-quality issues are surfaced instead of silently hidden.

## 4. Users and Stakeholders

### Primary users
- **Operations analyst:** compares factory and route performance and investigates slow shipments.
- **Supply-chain manager:** reviews product reassignment suggestions and operational risk.
- **Finance or commercial stakeholder:** checks margin sensitivity and guards against recommendations that could harm profitability.
- **Executive reviewer:** uses a concise summary of potential benefits, risks, and assumptions.

### User needs
- A quick way to identify promising factory alternatives.
- An understandable explanation of why each alternative ranks highly.
- The ability to change the optimization priority between speed and profit protection.
- Clear warnings when evidence is limited or data is inconsistent.

## 5. Scope

### In scope for the MVP
- CSV upload or loading of the supplied Nassau Candy dataset.
- Data validation, cleaning, feature engineering, and exploratory analysis.
- Historical lead-time calculation from order date and ship date.
- Predictive-model training and evaluation.
- Product-to-factory mapping using the supplied correlation table.
- Distance calculation between factory coordinates and customer locations when customer geocoding is available.
- Scenario simulation across eligible factories.
- Ranked recommendations, risk flags, KPI cards, and what-if comparisons.
- Streamlit interface with filters and downloadable results.
- Documentation of assumptions, limitations, and model performance.

### Out of scope for the MVP
- Automatically changing production or factory assignments in live systems.
- Live carrier rates, freight invoices, capacity, inventory, or factory throughput integrations unless supplied later.
- Guaranteed delivery-date prediction or carrier service-level commitments.
- A fully validated profit optimizer without incremental shipping costs and factory-specific costs.
- Government-policy conclusions; the executive summary should be written for business and public-sector stakeholders without implying government endorsement.

## 6. Dataset and Data Dictionary

The supplied CSV contains **10,194 rows and 18 columns** at the time of initial inspection.

| Field | Meaning | Use in the system |
|---|---|---|
| Row ID | Unique row identifier | Traceability; not a predictive feature |
| Order ID | Order identifier | Grouping and leakage checks |
| Order Date | Order date | Lead-time calculation and time features |
| Ship Date | Shipment date | Lead-time target |
| Ship Mode | Shipping service/method | Model feature and dashboard filter |
| Customer ID | Customer identifier | Grouping/leakage checks; exclude as a direct model feature by default |
| Country/Region | Customer country or region field | Destination context |
| City | Customer city | Optional geocoding input |
| State/Province | Customer state/province | Destination context |
| Postal Code | Customer postal code | Optional geocoding input; treat as text |
| Division | Product division | Product grouping |
| Region | Customer destination region | Model feature and dashboard filter |
| Product ID | Product identifier | Product grouping and mapping |
| Product Name | Product name | User-facing product selector |
| Sales | Sales value | Financial context |
| Units | Number of units | Demand and scale context |
| Gross Profit | Gross profit | Profitability context |
| Cost | Product cost | Financial context |

### Supplied factory reference data

| Factory | Latitude | Longitude |
|---|---:|---:|
| Lot's O' Nuts | 32.881893 | -111.768036 |
| Wicked Choccy's | 32.076176 | -81.088371 |
| Sugar Shack | 48.119140 | -96.181150 |
| Secret Factory | 41.446333 | -90.565487 |
| The Other Factory | 35.117500 | -89.971107 |

### Supplied product-to-factory mapping

| Division | Product | Current/default factory |
|---|---|---|
| Chocolate | Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts |
| Chocolate | Wonka Bar - Fudge Mallows | Lot's O' Nuts |
| Chocolate | Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts |
| Chocolate | Wonka Bar - Milk Chocolate | Wicked Choccy's |
| Chocolate | Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's |
| Sugar | Laffy Taffy | Sugar Shack |
| Sugar | SweeTARTS | Sugar Shack |
| Sugar | Nerds | Sugar Shack |
| Sugar | Fun Dip | Sugar Shack |
| Other | Fizzy Lifting Drinks | Sugar Shack |
| Sugar | Everlasting Gobstopper | Secret Factory |
| Sugar | Hair Toffee | The Other Factory |
| Other | Lickable Wallpaper | Secret Factory |
| Other | Wonka Gum | Secret Factory |
| Other | Kazookles | The Other Factory |

Treat this table as the initial current/default assignment configuration. Before production use, a business owner must confirm that it reflects actual factory assignments and that every listed product can legally and operationally be made at every alternative factory. A product should only be simulated at a factory if that factory is marked eligible.

## 7. Critical Data-Quality and Modeling Considerations

Initial inspection indicates that the dataset's date fields require validation: the sample rows contain Order Dates in 2024 and Ship Dates in 2026. This may represent a genuine long delay, a date parsing problem, synthetic data, or a source-system issue. The application must not assume these values are correct without profiling the full dataset.

Required checks:
- Parse dates explicitly and report invalid or missing values.
- Calculate `LeadTimeDays = Ship Date - Order Date`.
- Flag negative, zero, and unusually long lead times for review.
- Display date-range summaries and a lead-time distribution before training.
- Do not silently delete extreme values. Compare model performance with and without clearly documented outlier treatment.
- Check duplicate Row IDs, duplicate records, missing values, inconsistent product names, and invalid financial values.
- Verify the arithmetic relationship between Sales, Cost, and Gross Profit. If Gross Profit is not consistently equal to Sales minus Cost, preserve source values and report the discrepancy rather than rewriting them.
- Treat Postal Code as a string, not a numeric quantity.
- Do not use post-outcome or identifier fields in a way that leaks the target into the model.

### Important limitation: factory origin is not in the CSV
The supplied order dataset does not contain an explicit origin-factory field or shipping distance. The current factory must therefore be joined from the provided product-to-factory mapping. This is an assumption, not an observed shipment fact. The data also does not include actual freight charges, carrier rates, factory capacity, inventory, or factory-specific production cost.

### Important limitation: customer coordinates are not supplied
Factory coordinates are available, but customer latitude/longitude is not. Accurate route distance cannot be calculated from City/State alone without a geocoding source. For the MVP, distance-based analysis is optional and should only run when valid customer coordinates are supplied or obtained from an approved geocoding process. Do not present straight-line distance as actual road distance.

## 8. Functional Requirements

### FR-1: Data ingestion and validation
- Load the supplied CSV and validate required columns.
- Show row count, column count, date coverage, missing-value counts, duplicate counts, and validation warnings.
- Allow the user to download a validation report.
- Stop model training if critical target fields are unusable, while still allowing basic descriptive analysis where possible.

### FR-2: Exploratory data analysis
- Summarize orders, sales, units, gross profit, ship modes, regions, divisions, and products.
- Show lead-time distribution and lead time by region, ship mode, product, and division.
- Identify slow product-region-ship-mode combinations, with minimum sample-size warnings.
- Clearly distinguish correlation and descriptive patterns from causal explanations.

### FR-3: Lead-time prediction
Predict `LeadTimeDays` from features available at decision time, such as:
- Product ID or product category
- Destination Region and, if appropriate, State/Province
- Ship Mode
- Division
- Units or order size, if available before shipment
- Candidate factory, introduced through the scenario feature design

Do not use Ship Date, calculated lead time, or other post-shipment information as model inputs. Avoid using Order ID, Row ID, or Customer ID as ordinary numeric predictors. Evaluate whether Order Date features are useful and ensure the evaluation setup reflects future use.

### FR-4: Model training and comparison
Train and compare:
- Baseline: median lead time by relevant group or a simple global median
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Use a reproducible split. Prefer a chronological holdout when enough valid dates exist; otherwise document the chosen split. Use cross-validation on the training data where practical. Report MAE, RMSE, and R² for train/validation/test as appropriate. Select the model using test-safe validation metrics and operational interpretability, not training score alone. If the dataset is too small or dates are invalid, report that limitation rather than claiming a reliable model.

### FR-5: Factory assignment and eligibility
- Load the supplied product-to-factory mapping as the initial assignment.
- Maintain an explicit product-factory eligibility table.
- Exclude the current factory from the alternative list when showing reassignment options, but always retain it as the comparison baseline.
- If eligibility is unknown, label the alternative as “requires validation” and do not rank it as an actionable recommendation by default.

### FR-6: Scenario simulator
For a selected product, destination region, ship mode, and order-size context:
1. Predict lead time using the current/default factory.
2. Predict lead time for each eligible alternative factory.
3. Calculate absolute lead-time change in days and percentage change.
4. Compare predicted outcomes against the same input context so the comparison is fair.
5. Display model confidence/evidence indicators, sample size, and extrapolation warnings.
6. Let the user compare current and recommended assignments side by side.

`LeadTimeReductionPct = (CurrentPredictedDays - AlternativePredictedDays) / CurrentPredictedDays × 100`, when current predicted days is greater than zero. Negative values indicate a predicted deterioration.

### FR-7: Profit-impact and financial safety
- Show observed Sales, Cost, Gross Profit, and Gross Profit Margin where valid.
- Do not claim a precise profit improvement from a faster predicted shipment unless freight savings, penalties, spoilage, production cost differences, or another approved financial relationship is supplied.
- Until cost inputs exist, report profit impact as “not quantified” or as a transparent sensitivity scenario using user-entered assumptions.
- Flag any scenario that requires unknown factory costs or shipping charges.
- Preserve the difference between product gross profit and incremental logistics profit.

### FR-8: Recommendation ranking
Generate a ranked list of product/factory alternatives. The ranking should support a user-controlled priority between speed and profit protection.

Suggested MVP scoring:
- Normalize expected lead-time reduction to a 0–1 score.
- Normalize available financial impact to a 0–1 score only when valid cost inputs exist.
- Calculate a risk penalty from low sample size, high prediction uncertainty, invalid/missing inputs, and unverified factory eligibility.
- `RecommendationScore = speed_weight × speed_score + profit_weight × profit_score − risk_weight × risk_penalty`.
- If financial impact is unavailable, do not fabricate a profit score. Renormalize available weights and clearly label the recommendation as speed-focused.
- Show the component scores and reason codes behind every rank.

### FR-9: Risk and confidence indicators
Use interpretable labels such as:
- **Higher evidence:** adequate comparable historical records and stable validation performance.
- **Limited evidence:** small sample size, unfamiliar product-region combination, or disagreement among models.
- **Requires validation:** unverified factory eligibility, missing financial inputs, or unsupported scenario.

A model prediction interval may be added if the chosen approach supports defensible intervals. Do not call a heuristic score a calibrated probability. Define the “Scenario Confidence Score” transparently and label it as an evidence score unless statistically calibrated.

### FR-10: Export and auditability
- Download filtered recommendations as CSV.
- Allow export of scenario assumptions, predicted results, model version/date, and reason codes.
- Record the selected filters and configuration used for a scenario.
- Show that recommendations are advisory and require human review.

## 9. Streamlit Dashboard Specification

### Module A — Overview
- KPI cards: valid records, median lead time, model MAE, recommended opportunities, and data-quality warnings.
- Charts: lead-time distribution, lead time by region and ship mode, and product/division summaries.
- A visible note describing any unresolved data-quality issue.

### Module B — Factory Optimization Simulator
Controls:
- Product selector
- Destination region selector
- Ship mode selector
- Units/order-size input if needed
- Current factory display
- Optimization priority slider: speed ↔ profit protection

Results:
- Current factory predicted lead time
- Alternative factory predictions
- Lead-time reduction in days and percentage
- Eligibility and evidence flags
- Ranked comparison chart/table

### Module C — What-If Scenario Analysis
- Compare baseline and selected alternative side by side.
- Display a before/after bar chart.
- Show assumptions and input values.
- Show predicted deterioration as well as improvement; do not hide negative outcomes.
- Allow comparison of multiple alternatives.

### Module D — Recommendation Dashboard
- Ranked product/factory suggestions.
- Filters for region, ship mode, division, minimum expected lead-time gain, evidence level, and eligibility.
- Columns: product, current factory, candidate factory, predicted current lead time, predicted alternative lead time, days saved, percentage change, evidence level, risk flags, financial impact status, and explanation.
- CSV export.

### Module E — Risk & Impact Panel
- Highlight low-evidence recommendations and unverified assignments.
- Display missing cost inputs and data-quality warnings.
- Show profit margin only where it can be calculated correctly.
- Provide a plain-language explanation of why each scenario is flagged.

### Module F — Model & Data Quality
- Show training date, dataset version, feature list, split strategy, MAE, RMSE, R², and baseline comparison.
- Display date anomalies, missing values, outlier treatment, and feature leakage checks.
- Make it clear when the model has not been trained successfully.

## 10. User Stories and Acceptance Criteria

### User story 1: Compare factories for a product
As an operations analyst, I want to select a product, destination region, and ship mode so that I can compare predicted lead times across eligible factories.

**Acceptance criteria**
- The current factory is shown as the baseline.
- Each eligible alternative has a predicted lead time and difference from baseline.
- Invalid or unsupported alternatives are visibly flagged.
- The calculation uses the same destination and ship-mode inputs for each comparison.

### User story 2: Review ranked recommendations
As a supply-chain manager, I want to see the highest-value candidate reassignments so that I can prioritize investigation.

**Acceptance criteria**
- Results can be filtered and sorted.
- Each row includes the predicted gain, evidence level, risk flags, and a plain-language reason.
- Financial impact is marked “not quantified” when required cost data is missing.
- The user can export the current filtered table.

### User story 3: Test speed-versus-profit priorities
As a business stakeholder, I want to adjust the optimization priority so that I can explore different trade-offs.

**Acceptance criteria**
- Changing the slider updates rankings and visible score components.
- The application does not invent financial impact when cost inputs are absent.
- The UI explains which weights and assumptions are active.

### User story 4: Understand model reliability
As a reviewer, I want to inspect model metrics and data-quality warnings so that I can judge whether to trust a recommendation.

**Acceptance criteria**
- MAE, RMSE, and R² are visible with a baseline comparison.
- Date and target-quality issues are shown.
- The test-set evaluation method is documented.
- Unsupported or low-evidence predictions are flagged.

## 11. Analytical Design

### Target variable
`LeadTimeDays = Ship Date − Order Date`, measured in calendar days unless the business explicitly defines a different convention.

### Feature preparation
- Parse dates using a documented day/month interpretation and verify it against the source owner.
- Encode categorical fields using a pipeline, such as one-hot encoding.
- Scale numeric features for Linear Regression where useful; tree-based models generally do not require scaling.
- Impute or exclude missing values through a documented pipeline.
- Keep preprocessing inside the model pipeline to avoid data leakage.
- Use outlier flags and sensitivity analysis rather than arbitrary removal alone.

### Validation design
- Split by time where possible to mimic predicting future shipments.
- If orders belonging to the same Order ID can appear across multiple rows, keep all rows for an order in the same split.
- Compare model performance with a simple baseline.
- Report metrics in days for MAE and RMSE, making them interpretable to operations users.
- Check residuals by region, ship mode, product, and division to identify systematic bias.

### Route clustering
Cluster product-region-ship-mode or route-level performance summaries only where sample sizes are sufficient. Candidate features include average/median lead time, lead-time variability, order volume, and prediction error. Explain clusters in operational terms (for example, “slow and variable”) rather than presenting cluster numbers alone.

### Scenario and optimization logic
The factory name should be a model input only if historical factory labels or defensible proxy data exist. Since factory origin is not in the CSV, a model cannot learn a real factory effect from the provided order data alone. For the MVP, factory-specific prediction differences must be treated as scenario estimates that require supporting data or a clearly disclosed proxy such as validated distance features. The application must not manufacture different predictions merely by swapping a factory label that the model has never learned.

For meaningful factory-level comparisons, the next data increment should include actual origin factory, customer coordinates or validated route distance, ship/pickup timestamps if distinct, freight charges, factory capacity, product-factory eligibility, and factory-specific handling/production cost.

## 12. KPIs and Definitions

| KPI | Definition |
|---|---|
| Lead-time reduction (days) | Baseline predicted lead time minus alternative predicted lead time |
| Lead-time reduction (%) | Days saved divided by baseline predicted lead time, × 100 |
| Model MAE | Mean absolute difference between predicted and actual lead time on held-out data |
| Model RMSE | Square root of mean squared prediction error on held-out data |
| Model R² | Proportion of target variance explained on held-out data; may be negative |
| Profit impact stability | Variation in estimated financial impact across documented sensitivity assumptions; unavailable until cost inputs exist |
| Scenario evidence score | Transparent evidence indicator based on sample size, model error, eligibility, and input coverage; not a probability unless calibrated |
| Recommendation coverage | Eligible product-region-ship-mode scenarios with usable predictions divided by all requested scenarios |
| Risk flag rate | Share of displayed recommendations with one or more risk warnings |

## 13. Non-Functional Requirements

- **Usability:** simple filters, readable labels, units on all KPIs, and plain-language explanations.
- **Performance:** typical filter changes and scenario comparisons should respond within a few seconds on a standard laptop for this dataset.
- **Reproducibility:** fixed random seeds where applicable; saved configuration and model metrics.
- **Maintainability:** modular code for ingestion, preprocessing, training, simulation, ranking, and UI.
- **Privacy:** avoid displaying unnecessary customer-level identifiers; do not expose personal data in exports unless needed.
- **Reliability:** errors should be actionable and must not silently produce fabricated recommendations.
- **Accessibility:** use descriptive chart labels and do not rely on color alone to communicate risk.

## 14. Recommended Technical Architecture

- **Python:** pandas, NumPy, scikit-learn.
- **Visualization:** Plotly or Matplotlib/Seaborn for analysis; Streamlit for the application.
- **Model pipeline:** scikit-learn `Pipeline` and `ColumnTransformer`.
- **Data storage for MVP:** local CSV and serialized model artifacts, with clear versioning.
- **Testing:** pytest for data validation, lead-time calculations, score logic, and scenario calculations.
- **Deployment:** local execution first; hosted deployment only after data permissions and operational requirements are agreed.

Suggested module layout:
- `app.py` — Streamlit entry point and navigation
- `src/data_loader.py` — loading and validation
- `src/features.py` — target creation and feature engineering
- `src/train.py` — baseline and model training/evaluation
- `src/simulator.py` — factory scenario comparisons
- `src/recommendations.py` — scoring, ranking, risk flags
- `src/visualizations.py` — charts and KPI calculations
- `tests/` — validation and calculation tests
- `models/` — versioned model artifacts and metrics
- `reports/` — EDA and model evaluation outputs

## 15. Risks, Assumptions, and Mitigations

| Risk or assumption | Impact | Mitigation |
|---|---|---|
| Order and ship dates may be inconsistent | Target becomes unreliable | Validate date ranges and confirm date format/source meaning |
| Factory origin absent from source data | Real factory effect cannot be learned | Join confirmed assignment mapping; collect actual factory for each shipment |
| Customer coordinates absent | Accurate route distances unavailable | Add validated coordinates or approved geocoding |
| Freight and factory costs absent | Profit impact cannot be quantified | Display as unavailable or use user-entered, labeled sensitivity assumptions |
| Factory eligibility/capacity unknown | Unrealistic reassignment could be recommended | Maintain eligibility/capacity rules and require business validation |
| Small sample groups | Unstable predictions | Minimum sample thresholds and evidence warnings |
| Historical patterns may change | Model drift | Monitor errors and retrain on reviewed, current data |
| Correlation mistaken for causation | Misleading business conclusions | Label model output as predictive association and validate through pilots |

## 16. Delivery Plan

### Phase 1 — Data audit and EDA
- Load CSV, verify schema and date formats.
- Produce data-quality report.
- Explore lead time, order volume, sales, cost, gross profit, region, ship mode, and product patterns.
- Confirm current factory mappings and eligibility.

### Phase 2 — Predictive baseline
- Create lead-time target.
- Build leakage-safe preprocessing.
- Train baseline, Linear Regression, Random Forest, and Gradient Boosting models.
- Evaluate and document metrics and subgroup performance.

### Phase 3 — Scenario engine
- Implement eligible-factory comparisons.
- Add lead-time deltas, evidence labels, risk flags, and explicit assumptions.
- Add profit sensitivity only if valid cost inputs are available.

### Phase 4 — Streamlit dashboard
- Build Overview, Simulator, What-If, Recommendations, Risk & Impact, and Model & Data Quality modules.
- Add filters, charts, and CSV export.

### Phase 5 — Testing and review
- Validate formulas and edge cases.
- Review recommendations with operations and finance stakeholders.
- Run a limited pilot before making real assignment changes.

## 17. Definition of Done

The MVP is ready for review when:
- The CSV loads and validation findings are visible.
- Lead time is calculated and invalid date cases are handled explicitly.
- Model metrics are evaluated on a held-out set and compared with a baseline.
- The application does not use target leakage.
- Factory alternatives are limited to approved eligible options or clearly marked as unverified.
- Scenario calculations are reproducible and include negative as well as positive outcomes.
- Missing freight/production costs are not replaced with fabricated profit claims.
- The dashboard supports product, region, ship mode, and speed-versus-profit priority controls.
- Recommendations are exportable and explainable.
- Documentation lists assumptions, limitations, and next data requirements.

## 18. Final Product Principle

The system should recommend a change only when the evidence supports a useful comparison. A lower predicted lead time is an opportunity to investigate, not proof that a factory transfer will improve actual delivery or profit. The application must make uncertainty, missing information, and business constraints visible so that users can make responsible decisions.

---

## Appendix A — Required Follow-up Data for a Production-Grade Optimizer

1. Actual origin factory for each shipment.
2. Factory eligibility by product and any capacity/production constraints.
3. Customer latitude/longitude or approved route-distance data.
4. Freight charges, carrier/service costs, and relevant surcharges.
5. Factory-specific production, handling, packaging, and transfer costs.
6. Inventory availability, replenishment time, and production lead time.
7. Actual delivery date, if delivery speed (rather than time to ship) is the business target.
8. Business rules for service-level targets, minimum margin, and risk tolerance.

## Appendix B — Suggested Outputs for the Research Paper

- Dataset profile and data-quality findings.
- Lead-time distribution and segment-level analysis.
- Baseline-versus-model performance table.
- Residual analysis by region and ship mode.
- Route/product cluster interpretation.
- Scenario comparison examples with assumptions.
- Ranked recommendation examples, including at least one rejected or high-risk scenario.
- Limitations and required data to validate operational and financial impact.
