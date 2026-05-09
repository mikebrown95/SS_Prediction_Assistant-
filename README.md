# SSDI Predictor

A local prototype web application that estimates the probability of a Social Security Disability Insurance award from claimant demographics, education, location, work context, and medical condition inputs.

This is a decision-support prototype, not legal advice and not an SSA determination. The app can use SSA public aggregate workload data as a trained location baseline, then applies transparent claimant-level adjustments for fields not present in that public dataset.

## Run

```powershell
.\.venv\Scripts\python.exe app\server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Train From SSA Data

The training script uses the public SSA files in `data\`:

- Data.gov metadata: https://catalog.data.gov/dataset/ssa-disability-claim-data
- Annual state workload CSV: `data\SSA-SA-FYWL.csv`
- Monthly state workload CSV: `data\SSA-SA-MOWL.csv`
- ALJ disposition XML/CSV export: `data\ALJ Dispostion Data.csv`
- Hearing office processing time XML/CSV export: `data\AVG Processing Time by Office.csv`
- SSA data dictionary: https://www.ssa.gov/disability/data/ssa-sa-fywl.htm#DataDictionary

Run:

```powershell
.\.venv\Scripts\python.exe scripts\train_ssa_model.py
```

The script writes:

```text
app\artifacts\ssa_trained_model.json
```

That artifact is loaded automatically by `app/model.py`. If the artifact is absent, the application falls back to documented default baselines.

## Data Used

The current artifact was trained from four SSA public aggregate files:

| File | Rows used | Granularity | Main fields used |
| --- | ---: | --- | --- |
| `SSA-SA-FYWL.csv` | 1,248 | State agency by fiscal year, 2001-2024 | Adult determinations, favorable adult determinations, adult favorable determination rate, region, state |
| `SSA-SA-MOWL.csv` | 16,770 | State agency by month | Initial SSDI determinations/allowances, reconsideration SSDI determinations/allowances, closing pending inventory |
| `ALJ Dispostion Data.csv` | 1,409 | Administrative Law Judge disposition records | Fully favorable, partially favorable, denials, total decisions, hearing region |
| `AVG Processing Time by Office.csv` | 168 | Hearing office | Average processing time, dispositions, pending cases, hearing region |

The source files are aggregate operational statistics. They are not individual claimant records. They do not contain claimant diagnosis details, education, vocational history, earnings, functional capacity, attorney representation, application date, exact onset date, or case-level final outcomes.

## Model Notes

The model is a transparent aggregate-baseline estimator, not a black-box machine-learning classifier.

The trained artifact type is:

```text
aggregate_state_region_allowance_baseline
```

The main target is:

```text
adult_favorable_determination_rate
```

The app combines:

- State and region adult favorable-rate baselines from annual SSA workload data.
- Stage-specific award probabilities from monthly Initial SSDI and Reconsideration SSDI data.
- ALJ fully favorable, partially favorable, denial, and total award rates from ALJ disposition data.
- Decision timing estimates from monthly state workload data and hearing-office processing-time data.
- Transparent claimant-level adjustments for age, education, condition category, expected impairment duration, work level, specialist evidence, recent work attempt, and location.

The claimant-level adjustments are heuristic because the public SSA files are aggregate files. They should be replaced with coefficients learned from validated case-level data before the app is used for real operational decisions.

## Technical Model Details

This project does not currently train a supervised person-level classifier such as logistic regression, random forest, gradient boosting, or a neural network. The available SSA files are aggregate workload tables, so the trained model is an empirical rate model with hierarchical smoothing and deterministic adjustment layers.

The annual state/location model estimates the adult favorable determination probability as a weighted blend:

```text
predicted_state_rate =
  0.75 * historical_state_rate
  + 0.15 * historical_region_rate
  + 0.10 * national_rate
```

The rates are weighted by determination volume:

```text
rate = favorable_adult_determinations / all_adult_determinations
```

The app then converts the trained aggregate baseline into log-odds and applies transparent claimant-level log-odds adjustments:

```text
score = logit(trained_location_baseline)
  + age_adjustment
  + condition_category_adjustment
  + duration_adjustment
  + education_adjustment
  + work_level_adjustment
  + evidence_adjustment
  + recent_work_adjustment

overall_probability = logistic(score)
```

The stage model estimates where an award may occur using aggregate stage-specific rates:

```text
P(initial award) = initial_ssdi_allowances / initial_ssdi_determinations
P(recon award | initial denial) = recon_ssdi_allowances / recon_ssdi_determinations
P(ALJ fully favorable | hearing path) = fully_favorable / total_alj_decisions
P(ALJ partially favorable | hearing path) = partially_favorable / total_alj_decisions
```

The raw stage probabilities are scaled so the award-level probabilities sum to the overall award probability. The remaining probability is assigned to `Not awarded through hearing`.

Decision-time estimates are trained separately:

```text
initial_months = closing_pending_initial_ssdi / monthly_initial_ssdi_determinations
reconsideration_months = closing_pending_recon_ssdi / monthly_recon_ssdi_determinations
alj_months = weighted_average_hearing_office_processing_days / 30.4375
```

ALJ processing time is weighted by office dispositions.

## Current Metrics

Current trained artifact:

| Metric | Value |
| --- | ---: |
| Annual workload rows | 1,248 |
| Annual training rows | 1,196 |
| Monthly workload rows | 16,770 |
| ALJ disposition rows | 1,409 |
| Hearing processing rows | 168 |
| States/territories | 52 |
| SSA regions | 10 |
| Fiscal year range | 2001-2024 |
| Holdout year | 2024 |
| Holdout rows | 52 |
| Holdout MAE | 0.032177 |

The holdout MAE is calculated by training state/region/national aggregate baselines on pre-2024 annual workload data, predicting 2024 adult favorable determination rates by state, and comparing those predictions to observed 2024 rates. It measures aggregate location-baseline error, not person-level classification accuracy.

Interpreted as accuracy for the available aggregate target, the current holdout mean absolute error is about 3.22 percentage points. For example, if a state's observed 2024 adult favorable determination rate were 35.0%, a typical absolute error at this level would be roughly 3.22 percentage points, not 3.22% relative error.

Classification metrics such as accuracy, precision, recall, ROC AUC, and F1 are not reported because the training data does not contain individual approved/denied claimant records. Those metrics require person-level labels.

Current stage baselines:

| Stage or outcome | Probability |
| --- | ---: |
| Initial SSDI allowance | 0.393288 |
| Reconsideration SSDI allowance | 0.183301 |
| ALJ award | 0.582709 |
| ALJ fully favorable | 0.495596 |
| ALJ partially favorable | 0.087113 |
| ALJ denial | 0.417285 |

Current decision-time baselines:

| Stage | Estimated months |
| --- | ---: |
| Initial SSDI | 5.3 |
| Reconsideration SSDI | 6.3 |
| ALJ hearing | 9.3 |

Timing estimates are cumulative in the app. For example, an ALJ estimate starts with the initial estimate, adds reconsideration timing, then adds hearing-office ALJ processing time.

## Limitations

Key limitations:

- State/location baseline: trained from SSA aggregate adult favorable determination rates.
- Award level probabilities: trained from aggregate Initial SSDI, Reconsideration SSDI, and ALJ fully/partially favorable disposition rates.
- Decision timing: initial and reconsideration estimates use recent monthly closing pending divided by monthly determinations. ALJ timing uses the hearing-office average processing-time file, weighted by dispositions.
- Demographics, education, condition, duration, work level, and evidence fields: transparent adjustment factors until person-level training data is available.
- Output: an estimate of favorable disability determination probability, not guaranteed SSDI award probability.

For production, train and evaluate against real case-level, consented, representative adjudication data. Track calibration, subgroup performance, false-positive/false-negative rates, and drift by state, condition category, age group, sex, race/ethnicity, and education level.
