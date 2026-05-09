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

## Model Notes

The SSA dataset is useful, but it is not person-level SSDI adjudication data. It is fiscal-year, state-agency aggregate data for initial disability claims referred to a state agency. It includes receipts, determinations, eligible populations, favorable determinations, and favorable determination rates by state and year. It does not include claimant age, sex, education, diagnosis, severity, residual functional capacity, earnings history, or final non-disability eligibility outcomes.

Because of that limitation:

- State/location baseline: trained from SSA aggregate adult favorable determination rates.
- Award level probabilities: trained from aggregate Initial SSDI, Reconsideration SSDI, and ALJ fully/partially favorable disposition rates.
- Decision timing: initial and reconsideration estimates use recent monthly closing pending divided by monthly determinations. ALJ timing uses the hearing-office average processing-time file, weighted by dispositions.
- Demographics, education, condition, duration, work level, and evidence fields: transparent adjustment factors until person-level training data is available.
- Output: an estimate of favorable disability determination probability, not guaranteed SSDI award probability.

For production, train and evaluate against real case-level, consented, representative adjudication data. Track calibration, subgroup performance, false-positive/false-negative rates, and drift by state, condition category, age group, sex, race/ethnicity, and education level.
