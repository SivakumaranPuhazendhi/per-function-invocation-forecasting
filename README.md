# Predictive Cold-Start Mitigation for Serverless Functions

## Stage 1: Per-function prediction pipeline

This stage builds the real data/ML core before connecting it to OpenFaaS.

### Input dataset
Use the official Microsoft Azure Functions Invocation Trace 2021 dataset.
Expected columns:
- app
- func
- end_timestamp
- duration

The trace contains two weeks of function invocations. `end_timestamp` and `duration` are used to derive the invocation start time.

### Windows setup
```powershell
cd predictive-cold-start
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put an extracted CSV from the Microsoft trace into `data/raw/`. The script will locate the first CSV containing the required columns.

### Run
```powershell
python src\train_predictor.py
```

Outputs:
- `data/processed/invocation_bins.csv`
- `models/per_function_models.joblib`
- `results/predictions.csv`
- `results/model_metrics.csv`
- `results/prediction_timeline.png`

The model uses 10-second bins, lag-1/2/3 features, recent counts, and EMA. It predicts whether at least one invocation will occur in the next 30 seconds. Each sufficiently active function gets its own logistic-regression model; functions with too little class diversity use a transparent EMA-based fallback rather than fabricating a model.
