"""Stage 1: Build a real per-function invocation predictor.

This script:
1. Loads an extracted Azure Functions Invocation Trace CSV.
2. Derives invocation start timestamps.
3. Bins invocations into 10-second intervals per function.
4. Builds lag and EMA features.
5. Creates a 30-second-ahead binary target.
6. Trains a separate lightweight model per sufficiently active function.
7. Produces predictions, metrics, and a timeline plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Keep all project directories relative to the repository root.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for directory in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Feature configuration: small bins make short-term behavior visible.
BIN_SECONDS = 10
HORIZON_SECONDS = 30
MIN_ROWS_FOR_MODEL = 40
MIN_CLASS_COUNT = 5

FEATURES = [
    "count",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_30",
    "rolling_60",
    "ema",
    "seconds_since_previous",
]


def find_input_csv() -> Path:
    """Find the first CSV that has the Azure Functions trace schema."""
    candidates = sorted(RAW_DIR.rglob("*.csv"))
    required = {"app", "func", "end_timestamp", "duration"}

    for path in candidates:
        try:
            sample = pd.read_csv(path, nrows=10)
        except Exception:
            # Ignore unreadable CSVs and continue searching.
            continue
        if required.issubset(sample.columns):
            return path

    raise FileNotFoundError(
        "No Azure Functions trace CSV was found in data/raw/. "
        "Extract the Microsoft 2021 trace and place a matching CSV there."
    )


def load_events(path: Path) -> pd.DataFrame:
    """Load and normalize the invocation trace."""
    df = pd.read_csv(path)
    required = {"app", "func", "end_timestamp", "duration"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Convert timing columns to numeric and discard malformed records.
    df["end_timestamp"] = pd.to_numeric(df["end_timestamp"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["app", "func", "end_timestamp", "duration"]).copy()

    # The official trace stores both completion time and execution duration.
    df["start_timestamp"] = df["end_timestamp"] - df["duration"]
    df["function_id"] = df["app"].astype(str) + "::" + df["func"].astype(str)
    df = df.sort_values(["function_id", "start_timestamp"])

    return df[["function_id", "start_timestamp", "end_timestamp", "duration"]]


def build_binned_series(events: pd.DataFrame) -> pd.DataFrame:
    """Convert irregular invocations into regular 10-second per-function bins."""
    frames = []

    # Process functions independently so the model is genuinely per-function.
    for function_id, group in events.groupby("function_id", sort=False):
        start = float(group["start_timestamp"].min())
        end = float(group["start_timestamp"].max())
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue

        first_bin = np.floor(start / BIN_SECONDS) * BIN_SECONDS
        last_bin = np.floor(end / BIN_SECONDS) * BIN_SECONDS
        bins = np.arange(first_bin, last_bin + BIN_SECONDS, BIN_SECONDS)

        counts, _ = np.histogram(group["start_timestamp"].to_numpy(), bins=np.append(bins, bins[-1] + BIN_SECONDS))
        frame = pd.DataFrame({"function_id": function_id, "bin_start": bins, "count": counts})
        frames.append(frame)

    if not frames:
        raise ValueError("No valid function time series could be constructed from the trace.")

    return pd.concat(frames, ignore_index=True)


def add_features(bins: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, EMA, recency, and future-invocation target features."""
    output = []

    for function_id, group in bins.groupby("function_id", sort=False):
        g = group.sort_values("bin_start").copy()

        # Lag counts capture short-term temporal dependence.
        g["lag_1"] = g["count"].shift(1).fillna(0)
        g["lag_2"] = g["count"].shift(2).fillna(0)
        g["lag_3"] = g["count"].shift(3).fillna(0)

        # Rolling recent activity provides short context beyond individual lags.
        g["rolling_30"] = g["count"].rolling(3, min_periods=1).sum()
        g["rolling_60"] = g["count"].rolling(6, min_periods=1).sum()

        # EMA emphasizes recent demand while smoothing noise.
        g["ema"] = g["count"].ewm(alpha=0.30, adjust=False).mean()

        # Compute time since the previous non-zero invocation bin.
        nonzero_times = g.loc[g["count"] > 0, "bin_start"].to_numpy()
        previous = None
        recencies = []
        for t in g["bin_start"]:
            if previous is None:
                recencies.append(float(HORIZON_SECONDS))
            else:
                recencies.append(min(float(t - previous), 3600.0))
            if t in nonzero_times:
                previous = t
        g["seconds_since_previous"] = recencies

        # Target is 1 when at least one event appears in the next 30 seconds.
        future = g["count"].shift(-1).fillna(0)
        future2 = g["count"].shift(-2).fillna(0)
        future3 = g["count"].shift(-3).fillna(0)
        g["target"] = ((future + future2 + future3) > 0).astype(int)

        output.append(g)

    return pd.concat(output, ignore_index=True)


def train_per_function(dataset: pd.DataFrame) -> Tuple[Dict[str, Pipeline], list]:
    """Train a separate logistic regression for each sufficiently active function."""
    models: Dict[str, Pipeline] = {}
    metrics = []

    for function_id, group in dataset.groupby("function_id", sort=False):
        g = group.dropna(subset=FEATURES + ["target"]).copy()
        if len(g) < MIN_ROWS_FOR_MODEL or g["target"].nunique() < 2:
            continue
        if int(g["target"].sum()) < MIN_CLASS_COUNT or int((1 - g["target"]).sum()) < MIN_CLASS_COUNT:
            continue

        # Use chronological train/test separation to avoid leaking future data.
        split = max(int(len(g) * 0.8), 1)
        train = g.iloc[:split]
        test = g.iloc[split:]
        if test.empty or test["target"].nunique() < 2:
            continue

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        model.fit(train[FEATURES], train["target"])

        probabilities = model.predict_proba(test[FEATURES])[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        y_true = test["target"].to_numpy()

        metrics.append(
            {
                "function_id": function_id,
                "samples": len(g),
                "test_samples": len(test),
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )
        models[function_id] = model

    return models, metrics


def make_predictions(dataset: pd.DataFrame, models: Dict[str, Pipeline]) -> pd.DataFrame:
    """Generate probability + pre-warm decision for the latest rows."""
    rows = []

    for function_id, group in dataset.groupby("function_id", sort=False):
        g = group.sort_values("bin_start").dropna(subset=FEATURES).copy()
        if g.empty:
            continue

        latest = g.tail(min(50, len(g))).copy()
        if function_id in models:
            latest["probability"] = models[function_id].predict_proba(latest[FEATURES])[:, 1]
            latest["model_type"] = "per-function-logistic"
        else:
            # Transparent fallback for sparse functions: normalize EMA against
            # their observed maximum activity instead of inventing a model score.
            max_ema = max(float(g["ema"].max()), 1.0)
            latest["probability"] = (latest["ema"] / max_ema).clip(0, 1)
            latest["model_type"] = "ema-fallback"

        # The 0.50 action threshold is configurable and deliberately explicit.
        latest["decision"] = np.where(latest["probability"] >= 0.50, "PRE-WARM", "NO PRE-WARM")
        rows.append(latest[["function_id", "bin_start", "ema", "probability", "decision", "model_type"]])

    if not rows:
        raise ValueError("No predictions could be produced.")

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Run the complete Stage 1 pipeline."""
    input_path = find_input_csv()
    print(f"[1/6] Using trace: {input_path}")

    events = load_events(input_path)
    print(f"[2/6] Loaded {len(events):,} invocation records")

    bins = build_binned_series(events)
    bins.to_csv(PROCESSED_DIR / "invocation_bins.csv", index=False)
    print(f"[3/6] Built {len(bins):,} function-time bins")

    dataset = add_features(bins)
    print(f"[4/6] Built lag/EMA features for {dataset['function_id'].nunique():,} functions")

    models, metrics = train_per_function(dataset)
    joblib.dump({"models": models, "features": FEATURES}, MODELS_DIR / "per_function_models.joblib")
    pd.DataFrame(metrics).to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    print(f"[5/6] Trained {len(models):,} per-function logistic models")

    predictions = make_predictions(dataset, models)
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    # Plot the highest-activity function's recent EMA and predicted probability.
    best_function = (
        dataset.groupby("function_id")["count"].sum().sort_values(ascending=False).index[0]
    )
    plot_df = predictions[predictions["function_id"] == best_function].copy()
    if not plot_df.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(plot_df["bin_start"], plot_df["ema"], label="EMA")
        plt.plot(plot_df["bin_start"], plot_df["probability"], label="Prediction probability")
        plt.title(f"Per-function prediction: {best_function}")
        plt.xlabel("Time (trace seconds)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "prediction_timeline.png", dpi=160)
        plt.close()

    print("[6/6] Complete")
    print(f"      Predictions: {RESULTS_DIR / 'predictions.csv'}")
    print(f"      Metrics:     {RESULTS_DIR / 'model_metrics.csv'}")
    print(f"      Models:      {MODELS_DIR / 'per_function_models.joblib'}")


if __name__ == "__main__":
    main()
