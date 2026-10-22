"""
Train the real per-function prediction layer.

This script uses ONLY the real feature dataset produced from the
Microsoft Azure Functions Invocation Trace 2021.

Design:
- One lightweight model per (app, func).
- Chronological 80/20 split to avoid future-to-past leakage.
- StandardScaler + LogisticRegression where both classes exist.
- EMA-only fallback for functions whose training labels contain one class.
- Probability output for next-30-second invocation.
- A configurable decision threshold converts probability to PRE-WARM/NO PRE-WARM.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_FILE = Path("data/processed/features_top10.csv")
MODEL_DIR = Path("models/per_function")
RESULT_DIR = Path("results")

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "ema",
    "recent_2min_count",
    "time_since_last_invocation_bins",
]

TARGET = "target_next_30s"

# This is a policy parameter, not a measured result.
# We will calibrate it later using the real cloud benchmark.
PREWARM_THRESHOLD = 0.50

TEST_FRACTION = 0.20
RANDOM_STATE = 42


def make_model() -> Pipeline:
    """Create the lightweight per-function probability model."""
    return Pipeline(
        steps=[
            # Scaling prevents large-count features from dominating the model.
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_binary(y_true: pd.Series, probability: np.ndarray) -> dict:
    """Calculate honest classification metrics without inventing results."""
    prediction = (probability >= PREWARM_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
    }

    # ROC-AUC is only defined when both classes appear in the test set.
    if y_true.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probability)
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def train_one_function(group: pd.DataFrame, app: str, func: str) -> dict:
    """Train/evaluate one model for one real (app, func) pair."""
    group = group.sort_values("bin_id").reset_index(drop=True)

    split_index = int(len(group) * (1 - TEST_FRACTION))

    # Keep the split chronological: training uses the past,
    # testing uses a later unseen portion.
    train = group.iloc[:split_index].copy()
    test = group.iloc[split_index:].copy()

    x_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    x_test = test[FEATURES]
    y_test = test[TARGET].astype(int)

    model_path = MODEL_DIR / f"{app}_{func}.joblib"

    # A logistic classifier cannot be trained if the training labels
    # contain only 0s or only 1s. In that rare case, use EMA as an
    # explicit lightweight fallback rather than fabricating a model.
    if y_train.nunique() < 2:
        fallback_probability = float(y_train.mean())

        # Save metadata so the controller knows this is a fallback.
        joblib.dump(
            {
                "type": "ema_fallback",
                "probability": fallback_probability,
                "features": FEATURES,
                "threshold": PREWARM_THRESHOLD,
                "app": app,
                "func": func,
            },
            model_path,
        )

        probability = np.full(len(test), fallback_probability)

        metrics = evaluate_binary(y_test, probability)

        return {
            "app": app,
            "func": func,
            "model_type": "ema_fallback",
            "train_rows": len(train),
            "test_rows": len(test),
            "positive_rate_train": y_train.mean(),
            "positive_rate_test": y_test.mean(),
            **metrics,
        }

    model = make_model()
    model.fit(x_train, y_train)

    probability = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_binary(y_test, probability)

    joblib.dump(
        {
            "type": "logistic_regression",
            "model": model,
            "features": FEATURES,
            "threshold": PREWARM_THRESHOLD,
            "app": app,
            "func": func,
        },
        model_path,
    )

    return {
        "app": app,
        "func": func,
        "model_type": "logistic_regression",
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_rate_train": y_train.mean(),
        "positive_rate_test": y_test.mean(),
        **metrics,
    }


def make_latest_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the latest prediction for each function."""
    outputs = []

    for (app, func), group in df.groupby(["app", "func"], sort=False):
        latest = group.sort_values("bin_id").iloc[-1].copy()

        model_path = MODEL_DIR / f"{app}_{func}.joblib"
        package = joblib.load(model_path)

        if package["type"] == "logistic_regression":
            features = pd.DataFrame(
                [[latest[col] for col in FEATURES]],
                columns=FEATURES,
            )
            probability = float(package["model"].predict_proba(features)[0, 1])
        else:
            probability = float(package["probability"])

        decision = (
            "PRE-WARM"
            if probability >= PREWARM_THRESHOLD
            else "NO PRE-WARM"
        )

        outputs.append(
            {
                "app": app,
                "func": func,
                "bin_start_sec": latest["bin_start_sec"],
                "lag_1": latest["lag_1"],
                "lag_2": latest["lag_2"],
                "lag_3": latest["lag_3"],
                "ema": latest["ema"],
                "recent_2min_count": latest["recent_2min_count"],
                "time_since_last_invocation_bins":
                    latest["time_since_last_invocation_bins"],
                "predicted_probability": probability,
                "decision": decision,
            }
        )

    return pd.DataFrame(outputs).sort_values(
        "predicted_probability",
        ascending=False,
    )


def main() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FEATURE_FILE}. Run "
            "`python src\\build_features.py` first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading REAL feature data...")
    df = pd.read_csv(FEATURE_FILE)

    required = set(FEATURES + [TARGET, "app", "func", "bin_id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Feature rows: {len(df):,}")
    print(
        "Functions:",
        df[["app", "func"]].drop_duplicates().shape[0],
    )

    metrics = []

    for (app, func), group in df.groupby(["app", "func"], sort=False):
        print(
            f"Training {str(app)[:10]}... / "
            f"{str(func)[:10]}... | rows={len(group):,}"
        )
        metrics.append(
            train_one_function(group, str(app), str(func))
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_path = RESULT_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    latest_df = make_latest_predictions(df)
    latest_path = RESULT_DIR / "latest_predictions.csv"
    latest_df.to_csv(latest_path, index=False)

    metadata = {
        "feature_file": str(FEATURE_FILE),
        "feature_columns": FEATURES,
        "target": TARGET,
        "model": "per-function Logistic Regression with EMA fallback",
        "forecast_horizon_seconds": 30,
        "prewarm_threshold": PREWARM_THRESHOLD,
        "test_fraction": TEST_FRACTION,
        "split": "chronological",
    }

    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDONE")
    print(f"Metrics: {metrics_path}")
    print(f"Latest predictions: {latest_path}")
    print("\nLatest decisions:")
    print(
        latest_df[
            ["app", "func", "predicted_probability", "decision"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
