"""
Predictive pre-warming controller for the serverless cold-start experiment.

The controller:
1. Reads the real predictions produced by train_models.py.
2. Selects a real per-function prediction.
3. Applies a probability + latency/resource trade-off policy.
4. Sends a real pre-warm request to OpenFaaS when PRE-WARM is selected.
5. Records the action and measured latency in results/controller_log.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# Keep paths relative to the project root.
ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = ROOT / "results" / "latest_predictions.csv"
RESULTS_DIR = ROOT / "results"
LOG_FILE = RESULTS_DIR / "controller_log.csv"

# OpenFaaS Gateway and deployed function endpoint.
FUNCTION_URL = "http://127.0.0.1:8080/function/burst-function"

# Prediction configuration from the trained model metadata.
PREWARM_THRESHOLD = 0.50
FORECAST_HORIZON_SECONDS = 30

# Real measurements collected during our experiment.
WARM_LATENCY_MS = 31.8003
COLD_PATH_LATENCY_MS = 3039.6398

# Lambda converts resource-idle time into a latency-equivalent policy cost.
# This is a tunable policy parameter, not a fabricated measurement.
LAMBDA_MS_PER_CONTAINER_SECOND = 50.0

# Network timeout for the actual function invocation.
REQUEST_TIMEOUT_SECONDS = 10


def load_prediction() -> pd.Series:
    """Load the first real prediction from the generated prediction file."""

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)

    if predictions.empty:
        raise ValueError("latest_predictions.csv contains no predictions.")

    required_columns = {
        "app",
        "func",
        "bin_start_sec",
        "predicted_probability",
    }

    missing = required_columns.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    # Use the first real prediction for this live demonstration.
    return predictions.iloc[0]


def make_decision(probability: float) -> tuple[str, float, float]:
    """
    Calculate whether pre-warming is worthwhile.

    expected_cold_penalty:
        Expected user-visible delay caused by a cold path.

    prewarm_cost_proxy:
        Latency-equivalent cost of keeping the container available
        during the 30-second prediction horizon.
    """

    # Only the excess latency over a warm request represents
    # the cold-start penalty we are trying to avoid.
    cold_penalty_ms = max(
        COLD_PATH_LATENCY_MS - WARM_LATENCY_MS,
        0.0,
    )

    # Convert probability into expected cold-start penalty.
    expected_cold_penalty_ms = probability * cold_penalty_ms

    # Treat the 30-second warm window as the resource-cost horizon.
    prewarm_cost_proxy = (
        FORECAST_HORIZON_SECONDS * LAMBDA_MS_PER_CONTAINER_SECOND
    )

    # Pre-warm only when:
    # 1. The model probability crosses the prediction threshold.
    # 2. Expected cold-start penalty exceeds the resource-cost proxy.
    should_prewarm = (
        probability >= PREWARM_THRESHOLD
        and expected_cold_penalty_ms > prewarm_cost_proxy
    )

    decision = "PRE-WARM" if should_prewarm else "NO PRE-WARM"

    return decision, expected_cold_penalty_ms, prewarm_cost_proxy


def execute_prewarm() -> tuple[float, str]:
    """Send a real HTTP request to warm the deployed OpenFaaS function."""

    import time

    start = time.perf_counter()

    response = requests.post(
        FUNCTION_URL,
        data="predictive-prewarm",
        headers={"Content-Type": "text/plain"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return elapsed_ms, response.text.strip()


def save_log(
    prediction: pd.Series,
    probability: float,
    decision: str,
    expected_cold_penalty_ms: float,
    prewarm_cost_proxy: float,
    prewarm_latency_ms: float | None,
    response_text: str,
) -> None:
    """Append the controller decision and real execution result to CSV."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app": prediction["app"],
        "func": prediction["func"],
        "bin_start_sec": prediction["bin_start_sec"],
        "predicted_probability": probability,
        "prewarm_threshold": PREWARM_THRESHOLD,
        "decision": decision,
        "expected_cold_penalty_ms": expected_cold_penalty_ms,
        "prewarm_cost_proxy": prewarm_cost_proxy,
        "prewarm_latency_ms": prewarm_latency_ms,
        "response": response_text,
    }

    log_df = pd.DataFrame([row])

    # Append to the existing log instead of overwriting previous experiments.
    if LOG_FILE.exists():
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_FILE, index=False)


def main() -> None:
    """Run one real predictive pre-warming control cycle."""

    prediction = load_prediction()

    # Convert the model's probability output to a normal Python float.
    probability = float(prediction["predicted_probability"])

    decision, expected_cold_penalty_ms, prewarm_cost_proxy = make_decision(
        probability
    )

    print("=" * 60)
    print("PREDICTIVE COLD-START CONTROLLER")
    print("=" * 60)

    print(f"Real app:                 {prediction['app']}")
    print(f"Real function:            {prediction['func']}")
    print(f"Prediction probability:   {probability:.6f}")
    print(f"Threshold:                {PREWARM_THRESHOLD:.2f}")
    print(f"Expected cold penalty:    {expected_cold_penalty_ms:.2f} ms")
    print(f"Pre-warm cost proxy:      {prewarm_cost_proxy:.2f}")
    print(f"Decision:                 {decision}")

    prewarm_latency_ms = None
    response_text = ""

    # Execute a real pre-warm request when the controller decides to do so.
    if decision == "PRE-WARM":
        print("\nExecuting REAL OpenFaaS pre-warm request...")

        try:
            prewarm_latency_ms, response_text = execute_prewarm()

            print(f"Pre-warm latency:         {prewarm_latency_ms:.2f} ms")
            print(f"Function response:        {response_text}")

        except requests.RequestException as exc:
            response_text = f"REQUEST_ERROR: {exc}"
            print(f"Pre-warm failed:           {exc}")

    else:
        print("\nNo pre-warm request sent.")

    # Persist the real controller result for later evaluation.
    save_log(
        prediction=prediction,
        probability=probability,
        decision=decision,
        expected_cold_penalty_ms=expected_cold_penalty_ms,
        prewarm_cost_proxy=prewarm_cost_proxy,
        prewarm_latency_ms=prewarm_latency_ms,
        response_text=response_text,
    )

    print(f"\nController log:           {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()