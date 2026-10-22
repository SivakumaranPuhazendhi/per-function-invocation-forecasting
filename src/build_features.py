"""
Build real per-function features from the Microsoft Azure Functions Invocation Trace 2021.

No synthetic/demo data is generated here.
The script:
1. Streams the 291 MB trace in chunks.
2. Finds the most active (app, func) pairs.
3. Replays only those real invocation records.
4. Builds 30-second invocation bins.
5. Creates LAG-1/2/3, EMA, recent count, time-since-last-invocation.
6. Creates a next-30-second binary target.
7. Saves a compact feature dataset for model training.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
OUT_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

CHUNK_SIZE = 200_000
TOP_N_FUNCTIONS = 10
BIN_SECONDS = 30
EMA_ALPHA = 0.30

REQUIRED_COLUMNS = ["app", "func", "end_timestamp", "duration"]


def profile_top_functions() -> Counter[tuple[str, str]]:
    """Pass 1: count real invocations per (app, func) without loading the whole file."""
    counts: Counter[tuple[str, str]] = Counter()

    with INPUT.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(
                f"Unexpected columns: {reader.fieldnames}. "
                f"Expected: {REQUIRED_COLUMNS}"
            )

        for row in reader:
            counts[(row["app"], row["func"])] += 1

    return counts


def collect_top_function_events(
    selected: set[tuple[str, str]],
) -> dict[tuple[str, str], list[float]]:
    """Pass 2: collect real arrival-time timestamps for selected functions."""
    events: dict[tuple[str, str], list[float]] = defaultdict(list)

    for chunk in pd.read_csv(
        INPUT,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        # The trace gives end_timestamp and duration in seconds.
        chunk["end_timestamp"] = pd.to_numeric(
            chunk["end_timestamp"], errors="coerce"
        )
        chunk["duration"] = pd.to_numeric(chunk["duration"], errors="coerce")
        chunk = chunk.dropna(subset=["app", "func", "end_timestamp", "duration"])

        # Approximate request/arrival time from completion time - execution duration.
        chunk["arrival_time"] = (
            chunk["end_timestamp"] - chunk["duration"]
        )

        for (app, func), group in chunk.groupby(["app", "func"], sort=False):
            key = (str(app), str(func))
            if key in selected:
                events[key].extend(group["arrival_time"].astype(float).tolist())

    return events


def build_function_features(
    app: str,
    func: str,
    timestamps: list[float],
) -> pd.DataFrame:
    """Convert event timestamps into 30-second bins and temporal features."""
    if not timestamps:
        return pd.DataFrame()

    ts = np.asarray(timestamps, dtype=np.float64)
    bins = np.floor(ts / BIN_SECONDS).astype(np.int64)

    counts = (
        pd.Series(bins)
        .value_counts()
        .sort_index()
        .rename("invocation_count")
    )

    # Fill the time between real events with zero-invocation bins.
    full_index = pd.RangeIndex(counts.index.min(), counts.index.max() + 1)
    counts = counts.reindex(full_index, fill_value=0).astype(float)

    frame = pd.DataFrame({"bin_id": counts.index.astype(np.int64)})
    frame["app"] = app
    frame["func"] = func
    frame["invocation_count"] = counts.to_numpy()

    # Historical temporal features; all are based only on current/past bins.
    frame["lag_1"] = frame["invocation_count"].shift(1)
    frame["lag_2"] = frame["invocation_count"].shift(2)
    frame["lag_3"] = frame["invocation_count"].shift(3)

    frame["ema"] = (
        frame["invocation_count"]
        .ewm(alpha=EMA_ALPHA, adjust=False)
        .mean()
    )

    frame["recent_2min_count"] = (
        frame["invocation_count"]
        .rolling(window=4, min_periods=1)
        .sum()
    )

    # Number of 30-second bins since the last observed invocation.
    last_seen = frame["bin_id"].where(
        frame["invocation_count"] > 0
    ).ffill()
    frame["time_since_last_invocation_bins"] = (
        frame["bin_id"] - last_seen
    ).fillna(frame["bin_id"] - frame["bin_id"].min())

    # Target = whether the function receives >=1 invocation in the NEXT 30s bin.
    frame["target_next_30s"] = (
        frame["invocation_count"].shift(-1) > 0
    ).astype(int)

    frame["bin_start_sec"] = frame["bin_id"] * BIN_SECONDS

    # Remove rows where the lag history is not yet available and the final
    # row whose future target is not observable.
    frame = frame.dropna(
        subset=["lag_1", "lag_2", "lag_3"]
    ).iloc[:-1]

    return frame.reset_index(drop=True)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT}\n"
            "Place the original AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt "
            "inside data/raw/."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("PASS 1/2: counting real (app, func) invocations...")
    counts = profile_top_functions()
    top = counts.most_common(TOP_N_FUNCTIONS)

    top_df = pd.DataFrame(
        [
            {
                "rank": rank,
                "app": app,
                "func": func,
                "invocations": count,
            }
            for rank, ((app, func), count) in enumerate(top, start=1)
        ]
    )
    top_df.to_csv(RESULTS_DIR / "top_functions.csv", index=False)

    print("\nTop functions:")
    print(top_df[["rank", "invocations"]].to_string(index=False))

    selected = {key for key, _ in top}

    print("\nPASS 2/2: collecting real timestamps for top functions...")
    events = collect_top_function_events(selected)

    frames = []
    for rank, ((app, func), count) in enumerate(top, start=1):
        print(f"Building features {rank}/{len(top)} | invocations={count:,}")
        built = build_function_features(app, func, events[(app, func)])
        frames.append(built)

    features = pd.concat(frames, ignore_index=True)

    output = OUT_DIR / "features_top10.csv"
    features.to_csv(output, index=False)

    print("\nDONE")
    print(f"Feature rows: {len(features):,}")
    print(f"Functions modeled: {features[['app', 'func']].drop_duplicates().shape[0]}")
    print(f"Output: {output}")
    print(f"Target positive rate: {features['target_next_30s'].mean():.4f}")


if __name__ == "__main__":
    main()
