from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

ROOT = Path(r"D:\MissionShield")
DATA = ROOT / "data" / "processed" / "missionshield_e0_dataset.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
HORIZONS = [10, 30, 60]

FEATURES = [
    "latency",
    "packet_loss",
    "utilization",
    "availability",
    "sensor_quality",
    "detector_score",
]


def aggregate_time_step(df):
    """
    Convert node-level rows into one system-level vector per time step.
    Baseline deliberately ignores graph structure.
    """

    agg = (
        df.groupby(
            ["run_id", "attack_family", "time"],
            as_index=False
        )[FEATURES]
        .agg(["mean", "max", "std"])
    )

    agg.columns = [
        "_".join(x).strip("_")
        if isinstance(x, tuple)
        else x
        for x in agg.columns
    ]

    # Restore grouping-column names produced by multi-index aggregation.
    rename = {}
    for c in agg.columns:
        if c.startswith("run_id"):
            rename[c] = "run_id"
        elif c.startswith("attack_family"):
            rename[c] = "attack_family"
        elif c.startswith("time"):
            rename[c] = "time"

    agg = agg.rename(columns=rename)

    return agg


def get_labels(df):
    cols = [
        "run_id",
        "time",
        "mission_effectiveness",
        "failure",
    ] + [
        f"failure_within_{h}"
        for h in HORIZONS
    ]

    return (
        df[cols]
        .drop_duplicates(["run_id", "time"])
        .copy()
    )


def split_by_run(df):
    """
    70/15/15 split by complete run.
    No time steps from one run can appear in multiple splits.
    """

    rng = np.random.default_rng(SEED)

    run_info = (
        df[["run_id", "attack_family"]]
        .drop_duplicates()
    )

    train_runs = []
    val_runs = []
    test_runs = []

    # Stratify at run level by attack family.
    for family, g in run_info.groupby("attack_family"):

        runs = g["run_id"].to_numpy().copy()
        rng.shuffle(runs)

        n = len(runs)

        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))

        train_runs.extend(runs[:n_train])
        val_runs.extend(
            runs[n_train:n_train + n_val]
        )
        test_runs.extend(
            runs[n_train + n_val:]
        )

    return (
        set(train_runs),
        set(val_runs),
        set(test_runs),
    )


def expected_calibration_error(y, p, bins=10):

    edges = np.linspace(0, 1, bins + 1)

    ece = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):

        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not np.any(mask):
            continue

        accuracy = y[mask].mean()
        confidence = p[mask].mean()

        ece += (
            mask.mean()
            * abs(accuracy - confidence)
        )

    return float(ece)


def main():

    print("=" * 72)
    print("MissionShield E1 | Baseline Sanity Check")
    print("=" * 72)

    df = pd.read_csv(DATA)

    print(f"Loaded node rows : {len(df):,}")
    print(f"Runs             : {df.run_id.nunique()}")

    Xdf = aggregate_time_step(df)
    Ydf = get_labels(df)

    data = Xdf.merge(
        Ydf,
        on=["run_id", "time"],
        how="inner",
    )

    train_runs, val_runs, test_runs = split_by_run(data)

    # Hard leakage checks
    assert train_runs.isdisjoint(val_runs)
    assert train_runs.isdisjoint(test_runs)
    assert val_runs.isdisjoint(test_runs)

    train = data[data.run_id.isin(train_runs)].copy()
    val = data[data.run_id.isin(val_runs)].copy()
    test = data[data.run_id.isin(test_runs)].copy()

    print("\nRun split")
    print(f"Train : {len(train_runs)}")
    print(f"Val   : {len(val_runs)}")
    print(f"Test  : {len(test_runs)}")

    model_features = [
        c for c in data.columns
        if any(
            c.startswith(f + "_")
            for f in FEATURES
        )
    ]

    # std can be NaN in pathological aggregation cases.
    train[model_features] = train[model_features].fillna(0)
    val[model_features] = val[model_features].fillna(0)
    test[model_features] = test[model_features].fillna(0)

    results = []

    for H in HORIZONS:

        target = f"failure_within_{H}"

        X_train = train[model_features].to_numpy()
        y_train = train[target].to_numpy().astype(int)

        X_test = test[model_features].to_numpy()
        y_test = test[target].to_numpy().astype(int)

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        )

        model.fit(X_train, y_train)

        p = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, p)
        auprc = average_precision_score(y_test, p)
        ece = expected_calibration_error(y_test, p)
        brier = brier_score_loss(y_test, p)

        prevalence = y_test.mean()

        results.append({
            "horizon": H,
            "positive_rate": prevalence,
            "auroc": auroc,
            "auprc": auprc,
            "ece": ece,
            "brier": brier,
        })

        print("\n" + "-" * 72)
        print(f"H = {H}")
        print("-" * 72)

        print(f"Positive rate : {prevalence:.4f}")
        print(f"AUROC         : {auroc:.4f}")
        print(f"AUPRC         : {auprc:.4f}")
        print(f"ECE           : {ece:.4f}")
        print(f"Brier         : {brier:.4f}")

    out = pd.DataFrame(results)

    csv_file = RESULTS / "e1_baseline_sanity.csv"
    out.to_csv(csv_file, index=False)

    split_file = RESULTS / "e1_run_split.json"

    with open(split_file, "w") as f:
        json.dump({
            "train": sorted(train_runs),
            "validation": sorted(val_runs),
            "test": sorted(test_runs),
        }, f, indent=2)

    print("\n" + "=" * 72)
    print(out.to_string(index=False))
    print("=" * 72)

    print(f"\nSaved: {csv_file}")
    print(f"Saved: {split_file}")

    print("\nE1 BASELINE SANITY COMPLETE")


if __name__ == "__main__":
    main()