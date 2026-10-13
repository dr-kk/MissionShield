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

DATA = (
    ROOT / "data" / "processed" /
    "missionshield_e0_v2_dataset.csv"
)

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

    rename = {}

    for c in agg.columns:
        if c.startswith("run_id"):
            rename[c] = "run_id"
        elif c.startswith("attack_family"):
            rename[c] = "attack_family"
        elif c.startswith("time"):
            rename[c] = "time"

    return agg.rename(columns=rename)


def get_labels(df):

    cols = [
        "run_id",
        "time",
        "mission_effectiveness",
        "failure",
    ]

    cols += [
        f"failure_within_{h}"
        for h in HORIZONS
    ]

    return (
        df[cols]
        .drop_duplicates(["run_id", "time"])
        .copy()
    )


def split_by_run(df):

    rng = np.random.default_rng(SEED)

    run_info = (
        df[["run_id", "attack_family"]]
        .drop_duplicates()
    )

    train_runs = []
    val_runs = []
    test_runs = []

    for family, g in run_info.groupby(
        "attack_family"
    ):

        runs = g["run_id"].to_numpy().copy()

        rng.shuffle(runs)

        n = len(runs)

        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))

        train_runs.extend(
            runs[:n_train]
        )

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

    for lo, hi in zip(
        edges[:-1],
        edges[1:]
    ):

        if hi == 1.0:
            mask = (
                (p >= lo) &
                (p <= hi)
            )
        else:
            mask = (
                (p >= lo) &
                (p < hi)
            )

        if not np.any(mask):
            continue

        accuracy = y[mask].mean()
        confidence = p[mask].mean()

        ece += (
            mask.mean() *
            abs(accuracy - confidence)
        )

    return float(ece)


def main():

    print("=" * 72)
    print("MissionShield E1-v2 | Strict Pre-Failure Forecasting")
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

    # ========================================================
    # STRICT EARLY-WARNING PROTOCOL
    #
    # Only evaluate observations at which the mission is
    # currently operational.
    # ========================================================

    before_filter = len(data)

    data = data[
        data["failure"] == 0
    ].copy()

    after_filter = len(data)

    print("\nStrict pre-failure forecasting protocol")
    print(
        f"Samples before filter : "
        f"{before_filter:,}"
    )
    print(
        f"Samples retained      : "
        f"{after_filter:,}"
    )
    print(
        f"Already-failed removed: "
        f"{before_filter - after_filter:,}"
    )

    assert (
        data["failure"] == 0
    ).all()

    # ========================================================
    # Run-level split AFTER protocol filtering
    # ========================================================

    train_runs, val_runs, test_runs = (
        split_by_run(data)
    )

    assert train_runs.isdisjoint(
        val_runs
    )

    assert train_runs.isdisjoint(
        test_runs
    )

    assert val_runs.isdisjoint(
        test_runs
    )

    train = data[
        data.run_id.isin(train_runs)
    ].copy()

    val = data[
        data.run_id.isin(val_runs)
    ].copy()

    test = data[
        data.run_id.isin(test_runs)
    ].copy()

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

    train[model_features] = (
        train[model_features]
        .fillna(0)
    )

    val[model_features] = (
        val[model_features]
        .fillna(0)
    )

    test[model_features] = (
        test[model_features]
        .fillna(0)
    )

    results = []

    for H in HORIZONS:

        target = (
            f"failure_within_{H}"
        )

        X_train = (
            train[model_features]
            .to_numpy()
        )

        y_train = (
            train[target]
            .to_numpy(dtype=int)
        )

        X_test = (
            test[model_features]
            .to_numpy()
        )

        y_test = (
            test[target]
            .to_numpy(dtype=int)
        )

        print("\n" + "-" * 72)
        print(f"H = {H}")
        print("-" * 72)

        print(
            f"Train positives : "
            f"{y_train.sum():,}/"
            f"{len(y_train):,} "
            f"({y_train.mean():.4f})"
        )

        print(
            f"Test positives  : "
            f"{y_test.sum():,}/"
            f"{len(y_test):,} "
            f"({y_test.mean():.4f})"
        )

        if len(
            np.unique(y_train)
        ) < 2:

            print(
                "[ERROR] Training target "
                "contains only one class."
            )

            continue

        if len(
            np.unique(y_test)
        ) < 2:

            print(
                "[ERROR] Test target "
                "contains only one class."
            )

            continue

        model = (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=SEED,
            )
        )

        model.fit(
            X_train,
            y_train,
        )

        p = model.predict_proba(
            X_test
        )[:, 1]

        auroc = roc_auc_score(
            y_test,
            p,
        )

        auprc = average_precision_score(
            y_test,
            p,
        )

        ece = expected_calibration_error(
            y_test,
            p,
        )

        brier = brier_score_loss(
            y_test,
            p,
        )

        prevalence = float(
            y_test.mean()
        )

        results.append({
            "horizon": H,
            "positive_rate": prevalence,
            "auroc": auroc,
            "auprc": auprc,
            "ece": ece,
            "brier": brier,
        })

        print(
            f"Positive rate : "
            f"{prevalence:.4f}"
        )

        print(
            f"AUROC         : "
            f"{auroc:.4f}"
        )

        print(
            f"AUPRC         : "
            f"{auprc:.4f}"
        )

        print(
            f"ECE           : "
            f"{ece:.4f}"
        )

        print(
            f"Brier         : "
            f"{brier:.4f}"
        )

    out = pd.DataFrame(
        results
    )

    csv_file = (
        RESULTS /
        "e1_v2_prefailure_baseline.csv"
    )

    out.to_csv(
        csv_file,
        index=False,
    )

    split_file = (
        RESULTS /
        "e1_v2_prefailure_run_split.json"
    )

    with open(
        split_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump({
            "train": sorted(train_runs),
            "validation": sorted(val_runs),
            "test": sorted(test_runs),
        }, f, indent=2)

    print("\n" + "=" * 72)

    if len(out) > 0:
        print(
            out.to_string(
                index=False
            )
        )

    print("=" * 72)

    print(
        f"\nSaved: "
        f"{csv_file}"
    )

    print(
        f"Saved: "
        f"{split_file}"
    )

    print(
        "\nE1-v2 STRICT "
        "PRE-FAILURE BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()