from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(r"D:\MissionShield")

DATA = (
    ROOT / "data" / "processed" /
    "missionshield_e0_v2_dataset.csv"
)

OUT = (
    ROOT / "results" /
    "e1_v2_feature_diagnostic.csv"
)

SEED = 42
HORIZONS = [10, 30, 60]

ALL_FEATURES = [
    "latency",
    "packet_loss",
    "utilization",
    "availability",
    "sensor_quality",
    "detector_score",
]

FEATURE_SETS = {
    "all": ALL_FEATURES,

    "no_detector": [
        "latency",
        "packet_loss",
        "utilization",
        "availability",
        "sensor_quality",
    ],

    "network_only": [
        "latency",
        "packet_loss",
        "utilization",
    ],

    "functional_only": [
        "availability",
        "sensor_quality",
    ],

    "detector_only": [
        "detector_score",
    ],

    "latency_only": [
        "latency",
    ],

    "availability_only": [
        "availability",
    ],

    "sensor_quality_only": [
        "sensor_quality",
    ],
}


def aggregate(df):

    agg = (
        df.groupby(
            ["run_id", "attack_family", "time"],
            as_index=False
        )[ALL_FEATURES]
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


def labels(df):

    cols = [
        "run_id",
        "time",
    ]

    cols += [
        f"failure_within_{h}"
        for h in HORIZONS
    ]

    return (
        df[cols]
        .drop_duplicates(["run_id", "time"])
    )


def make_split(data):

    rng = np.random.default_rng(SEED)

    info = (
        data[
            ["run_id", "attack_family"]
        ]
        .drop_duplicates()
    )

    train = []
    val = []
    test = []

    for family, g in info.groupby(
        "attack_family"
    ):

        runs = g.run_id.to_numpy().copy()

        rng.shuffle(runs)

        n = len(runs)

        # 30 runs/family:
        # 21 train, 4 validation, 5 test
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))

        train.extend(
            runs[:n_train]
        )

        val.extend(
            runs[n_train:n_train+n_val]
        )

        test.extend(
            runs[n_train+n_val:]
        )

    return set(train), set(val), set(test)


def columns_for(feature_set, columns):

    selected = []

    for feature in feature_set:

        selected.extend([
            c for c in columns
            if c.startswith(feature + "_")
        ])

    return selected


def main():

    print("=" * 76)
    print("MissionShield E1-v2 | Feature Shortcut Diagnostic")
    print("=" * 76)

    df = pd.read_csv(DATA)

    X = aggregate(df)
    Y = labels(df)

    data = X.merge(
        Y,
        on=["run_id", "time"],
        how="inner",
    )

    train_runs, val_runs, test_runs = make_split(
        data
    )

    assert train_runs.isdisjoint(test_runs)
    assert train_runs.isdisjoint(val_runs)
    assert val_runs.isdisjoint(test_runs)

    train = data[
        data.run_id.isin(train_runs)
    ].copy()

    test = data[
        data.run_id.isin(test_runs)
    ].copy()

    print(f"Train runs : {len(train_runs)}")
    print(f"Test runs  : {len(test_runs)}")

    results = []

    for name, feature_set in FEATURE_SETS.items():

        cols = columns_for(
            feature_set,
            data.columns,
        )

        train[cols] = (
            train[cols]
            .fillna(0)
        )

        test[cols] = (
            test[cols]
            .fillna(0)
        )

        print("\n" + "-" * 76)
        print(f"FEATURE SET: {name}")
        print("-" * 76)

        for H in HORIZONS:

            target = f"failure_within_{H}"

            X_train = (
                train[cols]
                .to_numpy()
            )

            y_train = (
                train[target]
                .to_numpy(dtype=int)
            )

            X_test = (
                test[cols]
                .to_numpy()
            )

            y_test = (
                test[target]
                .to_numpy(dtype=int)
            )

            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=SEED,
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

            prevalence = y_test.mean()

            results.append({
                "feature_set": name,
                "horizon": H,
                "n_features": len(cols),
                "positive_rate": prevalence,
                "auroc": auroc,
                "auprc": auprc,
            })

            print(
                f"H={H:>2} | "
                f"AUROC={auroc:.4f} | "
                f"AUPRC={auprc:.4f}"
            )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUT,
        index=False,
    )

    print("\n" + "=" * 76)
    print("H=10 ranking")
    print("=" * 76)

    print(
        result_df[
            result_df.horizon == 10
        ]
        .sort_values(
            "auroc",
            ascending=False,
        )[
            [
                "feature_set",
                "auroc",
                "auprc",
            ]
        ]
        .to_string(index=False)
    )

    print(f"\nSaved: {OUT}")
    print("\nFEATURE DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()