from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ============================================================
# MissionShield E1-B
# LSTM Temporal Baseline
# Strict Pre-Failure Forecasting
# ============================================================

ROOT = Path(r"D:\MissionShield")

DATA = (
    ROOT / "data" / "processed" /
    "missionshield_e0_v2_dataset.csv"
)

SPLIT_FILE = (
    ROOT / "results" /
    "e1_v2_prefailure_run_split.json"
)

RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

RESULTS.mkdir(exist_ok=True)
CHECKPOINTS.mkdir(exist_ok=True)

SEED = 42

HORIZONS = [10, 30, 60]

HISTORY = 10

FEATURES = [
    "latency",
    "packet_loss",
    "utilization",
    "availability",
    "sensor_quality",
    "detector_score",
]

NODES = [
    "sensor",
    "tactical_link",
    "edge_processing",
    "fusion",
    "command",
    "actuator",
]

BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 5

HIDDEN = 128
LAYERS = 2
DROPOUT = 0.20

LR = 1e-3


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ECE
# ============================================================

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

        acc = y[mask].mean()
        conf = p[mask].mean()

        ece += (
            mask.mean() *
            abs(acc - conf)
        )

    return float(ece)


# ============================================================
# Convert node telemetry into fixed system vector
# ============================================================

def make_timeline(df):

    print("Building ordered node-level feature vectors...")

    # Ensure deterministic node ordering
    node_order = {
        node: i
        for i, node in enumerate(NODES)
    }

    df = df.copy()

    df["node_order"] = (
        df["node_id"]
        .map(node_order)
    )

    df = df.sort_values(
        [
            "run_id",
            "time",
            "node_order",
        ]
    )

    records = []

    for (run_id, time), g in df.groupby(
        ["run_id", "time"],
        sort=False,
    ):

        g = g.sort_values(
            "node_order"
        )

        if len(g) != len(NODES):
            continue

        vector = []

        for node in NODES:

            row = g[
                g.node_id == node
            ]

            if len(row) != 1:
                raise RuntimeError(
                    f"Node problem: "
                    f"{run_id}, t={time}, "
                    f"node={node}"
                )

            row = row.iloc[0]

            for feature in FEATURES:
                vector.append(
                    float(row[feature])
                )

        first = g.iloc[0]

        record = {
            "run_id": run_id,
            "attack_family":
                first["attack_family"],
            "time": int(time),
            "failure":
                int(first["failure"]),
        }

        for H in HORIZONS:
            record[
                f"failure_within_{H}"
            ] = int(
                first[
                    f"failure_within_{H}"
                ]
            )

        for i, value in enumerate(vector):
            record[f"x_{i}"] = value

        records.append(record)

    timeline = pd.DataFrame(records)

    return timeline


# ============================================================
# Normalization
# Fit ONLY on training runs
# ============================================================

def normalize(
    timeline,
    train_runs,
):

    feature_cols = [
        c for c in timeline.columns
        if c.startswith("x_")
    ]

    train_mask = (
        timeline.run_id
        .isin(train_runs)
    )

    mean = (
        timeline.loc[
            train_mask,
            feature_cols
        ]
        .mean()
    )

    std = (
        timeline.loc[
            train_mask,
            feature_cols
        ]
        .std()
        .replace(0, 1)
    )

    timeline = timeline.copy()

    timeline[feature_cols] = (
        timeline[feature_cols] - mean
    ) / std

    return (
        timeline,
        feature_cols,
        mean,
        std,
    )


# ============================================================
# Sequence creation
#
# IMPORTANT:
# Current observation must be pre-failure.
# History may contain only operational observations as well.
# ============================================================

def build_sequences(
    timeline,
    runs,
    feature_cols,
    horizon,
):

    X = []
    y = []
    metadata = []

    subset = timeline[
        timeline.run_id.isin(runs)
    ].copy()

    for run_id, g in subset.groupby(
        "run_id"
    ):

        g = g.sort_values(
            "time"
        ).reset_index(drop=True)

        values = (
            g[feature_cols]
            .to_numpy(
                dtype=np.float32
            )
        )

        current_failure = (
            g["failure"]
            .to_numpy(dtype=int)
        )

        labels = (
            g[
                f"failure_within_{horizon}"
            ]
            .to_numpy(dtype=int)
        )

        times = (
            g["time"]
            .to_numpy(dtype=int)
        )

        family = (
            g["attack_family"]
            .iloc[0]
        )

        for i in range(
            HISTORY - 1,
            len(g)
        ):

            start = (
                i - HISTORY + 1
            )

            # Strict protocol:
            # every point in the input history
            # must precede mission failure.
            if np.any(
                current_failure[
                    start:i + 1
                ] != 0
            ):
                continue

            X.append(
                values[
                    start:i + 1
                ]
            )

            y.append(
                labels[i]
            )

            metadata.append({
                "run_id": run_id,
                "time": int(times[i]),
                "attack_family": family,
            })

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    return X, y, metadata


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx],
        )


# ============================================================
# LSTM
# ============================================================

class LSTMForecaster(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden=128,
        layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(
                dropout
                if layers > 1
                else 0.0
            ),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        h = out[:, -1, :]

        return (
            self.head(h)
            .squeeze(1)
        )


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device,
):

    model.eval()

    probs = []
    labels = []

    for X, y in loader:

        X = X.to(device)

        logits = model(X)

        p = torch.sigmoid(
            logits
        )

        probs.append(
            p.cpu().numpy()
        )

        labels.append(
            y.numpy()
        )

    return (
        np.concatenate(labels),
        np.concatenate(probs),
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(SEED)

    print("=" * 76)
    print(
        "MissionShield E1-B | "
        "LSTM Strict Pre-Failure Forecasting"
    )
    print("=" * 76)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device : {device}")

    if torch.cuda.is_available():
        print(
            "GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    df = pd.read_csv(DATA)

    print(
        f"Node rows : {len(df):,}"
    )

    print(
        f"Runs      : "
        f"{df.run_id.nunique()}"
    )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_runs = set(
        split["train"]
    )

    val_runs = set(
        split["validation"]
    )

    test_runs = set(
        split["test"]
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

    print("\nFrozen run split")
    print(
        f"Train : {len(train_runs)}"
    )
    print(
        f"Val   : {len(val_runs)}"
    )
    print(
        f"Test  : {len(test_runs)}"
    )

    timeline = make_timeline(
        df
    )

    print(
        f"\nTimeline samples : "
        f"{len(timeline):,}"
    )

    timeline, feature_cols, mean, std = (
        normalize(
            timeline,
            train_runs,
        )
    )

    print(
        f"Input features/time-step : "
        f"{len(feature_cols)}"
    )

    print(
        f"History length           : "
        f"{HISTORY}"
    )

    results = []

    for H in HORIZONS:

        print("\n" + "=" * 76)
        print(
            f"Forecast horizon H={H}"
        )
        print("=" * 76)

        X_train, y_train, _ = (
            build_sequences(
                timeline,
                train_runs,
                feature_cols,
                H,
            )
        )

        X_val, y_val, _ = (
            build_sequences(
                timeline,
                val_runs,
                feature_cols,
                H,
            )
        )

        X_test, y_test, meta_test = (
            build_sequences(
                timeline,
                test_runs,
                feature_cols,
                H,
            )
        )

        print(
            f"Train sequences : "
            f"{len(y_train):,}"
        )

        print(
            f"Val sequences   : "
            f"{len(y_val):,}"
        )

        print(
            f"Test sequences  : "
            f"{len(y_test):,}"
        )

        print(
            f"Train positive  : "
            f"{y_train.mean():.4f}"
        )

        print(
            f"Val positive    : "
            f"{y_val.mean():.4f}"
        )

        print(
            f"Test positive   : "
            f"{y_test.mean():.4f}"
        )

        train_ds = SequenceDataset(
            X_train,
            y_train,
        )

        val_ds = SequenceDataset(
            X_val,
            y_val,
        )

        test_ds = SequenceDataset(
            X_test,
            y_test,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = LSTMForecaster(
            input_dim=len(
                feature_cols
            ),
            hidden=HIDDEN,
            layers=LAYERS,
            dropout=DROPOUT,
        ).to(device)

        # Handle class imbalance using
        # training data only.
        positives = float(
            y_train.sum()
        )

        negatives = float(
            len(y_train)
            - positives
        )

        pos_weight = (
            negatives /
            max(positives, 1.0)
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    pos_weight,
                    device=device,
                )
            )
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=1e-4,
        )

        best_val = -np.inf
        best_epoch = 0
        bad_epochs = 0

        checkpoint = (
            CHECKPOINTS /
            f"e1_lstm_H{H}.pt"
        )

        for epoch in range(
            1,
            EPOCHS + 1
        ):

            model.train()

            running_loss = 0.0

            for Xb, yb in train_loader:

                Xb = Xb.to(
                    device,
                    non_blocking=True,
                )

                yb = yb.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(Xb)

                loss = criterion(
                    logits,
                    yb,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

                running_loss += (
                    loss.item()
                    * len(yb)
                )

            train_loss = (
                running_loss /
                len(train_ds)
            )

            val_y, val_p = predict(
                model,
                val_loader,
                device,
            )

            if len(
                np.unique(val_y)
            ) < 2:

                raise RuntimeError(
                    "Validation set has "
                    "only one target class."
                )

            val_auprc = (
                average_precision_score(
                    val_y,
                    val_p,
                )
            )

            print(
                f"Epoch "
                f"{epoch:02d} | "
                f"loss={train_loss:.4f} | "
                f"val AUPRC="
                f"{val_auprc:.4f}"
            )

            if (
                val_auprc >
                best_val + 1e-4
            ):

                best_val = val_auprc
                best_epoch = epoch
                bad_epochs = 0

                torch.save(
                    model.state_dict(),
                    checkpoint,
                )

            else:

                bad_epochs += 1

                if (
                    bad_epochs >= PATIENCE
                ):
                    print(
                        "Early stopping."
                    )
                    break

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
        )

        test_y, test_p = predict(
            model,
            test_loader,
            device,
        )

        auroc = roc_auc_score(
            test_y,
            test_p,
        )

        auprc = (
            average_precision_score(
                test_y,
                test_p,
            )
        )

        ece = (
            expected_calibration_error(
                test_y,
                test_p,
            )
        )

        brier = (
            brier_score_loss(
                test_y,
                test_p,
            )
        )

        print("\nTEST RESULT")
        print(
            f"Best epoch    : "
            f"{best_epoch}"
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

        results.append({
            "model": "LSTM",
            "history": HISTORY,
            "horizon": H,
            "best_epoch":
                best_epoch,
            "positive_rate":
                float(test_y.mean()),
            "auroc": auroc,
            "auprc": auprc,
            "ece": ece,
            "brier": brier,
        })

        # Save predictions for later
        # lead-time analysis.
        pred_df = pd.DataFrame(
            meta_test
        )

        pred_df["label"] = (
            test_y.astype(int)
        )

        pred_df["probability"] = (
            test_p
        )

        pred_file = (
            RESULTS /
            f"e1_lstm_H{H}_predictions.csv"
        )

        pred_df.to_csv(
            pred_file,
            index=False,
        )

    # ========================================================
    # Final table
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    output = (
        RESULTS /
        "e1_lstm_prefailure_results.csv"
    )

    result_df.to_csv(
        output,
        index=False,
    )

    print("\n" + "=" * 76)
    print("FINAL LSTM RESULTS")
    print("=" * 76)

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output}"
    )

    print(
        "\nE1-B LSTM COMPLETE"
    )


if __name__ == "__main__":
    main()