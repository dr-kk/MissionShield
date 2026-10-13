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
# MissionShield E1-C
# Transformer Temporal Baseline
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

RESULTS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

SEED = 42

HORIZONS = [10, 30, 60]

# Same temporal history as LSTM
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

# Same training protocol as LSTM
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 5
LR = 1e-3

# Transformer configuration
D_MODEL = 128
NHEAD = 4
LAYERS = 2
DROPOUT = 0.20
FF_DIM = 256


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
# Expected Calibration Error
# ============================================================

def expected_calibration_error(y, p, bins=10):

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for lo, hi in zip(
        edges[:-1],
        edges[1:],
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
            mask.mean()
            * abs(
                accuracy -
                confidence
            )
        )

    return float(ece)


# ============================================================
# Convert node telemetry into ordered system vector
#
# Each time step:
# 6 nodes x 6 telemetry features = 36 dimensions
#
# Node ordering is fixed and identical to LSTM experiment.
# ============================================================

def make_timeline(df):

    print(
        "Building ordered node-level "
        "feature vectors..."
    )

    node_order = {
        node: i
        for i, node
        in enumerate(NODES)
    }

    df = df.copy()

    df["node_order"] = (
        df["node_id"]
        .map(node_order)
    )

    if df["node_order"].isna().any():

        unknown = (
            df.loc[
                df["node_order"].isna(),
                "node_id"
            ]
            .unique()
        )

        raise RuntimeError(
            f"Unknown node IDs: {unknown}"
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

            raise RuntimeError(
                f"{run_id}, t={time}: "
                f"expected {len(NODES)} nodes, "
                f"found {len(g)}."
            )

        vector = []

        for node in NODES:

            row = g[
                g["node_id"] == node
            ]

            if len(row) != 1:

                raise RuntimeError(
                    f"Node problem: "
                    f"{run_id}, "
                    f"t={time}, "
                    f"node={node}"
                )

            row = row.iloc[0]

            for feature in FEATURES:

                vector.append(
                    float(
                        row[feature]
                    )
                )

        first = g.iloc[0]

        record = {
            "run_id":
                run_id,

            "attack_family":
                first["attack_family"],

            "time":
                int(time),

            "failure":
                int(
                    first["failure"]
                ),
        }

        for H in HORIZONS:

            record[
                f"failure_within_{H}"
            ] = int(
                first[
                    f"failure_within_{H}"
                ]
            )

        for i, value in enumerate(
            vector
        ):

            record[
                f"x_{i}"
            ] = value

        records.append(
            record
        )

    timeline = pd.DataFrame(
        records
    )

    return timeline


# ============================================================
# Normalization
#
# IMPORTANT:
# Mean/std are fitted ONLY on frozen training runs.
# Validation/test information is never used.
# ============================================================

def normalize(
    timeline,
    train_runs,
):

    feature_cols = [
        c
        for c in timeline.columns
        if c.startswith("x_")
    ]

    train_mask = (
        timeline["run_id"]
        .isin(train_runs)
    )

    mean = (
        timeline.loc[
            train_mask,
            feature_cols,
        ]
        .mean()
    )

    std = (
        timeline.loc[
            train_mask,
            feature_cols,
        ]
        .std()
        .replace(0, 1)
    )

    timeline = (
        timeline.copy()
    )

    timeline[
        feature_cols
    ] = (
        timeline[
            feature_cols
        ] - mean
    ) / std

    return (
        timeline,
        feature_cols,
        mean,
        std,
    )


# ============================================================
# Sequence construction
#
# Strict early-warning protocol:
#
# 1. Sequence length = HISTORY
# 2. Every point in the input sequence must be pre-failure.
# 3. Current mission must therefore still be operational.
# 4. Target asks whether failure occurs within future H.
#
# Mission effectiveness itself is NOT an input feature.
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

    subset = (
        timeline[
            timeline["run_id"]
            .isin(runs)
        ]
        .copy()
    )

    for run_id, g in subset.groupby(
        "run_id"
    ):

        g = (
            g.sort_values(
                "time"
            )
            .reset_index(
                drop=True
            )
        )

        values = (
            g[
                feature_cols
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        current_failure = (
            g["failure"]
            .to_numpy(
                dtype=int
            )
        )

        labels = (
            g[
                f"failure_within_{horizon}"
            ]
            .to_numpy(
                dtype=int
            )
        )

        times = (
            g["time"]
            .to_numpy(
                dtype=int
            )
        )

        family = (
            g[
                "attack_family"
            ]
            .iloc[0]
        )

        for i in range(
            HISTORY - 1,
            len(g),
        ):

            start = (
                i -
                HISTORY +
                1
            )

            # Every observation in the
            # history must be pre-failure.
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
                "run_id":
                    run_id,

                "time":
                    int(
                        times[i]
                    ),

                "attack_family":
                    family,
            })

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    return (
        X,
        y,
        metadata,
    )


# ============================================================
# PyTorch Dataset
# ============================================================

class SequenceDataset(Dataset):

    def __init__(
        self,
        X,
        y,
    ):

        self.X = (
            torch.from_numpy(X)
        )

        self.y = (
            torch.from_numpy(y)
        )

    def __len__(self):

        return len(
            self.y
        )

    def __getitem__(
        self,
        idx,
    ):

        return (
            self.X[idx],
            self.y[idx],
        )


# ============================================================
# Transformer Temporal Forecaster
# ============================================================

class TransformerForecaster(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        layers=2,
        dropout=0.20,
        ff_dim=256,
        max_len=64,
    ):

        super().__init__()

        self.input_proj = (
            nn.Linear(
                input_dim,
                d_model,
            )
        )

        # Learnable temporal positional embedding
        self.pos_embedding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    max_len,
                    d_model,
                )
            )
        )

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
            )
        )

        self.norm = (
            nn.LayerNorm(
                d_model
            )
        )

        self.head = (
            nn.Sequential(
                nn.Linear(
                    d_model,
                    64,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    64,
                    1,
                ),
            )
        )

    def forward(
        self,
        x,
    ):

        # x:
        # [batch, history, 36]

        z = self.input_proj(
            x
        )

        seq_len = (
            z.size(1)
        )

        if seq_len > (
            self.pos_embedding
            .size(1)
        ):

            raise RuntimeError(
                "Sequence length exceeds "
                "positional embedding size."
            )

        z = (
            z +
            self.pos_embedding[
                :,
                :seq_len,
                :
            ]
        )

        z = self.encoder(
            z
        )

        # Use representation at current
        # (last) time step.
        z = self.norm(
            z[:, -1, :]
        )

        logits = (
            self.head(z)
            .squeeze(1)
        )

        return logits


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

        X = X.to(
            device,
            non_blocking=True,
        )

        logits = model(
            X
        )

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
        np.concatenate(
            labels
        ),
        np.concatenate(
            probs
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(
        SEED
    )

    print("=" * 76)
    print(
        "MissionShield E1-C | "
        "Transformer Strict Pre-Failure Forecasting"
    )
    print("=" * 76)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load frozen E0-v2 dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA
    )

    print(
        f"Node rows : "
        f"{len(df):,}"
    )

    print(
        f"Runs      : "
        f"{df.run_id.nunique()}"
    )

    # --------------------------------------------------------
    # Load EXACT SAME frozen split as strict GBDT/LSTM
    # --------------------------------------------------------

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        split = json.load(
            f
        )

    train_runs = set(
        split["train"]
    )

    val_runs = set(
        split["validation"]
    )

    test_runs = set(
        split["test"]
    )

    assert (
        train_runs.isdisjoint(
            val_runs
        )
    )

    assert (
        train_runs.isdisjoint(
            test_runs
        )
    )

    assert (
        val_runs.isdisjoint(
            test_runs
        )
    )

    print(
        "\nFrozen run split"
    )

    print(
        f"Train : "
        f"{len(train_runs)}"
    )

    print(
        f"Val   : "
        f"{len(val_runs)}"
    )

    print(
        f"Test  : "
        f"{len(test_runs)}"
    )

    # --------------------------------------------------------
    # Build ordered timeline
    # --------------------------------------------------------

    timeline = (
        make_timeline(
            df
        )
    )

    print(
        f"\nTimeline samples : "
        f"{len(timeline):,}"
    )

    # --------------------------------------------------------
    # Normalize using training runs only
    # --------------------------------------------------------

    (
        timeline,
        feature_cols,
        mean,
        std,
    ) = normalize(
        timeline,
        train_runs,
    )

    print(
        "Input features/time-step : "
        f"{len(feature_cols)}"
    )

    print(
        "History length           : "
        f"{HISTORY}"
    )

    print(
        "Transformer d_model      : "
        f"{D_MODEL}"
    )

    print(
        "Attention heads          : "
        f"{NHEAD}"
    )

    print(
        "Encoder layers           : "
        f"{LAYERS}"
    )

    results = []

    # ========================================================
    # Train independent model for each forecast horizon
    # ========================================================

    for H in HORIZONS:

        # Reset random state so each horizon is reproducible.
        seed_everything(
            SEED + H
        )

        print(
            "\n" +
            "=" * 76
        )

        print(
            f"Forecast horizon H={H}"
        )

        print(
            "=" * 76
        )

        (
            X_train,
            y_train,
            _,
        ) = build_sequences(
            timeline,
            train_runs,
            feature_cols,
            H,
        )

        (
            X_val,
            y_val,
            _,
        ) = build_sequences(
            timeline,
            val_runs,
            feature_cols,
            H,
        )

        (
            X_test,
            y_test,
            meta_test,
        ) = build_sequences(
            timeline,
            test_runs,
            feature_cols,
            H,
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

        # Safety checks
        if len(
            np.unique(
                y_train
            )
        ) < 2:

            raise RuntimeError(
                f"H={H}: training set "
                "contains only one class."
            )

        if len(
            np.unique(
                y_val
            )
        ) < 2:

            raise RuntimeError(
                f"H={H}: validation set "
                "contains only one class."
            )

        if len(
            np.unique(
                y_test
            )
        ) < 2:

            raise RuntimeError(
                f"H={H}: test set "
                "contains only one class."
            )

        # ----------------------------------------------------
        # Data loaders
        # ----------------------------------------------------

        train_ds = (
            SequenceDataset(
                X_train,
                y_train,
            )
        )

        val_ds = (
            SequenceDataset(
                X_val,
                y_val,
            )
        )

        test_ds = (
            SequenceDataset(
                X_test,
                y_test,
            )
        )

        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            SEED + H
        )

        train_loader = (
            DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=(
                    device.type
                    == "cuda"
                ),
                generator=generator,
            )
        )

        val_loader = (
            DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(
                    device.type
                    == "cuda"
                ),
            )
        )

        test_loader = (
            DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(
                    device.type
                    == "cuda"
                ),
            )
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        model = (
            TransformerForecaster(
                input_dim=len(
                    feature_cols
                ),
                d_model=D_MODEL,
                nhead=NHEAD,
                layers=LAYERS,
                dropout=DROPOUT,
                ff_dim=FF_DIM,
                max_len=max(
                    64,
                    HISTORY,
                ),
            )
            .to(device)
        )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            f"Parameters      : "
            f"{total_params:,}"
        )

        print(
            f"Trainable       : "
            f"{trainable_params:,}"
        )

        # ----------------------------------------------------
        # Class imbalance
        # ----------------------------------------------------

        positives = float(
            y_train.sum()
        )

        negatives = float(
            len(y_train) -
            positives
        )

        pos_weight = (
            negatives /
            max(
                positives,
                1.0,
            )
        )

        print(
            f"Positive weight : "
            f"{pos_weight:.3f}"
        )

        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=(
                    torch.tensor(
                        pos_weight,
                        dtype=torch.float32,
                        device=device,
                    )
                )
            )
        )

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=LR,
                weight_decay=1e-4,
            )
        )

        # ----------------------------------------------------
        # Training / validation
        # ----------------------------------------------------

        best_val = -np.inf
        best_epoch = 0
        bad_epochs = 0

        checkpoint = (
            CHECKPOINTS /
            f"e1_transformer_H{H}.pt"
        )

        for epoch in range(
            1,
            EPOCHS + 1,
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

                logits = model(
                    Xb
                )

                loss = criterion(
                    logits,
                    yb,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
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

            val_auprc = (
                average_precision_score(
                    val_y,
                    val_p,
                )
            )

            val_auroc = (
                roc_auc_score(
                    val_y,
                    val_p,
                )
            )

            print(
                f"Epoch "
                f"{epoch:02d} | "
                f"loss="
                f"{train_loss:.4f} | "
                f"val AUROC="
                f"{val_auroc:.4f} | "
                f"val AUPRC="
                f"{val_auprc:.4f}"
            )

            # Primary model-selection metric = AUPRC
            if (
                val_auprc >
                best_val + 1e-4
            ):

                best_val = (
                    val_auprc
                )

                best_epoch = (
                    epoch
                )

                bad_epochs = 0

                torch.save(
                    model.state_dict(),
                    checkpoint,
                )

            else:

                bad_epochs += 1

                if (
                    bad_epochs >=
                    PATIENCE
                ):

                    print(
                        "Early stopping."
                    )

                    break

        # ----------------------------------------------------
        # Restore best validation checkpoint
        # ----------------------------------------------------

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
        )

        # ----------------------------------------------------
        # Final held-out test evaluation
        # ----------------------------------------------------

        test_y, test_p = predict(
            model,
            test_loader,
            device,
        )

        auroc = (
            roc_auc_score(
                test_y,
                test_p,
            )
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

        print(
            "\nTEST RESULT"
        )

        print(
            f"Best epoch    : "
            f"{best_epoch}"
        )

        print(
            f"Best val AUPRC: "
            f"{best_val:.4f}"
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
            "model":
                "Transformer",

            "history":
                HISTORY,

            "horizon":
                H,

            "best_epoch":
                best_epoch,

            "positive_rate":
                float(
                    test_y.mean()
                ),

            "auroc":
                float(auroc),

            "auprc":
                float(auprc),

            "ece":
                float(ece),

            "brier":
                float(brier),

            "parameters":
                total_params,
        })

        # ----------------------------------------------------
        # Save prediction-level results.
        # Needed later for lead-time/FAR/statistical analysis.
        # ----------------------------------------------------

        pred_df = pd.DataFrame(
            meta_test
        )

        pred_df[
            "label"
        ] = (
            test_y.astype(int)
        )

        pred_df[
            "probability"
        ] = test_p

        pred_file = (
            RESULTS /
            f"e1_transformer_H{H}_predictions.csv"
        )

        pred_df.to_csv(
            pred_file,
            index=False,
        )

        print(
            f"Predictions saved: "
            f"{pred_file}"
        )

    # ========================================================
    # Final Transformer table
    # ========================================================

    result_df = (
        pd.DataFrame(
            results
        )
    )

    output = (
        RESULTS /
        "e1_transformer_prefailure_results.csv"
    )

    result_df.to_csv(
        output,
        index=False,
    )

    print(
        "\n" +
        "=" * 76
    )

    print(
        "FINAL TRANSFORMER RESULTS"
    )

    print(
        "=" * 76
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: "
        f"{output}"
    )

    print(
        "\nE1-C TRANSFORMER COMPLETE"
    )


if __name__ == "__main__":
    main()