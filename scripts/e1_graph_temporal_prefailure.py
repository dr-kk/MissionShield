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
# MissionShield E1-D
# Graph-Temporal Baseline
#
# Spatial graph message passing + temporal Transformer
#
# Input:
# [B, T=10, N=6, F=6]
#
# Strict pre-failure forecasting
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

NUM_NODES = len(NODES)
NUM_FEATURES = len(FEATURES)

# ------------------------------------------------------------
# Mission dependency graph
#
# sensor -> tactical_link -> edge_processing
#        -> fusion -> command -> actuator
#
# For the forecasting baseline we use bidirectional message
# exchange along physical/dependency links plus self-loops.
# This does NOT alter the underlying directed mission semantics;
# it simply allows neighboring node states to be aggregated.
# ------------------------------------------------------------

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]

BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 5
LR = 1e-3

GRAPH_HIDDEN = 64
TEMPORAL_DIM = 128
NHEAD = 4
TEMPORAL_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.20


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
# Build normalized adjacency
# ============================================================

def build_adjacency():

    A = np.zeros(
        (NUM_NODES, NUM_NODES),
        dtype=np.float32,
    )

    # self-loops
    for i in range(NUM_NODES):
        A[i, i] = 1.0

    # neighboring dependency links
    for u, v in EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0

    degree = A.sum(axis=1)

    d_inv_sqrt = np.power(
        degree,
        -0.5,
        where=(degree > 0),
    )

    D = np.diag(
        d_inv_sqrt
    )

    A_norm = (
        D @ A @ D
    )

    return torch.tensor(
        A_norm,
        dtype=torch.float32,
    )


# ============================================================
# Construct timeline
#
# Each timestamp retains:
# [nodes=6, features=6]
# ============================================================

def make_timeline(df):

    print(
        "Building graph-structured "
        "node telemetry..."
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

        if len(g) != NUM_NODES:

            raise RuntimeError(
                f"{run_id}, t={time}: "
                f"expected {NUM_NODES} nodes, "
                f"found {len(g)}."
            )

        node_matrix = np.zeros(
            (
                NUM_NODES,
                NUM_FEATURES,
            ),
            dtype=np.float32,
        )

        for node_idx, node in enumerate(
            NODES
        ):

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

            for feature_idx, feature in enumerate(
                FEATURES
            ):

                node_matrix[
                    node_idx,
                    feature_idx
                ] = float(
                    row[feature]
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

            "node_matrix":
                node_matrix,
        }

        for H in HORIZONS:

            record[
                f"failure_within_{H}"
            ] = int(
                first[
                    f"failure_within_{H}"
                ]
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Training-only normalization
#
# Mean/std estimated per telemetry feature across:
# training runs x time x nodes
# ============================================================

def get_normalization(
    timeline,
    train_runs,
):

    matrices = np.stack(
        timeline.loc[
            timeline["run_id"].isin(
                train_runs
            ),
            "node_matrix",
        ].to_list()
    )

    # [samples, nodes, features]
    # normalize each telemetry feature globally across
    # training timestamps and nodes.
    mean = matrices.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = matrices.std(
        axis=(0, 1),
        keepdims=True,
    )

    std[
        std < 1e-8
    ] = 1.0

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


# ============================================================
# Sequence construction
#
# X:
# [samples, history, nodes, features]
#
# Strict protocol:
# every observation in input history must have failure == 0.
# ============================================================

def build_sequences(
    timeline,
    runs,
    horizon,
    mean,
    std,
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

        matrices = np.stack(
            g[
                "node_matrix"
            ].to_list()
        ).astype(
            np.float32
        )

        matrices = (
            matrices - mean
        ) / std

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

            # Strict pre-failure history
            if np.any(
                current_failure[
                    start:i + 1
                ] != 0
            ):
                continue

            sequence = (
                matrices[
                    start:i + 1
                ]
            )

            X.append(
                sequence
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
# Dataset
# ============================================================

class GraphSequenceDataset(Dataset):

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
# Graph convolution
#
# H' = activation(A_norm H W)
# ============================================================

class GraphConv(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        adjacency,
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features,
        )

        self.register_buffer(
            "adjacency",
            adjacency,
        )

        self.norm = nn.LayerNorm(
            out_features
        )

        self.activation = nn.GELU()

    def forward(
        self,
        x,
    ):

        # x:
        # [B, N, F]

        h = self.linear(
            x
        )

        # A @ H for every batch
        h = torch.einsum(
            "ij,bjd->bid",
            self.adjacency,
            h,
        )

        h = self.norm(
            h
        )

        return self.activation(
            h
        )


# ============================================================
# Spatial graph encoder
# ============================================================

class SpatialGraphEncoder(nn.Module):

    def __init__(
        self,
        adjacency,
        hidden=64,
        dropout=0.20,
    ):

        super().__init__()

        self.gcn1 = GraphConv(
            NUM_FEATURES,
            hidden,
            adjacency,
        )

        self.gcn2 = GraphConv(
            hidden,
            hidden,
            adjacency,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Attention pooling over mission nodes
        self.node_score = nn.Linear(
            hidden,
            1,
        )

    def forward(
        self,
        x,
    ):

        # x:
        # [B, N, F]

        h1 = self.gcn1(
            x
        )

        h1 = self.dropout(
            h1
        )

        h2 = self.gcn2(
            h1
        )

        # Residual connection
        h = h1 + h2

        scores = (
            self.node_score(
                h
            )
            .squeeze(-1)
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        graph_embedding = torch.sum(
            h * weights.unsqueeze(-1),
            dim=1,
        )

        return (
            graph_embedding,
            weights,
        )


# ============================================================
# Graph-Temporal Forecaster
#
# 1. Encode graph independently at each time step.
# 2. Produce graph embedding for each t.
# 3. Transformer models temporal evolution.
# 4. Last temporal state predicts future mission failure.
# ============================================================

class GraphTemporalForecaster(
    nn.Module
):

    def __init__(
        self,
        adjacency,
        graph_hidden=64,
        temporal_dim=128,
        nhead=4,
        temporal_layers=2,
        ff_dim=256,
        dropout=0.20,
        max_len=64,
    ):

        super().__init__()

        self.graph_encoder = (
            SpatialGraphEncoder(
                adjacency=adjacency,
                hidden=graph_hidden,
                dropout=dropout,
            )
        )

        self.temporal_proj = (
            nn.Linear(
                graph_hidden,
                temporal_dim,
            )
        )

        self.pos_embedding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    max_len,
                    temporal_dim,
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
                d_model=temporal_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.temporal_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=temporal_layers,
            )
        )

        self.final_norm = (
            nn.LayerNorm(
                temporal_dim
            )
        )

        self.head = nn.Sequential(
            nn.Linear(
                temporal_dim,
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

    def forward(
        self,
        x,
    ):

        # x:
        # [B, T, N, F]

        B, T, N, F = (
            x.shape
        )

        # Process all graph snapshots
        # efficiently as one large batch.
        x_flat = x.reshape(
            B * T,
            N,
            F,
        )

        graph_embedding, node_weights = (
            self.graph_encoder(
                x_flat
            )
        )

        # [B*T, graph_hidden]
        # ->
        # [B, T, graph_hidden]
        graph_embedding = (
            graph_embedding.reshape(
                B,
                T,
                -1,
            )
        )

        z = self.temporal_proj(
            graph_embedding
        )

        if T > (
            self.pos_embedding
            .size(1)
        ):

            raise RuntimeError(
                "History exceeds positional "
                "embedding length."
            )

        z = (
            z +
            self.pos_embedding[
                :,
                :T,
                :
            ]
        )

        z = self.temporal_encoder(
            z
        )

        current_state = (
            self.final_norm(
                z[:, -1, :]
            )
        )

        logits = (
            self.head(
                current_state
            )
            .squeeze(1)
        )

        # Useful later for attribution analysis
        node_weights = (
            node_weights.reshape(
                B,
                T,
                N,
            )
        )

        return (
            logits,
            node_weights,
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

    probabilities = []
    labels = []

    for X, y in loader:

        X = X.to(
            device,
            non_blocking=True,
        )

        logits, _ = model(
            X
        )

        p = torch.sigmoid(
            logits
        )

        probabilities.append(
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
            probabilities
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(
        SEED
    )

    print("=" * 78)
    print(
        "MissionShield E1-D | "
        "Graph-Temporal Strict Pre-Failure Forecasting"
    )
    print("=" * 78)

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
    # Frozen dataset
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
    # Frozen run split
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

    assert train_runs.isdisjoint(
        val_runs
    )

    assert train_runs.isdisjoint(
        test_runs
    )

    assert val_runs.isdisjoint(
        test_runs
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
    # Graph
    # --------------------------------------------------------

    adjacency = (
        build_adjacency()
    )

    print(
        "\nMission graph"
    )

    print(
        f"Nodes : {NUM_NODES}"
    )

    print(
        f"Dependency links : "
        f"{len(EDGES)}"
    )

    print(
        "Topology : "
        "sensor -> tactical_link -> "
        "edge_processing -> fusion -> "
        "command -> actuator"
    )

    # --------------------------------------------------------
    # Timeline
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

    mean, std = (
        get_normalization(
            timeline,
            train_runs,
        )
    )

    print(
        "Input shape/time-step : "
        f"{NUM_NODES} nodes x "
        f"{NUM_FEATURES} features"
    )

    print(
        f"History length        : "
        f"{HISTORY}"
    )

    print(
        f"Graph hidden          : "
        f"{GRAPH_HIDDEN}"
    )

    print(
        f"Temporal dimension    : "
        f"{TEMPORAL_DIM}"
    )

    results = []

    # ========================================================
    # Each forecast horizon
    # ========================================================

    for H in HORIZONS:

        seed_everything(
            SEED + H
        )

        print(
            "\n" +
            "=" * 78
        )

        print(
            f"Forecast horizon H={H}"
        )

        print(
            "=" * 78
        )

        (
            X_train,
            y_train,
            _,
        ) = build_sequences(
            timeline,
            train_runs,
            H,
            mean,
            std,
        )

        (
            X_val,
            y_val,
            _,
        ) = build_sequences(
            timeline,
            val_runs,
            H,
            mean,
            std,
        )

        (
            X_test,
            y_test,
            meta_test,
        ) = build_sequences(
            timeline,
            test_runs,
            H,
            mean,
            std,
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
            f"Tensor shape    : "
            f"{X_train.shape}"
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

        # Important fairness checks:
        # these should match LSTM/Transformer.
        assert (
            X_train.shape[1]
            == HISTORY
        )

        assert (
            X_train.shape[2]
            == NUM_NODES
        )

        assert (
            X_train.shape[3]
            == NUM_FEATURES
        )

        if len(
            np.unique(
                y_train
            )
        ) < 2:

            raise RuntimeError(
                f"H={H}: training target "
                "has only one class."
            )

        if len(
            np.unique(
                y_val
            )
        ) < 2:

            raise RuntimeError(
                f"H={H}: validation target "
                "has only one class."
            )

        if len(
            np.unique(
                y_test
            )
        ) < 2:

            raise RuntimeError(
                f"H={H}: test target "
                "has only one class."
            )

        # ----------------------------------------------------
        # Data loaders
        # ----------------------------------------------------

        train_ds = (
            GraphSequenceDataset(
                X_train,
                y_train,
            )
        )

        val_ds = (
            GraphSequenceDataset(
                X_val,
                y_val,
            )
        )

        test_ds = (
            GraphSequenceDataset(
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
        # Model
        # ----------------------------------------------------

        model = (
            GraphTemporalForecaster(
                adjacency=adjacency,
                graph_hidden=GRAPH_HIDDEN,
                temporal_dim=TEMPORAL_DIM,
                nhead=NHEAD,
                temporal_layers=(
                    TEMPORAL_LAYERS
                ),
                ff_dim=FF_DIM,
                dropout=DROPOUT,
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

        print(
            f"Parameters      : "
            f"{total_params:,}"
        )

        # ----------------------------------------------------
        # Class weighting from training set only
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
        # Training
        # ----------------------------------------------------

        best_val = -np.inf
        best_epoch = 0
        bad_epochs = 0

        checkpoint = (
            CHECKPOINTS /
            f"e1_graph_temporal_H{H}.pt"
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

                logits, _ = model(
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

            val_auroc = (
                roc_auc_score(
                    val_y,
                    val_p,
                )
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
                f"loss="
                f"{train_loss:.4f} | "
                f"val AUROC="
                f"{val_auroc:.4f} | "
                f"val AUPRC="
                f"{val_auprc:.4f}"
            )

            # Selection ONLY on validation AUPRC
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
        # Best checkpoint
        # ----------------------------------------------------

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
        )

        # ----------------------------------------------------
        # Held-out test
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
                "GraphTemporal",

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
                float(
                    auroc
                ),

            "auprc":
                float(
                    auprc
                ),

            "ece":
                float(
                    ece
                ),

            "brier":
                float(
                    brier
                ),

            "parameters":
                int(
                    total_params
                ),
        })

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        pred_df = pd.DataFrame(
            meta_test
        )

        pred_df[
            "label"
        ] = (
            test_y.astype(
                int
            )
        )

        pred_df[
            "probability"
        ] = test_p

        pred_file = (
            RESULTS /
            f"e1_graph_temporal_H{H}_predictions.csv"
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
    # Final results
    # ========================================================

    result_df = (
        pd.DataFrame(
            results
        )
    )

    output = (
        RESULTS /
        "e1_graph_temporal_prefailure_results.csv"
    )

    result_df.to_csv(
        output,
        index=False,
    )

    print(
        "\n" +
        "=" * 78
    )

    print(
        "FINAL GRAPH-TEMPORAL RESULTS"
    )

    print(
        "=" * 78
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
        "\nE1-D GRAPH-TEMPORAL COMPLETE"
    )


if __name__ == "__main__":
    main()