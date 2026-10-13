from pathlib import Path
import importlib.util
import random
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# ============================================================
# MissionShield E2
# FAIR BASELINE RERUN ON FINAL FROZEN MANIFESTS
#
# Baseline architectures/hyperparameters are imported from:
#   e1_lstm_prefailure.py
#   e1_transformer_prefailure.py
#   e1_graph_temporal_prefailure.py
#
# GBDT hyperparameters are preserved from:
#   e1_v2_prefailure_baseline.py
#
# The ONLY evaluation protocol change is that all models now use
# the exact censoring-safe frozen H-specific sample manifests:
#   e1_final_samples_H10.csv
#   e1_final_samples_H30.csv
#   e1_final_samples_H60.csv
#
# No MissionShield result is used for tuning these baselines.
# ============================================================

ROOT = Path(r"D:\MissionShield")
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data" / "processed" / "missionshield_e0_v2_dataset.csv"
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

BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4


def load_module(name, filename):
    path = SCRIPTS / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required original baseline file not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
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

        ece += mask.mean() * abs(accuracy - confidence)

    return float(ece)


def resolve_node_column(df):
    for c in ["node_id", "node", "node_name"]:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Could not find node column. Expected node_id, node, or node_name."
    )


def build_node_lookup(df):
    print("Building frozen telemetry lookup...")

    node_col = resolve_node_column(df)
    node_order = {node: i for i, node in enumerate(NODES)}

    work = df.copy()
    work[node_col] = work[node_col].astype(str)
    work["node_order"] = work[node_col].map(node_order)

    if work["node_order"].isna().any():
        unknown = work.loc[
            work["node_order"].isna(),
            node_col,
        ].unique()

        raise RuntimeError(f"Unknown node IDs: {unknown}")

    lookup = {}

    for (run_id, time), g in work.groupby(
        ["run_id", "time"],
        sort=False,
    ):
        g = g.sort_values("node_order")

        if len(g) != NUM_NODES:
            raise RuntimeError(
                f"{run_id}, t={time}: expected {NUM_NODES} nodes, "
                f"found {len(g)}"
            )

        matrix = np.zeros(
            (NUM_NODES, NUM_FEATURES),
            dtype=np.float32,
        )

        for node_idx, node in enumerate(NODES):
            row = g[g[node_col] == node]

            if len(row) != 1:
                raise RuntimeError(
                    f"Node problem: run={run_id}, time={time}, node={node}"
                )

            row = row.iloc[0]

            for feature_idx, feature in enumerate(FEATURES):
                matrix[node_idx, feature_idx] = float(row[feature])

        lookup[(str(run_id), int(time))] = matrix

    print(f"Telemetry timestamps : {len(lookup):,}")
    return lookup


def training_history_keys(manifest):
    keys = set()

    train = manifest[manifest["split"] == "train"]

    for row in train.itertuples():
        run_id = str(row.run_id)
        start = int(row.history_start_time)
        end = int(row.time)

        times = list(range(start, end + 1))

        if len(times) != HISTORY:
            raise RuntimeError(
                f"Frozen history length error for {row.sample_id}: {len(times)}"
            )

        for t in times:
            keys.add((run_id, t))

    return keys


def vector_normalization(manifest, lookup):
    """
    LSTM/Transformer preserve the original 36-D ordered system vector:
    6 nodes x 6 telemetry features.

    Normalization is fitted using training data only.
    """
    keys = training_history_keys(manifest)

    vectors = []

    for key in keys:
        if key not in lookup:
            raise RuntimeError(f"Missing telemetry key: {key}")

        vectors.append(
            lookup[key].reshape(-1)
        )

    vectors = np.stack(vectors).astype(np.float32)

    mean = vectors.mean(axis=0).astype(np.float32)
    std = vectors.std(axis=0).astype(np.float32)

    std[std < 1e-8] = 1.0

    return mean, std


def graph_normalization(manifest, lookup):
    """
    GraphTemporal preserves node x feature input.

    Feature normalization is fitted on training histories only,
    pooling over training timestamps and nodes.
    """
    keys = training_history_keys(manifest)

    matrices = []

    for key in keys:
        if key not in lookup:
            raise RuntimeError(f"Missing telemetry key: {key}")

        matrices.append(lookup[key])

    matrices = np.stack(matrices).astype(np.float32)

    mean = matrices.mean(axis=(0, 1)).astype(np.float32)
    std = matrices.std(axis=(0, 1)).astype(np.float32)

    std[std < 1e-8] = 1.0

    return mean, std


def build_vector_sequences(
    manifest,
    split_name,
    lookup,
    mean,
    std,
):
    subset = (
        manifest[manifest["split"] == split_name]
        .copy()
        .sort_values(["run_id", "time"])
    )

    X = []
    y = []
    metadata = []

    for row in subset.itertuples():
        run_id = str(row.run_id)
        start = int(row.history_start_time)
        end = int(row.time)

        times = list(range(start, end + 1))

        if len(times) != HISTORY:
            raise RuntimeError(
                f"History error for {row.sample_id}: {len(times)}"
            )

        seq = []

        for t in times:
            key = (run_id, t)

            if key not in lookup:
                raise RuntimeError(f"Missing telemetry: {key}")

            vector = lookup[key].reshape(-1)
            vector = (vector - mean) / std
            seq.append(vector.astype(np.float32, copy=False))

        seq = np.stack(seq)

        if seq.shape != (HISTORY, NUM_NODES * NUM_FEATURES):
            raise RuntimeError(
                f"Vector sequence shape error: {seq.shape}"
            )

        X.append(seq)
        y.append(float(row.label))

        metadata.append(
            {
                "sample_id": row.sample_id,
                "run_id": row.run_id,
                "time": int(row.time),
                "attack_family": row.attack_family,
                "horizon": int(row.horizon),
                "steps_to_failure": row.steps_to_failure,
            }
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        metadata,
    )


def build_graph_sequences(
    manifest,
    split_name,
    lookup,
    mean,
    std,
):
    subset = (
        manifest[manifest["split"] == split_name]
        .copy()
        .sort_values(["run_id", "time"])
    )

    X = []
    y = []
    metadata = []

    for row in subset.itertuples():
        run_id = str(row.run_id)
        start = int(row.history_start_time)
        end = int(row.time)

        times = list(range(start, end + 1))

        if len(times) != HISTORY:
            raise RuntimeError(
                f"History error for {row.sample_id}: {len(times)}"
            )

        seq = []

        for t in times:
            key = (run_id, t)

            if key not in lookup:
                raise RuntimeError(f"Missing telemetry: {key}")

            matrix = (lookup[key] - mean) / std

            if matrix.shape != (NUM_NODES, NUM_FEATURES):
                raise RuntimeError(
                    f"Graph matrix shape error: {matrix.shape}"
                )

            seq.append(matrix.astype(np.float32, copy=False))

        seq = np.stack(seq)

        if seq.shape != (HISTORY, NUM_NODES, NUM_FEATURES):
            raise RuntimeError(
                f"Graph sequence shape error: {seq.shape}"
            )

        X.append(seq)
        y.append(float(row.label))

        metadata.append(
            {
                "sample_id": row.sample_id,
                "run_id": row.run_id,
                "time": int(row.time),
                "attack_family": row.attack_family,
                "horizon": int(row.horizon),
                "steps_to_failure": row.steps_to_failure,
            }
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        metadata,
    )


def aggregate_current_step(df):
    """
    Exact GBDT representation from e1_v2_prefailure_baseline:
    mean/max/std for each of the six telemetry features across nodes.
    """
    agg = (
        df.groupby(
            ["run_id", "attack_family", "time"],
            as_index=False,
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

    agg = agg.rename(columns=rename)
    agg["run_id"] = agg["run_id"].astype(str)
    agg["time"] = agg["time"].astype(int)

    return agg


def gbdt_dataset(manifest, split_name, aggregate_df, model_features):
    subset = manifest[manifest["split"] == split_name].copy()

    subset["run_id"] = subset["run_id"].astype(str)
    subset["time"] = subset["time"].astype(int)

    merged = subset.merge(
        aggregate_df[
            ["run_id", "time"] + model_features
        ],
        on=["run_id", "time"],
        how="left",
        validate="one_to_one",
    )

    if merged[model_features].isna().all(axis=1).any():
        raise RuntimeError(
            f"GBDT telemetry merge failed for split={split_name}"
        )

    X = (
        merged[model_features]
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    y = merged["label"].to_numpy(dtype=int)

    metadata = merged[
        [
            "sample_id",
            "run_id",
            "time",
            "attack_family",
            "horizon",
            "steps_to_failure",
        ]
    ].to_dict("records")

    return X, y, metadata


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@torch.no_grad()
def predict_neural(model, loader, device, graph_model=False):
    model.eval()

    labels = []
    probabilities = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)

        output = model(X)

        if graph_model:
            logits = output[0]
        else:
            logits = output

        p = torch.sigmoid(logits)

        labels.append(y.numpy())
        probabilities.append(p.cpu().numpy())

    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
    )


def metrics(y, p):
    return {
        "positive_rate": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "ece": float(expected_calibration_error(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def save_predictions(model_name, H, metadata, y, p):
    pred = pd.DataFrame(metadata)
    pred["label"] = np.asarray(y).astype(int)
    pred["probability"] = np.asarray(p)

    filename = (
        RESULTS /
        f"e2_frozen_{model_name.lower()}_H{H}_predictions.csv"
    )

    pred.to_csv(filename, index=False)

    print(f"Predictions saved: {filename}")


def run_gbdt(
    H,
    manifest,
    aggregate_df,
    model_features,
):
    print("\n" + "-" * 80)
    print(f"GBDT | H={H}")
    print("-" * 80)

    X_train, y_train, _ = gbdt_dataset(
        manifest, "train", aggregate_df, model_features
    )
    X_val, y_val, _ = gbdt_dataset(
        manifest, "validation", aggregate_df, model_features
    )
    X_test, y_test, meta_test = gbdt_dataset(
        manifest, "test", aggregate_df, model_features
    )

    print(
        f"Train/Val/Test : "
        f"{len(y_train):,}/{len(y_val):,}/{len(y_test):,}"
    )
    print(
        f"Positive rates : "
        f"{y_train.mean():.4f}/"
        f"{y_val.mean():.4f}/"
        f"{y_test.mean():.4f}"
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=SEED,
    )

    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    m = metrics(y_test, p)

    print(
        f"TEST | AUROC={m['auroc']:.4f} | "
        f"AUPRC={m['auprc']:.4f} | "
        f"ECE={m['ece']:.4f} | "
        f"Brier={m['brier']:.4f}"
    )

    save_predictions(
        "GBDT",
        H,
        meta_test,
        y_test,
        p,
    )

    return {
        "model": "GBDT",
        "history": 1,
        "horizon": H,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "best_epoch": np.nan,
        **m,
    }


def train_neural_baseline(
    model_name,
    H,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    meta_test,
    device,
    graph_model=False,
):
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(SEED + H)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = negatives / max(positives, 1.0)

    print(f"Parameters      : {total_params:,}")
    print(f"Positive weight : {pos_weight:.3f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = -np.inf
    best_epoch = 0
    bad_epochs = 0

    checkpoint = (
        CHECKPOINTS /
        f"e2_frozen_{model_name.lower()}_H{H}.pt"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output = model(Xb)

            if graph_model:
                logits = output[0]
            else:
                logits = output

            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            running_loss += loss.item() * len(yb)

        train_loss = running_loss / len(train_ds)

        val_y, val_p = predict_neural(
            model,
            val_loader,
            device,
            graph_model=graph_model,
        )

        val_auprc = average_precision_score(
            val_y,
            val_p,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val AUPRC={val_auprc:.4f}"
        )

        if val_auprc > best_val + 1e-4:
            best_val = val_auprc
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                model.state_dict(),
                checkpoint,
            )
        else:
            bad_epochs += 1

            if bad_epochs >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    )

    test_y, test_p = predict_neural(
        model,
        test_loader,
        device,
        graph_model=graph_model,
    )

    m = metrics(test_y, test_p)

    print("\nFROZEN TEST RESULT")
    print(f"Best epoch     : {best_epoch}")
    print(f"Best val AUPRC : {best_val:.4f}")
    print(f"AUROC          : {m['auroc']:.4f}")
    print(f"AUPRC          : {m['auprc']:.4f}")
    print(f"ECE            : {m['ece']:.4f}")
    print(f"Brier          : {m['brier']:.4f}")

    save_predictions(
        model_name,
        H,
        meta_test,
        test_y,
        test_p,
    )

    result = {
        "model": model_name,
        "history": HISTORY,
        "horizon": H,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "best_epoch": best_epoch,
        **m,
        "parameters": total_params,
    }

    del model
    del train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    seed_everything(SEED)

    print("=" * 80)
    print("MissionShield E2 | FAIR BASELINES ON FINAL FROZEN MANIFESTS")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device : {device}")

    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("\nLoading ORIGINAL baseline architectures...")

    lstm_mod = load_module(
        "e1_lstm_original",
        "e1_lstm_prefailure.py",
    )

    transformer_mod = load_module(
        "e1_transformer_original",
        "e1_transformer_prefailure.py",
    )

    graph_mod = load_module(
        "e1_graph_original",
        "e1_graph_temporal_prefailure.py",
    )

    print("Original baseline modules loaded.")

    df = pd.read_csv(DATA)

    print(f"\nNode rows : {len(df):,}")
    print(f"Runs      : {df.run_id.nunique()}")

    lookup = build_node_lookup(df)

    aggregate_df = aggregate_current_step(df)

    model_features = [
        c for c in aggregate_df.columns
        if any(
            c.startswith(feature + "_")
            for feature in FEATURES
        )
    ]

    print(f"GBDT aggregate features : {len(model_features)}")
    print(f"Temporal vector size    : {NUM_NODES * NUM_FEATURES}")
    print(f"History                 : {HISTORY}")

    all_results = []

    for H in HORIZONS:
        print("\n" + "=" * 80)
        print(f"FINAL FROZEN HORIZON H={H}")
        print("=" * 80)

        manifest_file = (
            RESULTS /
            f"e1_final_samples_H{H}.csv"
        )

        manifest = pd.read_csv(manifest_file)
        manifest["run_id"] = manifest["run_id"].astype(str)

        n_train = int((manifest["split"] == "train").sum())
        n_val = int((manifest["split"] == "validation").sum())
        n_test = int((manifest["split"] == "test").sum())

        print(f"Frozen samples : {len(manifest):,}")
        print(
            f"Train/Val/Test : "
            f"{n_train:,}/{n_val:,}/{n_test:,}"
        )

        # ----------------------------------------------------
        # 1. GBDT
        # ----------------------------------------------------
        all_results.append(
            run_gbdt(
                H,
                manifest,
                aggregate_df,
                model_features,
            )
        )

        # ----------------------------------------------------
        # Common exact frozen temporal samples
        # ----------------------------------------------------
        vector_mean, vector_std = vector_normalization(
            manifest,
            lookup,
        )

        X_train, y_train, _ = build_vector_sequences(
            manifest,
            "train",
            lookup,
            vector_mean,
            vector_std,
        )

        X_val, y_val, _ = build_vector_sequences(
            manifest,
            "validation",
            lookup,
            vector_mean,
            vector_std,
        )

        X_test, y_test, meta_test = build_vector_sequences(
            manifest,
            "test",
            lookup,
            vector_mean,
            vector_std,
        )

        assert len(y_train) == n_train
        assert len(y_val) == n_val
        assert len(y_test) == n_test

        print(
            f"\nFrozen temporal tensor : "
            f"{X_train.shape}"
        )

        # ----------------------------------------------------
        # 2. LSTM - ORIGINAL architecture/hyperparameters
        # ----------------------------------------------------
        print("\n" + "-" * 80)
        print(f"LSTM | H={H}")
        print("-" * 80)

        seed_everything(SEED + H)

        lstm_model = lstm_mod.LSTMForecaster(
            input_dim=NUM_NODES * NUM_FEATURES,
            hidden=lstm_mod.HIDDEN,
            layers=lstm_mod.LAYERS,
            dropout=lstm_mod.DROPOUT,
        )

        all_results.append(
            train_neural_baseline(
                "LSTM",
                H,
                lstm_model,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                meta_test,
                device,
                graph_model=False,
            )
        )

        # ----------------------------------------------------
        # 3. Transformer - ORIGINAL architecture/hyperparameters
        # ----------------------------------------------------
        print("\n" + "-" * 80)
        print(f"Transformer | H={H}")
        print("-" * 80)

        seed_everything(SEED + H)

        transformer_model = (
            transformer_mod.TransformerForecaster(
                input_dim=NUM_NODES * NUM_FEATURES,
                d_model=transformer_mod.D_MODEL,
                nhead=transformer_mod.NHEAD,
                layers=transformer_mod.LAYERS,
                dropout=transformer_mod.DROPOUT,
                ff_dim=transformer_mod.FF_DIM,
                max_len=max(64, HISTORY),
            )
        )

        all_results.append(
            train_neural_baseline(
                "Transformer",
                H,
                transformer_model,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                meta_test,
                device,
                graph_model=False,
            )
        )

        del X_train, X_val, X_test
        gc.collect()

        # ----------------------------------------------------
        # 4. GraphTemporal - ORIGINAL architecture/hyperparameters
        # ----------------------------------------------------
        graph_mean, graph_std = graph_normalization(
            manifest,
            lookup,
        )

        GX_train, Gy_train, _ = build_graph_sequences(
            manifest,
            "train",
            lookup,
            graph_mean,
            graph_std,
        )

        GX_val, Gy_val, _ = build_graph_sequences(
            manifest,
            "validation",
            lookup,
            graph_mean,
            graph_std,
        )

        GX_test, Gy_test, Gmeta_test = build_graph_sequences(
            manifest,
            "test",
            lookup,
            graph_mean,
            graph_std,
        )

        assert len(Gy_train) == n_train
        assert len(Gy_val) == n_val
        assert len(Gy_test) == n_test

        print(
            f"\nFrozen graph tensor    : "
            f"{GX_train.shape}"
        )

        print("\n" + "-" * 80)
        print(f"GraphTemporal | H={H}")
        print("-" * 80)

        seed_everything(SEED + H)

        adjacency = graph_mod.build_adjacency()

        graph_model = graph_mod.GraphTemporalForecaster(
            adjacency=adjacency,
            graph_hidden=graph_mod.GRAPH_HIDDEN,
            temporal_dim=graph_mod.TEMPORAL_DIM,
            nhead=graph_mod.NHEAD,
            temporal_layers=graph_mod.TEMPORAL_LAYERS,
            ff_dim=graph_mod.FF_DIM,
            dropout=graph_mod.DROPOUT,
            max_len=max(64, HISTORY),
        )

        all_results.append(
            train_neural_baseline(
                "GraphTemporal",
                H,
                graph_model,
                GX_train,
                Gy_train,
                GX_val,
                Gy_val,
                GX_test,
                Gy_test,
                Gmeta_test,
                device,
                graph_model=True,
            )
        )

        del GX_train, GX_val, GX_test
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(all_results)

    result_file = (
        RESULTS /
        "e2_frozen_baselines_results.csv"
    )

    result_df.to_csv(
        result_file,
        index=False,
    )

    print("\n" + "=" * 80)
    print("E2 FROZEN BASELINE RESULTS")
    print("=" * 80)

    print(
        result_df[
            [
                "model",
                "horizon",
                "train_samples",
                "validation_samples",
                "test_samples",
                "best_epoch",
                "positive_rate",
                "auroc",
                "auprc",
                "ece",
                "brier",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {result_file}")
    print("\nE2 FROZEN BASELINE RERUN COMPLETE")


if __name__ == "__main__":
    main()
