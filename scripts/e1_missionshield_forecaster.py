from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# ============================================================
# MissionShield E1-PROPOSED
# Directed Dependency-Aware Temporal Forecaster
#
# Uses FINAL FROZEN censoring-safe manifests.
# No mission_effectiveness as model input.
# Validation AUPRC selects checkpoint.
# ============================================================

ROOT = Path(r"D:\MissionShield")
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

DIRECTED_EDGES = [
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
WEIGHT_DECAY = 1e-4

NODE_DIM = 64
GRAPH_HEADS = 4
TEMPORAL_DIM = 128
TEMPORAL_HEADS = 4
TEMPORAL_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.20


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


def build_directed_attention_mask():
    mask = np.zeros((NUM_NODES, NUM_NODES), dtype=bool)

    for i in range(NUM_NODES):
        mask[i, i] = True

    # destination attends to its direct upstream source
    for src, dst in DIRECTED_EDGES:
        mask[dst, src] = True

    return torch.tensor(mask, dtype=torch.bool)


def resolve_node_column(df):
    for c in ["node_id", "node", "node_name"]:
        if c in df.columns:
            return c
    raise RuntimeError(
        "Could not find node column. Expected one of: node_id, node, node_name"
    )


def build_telemetry_lookup(df):
    print("Building node-structured telemetry lookup...")

    node_col = resolve_node_column(df)
    node_order = {node: i for i, node in enumerate(NODES)}

    df = df.copy()
    df[node_col] = df[node_col].astype(str)
    df["node_order"] = df[node_col].map(node_order)

    if df["node_order"].isna().any():
        unknown = df.loc[df["node_order"].isna(), node_col].unique()
        raise RuntimeError(f"Unknown node IDs: {unknown}")

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"Missing telemetry features: {missing_features}")

    lookup = {}

    for (run_id, time), g in df.groupby(["run_id", "time"], sort=False):
        g = g.sort_values("node_order")

        if len(g) != NUM_NODES:
            raise RuntimeError(
                f"{run_id}, t={time}: expected {NUM_NODES} nodes, found {len(g)}"
            )

        matrix = np.zeros((NUM_NODES, NUM_FEATURES), dtype=np.float32)

        for node_idx, node in enumerate(NODES):
            row = g[g[node_col] == node]

            if len(row) != 1:
                raise RuntimeError(
                    f"Node error: run={run_id}, time={time}, node={node}"
                )

            row = row.iloc[0]

            for feature_idx, feature in enumerate(FEATURES):
                matrix[node_idx, feature_idx] = float(row[feature])

        lookup[(str(run_id), int(time))] = matrix

    return lookup


def calculate_normalization(manifest, lookup):
    """
    TRAINING-ONLY feature normalization.

    matrices shape:
        [unique training timestamps, nodes, features]

    mean/std shape:
        [features]

    This is intentionally [F], NOT [1,1,F].
    Therefore normalization of a [N,F] matrix remains [N,F]
    and does not introduce an unwanted singleton dimension.
    """
    train_manifest = manifest[manifest["split"] == "train"]

    keys = set()

    for row in train_manifest.itertuples():
        run_id = str(row.run_id)
        start_time = int(row.history_start_time)
        end_time = int(row.time)

        for t in range(start_time, end_time + 1):
            keys.add((run_id, t))

    matrices = []

    for key in keys:
        if key not in lookup:
            raise RuntimeError(f"Missing telemetry key: {key}")
        matrices.append(lookup[key])

    matrices = np.stack(matrices)  # [timestamps, nodes, features]

    # CORRECTED NORMALIZATION:
    # Reduce timestamp and node axes, leaving only feature axis.
    mean = matrices.mean(axis=(0, 1)).astype(np.float32)
    std = matrices.std(axis=(0, 1)).astype(np.float32)

    std[std < 1e-8] = 1.0

    if mean.shape != (NUM_FEATURES,) or std.shape != (NUM_FEATURES,):
        raise RuntimeError(
            f"Normalization shape error: mean={mean.shape}, std={std.shape}"
        )

    return mean, std


def build_sequences_from_manifest(
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
        current_time = int(row.time)
        history_start = int(row.history_start_time)

        times = list(range(history_start, current_time + 1))

        if len(times) != HISTORY:
            raise RuntimeError(
                f"History length error: sample={row.sample_id}, found={len(times)}"
            )

        sequence = []

        for t in times:
            key = (run_id, t)

            if key not in lookup:
                raise RuntimeError(f"Missing telemetry: {key}")

            matrix = lookup[key]

            # matrix [N,F], mean/std [F] -> normalized [N,F]
            matrix = (matrix - mean) / std

            if matrix.shape != (NUM_NODES, NUM_FEATURES):
                raise RuntimeError(
                    f"Normalized matrix shape error for {key}: {matrix.shape}"
                )

            sequence.append(matrix.astype(np.float32, copy=False))

        sequence = np.stack(sequence)  # [T,N,F]

        if sequence.shape != (HISTORY, NUM_NODES, NUM_FEATURES):
            raise RuntimeError(
                f"Sequence shape error for {row.sample_id}: {sequence.shape}"
            )

        X.append(sequence)
        y.append(float(row.label))

        metadata.append(
            {
                "sample_id": row.sample_id,
                "run_id": row.run_id,
                "time": current_time,
                "attack_family": row.attack_family,
                "horizon": int(row.horizon),
                "steps_to_failure": row.steps_to_failure,
            }
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    return X, y, metadata


class MissionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TelemetryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DirectedGraphAttention(nn.Module):
    def __init__(self, dim, heads, dropout, allowed_mask):
        super().__init__()

        if dim % heads != 0:
            raise ValueError("dim must be divisible by number of heads.")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "allowed_mask",
            allowed_mask.clone().detach().bool(),
        )

    def forward(self, x):
        # x [B,N,D]
        B, N, D = x.shape

        residual = x
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q = q.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        # [B, heads, destination, source]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        mask = self.allowed_mask[None, None, :, :]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        h = torch.matmul(attention, v)

        h = (
            h.transpose(1, 2)
            .contiguous()
            .reshape(B, N, D)
        )

        h = self.out_proj(h)

        x = residual + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x, attention


class DependencyEncoder(nn.Module):
    def __init__(self, dim, heads, dropout, mask):
        super().__init__()

        self.layer1 = DirectedGraphAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            allowed_mask=mask,
        )

        self.layer2 = DirectedGraphAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            allowed_mask=mask,
        )

        self.node_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        x, attn1 = self.layer1(x)
        x, attn2 = self.layer2(x)

        node_logits = self.node_gate(x).squeeze(-1)
        node_weights = torch.softmax(node_logits, dim=-1)

        mission_state = torch.sum(
            x * node_weights.unsqueeze(-1),
            dim=1,
        )

        return mission_state, node_weights, attn1, attn2


class MissionShieldForecaster(nn.Module):
    def __init__(self, directed_mask):
        super().__init__()

        self.telemetry_encoder = TelemetryEncoder(
            input_dim=NUM_FEATURES,
            hidden_dim=NODE_DIM,
            dropout=DROPOUT,
        )

        self.dependency_encoder = DependencyEncoder(
            dim=NODE_DIM,
            heads=GRAPH_HEADS,
            dropout=DROPOUT,
            mask=directed_mask,
        )

        self.temporal_projection = nn.Linear(NODE_DIM, TEMPORAL_DIM)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, HISTORY, TEMPORAL_DIM)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=TEMPORAL_DIM,
            nhead=TEMPORAL_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=TEMPORAL_LAYERS,
        )

        self.temporal_norm = nn.LayerNorm(TEMPORAL_DIM)
        self.current_projection = nn.Linear(NODE_DIM, TEMPORAL_DIM)

        self.fusion_gate = nn.Sequential(
            nn.Linear(TEMPORAL_DIM * 2, TEMPORAL_DIM),
            nn.GELU(),
            nn.Linear(TEMPORAL_DIM, TEMPORAL_DIM),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(TEMPORAL_DIM),
            nn.Linear(TEMPORAL_DIM, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x [B,T,N,F]
        B, T, N, F = x.shape

        if T != HISTORY or N != NUM_NODES or F != NUM_FEATURES:
            raise RuntimeError(
                f"Expected [B,{HISTORY},{NUM_NODES},{NUM_FEATURES}], "
                f"received {tuple(x.shape)}"
            )

        flat = x.reshape(B * T, N, F)

        node_embeddings = self.telemetry_encoder(flat)

        mission_state, node_weights, _, _ = self.dependency_encoder(
            node_embeddings
        )

        mission_state = mission_state.reshape(B, T, NODE_DIM)
        node_weights = node_weights.reshape(B, T, N)

        temporal = self.temporal_projection(mission_state)
        temporal = temporal + self.position_embedding[:, :T, :]
        temporal = self.temporal_encoder(temporal)

        temporal_state = self.temporal_norm(temporal[:, -1, :])

        current_state = self.current_projection(
            mission_state[:, -1, :]
        )

        fusion_input = torch.cat(
            [temporal_state, current_state],
            dim=-1,
        )

        gate = self.fusion_gate(fusion_input)

        fused = (
            gate * temporal_state
            + (1.0 - gate) * current_state
        )

        logits = self.head(fused).squeeze(1)

        return logits, node_weights, gate


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    return_explanations=False,
):
    model.eval()

    labels = []
    probabilities = []
    all_node_weights = []
    all_gate_values = []

    for X, y in loader:
        X = X.to(device, non_blocking=True)

        logits, node_weights, gate = model(X)
        p = torch.sigmoid(logits)

        labels.append(y.numpy())
        probabilities.append(p.cpu().numpy())

        if return_explanations:
            current_weights = node_weights[:, -1, :]

            all_node_weights.append(
                current_weights.cpu().numpy()
            )

            gate_mean = gate.mean(dim=1)

            all_gate_values.append(
                gate_mean.cpu().numpy()
            )

    y = np.concatenate(labels)
    p = np.concatenate(probabilities)

    if not return_explanations:
        return y, p

    node_weights = np.concatenate(all_node_weights, axis=0)
    gate_values = np.concatenate(all_gate_values, axis=0)

    return y, p, node_weights, gate_values


def main():
    seed_everything(SEED)

    print("=" * 80)
    print(
        "MissionShield E1-PROPOSED | "
        "Directed Dependency-Aware Temporal Forecaster"
    )
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device : {device}")

    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(DATA)

    print(f"Node rows : {len(df):,}")
    print(f"Runs      : {df.run_id.nunique()}")

    lookup = build_telemetry_lookup(df)

    print(f"Telemetry timestamps : {len(lookup):,}")

    directed_mask = build_directed_attention_mask()

    print("\nDirected mission graph:")
    print(
        "sensor -> tactical_link -> edge_processing -> "
        "fusion -> command -> actuator"
    )
    print(f"History : {HISTORY}")
    print(f"Nodes   : {NUM_NODES}")
    print(f"Features: {NUM_FEATURES}")

    results = []

    for H in HORIZONS:
        seed_everything(SEED + H)

        print("\n" + "=" * 80)
        print(f"FINAL FROZEN HORIZON H={H}")
        print("=" * 80)

        manifest_file = RESULTS / f"e1_final_samples_H{H}.csv"
        manifest = pd.read_csv(manifest_file)

        manifest["run_id"] = manifest["run_id"].astype(str)

        print(f"Frozen manifest samples : {len(manifest):,}")

        mean, std = calculate_normalization(
            manifest,
            lookup,
        )

        print(f"Normalization mean shape: {mean.shape}")
        print(f"Normalization std shape : {std.shape}")

        X_train, y_train, meta_train = build_sequences_from_manifest(
            manifest, "train", lookup, mean, std
        )
        X_val, y_val, meta_val = build_sequences_from_manifest(
            manifest, "validation", lookup, mean, std
        )
        X_test, y_test, meta_test = build_sequences_from_manifest(
            manifest, "test", lookup, mean, std
        )

        print(f"Train : {len(y_train):,}")
        print(f"Val   : {len(y_val):,}")
        print(f"Test  : {len(y_test):,}")
        print(f"Tensor: {X_train.shape}")

        print(f"Train positive : {y_train.mean():.4f}")
        print(f"Val positive   : {y_val.mean():.4f}")
        print(f"Test positive  : {y_test.mean():.4f}")

        expected_train = len(manifest[manifest["split"] == "train"])
        expected_val = len(manifest[manifest["split"] == "validation"])
        expected_test = len(manifest[manifest["split"] == "test"])

        assert len(y_train) == expected_train
        assert len(y_val) == expected_val
        assert len(y_test) == expected_test

        assert X_train.shape[1:] == (
            HISTORY,
            NUM_NODES,
            NUM_FEATURES,
        )

        assert X_val.shape[1:] == (
            HISTORY,
            NUM_NODES,
            NUM_FEATURES,
        )

        assert X_test.shape[1:] == (
            HISTORY,
            NUM_NODES,
            NUM_FEATURES,
        )

        if len(np.unique(y_train)) < 2:
            raise RuntimeError("Training set has one class.")
        if len(np.unique(y_val)) < 2:
            raise RuntimeError("Validation set has one class.")
        if len(np.unique(y_test)) < 2:
            raise RuntimeError("Test set has one class.")

        train_ds = MissionDataset(X_train, y_train)
        val_ds = MissionDataset(X_val, y_val)
        test_ds = MissionDataset(X_test, y_test)

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

        model = MissionShieldForecaster(
            directed_mask=directed_mask
        ).to(device)

        total_params = sum(
            p.numel() for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(f"Parameters      : {total_params:,}")
        print(f"Trainable       : {trainable_params:,}")

        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)

        pos_weight = negatives / max(positives, 1.0)

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
            f"e1_missionshield_H{H}.pt"
        )

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0

            for Xb, yb in train_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits, _, _ = model(Xb)

                loss = criterion(logits, yb)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

                running_loss += loss.item() * len(yb)

            train_loss = running_loss / len(train_ds)

            val_y, val_p = predict(
                model,
                val_loader,
                device,
            )

            val_auroc = roc_auc_score(
                val_y,
                val_p,
            )

            val_auprc = average_precision_score(
                val_y,
                val_p,
            )

            print(
                f"Epoch {epoch:02d} | "
                f"loss={train_loss:.4f} | "
                f"val AUROC={val_auroc:.4f} | "
                f"val AUPRC={val_auprc:.4f}"
            )

            # Checkpoint selection uses validation AUPRC only.
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

        test_y, test_p, node_weights, gate_values = predict(
            model,
            test_loader,
            device,
            return_explanations=True,
        )

        auroc = roc_auc_score(test_y, test_p)
        auprc = average_precision_score(test_y, test_p)

        ece = expected_calibration_error(
            test_y,
            test_p,
        )

        brier = brier_score_loss(
            test_y,
            test_p,
        )

        print("\nFINAL HELD-OUT TEST")
        print(f"Best epoch     : {best_epoch}")
        print(f"Best val AUPRC : {best_val:.4f}")
        print(f"AUROC          : {auroc:.4f}")
        print(f"AUPRC          : {auprc:.4f}")
        print(f"ECE            : {ece:.4f}")
        print(f"Brier          : {brier:.4f}")

        pred_df = pd.DataFrame(meta_test)

        pred_df["label"] = test_y.astype(int)
        pred_df["probability"] = test_p
        pred_df["fusion_temporal_gate"] = gate_values

        for node_idx, node in enumerate(NODES):
            pred_df[f"importance_{node}"] = (
                node_weights[:, node_idx]
            )

        prediction_file = (
            RESULTS /
            f"e1_missionshield_H{H}_predictions.csv"
        )

        pred_df.to_csv(
            prediction_file,
            index=False,
        )

        print(
            f"Predictions saved: {prediction_file}"
        )

        results.append(
            {
                "model": "MissionShield",
                "history": HISTORY,
                "horizon": H,
                "train_samples": len(y_train),
                "validation_samples": len(y_val),
                "test_samples": len(y_test),
                "best_epoch": best_epoch,
                "positive_rate": float(y_test.mean()),
                "auroc": float(auroc),
                "auprc": float(auprc),
                "ece": float(ece),
                "brier": float(brier),
                "parameters": int(total_params),
            }
        )

        # Release horizon tensors/model before next horizon.
        del X_train, X_val, X_test
        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(results)

    result_file = (
        RESULTS /
        "e1_missionshield_final_results.csv"
    )

    result_df.to_csv(
        result_file,
        index=False,
    )

    print("\n" + "=" * 80)
    print("MISSIONShield FINAL FORECASTING RESULTS")
    print("=" * 80)

    print(
        result_df.to_string(
            index=False
        )
    )

    print(f"\nSaved: {result_file}")
    print("\nE1 MISSIONShield PROPOSED MODEL COMPLETE")


if __name__ == "__main__":
    main()
