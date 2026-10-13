from pathlib import Path
import importlib.util
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(r"D:\MissionShield")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
DATA = ROOT / "data" / "processed" / "missionshield_e0_v2_dataset.csv"

HORIZONS = [10, 30, 60]
BATCH_SIZE = 256


def load_module(name, filename):
    path = SCRIPTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_predictions(path, metadata, y, p):
    df = pd.DataFrame(metadata)
    df["label"] = np.asarray(y).astype(int)
    df["probability"] = np.asarray(p, dtype=float)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def extract_missionshield(ms, H, df, device):
    print("\n" + "-" * 80)
    print(f"MissionShield validation predictions | H={H}")
    print("-" * 80)

    manifest = pd.read_csv(RESULTS / f"e1_final_samples_H{H}.csv")
    manifest["run_id"] = manifest["run_id"].astype(str)

    lookup = ms.build_telemetry_lookup(df)
    mean, std = ms.calculate_normalization(manifest, lookup)

    X_val, y_val, meta_val = ms.build_sequences_from_manifest(
        manifest, "validation", lookup, mean, std
    )

    expected = int((manifest["split"] == "validation").sum())
    assert len(y_val) == expected

    loader = DataLoader(
        ms.MissionDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = ms.MissionShieldForecaster(
        directed_mask=ms.build_directed_attention_mask()
    ).to(device)

    checkpoint = CHECKPOINTS / f"e1_missionshield_H{H}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    y, p = ms.predict(model, loader, device)

    assert np.array_equal(y.astype(int), y_val.astype(int))

    save_predictions(
        RESULTS / f"e3a_validation_missionshield_H{H}.csv",
        meta_val, y, p
    )

    del X_val, loader, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_baselines(e2, lstm_mod, transformer_mod, graph_mod, H, df, lookup, device):
    manifest = pd.read_csv(RESULTS / f"e1_final_samples_H{H}.csv")
    manifest["run_id"] = manifest["run_id"].astype(str)

    expected = int((manifest["split"] == "validation").sum())

    # ---------------- LSTM + Transformer ----------------
    vector_mean, vector_std = e2.vector_normalization(manifest, lookup)

    X_val, y_val, meta_val = e2.build_vector_sequences(
        manifest, "validation", lookup, vector_mean, vector_std
    )
    assert len(y_val) == expected

    loader = DataLoader(
        e2.SequenceDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("\n" + "-" * 80)
    print(f"LSTM validation predictions | H={H}")
    print("-" * 80)

    model = lstm_mod.LSTMForecaster(
        input_dim=e2.NUM_NODES * e2.NUM_FEATURES,
        hidden=lstm_mod.HIDDEN,
        layers=lstm_mod.LAYERS,
        dropout=lstm_mod.DROPOUT,
    ).to(device)

    ckpt = CHECKPOINTS / f"e2_frozen_lstm_H{H}.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )

    y, p = e2.predict_neural(model, loader, device, graph_model=False)
    save_predictions(
        RESULTS / f"e3a_validation_lstm_H{H}.csv",
        meta_val, y, p
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "-" * 80)
    print(f"Transformer validation predictions | H={H}")
    print("-" * 80)

    model = transformer_mod.TransformerForecaster(
        input_dim=e2.NUM_NODES * e2.NUM_FEATURES,
        d_model=transformer_mod.D_MODEL,
        nhead=transformer_mod.NHEAD,
        layers=transformer_mod.LAYERS,
        dropout=transformer_mod.DROPOUT,
        ff_dim=transformer_mod.FF_DIM,
        max_len=max(64, e2.HISTORY),
    ).to(device)

    ckpt = CHECKPOINTS / f"e2_frozen_transformer_H{H}.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )

    y, p = e2.predict_neural(model, loader, device, graph_model=False)
    save_predictions(
        RESULTS / f"e3a_validation_transformer_H{H}.csv",
        meta_val, y, p
    )

    del model, loader, X_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------- GraphTemporal ----------------
    graph_mean, graph_std = e2.graph_normalization(manifest, lookup)

    GX_val, Gy_val, Gmeta_val = e2.build_graph_sequences(
        manifest, "validation", lookup, graph_mean, graph_std
    )
    assert len(Gy_val) == expected

    gloader = DataLoader(
        e2.SequenceDataset(GX_val, Gy_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("\n" + "-" * 80)
    print(f"GraphTemporal validation predictions | H={H}")
    print("-" * 80)

    adjacency = graph_mod.build_adjacency()

    model = graph_mod.GraphTemporalForecaster(
        adjacency=adjacency,
        graph_hidden=graph_mod.GRAPH_HIDDEN,
        temporal_dim=graph_mod.TEMPORAL_DIM,
        nhead=graph_mod.NHEAD,
        temporal_layers=graph_mod.TEMPORAL_LAYERS,
        ff_dim=graph_mod.FF_DIM,
        dropout=graph_mod.DROPOUT,
        max_len=max(64, e2.HISTORY),
    ).to(device)

    ckpt = CHECKPOINTS / f"e2_frozen_graphtemporal_H{H}.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )

    y, p = e2.predict_neural(model, gloader, device, graph_model=True)
    save_predictions(
        RESULTS / f"e3a_validation_graphtemporal_H{H}.csv",
        Gmeta_val, y, p
    )

    del model, gloader, GX_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_gbdt(e2, H, df):
    """
    Refit GBDT using exactly the frozen training samples and the already-frozen
    original hyperparameters, solely to obtain validation probabilities.
    Test predictions from E2 are not touched.
    """
    print("\n" + "-" * 80)
    print(f"GBDT validation predictions | H={H}")
    print("-" * 80)

    manifest = pd.read_csv(RESULTS / f"e1_final_samples_H{H}.csv")
    manifest["run_id"] = manifest["run_id"].astype(str)

    aggregate_df = e2.aggregate_current_step(df)
    model_features = [
        c for c in aggregate_df.columns
        if any(c.startswith(feature + "_") for feature in e2.FEATURES)
    ]

    X_train, y_train, _ = e2.gbdt_dataset(
        manifest, "train", aggregate_df, model_features
    )
    X_val, y_val, meta_val = e2.gbdt_dataset(
        manifest, "validation", aggregate_df, model_features
    )

    model = e2.HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=e2.SEED,
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_val)[:, 1]

    save_predictions(
        RESULTS / f"e3a_validation_gbdt_H{H}.csv",
        meta_val, y_val, p
    )


def main():
    print("=" * 80)
    print("MissionShield E3-A | EXTRACT VALIDATION PREDICTIONS")
    print("NO NEURAL RETRAINING | EXISTING SELECTED CHECKPOINTS ONLY")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    ms = load_module(
        "missionshield_e1",
        "e1_missionshield_forecaster.py",
    )
    e2 = load_module(
        "frozen_e2",
        "e2_frozen_baselines.py",
    )
    lstm_mod = load_module(
        "lstm_original",
        "e1_lstm_prefailure.py",
    )
    transformer_mod = load_module(
        "transformer_original",
        "e1_transformer_prefailure.py",
    )
    graph_mod = load_module(
        "graph_original",
        "e1_graph_temporal_prefailure.py",
    )

    df = pd.read_csv(DATA)
    print(f"Node rows : {len(df):,}")
    print(f"Runs      : {df.run_id.nunique()}")

    lookup = e2.build_node_lookup(df)

    for H in HORIZONS:
        print("\n" + "=" * 80)
        print(f"HORIZON H={H}")
        print("=" * 80)

        extract_gbdt(e2, H, df)

        extract_baselines(
            e2,
            lstm_mod,
            transformer_mod,
            graph_mod,
            H,
            df,
            lookup,
            device,
        )

        extract_missionshield(ms, H, df, device)

    print("\n" + "=" * 80)
    print("E3-A COMPLETE")
    print("=" * 80)
    print("Validation predictions saved for:")
    print("GBDT, LSTM, Transformer, GraphTemporal, MissionShield")
    print("Next: validation-only calibration + fixed-FAR operational evaluation.")


if __name__ == "__main__":
    main()
