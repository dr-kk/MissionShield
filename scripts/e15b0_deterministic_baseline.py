import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e1_missionshield_forecaster as ms
import e12e_missionshield_multiseed as e12

H = 60
HELD = "service_degradation"
OUT = ROOT / "results" / "e15"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ms.DATA)
lookup = ms.build_telemetry_lookup(df)
base = pd.read_csv(ROOT / "results" / f"e1_final_samples_H{H}.csv")

tr, va, te = e12.splits(base, HELD)
manifest = pd.concat([tr, va, te], ignore_index=True)

train_manifest = manifest[manifest["split"] == "train"]

keys = set()
for row in train_manifest.itertuples():
    for t in range(int(row.history_start_time), int(row.time) + 1):
        keys.add((str(row.run_id), t))

matrices = np.stack([lookup[k] for k in sorted(keys)])
mean = matrices.mean(axis=(0, 1)).astype(np.float32)
std = matrices.std(axis=(0, 1)).astype(np.float32)
std[std < 1e-8] = 1.0

X, y, meta = ms.build_sequences_from_manifest(
    manifest, "test", lookup, mean, std
)

X = np.ascontiguousarray(X, dtype=np.float32)
y = np.asarray(y)

def sha256_array(a):
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()
    ).hexdigest()

summary = {
    "experiment": "E15-B0",
    "attack_family": HELD,
    "horizon": H,
    "normalization": "sorted_training_timestamp_keys_float32",
    "normalization_timestamp_count": len(keys),
    "test_samples": len(X),
    "X_shape": list(X.shape),
    "positive_labels": int(np.sum(y)),
    "normalization_mean": mean.tolist(),
    "normalization_std": std.tolist(),
    "X_sha256": sha256_array(X),
    "y_sha256": sha256_array(y),
    "status": "deterministic_input_baseline"
}

path = OUT / "e15b0_deterministic_baseline.json"
path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n=== E15-B0 DETERMINISTIC BASELINE ===")
print("Test samples:", len(X))
print("Positive labels:", int(np.sum(y)))
print("Input shape:", X.shape)
print("Input SHA256:", summary["X_sha256"])
print("Labels SHA256:", summary["y_sha256"])
print("Saved:", path)
