import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e1_missionshield_forecaster as ms
import e12e_missionshield_multiseed as e12

H = 60
HELD = "service_degradation"
SEEDS = [101,202,303,404,505,606,707,808,909,1010]

OUT = ROOT / "results" / "e15"
OUT.mkdir(parents=True, exist_ok=True)

baseline_spec = json.loads(
    (OUT / "e15b0_deterministic_baseline.json").read_text(
        encoding="utf-8"
    )
)

df = pd.read_csv(ms.DATA)
lookup = ms.build_telemetry_lookup(df)
base = pd.read_csv(ROOT / "results" / f"e1_final_samples_H{H}.csv")

tr, va, te = e12.splits(base, HELD)
manifest = pd.concat([tr, va, te], ignore_index=True)

keys = set()
for row in manifest[manifest["split"] == "train"].itertuples():
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

def digest(a):
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()
    ).hexdigest()

assert digest(X) == baseline_spec["X_sha256"], "INPUT HASH MISMATCH"
assert digest(y) == baseline_spec["y_sha256"], "LABEL HASH MISMATCH"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(model, values):
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(values), ms.BATCH_SIZE):
            batch = torch.as_tensor(
                np.ascontiguousarray(
                    values[start:start + ms.BATCH_SIZE]
                ),
                dtype=torch.float32,
                device=device
            )
            logits, _, _ = model(batch)
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy().reshape(-1)
            )
    return np.concatenate(probabilities)

feature_names = list(ms.FEATURES)
rows = []

for seed in SEEDS:
    print(f"\nSeed {seed}")

    checkpoint = (
        ROOT / "results" / "e12" / "e12e_checkpoints" /
        f"e12e_missionshield_service_degradation_H60_seed{seed}.pt"
    )

    model = ms.MissionShieldForecaster(
        ms.build_directed_attention_mask()
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    p0 = infer(model, X)

    for j, feature in enumerate(feature_names):
        Xpert = X.copy()
        Xpert[:, :, :, j] = 0.0

        p1 = infer(model, Xpert)

        rows.append({
            "seed": seed,
            "feature": feature,
            "mean_abs_probability_shift": float(
                np.mean(np.abs(p1 - p0))
            ),
            "max_abs_probability_shift": float(
                np.max(np.abs(p1 - p0))
            ),
            "baseline_auprc": float(
                average_precision_score(y, p0)
            ),
            "perturbed_auprc": float(
                average_precision_score(y, p1)
            ),
            "baseline_auroc": float(
                roc_auc_score(y, p0)
            ),
            "perturbed_auroc": float(
                roc_auc_score(y, p1)
            )
        })

result = pd.DataFrame(rows)

output_file = OUT / "e15b1_feature_masking.csv"
if output_file.exists():
    raise RuntimeError("Refusing to overwrite existing E15-B1 artifact")

result.to_csv(output_file, index=False)

print("\n=== E15-B1 FEATURE MASKING COMPLETE ===")
print("Rows:", len(result))
print("Output:", output_file)

print("\nMean probability shift by feature:")
print(
    result.groupby("feature")[
        "mean_abs_probability_shift"
    ].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    )
)
