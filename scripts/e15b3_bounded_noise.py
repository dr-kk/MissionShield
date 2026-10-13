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
NOISE_LEVELS = [0.01, 0.05, 0.10]
REPLICATES = 5

OUT = ROOT / "results" / "e15"
OUTPUT = OUT / "e15b3_bounded_noise.csv"

if OUTPUT.exists():
    raise RuntimeError("Refusing to overwrite existing E15-B3 results")

spec = json.loads(
    (OUT / "e15b0_deterministic_baseline.json")
    .read_text(encoding="utf-8")
)

df = pd.read_csv(ms.DATA)
lookup = ms.build_telemetry_lookup(df)

base = pd.read_csv(
    ROOT / "results" / f"e1_final_samples_H{H}.csv"
)

tr, va, te = e12.splits(base, HELD)
manifest = pd.concat([tr, va, te], ignore_index=True)

keys = set()

for row in manifest[manifest["split"] == "train"].itertuples():
    for t in range(
        int(row.history_start_time),
        int(row.time) + 1
    ):
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

assert digest(X) == spec["X_sha256"], "INPUT HASH MISMATCH"
assert digest(y) == spec["y_sha256"], "LABEL HASH MISMATCH"

assert X.shape == (405, 10, 6, 6)
assert int(y.sum()) == 26

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def infer(model, values):
    predictions = []

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

            predictions.append(
                torch.sigmoid(logits)
                .cpu().numpy().reshape(-1)
            )

    return np.concatenate(predictions)

# Generate the same noise realizations for every checkpoint.
# This permits paired across-seed comparisons.
noise_conditions = []

for level_index, sigma in enumerate(NOISE_LEVELS):
    for replicate in range(REPLICATES):
        rng = np.random.default_rng(
            15000 + level_index * 100 + replicate
        )

        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=X.shape
        ).astype(np.float32)

        Xpert = np.ascontiguousarray(X + noise)

        noise_conditions.append(
            (sigma, replicate, Xpert)
        )

rows = []

for seed in SEEDS:
    print(f"Processing seed {seed}...", flush=True)

    checkpoint = (
        ROOT / "results" / "e12" / "e12e_checkpoints" /
        f"e12e_missionshield_service_degradation_H60_seed{seed}.pt"
    )

    model = ms.MissionShieldForecaster(
        ms.build_directed_attention_mask()
    ).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    p0 = infer(model, X)

    baseline_auprc = float(average_precision_score(y, p0))
    baseline_auroc = float(roc_auc_score(y, p0))

    for sigma, replicate, Xpert in noise_conditions:
        p1 = infer(model, Xpert)

        perturbed_auprc = float(
            average_precision_score(y, p1)
        )
        perturbed_auroc = float(
            roc_auc_score(y, p1)
        )

        rows.append({
            "seed": seed,
            "noise_sigma": sigma,
            "replicate": replicate,
            "mean_abs_probability_shift": float(
                np.mean(np.abs(p1 - p0))
            ),
            "max_abs_probability_shift": float(
                np.max(np.abs(p1 - p0))
            ),
            "baseline_auprc": baseline_auprc,
            "perturbed_auprc": perturbed_auprc,
            "delta_auprc": perturbed_auprc - baseline_auprc,
            "baseline_auroc": baseline_auroc,
            "perturbed_auroc": perturbed_auroc,
            "delta_auroc": perturbed_auroc - baseline_auroc
        })

result = pd.DataFrame(rows)

assert len(result) == 150
assert not result.isna().any().any()

result.to_csv(OUTPUT, index=False)

print("\n=== E15-B3 BOUNDED NOISE COMPLETE ===")
print("Rows:", len(result))
print("Output:", OUTPUT)

# First average the five noise realizations within each seed.
# Then summarize across the ten checkpoint seeds.
per_seed = result.groupby(
    ["seed", "noise_sigma"], as_index=False
).agg(
    mean_probability_shift=("mean_abs_probability_shift", "mean"),
    mean_delta_auprc=("delta_auprc", "mean"),
    mean_delta_auroc=("delta_auroc", "mean")
)

summary = per_seed.groupby("noise_sigma").agg(
    mean_probability_shift=("mean_probability_shift", "mean"),
    sd_probability_shift=("mean_probability_shift", "std"),
    mean_delta_auprc=("mean_delta_auprc", "mean"),
    sd_delta_auprc=("mean_delta_auprc", "std"),
    mean_delta_auroc=("mean_delta_auroc", "mean"),
    sd_delta_auroc=("mean_delta_auroc", "std")
)

print("\nNoise sensitivity summary:")
print(summary.round(6).to_string())
