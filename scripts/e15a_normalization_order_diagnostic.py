import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e1_missionshield_forecaster as ms

HELD = "service_degradation"
H = 60

df = pd.read_csv(ms.DATA)
lookup = ms.build_telemetry_lookup(df)

base = pd.read_csv(ROOT / "results" / f"e1_final_samples_H{H}.csv")

train = base[
    (base["split"] == "train") &
    (base["attack_family"] != HELD)
].copy()

keys = set()
for row in train.itertuples():
    for t in range(int(row.history_start_time), int(row.time) + 1):
        keys.add((str(row.run_id), t))

def statistics(ordered_keys):
    matrices = np.stack([lookup[k] for k in ordered_keys])
    mean = matrices.mean(axis=(0, 1)).astype(np.float32)
    std = matrices.std(axis=(0, 1)).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std

mean_original, std_original = statistics(keys)
mean_sorted, std_sorted = statistics(sorted(keys))

print("\n=== E15-A NORMALIZATION ORDER DIAGNOSTIC ===")
print("Unique training timestamps:", len(keys))
print("Maximum mean difference:",
      float(np.max(np.abs(mean_original - mean_sorted))))
print("Maximum std difference:",
      float(np.max(np.abs(std_original - std_sorted))))

print("\nPer-feature differences:")
for i, feature in enumerate(ms.FEATURES):
    print(
        f"{feature}: "
        f"mean_delta={abs(float(mean_original[i])-float(mean_sorted[i])):.10f} "
        f"std_delta={abs(float(std_original[i])-float(std_sorted[i])):.10f}"
    )

# E15-A: Frozen-checkpoint normalization sensitivity
import torch

seed = 101
ckpt = (
    ROOT / "results" / "e12" / "e12e_checkpoints" /
    f"e12e_missionshield_service_degradation_H60_seed{seed}.pt"
)

prediction_file = (
    ROOT / "results" / "e12" / "e12e_predictions" /
    f"e12e_missionshield_service_degradation_H60_seed{seed}_predictions.csv"
)

validation = base[
    (base["split"] == "validation") &
    (base["attack_family"] != HELD)
].copy()

test = base[
    (base["split"] == "test") &
    (base["attack_family"] == HELD)
].copy()

manifest = pd.concat([train, validation, test], ignore_index=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ms.MissionShieldForecaster(
    ms.build_directed_attention_mask()
).to(device)

model.load_state_dict(
    torch.load(ckpt, map_location=device, weights_only=True)
)
model.eval()

reference = pd.read_csv(prediction_file)

def predict_with_stats(mean, std):
    X, y, metadata = ms.build_sequences_from_manifest(
        manifest, "test", lookup, mean, std
    )

    ids = pd.DataFrame(metadata)["sample_id"].astype(str).tolist()
    ref_ids = reference["sample_id"].astype(str).tolist()

    if ids != ref_ids:
        raise RuntimeError("Reference sample order mismatch")

    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(X), ms.BATCH_SIZE):
            batch = torch.as_tensor(
                X[start:start + ms.BATCH_SIZE],
                dtype=torch.float32,
                device=device
            )

            logits, _, _ = model(batch)
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy().reshape(-1)
            )

    return np.concatenate(probabilities)

p_original = predict_with_stats(mean_original, std_original)
p_sorted = predict_with_stats(mean_sorted, std_sorted)

p_reference = reference["probability"].to_numpy(dtype=np.float64)

print("\n=== FROZEN CHECKPOINT NORMALIZATION SENSITIVITY ===")
print("Set-order vs saved E12-E max error:",
      float(np.max(np.abs(p_original - p_reference))))
print("Sorted-order vs saved E12-E max error:",
      float(np.max(np.abs(p_sorted - p_reference))))
print("Set-order vs sorted-order max difference:",
      float(np.max(np.abs(p_original - p_sorted))))
print("Set-order vs sorted-order mean difference:",
      float(np.mean(np.abs(p_original - p_sorted))))
print("Verification tolerance:", 2e-5)

print("\n=== E15-A ALL-SEED SORTED NORMALIZATION AUDIT ===")

for seed in [101,202,303,404,505,606,707,808,909,1010]:
    ckpt = (
        ROOT / "results" / "e12" / "e12e_checkpoints" /
        f"e12e_missionshield_service_degradation_H60_seed{seed}.pt"
    )

    prediction_file = (
        ROOT / "results" / "e12" / "e12e_predictions" /
        f"e12e_missionshield_service_degradation_H60_seed{seed}_predictions.csv"
    )

    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )
    model.eval()

    reference = pd.read_csv(prediction_file)

    probabilities = predict_with_stats(mean_sorted, std_sorted)
    saved = reference["probability"].to_numpy(dtype=np.float64)

    max_error = float(np.max(np.abs(probabilities - saved)))

    print(
        f"seed={seed:4d} "
        f"max_error={max_error:.10f} "
        f"status={'PASS' if max_error <= 2e-5 else 'FAIL'}"
    )

print("\n=== E15-A SEED 1010 ERROR DISTRIBUTION ===")

seed = 1010

ckpt = (
    ROOT / "results" / "e12" / "e12e_checkpoints" /
    f"e12e_missionshield_service_degradation_H60_seed{seed}.pt"
)

prediction_file = (
    ROOT / "results" / "e12" / "e12e_predictions" /
    f"e12e_missionshield_service_degradation_H60_seed{seed}_predictions.csv"
)

model.load_state_dict(
    torch.load(ckpt, map_location=device, weights_only=True)
)
model.eval()

reference = pd.read_csv(prediction_file)
predicted = predict_with_stats(mean_sorted, std_sorted)

audit = reference[["sample_id", "label", "probability"]].copy()
audit["reconstructed"] = predicted
audit["abs_error"] = np.abs(
    audit["reconstructed"] - audit["probability"]
)

print("Samples:", len(audit))
print("Mean absolute error:", audit["abs_error"].mean())
print("Median absolute error:", audit["abs_error"].median())
print("95th percentile:", audit["abs_error"].quantile(0.95))
print("Maximum error:", audit["abs_error"].max())
print("Samples exceeding 2e-5:",
      int((audit["abs_error"] > 2e-5).sum()))

print("\nTop 10 discrepancies:")
print(
    audit.nlargest(10, "abs_error").to_string(index=False)
)
