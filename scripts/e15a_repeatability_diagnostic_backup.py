import sys
import numpy as np
import pandas as pd
import torch

from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

import e1_missionshield_forecaster as ms

ROOT = Path("results")
SEED = 101
HELD = "service_degradation"
H = 60

df = pd.read_csv(ms.DATA)
lookup = ms.build_telemetry_lookup(df)
mask = ms.build_directed_attention_mask()

base = pd.read_csv(ROOT / f"e1_final_samples_H{H}.csv")

tr = base[
    (base["split"] == "train") &
    (base["attack_family"] != HELD)
].copy()

va = base[
    (base["split"] == "validation") &
    (base["attack_family"] != HELD)
].copy()

te = base[
    (base["split"] == "test") &
    (base["attack_family"] == HELD)
].copy()

manifest = pd.concat([tr, va, te], ignore_index=True)

mean, std = ms.calculate_normalization(manifest, lookup)

Xt, yt, meta = ms.build_sequences_from_manifest(
    manifest, "test", lookup, mean, std
)

name = f"e12e_missionshield_{HELD}_H{H}_seed{SEED}"

ckpt = ROOT / "e12" / "e12e_checkpoints" / f"{name}.pt"
pred = ROOT / "e12" / "e12e_predictions" / f"{name}_predictions.csv"

ref = pd.read_csv(pred)

assert ref["sample_id"].tolist() == [
    m["sample_id"] for m in meta
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ms.MissionShieldForecaster(mask).to(device)

model.load_state_dict(
    torch.load(ckpt, map_location=device, weights_only=True)
)

model.eval()

X = torch.as_tensor(
    np.ascontiguousarray(Xt),
    dtype=torch.float32
)

runs = []

print("\nBatch configuration:")
print("ms.BATCH_SIZE =", ms.BATCH_SIZE)
print("Xt.shape =", Xt.shape)
print("torch =", torch.__version__)
print("CUDA =", torch.version.cuda)
print("TF32 matmul =", torch.backends.cuda.matmul.allow_tf32)
print("TF32 cuDNN =", torch.backends.cudnn.allow_tf32)


with torch.inference_mode():
    for repeat in range(3):
        probs = []

        for start in range(0, len(X), ms.BATCH_SIZE):
            xb = X[start:start + ms.BATCH_SIZE].to(device)
            logits, _, _ = model(xb)

            probs.append(
                torch.sigmoid(logits)
                .reshape(-1)
                .cpu()
                .numpy()
            )

        p = np.concatenate(probs)
        runs.append(p)

        error = np.abs(
            p - ref["probability"].to_numpy(dtype=float)
        )

        print(
            f"Repeat {repeat + 1}: "
            f"max_reference_error={error.max():.10f} "
            f"mean_reference_error={error.mean():.10f}"
        )

for i in range(1, 3):
    diff = np.abs(runs[0] - runs[i])

    print(
        f"Repeat 1 vs {i + 1}: "
        f"max_repeat_error={diff.max():.10f}"
    )

errors = np.abs(
    runs[0] - ref["probability"].to_numpy(dtype=float)
)

worst = np.argsort(errors)[-10:][::-1]

print("\nTop 10 reference discrepancies:")

for idx in worst:
    print(
        f"sample_id={ref.iloc[idx]['sample_id']} "
        f"label={ref.iloc[idx]['label']} "
        f"reference={ref.iloc[idx]['probability']:.8f} "
        f"current={runs[0][idx]:.8f} "
        f"error={errors[idx]:.8f}"
    )


print("\n=== BATCH-SIZE SENSITIVITY ===")

reference = ref["probability"].to_numpy(dtype=float)
results = {}

with torch.inference_mode():
    for bs in [1, 32, 128, 256, 405]:
        chunks = []

        for start in range(0, len(X), bs):
            xb = X[start:start + bs].to(device)
            logits, _, _ = model(xb)
            chunks.append(
                torch.sigmoid(logits).reshape(-1).cpu().numpy()
            )

        results[bs] = np.concatenate(chunks)
        err = np.abs(results[bs] - reference)

        print(
            f"batch={bs} "
            f"max_reference_error={err.max():.10f} "
            f"mean_reference_error={err.mean():.10f}",
            flush=True
        )

print("\n=== SAME-SAMPLE BATCH COMPARISON ===")

for bs in [1, 32, 128, 405]:
    diff = np.abs(results[bs] - results[256])

    print(
        f"batch={bs} vs 256 "
        f"max_difference={diff.max():.10f}",
        flush=True
    )
