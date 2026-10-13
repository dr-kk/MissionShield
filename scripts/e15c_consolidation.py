from pathlib import Path
import pandas as pd

root = Path("results/e15")

b1 = pd.read_csv(root / "e15b1_feature_masking.csv")
b2 = pd.read_csv(root / "e15b2_node_masking.csv")
b3 = pd.read_csv(root / "e15b3_bounded_noise.csv")

assert len(b1) == 60
assert len(b2) == 60
assert len(b3) == 150

b1["delta_auprc"] = b1.perturbed_auprc - b1.baseline_auprc
b1["delta_auroc"] = b1.perturbed_auroc - b1.baseline_auroc

def summarize(df, group):
    return df.groupby(group).agg(
        mean_shift=("mean_abs_probability_shift", "mean"),
        sd_shift=("mean_abs_probability_shift", "std"),
        mean_delta_auprc=("delta_auprc", "mean"),
        sd_delta_auprc=("delta_auprc", "std"),
        mean_delta_auroc=("delta_auroc", "mean"),
        sd_delta_auroc=("delta_auroc", "std")
    ).reset_index()

feature = summarize(b1, "feature")
node = summarize(b2, "node")

# Noise replicates are averaged within seed before
# across-seed statistics are calculated.
per_seed = b3.groupby(
    ["seed", "noise_sigma"], as_index=False
).agg(
    mean_abs_probability_shift=("mean_abs_probability_shift", "mean"),
    delta_auprc=("delta_auprc", "mean"),
    delta_auroc=("delta_auroc", "mean")
)

noise = summarize(per_seed, "noise_sigma")

outputs = {
    "e15c_feature_summary.csv": feature,
    "e15c_node_summary.csv": node,
    "e15c_noise_summary.csv": noise
}

for filename, data in outputs.items():
    path = root / filename
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")
    data.to_csv(path, index=False)
    print("Saved:", path)

print("\n=== E15-C FINAL CONSOLIDATION ===")
print("\nFEATURE MASKING")
print(feature.round(6).to_string(index=False))
print("\nNODE MASKING")
print(node.round(6).to_string(index=False))
print("\nNOISE SENSITIVITY")
print(noise.round(6).to_string(index=False))
