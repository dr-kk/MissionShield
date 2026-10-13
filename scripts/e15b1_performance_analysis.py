from pathlib import Path
import pandas as pd

path = Path("results/e15/e15b1_feature_masking.csv")
df = pd.read_csv(path)

assert len(df) == 60
assert df.groupby("seed")["feature"].nunique().eq(6).all()

df["delta_auprc"] = (
    df["perturbed_auprc"] - df["baseline_auprc"]
)

df["delta_auroc"] = (
    df["perturbed_auroc"] - df["baseline_auroc"]
)

summary = df.groupby("feature").agg(
    mean_probability_shift=("mean_abs_probability_shift", "mean"),
    mean_delta_auprc=("delta_auprc", "mean"),
    sd_delta_auprc=("delta_auprc", "std"),
    mean_delta_auroc=("delta_auroc", "mean"),
    sd_delta_auroc=("delta_auroc", "std"),
    seeds_auprc_decreased=("delta_auprc", lambda x: int((x < 0).sum())),
    seeds_auprc_increased=("delta_auprc", lambda x: int((x > 0).sum()))
).sort_values("mean_probability_shift", ascending=False)

print("\n=== E15-B1 PERFORMANCE IMPACT ===")
print(summary.round(6).to_string())

print("\n=== BASELINE PERFORMANCE ACROSS SEEDS ===")
baseline = df.drop_duplicates("seed")
print("Mean baseline AUPRC:", baseline["baseline_auprc"].mean())
print("Mean baseline AUROC:", baseline["baseline_auroc"].mean())

print("\n=== SEED 1010 EXCLUSION SENSITIVITY ===")
print(
    df[df["seed"] != 1010]
    .groupby("feature")["delta_auprc"]
    .mean()
    .sort_values()
    .round(6)
    .to_string()
)
