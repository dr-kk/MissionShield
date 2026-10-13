import pandas as pd
from pathlib import Path

path = Path("results/e12/e13/e13_seed_metrics.csv")
df = pd.read_csv(path)

print("\n=== E16-A DATASET VERIFICATION ===")
print("Total rows:", len(df))
print("Models:", sorted(df["model"].unique()))
print("Attack families:", sorted(df["family"].unique()))
print("Horizons:", sorted(df["horizon"].unique()))
print("Seeds:", sorted(df["seed"].unique()))

print("\n=== MODEL PERFORMANCE ===")
summary = df.groupby("model").agg(
    evaluations=("auprc", "count"),
    mean_auprc=("auprc", "mean"),
    sd_auprc=("auprc", "std"),
    mean_auroc=("auroc", "mean"),
    sd_auroc=("auroc", "std")
)

print(summary.round(6).to_string())

print("\n=== PERFORMANCE BY ATTACK FAMILY AND HORIZON ===")
comparison = df.pivot_table(
    index=["family", "horizon"],
    columns="model",
    values="auprc",
    aggfunc="mean"
)

print(comparison.round(6).to_string())

print("\n=== BEST MODEL PER CONDITION ===")
print(comparison.idxmax(axis=1).to_string())

print("\n=== CONDITION WINS ===")
print(comparison.idxmax(axis=1).value_counts().to_string())

print("\n=== DATA INTEGRITY ===")
print("Missing metrics:", int(df[["auprc", "auroc"]].isna().sum().sum()))
print(
    "Duplicate model/family/horizon/seed:",
    int(df.duplicated(
        ["model", "family", "horizon", "seed"]
    ).sum())
)
