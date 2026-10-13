from pathlib import Path
import pandas as pd

ROOT = Path("results/e16")
ROOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results/e12/e13/e13_seed_metrics.csv")

models = ["MissionShield", "graphwave", "lstm", "patch"]

assert len(df) == 360
assert set(df["model"]) == set(models)
assert not df.duplicated(
    ["model", "family", "horizon", "seed"]
).any()

summary = df.groupby(
    ["family", "horizon", "model"], as_index=False
).agg(
    n_seeds=("seed", "nunique"),
    mean_auprc=("auprc", "mean"),
    sd_auprc=("auprc", "std"),
    mean_auroc=("auroc", "mean"),
    sd_auroc=("auroc", "std")
)

assert len(summary) == 36
assert (summary["n_seeds"] == 10).all()

summary["auprc_display"] = summary.apply(
    lambda r: f'{r.mean_auprc:.4f} ± {r.sd_auprc:.4f}',
    axis=1
)

summary["auroc_display"] = summary.apply(
    lambda r: f'{r.mean_auroc:.4f} ± {r.sd_auroc:.4f}',
    axis=1
)

wide = summary.pivot(
    index=["family", "horizon"],
    columns="model",
    values="auprc_display"
).reset_index()

best = summary.loc[
    summary.groupby(["family", "horizon"])[
        "mean_auprc"
    ].idxmax(),
    ["family", "horizon", "model"]
].rename(columns={"model": "best_model"})

wide = wide.merge(
    best, on=["family", "horizon"]
)

wide = wide[
    ["family", "horizon"] + models + ["best_model"]
]

summary_path = ROOT / "e16c_full_model_summary.csv"
table_path = ROOT / "e16c_auprc_comparison_table.csv"

for path in [summary_path, table_path]:
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

summary.to_csv(summary_path, index=False)
wide.to_csv(table_path, index=False)

print("\n=== E16-C AUPRC COMPARISON TABLE ===")
print(wide.to_string(index=False))

print("\nSaved:", summary_path)
print("Saved:", table_path)
