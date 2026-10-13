from pathlib import Path
import pandas as pd

ROOT = Path("results")
OUT = ROOT / "e19"
OUT.mkdir(parents=True, exist_ok=True)

sources = {
    "forecasting": ROOT / "e16/e16c_full_model_summary.csv",
    "comparison": ROOT / "e16/e16c_auprc_comparison_table.csv",
    "operational": ROOT / "e17/e17d_overall_operational_table.csv",
    "operational_conditions": ROOT / "e17/e17d_condition_operational_table.csv",
    "independent": ROOT / "e8_confirmation_forecasting_summary.csv",
    "intervention": ROOT / "e8_confirmation_intervention_summary.csv",
    "features": ROOT / "e15/e15c_feature_summary.csv",
    "nodes": ROOT / "e15/e15c_node_summary.csv",
    "noise": ROOT / "e15/e15c_noise_summary.csv",
}

print("\n=== E19-B1 SOURCE VALIDATION ===")

for name, path in sources.items():
    df = pd.read_csv(path)
    print(f"{name:24} rows={len(df):4} columns={len(df.columns):3}")
    print("  ", ", ".join(df.columns))

    if df.empty:
        raise RuntimeError(f"Empty source: {path}")

    df.to_csv(OUT / f"table_{name}.csv", index=False)

print("\n=== E19-B1 OUTPUT ===")
print("Publication working tables:", len(sources))
print("Output directory:", OUT.resolve())
print("Frozen E1-E18 results unchanged.")
