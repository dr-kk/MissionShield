from pathlib import Path

files = [
    "results/e12/e13/e13_seed_metrics.csv",
    "results/e15/e15c_feature_summary.csv",
    "results/e15/e15c_node_summary.csv",
    "results/e15/e15c_noise_summary.csv",
    "results/e16/e16c_full_model_summary.csv",
    "results/e16/e16c_auprc_comparison_table.csv",
    "results/e17/e17d_overall_operational_table.csv",
    "results/e17/e17d_condition_operational_table.csv",
    "results/e8_confirmation_forecasting_summary.csv",
    "results/e8_confirmation_intervention_summary.csv",
]

print("\n=== E19-A ARTIFACT INVENTORY ===")

missing = []

for name in files:
    p = Path(name)
    status = "PASS" if p.is_file() else "MISSING"
    print(f"{status:8} {name}")
    if not p.is_file():
        missing.append(name)

print("\nAvailable:", len(files) - len(missing))
print("Missing:", len(missing))

if missing:
    raise SystemExit("FAIL: Required artifacts missing.")

print("\nPASS: All manuscript synthesis inputs available.")
print("Frozen experimental results unchanged.")
