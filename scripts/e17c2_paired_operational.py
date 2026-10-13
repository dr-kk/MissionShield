import pandas as pd

path = "results/e17/e17c1_operational_seed_metrics.csv"
df = pd.read_csv(path)

metrics = [
    "positive_window_recall",
    "false_alerts_per_100_negative_windows",
    "event_detection_rate"
]

base = df[df.model == "MissionShield"]

rows = []

for competitor in ["graphwave", "lstm", "patch"]:
    other = df[df.model == competitor]

    paired = base.merge(
        other,
        on=["family", "horizon", "seed"],
        suffixes=("_ms", "_baseline"),
        validate="one_to_one"
    )

    assert len(paired) == 90

    for metric in metrics:
        delta = (
            paired[f"{metric}_ms"]
            - paired[f"{metric}_baseline"]
        )

        rows.append({
            "competitor": competitor,
            "metric": metric,
            "mean_difference": delta.mean(),
            "median_difference": delta.median(),
            "missionShield_higher": int((delta > 0).sum()),
            "missionShield_lower": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
            "pairs": len(delta)
        })

result = pd.DataFrame(rows)

print("\n=== E17-C2 PAIRED COMPARISON ===")
print(result.round(4).to_string(index=False))

result.to_csv(
    "results/e17/e17c2_paired_operational_comparison.csv",
    index=False
)

print("\nSaved E17-C2. Frozen results unchanged.")
