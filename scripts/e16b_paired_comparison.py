import pandas as pd

df = pd.read_csv("results/e12/e13/e13_seed_metrics.csv")

key = ["family", "horizon", "seed"]

wide = df.pivot(
    index=key,
    columns="model",
    values="auprc"
).reset_index()

assert len(wide) == 90
assert not wide.isna().any().any()

baselines = ["graphwave", "lstm", "patch"]

rows = []

for (family, horizon), group in wide.groupby(
    ["family", "horizon"]
):
    for baseline in baselines:
        delta = (
            group["MissionShield"] - group[baseline]
        )

        rows.append({
            "family": family,
            "horizon": horizon,
            "baseline": baseline,
            "mean_delta_auprc": delta.mean(),
            "sd_delta_auprc": delta.std(),
            "positive_seed_pairs": int((delta > 0).sum()),
            "negative_seed_pairs": int((delta < 0).sum())
        })

result = pd.DataFrame(rows)

print("\n=== E16-B PAIRED SEED COMPARISON ===")
print(result.round(6).to_string(index=False))

print("\n=== OVERALL DESCRIPTIVE DIFFERENCES ===")
print(
    result.groupby("baseline")[
        "mean_delta_auprc"
    ].mean().round(6)
)

print(
    "\nNOTE: Seed pairing is descriptive; "
    "shared test missions limit independence."
)
