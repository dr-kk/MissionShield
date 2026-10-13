from pathlib import Path
import pandas as pd

ROOT = Path("results/e17")
ROOT.mkdir(parents=True, exist_ok=True)

c1 = pd.read_csv(
    ROOT / "e17c1_operational_seed_metrics.csv"
)

c3 = pd.read_csv(
    ROOT / "e17c3_alert_episode_diagnostics.csv"
)

KEYS = ["model", "family", "horizon", "seed"]

assert len(c1) == 360
assert len(c3) == 360

assert not c1.duplicated(KEYS).any()
assert not c3.duplicated(KEYS).any()

df = c1.merge(
    c3[
        KEYS + [
            "alert_episodes",
            "negative_window_alert_runs_per_mission",
            "missions_with_negative_window_alert_runs"
        ]
    ],
    on=KEYS,
    validate="one_to_one"
)

assert len(df) == 360

metrics = [
    "positive_window_recall",
    "false_alerts_per_100_negative_windows",
    "event_detection_rate",
    "mean_first_alert_lead_steps",
    "alert_episodes",
    "negative_window_alert_runs_per_mission",
    "missions_with_negative_window_alert_runs"
]

# Equal-weight macro-average across 9 conditions,
# with 10 seeds per condition.
overall = (
    df.groupby("model")[metrics]
      .agg(["mean", "std"])
)

overall.columns = [
    f"{metric}_{stat}"
    for metric, stat in overall.columns
]

overall = overall.reset_index()

# Condition-specific mean and SD across 10 seeds.
conditions = (
    df.groupby(
        ["family", "horizon", "model"]
    )[metrics]
    .agg(["mean", "std"])
)

conditions.columns = [
    f"{metric}_{stat}"
    for metric, stat in conditions.columns
]

conditions = conditions.reset_index()

# Check each model has exactly 10 seeds
# for each family-horizon condition.
counts = df.groupby(
    ["family", "horizon", "model"]
).size()

assert len(counts) == 36
assert counts.eq(10).all()

overall.to_csv(
    ROOT / "e17d_overall_operational_table.csv",
    index=False
)

conditions.to_csv(
    ROOT / "e17d_condition_operational_table.csv",
    index=False
)

print("\n=== E17-D OVERALL TABLE ===")

cols = [
    "model",
    "positive_window_recall_mean",
    "false_alerts_per_100_negative_windows_mean",
    "event_detection_rate_mean",
    "negative_window_alert_runs_per_mission_mean"
]

print(
    overall[cols].round(4).to_string(index=False)
)

print("\n=== E17-D SERVICE DEGRADATION H60 ===")

subset = conditions[
    (conditions.family == "service_degradation")
    & (conditions.horizon == 60)
]

print(
    subset[
        [
            "model",
            "positive_window_recall_mean",
            "false_alerts_per_100_negative_windows_mean",
            "event_detection_rate_mean",
            "negative_window_alert_runs_per_mission_mean"
        ]
    ].round(4).to_string(index=False)
)

print("\n=== VALIDATION ===")
print("Merged evaluations:", len(df))
print("Condition-model groups:", len(counts))
print("Seeds per group:", counts.unique().tolist())

print("\nSaved:")
print("e17d_overall_operational_table.csv")
print("e17d_condition_operational_table.csv")
print("\nFrozen E1-E16 unchanged.")
