from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/e12")
files = (
    list((ROOT / "predictions").glob("*_predictions.csv"))
    + list((ROOT / "e12e_predictions").glob("*_predictions.csv"))
)

selected = {}

for path in sorted(files):
    df = pd.read_csv(path)

    key = (
        str(df["attack_family"].iloc[0]),
        int(df["horizon"].iloc[0])
    )

    selected.setdefault(key, path)

assert len(selected) == 9

rows = []
issues = []

for (family, horizon), path in sorted(selected.items()):
    df = pd.read_csv(path)

    positive = df.loc[
        df["label"].eq(1)
    ].copy()

    positive["failure_time"] = (
        pd.to_numeric(positive["time"])
        + pd.to_numeric(positive["steps_to_failure"])
    )

    events = positive.groupby(
        ["run_id", "failure_time"]
    ).agg(
        positive_windows=("label", "size"),
        earliest_prediction=("time", "min"),
        latest_prediction=("time", "max"),
        maximum_lead=("steps_to_failure", "max"),
        minimum_lead=("steps_to_failure", "min")
    ).reset_index()

    for _, event in events.iterrows():
        rows.append({
            "family": family,
            "horizon": horizon,
            **event.to_dict()
        })

    print(f"\n=== {family} H{horizon} ===")
    print("Positive windows:", len(positive))
    print("Reconstructed failure events:", len(events))
    print("Runs with reconstructed failures:",
          events["run_id"].nunique())

    if len(events):
        print(events.to_string(index=False))

    # Check for contradictory failure times within
    # contiguous positive prediction windows.
    for run_id, group in positive.groupby("run_id"):
        group = group.sort_values("time")
        adjacent = group["time"].diff().eq(1)
        changed = group["failure_time"].diff().ne(0)

        n = int((adjacent & changed).sum())

        if n:
            issues.append(
                f"{family} H{horizon} {run_id}: "
                f"{n} adjacent-window failure-time changes"
            )

print("\n=== E17-B2 SUMMARY ===")
print("Total reconstructed event-condition records:", len(rows))
print("Consistency issues:", len(issues))

for issue in issues[:20]:
    print(issue)

print("\nRead-only audit. No files modified.")
