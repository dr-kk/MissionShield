from pathlib import Path
import pandas as pd

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

all_events = set()
data = {}

for key, path in selected.items():
    df = pd.read_csv(path)

    df["time"] = pd.to_numeric(df["time"])
    df["steps_to_failure"] = pd.to_numeric(
        df["steps_to_failure"], errors="coerce"
    )

    positive = df[df["label"].eq(1)].copy()

    positive["failure_time"] = (
        positive["time"] +
        positive["steps_to_failure"]
    )

    events = set(
        zip(
            positive["run_id"],
            positive["failure_time"]
        )
    )

    data[key] = (df, events)

    for run_id, failure_time in events:
        all_events.add(
            (key[0], run_id, failure_time)
        )

print("\n=== E17-B3 EVENT COVERAGE AUDIT ===")

rows = []

for family, run_id, failure_time in sorted(all_events):
    for horizon in [10, 30, 60]:
        df, events = data[(family, horizon)]

        run = df[df["run_id"].eq(run_id)]

        if run.empty:
            status = "RUN_NOT_PRESENT"
            first_time = None
            last_time = None
        else:
            first_time = run["time"].min()
            last_time = run["time"].max()

            if (run_id, failure_time) in events:
                status = "EVENT_REPRESENTED"
            elif first_time >= failure_time:
                status = "AFTER_EVENT"
            elif last_time < failure_time - horizon:
                status = "BEFORE_WARNING_WINDOW"
            else:
                status = "NOT_REPRESENTED_REVIEW"

        rows.append({
            "family": family,
            "run_id": run_id,
            "failure_time": failure_time,
            "horizon": horizon,
            "first_prediction_time": first_time,
            "last_prediction_time": last_time,
            "status": status
        })

result = pd.DataFrame(rows)

print(result.to_string(index=False))

print("\n=== COVERAGE SUMMARY ===")
print(
    result.groupby(
        ["horizon", "status"]
    ).size().to_string()
)

print("\nRead-only audit. No files modified.")
