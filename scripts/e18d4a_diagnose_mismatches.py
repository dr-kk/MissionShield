import numpy as np
import pandas as pd

R = "results/"

pred = pd.read_csv(R + "e8_confirmation_predictions.csv")
inter = pd.read_csv(R + "e8_confirmation_intervention_results.csv")

eligible = ["edge_processing", "fusion", "command"]

errors = []
target_errors = []
time_errors = []
trigger_errors = []

for H, records in inter.groupby("horizon"):

    p = pred[pred.horizon == H]

    first = (
        p[p.alarm_fixed_far == 1]
        .sort_values(["run_id", "time"])
        .groupby("run_id", as_index=False)
        .first()
        .set_index("run_id")
    )

    for r in records.itertuples(index=False):

        rid = str(r.run_id)
        triggered = rid in first.index

        if int(r.triggered) != int(triggered):
            trigger_errors.append((H, rid))
            continue

        if not triggered:
            continue

        alarm = first.loc[rid]

        if int(r.trigger_time) != int(alarm.time):
            time_errors.append((H, rid))

        selected = max(
            eligible,
            key=lambda n: (
                float(alarm[f"importance_{n}"]),
                -eligible.index(n)
            )
        )

        if str(r.observable_target) != selected:
            target_errors.append((H, rid, r.observable_target, selected))

        for n in eligible:
            actual = float(getattr(r, f"alarm_importance_{n}"))
            expected = float(alarm[f"importance_{n}"])

            errors.append({
                "horizon": H,
                "run_id": rid,
                "node": n,
                "actual": actual,
                "expected": expected,
                "absolute_error": abs(actual - expected),
                "relative_error": (
                    abs(actual - expected) /
                    max(abs(expected), 1e-15)
                )
            })

err = pd.DataFrame(errors)

print("\n=== E18-D4A NUMERICAL DIAGNOSTIC ===")
print("Importance comparisons:", len(err))

print("\nAbsolute error distribution:")
print(
    err.absolute_error.describe(
        percentiles=[.50, .90, .95, .99]
    ).to_string()
)

print("\nRelative error distribution:")
print(
    err.relative_error.describe(
        percentiles=[.50, .90, .95, .99]
    ).to_string()
)

print("\nMismatch counts by tolerance:")
for tol in [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]:
    print(
        f"atol={tol:.0e}:",
        int((err.absolute_error > tol).sum())
    )

print("\nLargest differences:")
print(
    err.nlargest(10, "absolute_error")
    .to_string(index=False)
)

print("\nOther integrity errors:")
print("Trigger mismatches:", len(trigger_errors))
print("Alarm-time mismatches:", len(time_errors))
print("Target-selection mismatches:", len(target_errors))

if target_errors:
    print("Target examples:", target_errors[:10])

print("\nDiagnostic complete. Original files unchanged.")
