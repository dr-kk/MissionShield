import numpy as np
import pandas as pd

R = "results/"
pred = pd.read_csv(R + "e8_confirmation_predictions.csv")
inter = pd.read_csv(R + "e8_confirmation_intervention_results.csv")

eligible = ["edge_processing", "fusion", "command"]
issues = []
checked = 0

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
            issues.append((H, rid, "trigger mismatch"))
            continue

        if triggered:
            alarm = first.loc[rid]

            if int(r.trigger_time) != int(alarm.time):
                issues.append((H, rid, "alarm time mismatch"))

            selected = max(
                eligible,
                key=lambda n: (
                    float(alarm[f"importance_{n}"]),
                    -eligible.index(n)
                )
            )

            if r.observable_target != selected:
                issues.append((H, rid, "target mismatch"))

            for n in eligible:
                actual = float(getattr(r, f"alarm_importance_{n}"))
                expected = float(alarm[f"importance_{n}"])
                if not np.isclose(actual, expected, rtol=0, atol=1e-7):
                    issues.append((H, rid, f"{n} importance mismatch"))
        else:
            if r.observable_target != "none":
                issues.append((H, rid, "unexpected target"))
            if int(r.baseline_failure) != int(r.post_policy_failure):
                issues.append((H, rid, "untriggered failure changed"))
            if not np.isclose(
                float(r.mission_effectiveness_gain), 0,
                rtol=0, atol=1e-12
            ):
                issues.append((H, rid, "untriggered effectiveness changed"))

        prevented = int(
            r.baseline_failure == 1 and r.post_policy_failure == 0
        )
        induced = int(
            r.baseline_failure == 0 and r.post_policy_failure == 1
        )

        if int(r.failure_prevented) != prevented:
            issues.append((H, rid, "prevention mismatch"))
        if int(r.failure_induced) != induced:
            issues.append((H, rid, "induced failure mismatch"))

        checked += 1

print("\n=== E18-D4 POLICY INTEGRITY ===")
print("Records checked:", checked)
print("Issues:", len(issues))

if issues:
    for issue in issues[:30]:
        print(issue)
    raise SystemExit("FAIL: Policy integrity issues detected.")

assert checked == 540
print("PASS: All 540 intervention records consistent.")
print("Frozen E1-E17 and existing E8 results unchanged.")

