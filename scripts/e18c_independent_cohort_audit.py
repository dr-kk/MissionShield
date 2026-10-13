import pandas as pd
import numpy as np

R = "results/"

cohort = pd.read_csv(R + "e8_independent_cohort.csv")
pred = pd.read_csv(R + "e8_confirmation_predictions.csv")
inter = pd.read_csv(R + "e8_confirmation_intervention_results.csv")
forecast = pd.read_csv(R + "e8_confirmation_forecasting_summary.csv")

print("\n=== E18-C COHORT AUDIT ===")
print("Unique missions:", cohort.run_id.nunique())
print("Unique seeds:", cohort.seed.nunique())
print("Attack-family mission counts:")
print(
    cohort[["run_id", "attack_family"]]
    .drop_duplicates()
    .groupby("attack_family").size()
    .to_string()
)

assert cohort.run_id.nunique() == 180
assert cohort.seed.nunique() == 180

print("\n=== PREDICTION AUDIT ===")
assert not pred.duplicated(["sample_id"]).any()

for H, g in pred.groupby("horizon"):
    expected = forecast.loc[
        forecast.horizon.eq(H)
    ].iloc[0]

    assert len(g) == int(expected.samples)
    assert g.probability.between(0, 1).all()
    assert g.probability_calibrated.between(0, 1).all()

    calculated_alarm = (
        g.probability_calibrated >=
        float(expected.threshold_frozen)
    ).astype(int)

    assert np.array_equal(
        calculated_alarm.to_numpy(),
        g.alarm_fixed_far.to_numpy()
    )

    print(
        f"H{H}: samples={len(g)}, "
        f"missions={g.run_id.nunique()}, "
        f"alarm_rate={g.alarm_fixed_far.mean():.6f}"
    )

print("\n=== INTERVENTION AUDIT ===")
for H, g in inter.groupby("horizon"):
    assert len(g) == 180
    assert g.run_id.nunique() == 180

    prevented = (
        (g.baseline_failure == 1) &
        (g.post_policy_failure == 0)
    ).sum()

    induced = (
        (g.baseline_failure == 0) &
        (g.post_policy_failure == 1)
    ).sum()

    print(
        f"H{H}: baseline={g.baseline_failure.sum()}, "
        f"post={g.post_policy_failure.sum()}, "
        f"prevented={prevented}, induced={induced}"
    )

print("\nPASS: E18-C internal consistency checks.")
print("Existing files unchanged.")
