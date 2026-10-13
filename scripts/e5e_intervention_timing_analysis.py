from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\MissionShield")
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed" / "missionshield_e0_v2_dataset.csv"

HORIZONS = [10, 30, 60]

def first_failure_times():
    df = pd.read_csv(DATA, usecols=[
        "run_id", "time", "failure", "attack_onset", "attack_family"
    ])
    tl = (df.drop_duplicates(["run_id", "time"])
            .sort_values(["run_id", "time"]))

    first_fail = (tl[tl["failure"].astype(int) == 1]
                  .groupby("run_id", as_index=False)["time"]
                  .min()
                  .rename(columns={"time": "first_failure_time"}))

    run_info = (tl[["run_id", "attack_onset", "attack_family"]]
                .drop_duplicates("run_id"))

    return run_info.merge(first_fail, on="run_id", how="left")

def main():
    src = RESULTS / "e5d_actual_trigger_counterfactual_results.csv"
    if not src.exists():
        raise FileNotFoundError(src)

    d = pd.read_csv(src)
    truth = first_failure_times()

    d = d.merge(
        truth[["run_id", "first_failure_time"]],
        on="run_id",
        how="left"
    )

    # Timing quantities are retrospective evaluation only.
    d["trigger_lead_to_failure"] = np.where(
        (d["triggered"] == 1) & d["first_failure_time"].notna(),
        d["first_failure_time"] - d["trigger_time"],
        np.nan
    )

    d["trigger_delay_from_attack_onset"] = np.where(
        d["triggered"] == 1,
        d["trigger_time"] - d["attack_onset_eval_only"],
        np.nan
    )

    d["trigger_before_attack_onset"] = np.where(
        d["triggered"] == 1,
        (d["trigger_time"] < d["attack_onset_eval_only"]).astype(int),
        np.nan
    )

    d["trigger_before_first_failure"] = np.where(
        (d["triggered"] == 1) & d["first_failure_time"].notna(),
        (d["trigger_time"] < d["first_failure_time"]).astype(int),
        np.nan
    )

    # Primary analysis: baseline-failing runs only.
    f = d[d["baseline_failure"].astype(int) == 1].copy()

    rows = []
    for H in HORIZONS:
        h = f[f["horizon"] == H].copy()
        for status, name in [(1, "prevented"), (0, "not_prevented")]:
            g = h[h["failure_prevented"].astype(int) == status]
            lead = g["trigger_lead_to_failure"].dropna()
            delay = g["trigger_delay_from_attack_onset"].dropna()

            rows.append({
                "horizon": H,
                "outcome": name,
                "baseline_failing_runs": len(g),
                "triggered_runs": int(g["triggered"].sum()),
                "median_lead_to_failure": float(lead.median()) if len(lead) else np.nan,
                "mean_lead_to_failure": float(lead.mean()) if len(lead) else np.nan,
                "min_lead_to_failure": float(lead.min()) if len(lead) else np.nan,
                "max_lead_to_failure": float(lead.max()) if len(lead) else np.nan,
                "median_delay_from_attack_onset": float(delay.median()) if len(delay) else np.nan,
                "mean_delay_from_attack_onset": float(delay.mean()) if len(delay) else np.nan,
            })

    summary = pd.DataFrame(rows)

    # Run-level table for transparent inspection.
    cols = [
        "horizon", "run_id", "attack_family_eval_only",
        "attack_onset_eval_only", "first_failure_time",
        "triggered", "trigger_time", "trigger_probability",
        "trigger_delay_from_attack_onset",
        "trigger_lead_to_failure",
        "trigger_before_attack_onset",
        "trigger_before_first_failure",
        "selected_action", "baseline_failure",
        "post_policy_failure", "failure_prevented",
        "failure_induced", "mission_effectiveness_gain",
        "incurred_cost"
    ]
    runlevel = f[cols].sort_values(
        ["horizon", "failure_prevented", "run_id"],
        ascending=[True, False, True]
    )

    # Compact cross-horizon diagnostic.
    diagnostic = (f.groupby("horizon", as_index=False)
        .agg(
            failing_runs=("run_id", "count"),
            triggered_failing_runs=("triggered", "sum"),
            prevented=("failure_prevented", "sum"),
            median_trigger_lead=("trigger_lead_to_failure", "median"),
            mean_trigger_lead=("trigger_lead_to_failure", "mean"),
            median_trigger_delay=("trigger_delay_from_attack_onset", "median"),
            mean_trigger_delay=("trigger_delay_from_attack_onset", "mean"),
        ))

    p1 = RESULTS / "e5e_intervention_timing_runlevel.csv"
    p2 = RESULTS / "e5e_intervention_timing_summary.csv"
    p3 = RESULTS / "e5e_intervention_timing_diagnostic.csv"

    runlevel.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)
    diagnostic.to_csv(p3, index=False)

    print("=" * 100)
    print("MissionShield E5-E | INTERVENTION TIMING ANALYSIS")
    print("RETROSPECTIVE DIAGNOSTIC ONLY | NO POLICY/THRESHOLD/ACTION TUNING")
    print("=" * 100)

    print("\nPRIMARY: PREVENTED vs NOT PREVENTED")
    print(summary.to_string(index=False))

    print("\nCROSS-HORIZON DIAGNOSTIC")
    print(diagnostic.to_string(index=False))

    print("\nRUN-LEVEL FAILING CASES")
    print(runlevel[[
        "horizon", "run_id", "attack_family_eval_only",
        "attack_onset_eval_only", "trigger_time",
        "first_failure_time", "trigger_lead_to_failure",
        "trigger_delay_from_attack_onset", "failure_prevented"
    ]].to_string(index=False))

    print("\nSaved:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)

    print("\nE5-E COMPLETE")
    print("Interpret timing descriptively; n=2 prevented cases per horizon is too small")
    print("for a strong inferential claim. Do not tune the intervention from this analysis.")

if __name__ == "__main__":
    main()
