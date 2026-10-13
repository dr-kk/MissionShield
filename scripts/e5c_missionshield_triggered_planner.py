from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\MissionShield")
RESULTS = ROOT / "results"

HORIZONS = [10, 30, 60]
ACTIONS = ["rate_limit", "reroute", "isolate", "restore_service", "safe_mode"]

def first_alarm_table(H):
    p = RESULTS / f"e3b_missionshield_H{H}_operational_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    req = {"run_id","time","alarm_fixed_far","probability_calibrated","label"}
    miss = req-set(df.columns)
    if miss:
        raise RuntimeError(f"{p.name} missing {sorted(miss)}")

    # E3-B file already contains the validation-frozen calibrated decision.
    alarm = df[df["alarm_fixed_far"].astype(int)==1].copy()
    first = (alarm.sort_values(["run_id","time"])
                  .groupby("run_id",as_index=False).first())
    return first[["run_id","time","probability_calibrated"]].rename(
        columns={"time":"trigger_time","probability_calibrated":"trigger_probability"})

def choose_action_validation_only(e5b, train_runs, val_runs):
    """
    E5-C v1 deliberately uses a GLOBAL action policy.
    It does not use attack_family, attack_onset, test labels, or test outcomes.

    Action selection:
      maximize validation mean mission-effectiveness gain - intervention cost.
    Cost weight is fixed a priori at 0.05 (not tuned on test).
    """
    LAMBDA_COST = 0.05
    dev = e5b[e5b["run_id"].isin(set(train_runs)|set(val_runs))].copy()
    dev = dev[dev["action"].isin(ACTIONS)]
    s=(dev.groupby("action",as_index=False)
       .agg(mean_gain=("mission_effectiveness_gain","mean"),
            cost=("intervention_cost","first"),
            failure_prevention=("failure_prevented","mean")))
    s["utility"]=s["mean_gain"]-LAMBDA_COST*s["cost"]
    s=s.sort_values(["utility","failure_prevention"],ascending=False).reset_index(drop=True)
    return s.iloc[0]["action"], s, LAMBDA_COST

def nearest_counterfactual(e5b, run_id, action):
    """
    E5-B was replayed at attack onset. It cannot validly provide a new
    counterfactual for an arbitrary later trigger time. Therefore E5-C
    reports planner *eligibility/coverage* and uses E5-B only as an
    action-efficacy reference. It does NOT relabel E5-B onset outcomes as
    MissionShield-triggered causal outcomes.
    """
    g=e5b[(e5b.run_id==run_id)&(e5b.action==action)]
    return None if g.empty else g.iloc[0]

def main():
    e5b_file=RESULTS/"e5b_counterfactual_intervention_results.csv"
    e5b=pd.read_csv(e5b_file)

    print("="*96)
    print("MissionShield E5-C | VALIDATION-ONLY ACTION POLICY + FORECAST TRIGGER AUDIT")
    print("NO ATTACK-FAMILY INPUT | NO TRUE ATTACK-ONSET INPUT | NO TEST ACTION TUNING")
    print("="*96)

    all_rows=[]
    policy_rows=[]

    for H in HORIZONS:
        manifest=pd.read_csv(RESULTS/f"e1_final_samples_H{H}.csv")
        manifest["run_id"]=manifest["run_id"].astype(str)
        split_runs={s:manifest.loc[manifest.split==s,"run_id"].unique().tolist()
                    for s in ["train","validation","test"]}

        action, scores, lam=choose_action_validation_only(
            e5b,split_runs["train"],split_runs["validation"])
        scores.insert(0,"horizon",H)
        scores["selected"]=(scores.action==action).astype(int)
        scores["lambda_cost"]=lam
        policy_rows.append(scores)

        alarms=first_alarm_table(H)
        test_runs=pd.DataFrame({"run_id":split_runs["test"]})
        test_runs=test_runs.merge(alarms,on="run_id",how="left")
        test_runs["horizon"]=H
        test_runs["selected_action"]=action
        test_runs["triggered"]=test_runs["trigger_time"].notna().astype(int)

        # Ground truth is retained only for retrospective evaluation, never action choice.
        truth=(e5b[["run_id","attack_family","baseline_post_failure"]]
               .drop_duplicates("run_id"))
        test_runs=test_runs.merge(truth,on="run_id",how="left")

        # E5-B onset efficacy is explicitly marked as ORACLE-ONSET REFERENCE.
        refs=[]
        for r in test_runs.itertuples():
            z=nearest_counterfactual(e5b,r.run_id,action)
            refs.append({
                "e5b_oracle_onset_post_failure": np.nan if z is None else z.post_intervention_failure,
                "e5b_oracle_onset_failure_prevented": np.nan if z is None else z.failure_prevented,
                "e5b_oracle_onset_effectiveness_gain": np.nan if z is None else z.mission_effectiveness_gain,
            })
        test_runs=pd.concat([test_runs,pd.DataFrame(refs)],axis=1)
        all_rows.append(test_runs)

        n=len(test_runs); trig=int(test_runs.triggered.sum())
        failing=test_runs[test_runs.baseline_post_failure==1]
        detected=int(failing.triggered.sum()) if len(failing) else 0
        print(f"\nH={H}")
        print(f"Selected validation-only global action : {action}")
        print(f"Test runs                            : {n}")
        print(f"Runs triggering intervention         : {trig}/{n} ({trig/n:.3f})")
        if len(failing):
            print(f"Failing runs triggered               : {detected}/{len(failing)} ({detected/len(failing):.3f})")
        print("NOTE: causal outcome at actual trigger time is NOT claimed in E5-C.")

    policy=pd.concat(policy_rows,ignore_index=True)
    audit=pd.concat(all_rows,ignore_index=True)

    p1=RESULTS/"e5c_validation_action_policy.csv"
    p2=RESULTS/"e5c_trigger_audit.csv"
    policy.to_csv(p1,index=False)
    audit.to_csv(p2,index=False)

    print("\n"+"="*96)
    print("E5-C COMPLETE")
    print("="*96)
    print("Saved:")
    print(" ",p1)
    print(" ",p2)
    print("\nScientific boundary:")
    print("E5-C freezes a telemetry-triggered planner policy and audits when it fires.")
    print("E5-D must replay the emulator from each ACTUAL MissionShield trigger time")
    print("before claiming MissionShield-triggered failure prevention.")

if __name__=="__main__":
    main()
