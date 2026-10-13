import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r"D:\MissionShield")
SCRIPTS=ROOT/"scripts"
RESULTS=ROOT/"results"

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

e0=load_module("e0v2",SCRIPTS/"e0_missionshield_setup_v2.py")
e5b=load_module("e5b",SCRIPTS/"e5b_counterfactual_intervention_emulator.py")

HORIZONS=[10,30,60]

def run_info_from_original():
    df=pd.read_csv(e0.PROC_DIR/"missionshield_e0_v2_dataset.csv")
    info=(df[["run_id","seed","attack_family","attack_onset"]]
          .drop_duplicates("run_id"))
    return df,info

def main():
    audit=pd.read_csv(RESULTS/"e5c_trigger_audit.csv")
    policy=pd.read_csv(RESULTS/"e5c_validation_action_policy.csv")
    original,info=run_info_from_original()

    print("="*100)
    print("MissionShield E5-D | ACTUAL FORECAST-TRIGGER COUNTERFACTUAL REPLAY")
    print("FROZEN E5-C ACTION | ACTUAL MISSIONShield ALARM TIME | NO TEST TUNING")
    print("="*100)

    rows=[]
    for H in HORIZONS:
        selected=policy[(policy.horizon==H)&(policy.selected==1)]
        if len(selected)!=1:
            raise RuntimeError(f"H={H}: expected exactly one frozen selected action")
        action=str(selected.iloc[0].action)
        cost=float(selected.iloc[0].cost)

        a=audit[audit.horizon==H].copy()
        print(f"\nH={H} | frozen action={action} | test runs={len(a)}")

        for r in a.itertuples():
            z=info[info.run_id==r.run_id]
            if z.empty:
                raise RuntimeError(f"Missing run info {r.run_id}")
            z=z.iloc[0]
            seed=int(z.seed); attack=str(z.attack_family)
            onset=int(z.attack_onset)

            # Exact A0 reference
            _,_,_,base=e5b.simulate_counterfactual(
                r.run_id,attack,seed,"no_action",onset)

            triggered=int(r.triggered)
            if triggered:
                t0=int(r.trigger_time)
                _,_,targets,cf=e5b.simulate_counterfactual(
                    r.run_id,attack,seed,action,t0)
                s=e5b.summarize_pair(
                    r.run_id,attack,seed,action,cost,t0,base,cf,targets)
            else:
                t0=np.nan
                targets=[]
                cf=base.copy()
                # No alarm means no intervention and zero incurred cost.
                s=e5b.summarize_pair(
                    r.run_id,attack,seed,"no_action",0.0,
                    e0.STEPS_PER_RUN,base,cf,targets)

            # Whole-run ground truth; evaluation only, never used for action choice.
            original_run=(original[original.run_id==r.run_id][
                ["time","mission_effectiveness","failure"]]
                .drop_duplicates("time").sort_values("time"))
            original_failure=int(original_run.failure.max())

            cf_failure=int(cf.failure.max())
            base_failure=int(base.failure.max())

            rows.append({
                "horizon":H,
                "run_id":r.run_id,
                "attack_family_eval_only":attack,
                "attack_onset_eval_only":onset,
                "triggered":triggered,
                "trigger_time":t0,
                "trigger_probability":getattr(r,"trigger_probability",np.nan),
                "selected_action":action if triggered else "no_action",
                "incurred_cost":cost if triggered else 0.0,
                "baseline_failure":base_failure,
                "post_policy_failure":cf_failure,
                "failure_prevented":int(base_failure==1 and cf_failure==0),
                "failure_induced":int(base_failure==0 and cf_failure==1),
                "mission_effectiveness_gain":float(
                    cf.mission_effectiveness.mean()-base.mission_effectiveness.mean()),
                "a0_matches_original":int(base_failure==original_failure),
            })

    out=pd.DataFrame(rows)
    if out.a0_matches_original.min()!=1:
        raise RuntimeError("A0 replay integrity failure.")

    detail=RESULTS/"e5d_actual_trigger_counterfactual_results.csv"
    out.to_csv(detail,index=False)

    summary=(out.groupby("horizon",as_index=False)
      .agg(test_runs=("run_id","count"),
           triggered_runs=("triggered","sum"),
           baseline_failures=("baseline_failure","sum"),
           post_policy_failures=("post_policy_failure","sum"),
           failures_prevented=("failure_prevented","sum"),
           failures_induced=("failure_induced","sum"),
           mean_effectiveness_gain=("mission_effectiveness_gain","mean"),
           mean_incurred_cost=("incurred_cost","mean")))
    summary["trigger_rate"]=summary.triggered_runs/summary.test_runs
    summary["failure_rate_before"]=summary.baseline_failures/summary.test_runs
    summary["failure_rate_after"]=summary.post_policy_failures/summary.test_runs
    summary["net_failure_rate_reduction"]=(
        summary.failure_rate_before-summary.failure_rate_after)
    summary["prevention_among_baseline_failures"]=np.where(
        summary.baseline_failures>0,
        summary.failures_prevented/summary.baseline_failures,np.nan)

    sf=RESULTS/"e5d_actual_trigger_counterfactual_summary.csv"
    summary.to_csv(sf,index=False)

    family=(out.groupby(["horizon","attack_family_eval_only"],as_index=False)
      .agg(runs=("run_id","count"),
           triggered=("triggered","sum"),
           baseline_failures=("baseline_failure","sum"),
           post_failures=("post_policy_failure","sum"),
           prevented=("failure_prevented","sum"),
           induced=("failure_induced","sum"),
           mean_gain=("mission_effectiveness_gain","mean")))
    ff=RESULTS/"e5d_actual_trigger_family_summary.csv"
    family.to_csv(ff,index=False)

    print("\n"+"="*100)
    print("E5-D PRIMARY SUMMARY")
    print("="*100)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(" ",detail)
    print(" ",sf)
    print(" ",ff)
    print("\nE5-D COMPLETE")
    print("These are actual forecast-trigger-time counterfactual results under the frozen E5-C policy.")

if __name__=="__main__":
    main()
