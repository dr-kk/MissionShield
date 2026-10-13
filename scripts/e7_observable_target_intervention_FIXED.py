from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT=Path(r"D:\MissionShield")
SCRIPTS=ROOT/"scripts"
RESULTS=ROOT/"results"
E0=SCRIPTS/"e0_missionshield_setup_v2.py"
E5B=SCRIPTS/"e5b_counterfactual_intervention_emulator.py"
PROTO=RESULTS/"e5a_intervention_protocol.json"
HORIZONS=[10,30,60]
ELIGIBLE_RESTORE=["edge_processing","fusion","command"]

def load_module(name,path):
    if not path.exists(): raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

e0=load_module("e0v2_e7",E0)
e5b=load_module("e5b_e7",E5B)

def simulate_explicit_target(run_id,attack,seed,action,intervention_time,targets):
    """Exact E5-B dynamics, except target nodes are supplied explicitly."""
    rng=np.random.default_rng(seed)
    onset,severity=e0.generate_run_profile(attack,rng)
    latent={n:float(rng.uniform(0.0,0.03)) for n in e0.BASE_NODES}
    disturbance={"remaining":0,"strength":0.0}
    task_history=[]; timeline=[]
    for t in range(e0.STEPS_PER_RUN):
        level=e0.attack_intensity(attack,t,onset,severity)
        benign=e0.benign_disturbance(t,rng,disturbance)
        active=(action!="no_action" and t>=intervention_time)
        latent=e5b.update_damage_cf(
            latent,attack,level,benign,rng,action,targets,active
        )
        for node in e0.BASE_NODES:
            e0.generate_telemetry(node,attack,level,benign,latent[node],rng)
        p=e0.mission_task_probability(latent,rng)
        if active:
            if action=="rate_limit": p-=0.005
            elif action=="reroute": p-=0.010
            elif action=="isolate": p-=0.025
            elif action=="restore_service": p-=0.005
            elif action=="safe_mode": p-=0.020
        p=float(np.clip(p,0.02,0.995))
        tasks=rng.binomial(1,p,size=8)
        task_history.extend(tasks.tolist())
        recent=task_history[-e0.MISSION_WINDOW*8:]
        me=float(np.mean(recent))
        failure=int(me<e0.MISSION_CRIT)
        timeline.append((t,me,failure))
    return onset,severity,pd.DataFrame(
        timeline,columns=["time","mission_effectiveness","failure"]
    )

def choose_observable_target(pred_row):
    # pred_row comes from itertuples(), so access named fields as attributes.
    vals={n:float(getattr(pred_row,f"importance_{n}")) for n in ELIGIBLE_RESTORE}
    # deterministic tie break follows ELIGIBLE_RESTORE order
    return max(ELIGIBLE_RESTORE,key=lambda n:(vals[n],-ELIGIBLE_RESTORE.index(n))), vals

def main():
    print("="*104)
    print("MissionShield E7 | OBSERVABLE-TARGET INTERVENTION INTEGRITY CORRECTION")
    print("Target = highest MissionShield node importance at first fixed-FAR alarm")
    print("No attack-family/attack-onset information used for target selection")
    print("="*104)

    with open(PROTO,encoding="utf-8") as f: proto=json.load(f)
    costs={x["action"]:float(x["cost"]) for x in proto["actions"]}

    original=pd.read_csv(e0.PROC_DIR/"missionshield_e0_v2_dataset.csv")
    run_info=(original[["run_id","seed","attack_family","attack_onset"]]
              .drop_duplicates("run_id").set_index("run_id"))

    rows=[]
    for H in HORIZONS:
        audit=pd.read_csv(RESULTS/"e5c_trigger_audit.csv")
        audit=audit[audit.horizon==H].copy()
        if audit.empty: raise RuntimeError(f"No E5-C rows for H={H}")
        actions=audit["selected_action"].dropna().unique()
        if len(actions)!=1: raise RuntimeError(f"H={H}: expected one selected action, got {actions}")
        action=str(actions[0])
        if action!="restore_service":
            raise RuntimeError(f"E7 explicit observable target currently frozen for restore_service; got {action}")

        pred=pd.read_csv(RESULTS/f"e3b_missionshield_H{H}_operational_predictions.csv")
        needed={"run_id","time","alarm_fixed_far"} | {f"importance_{n}" for n in ELIGIBLE_RESTORE}
        missing=needed-set(pred.columns)
        if missing:
            raise RuntimeError(
                f"H={H}: operational prediction file lacks observable target columns {sorted(missing)}. "
                "Do not substitute ground truth."
            )

        # first alarm row is the only source for target selection
        alarms=(pred[pred.alarm_fixed_far.astype(int)==1]
                .sort_values(["run_id","time"]).groupby("run_id",as_index=False).first())
        alarm_map={str(r.run_id):r for r in alarms.itertuples(index=False)}

        for r in audit.itertuples(index=False):
            rid=str(r.run_id)
            if rid not in run_info.index: raise RuntimeError(f"Unknown run {rid}")
            info=run_info.loc[rid]
            seed=int(info.seed); attack=str(info.attack_family)

            # Exact no-action replay and integrity check.
            onset,_,base=simulate_explicit_target(rid,attack,seed,"no_action",0,[])
            orig=(original[original.run_id==rid][["time","mission_effectiveness","failure"]]
                  .drop_duplicates("time").sort_values("time").reset_index(drop=True))
            if not np.allclose(base.mission_effectiveness,orig.mission_effectiveness,atol=1e-12):
                raise RuntimeError(f"A0 mission-effectiveness mismatch: {rid}")
            if not np.array_equal(base.failure.to_numpy(),orig.failure.to_numpy()):
                raise RuntimeError(f"A0 failure mismatch: {rid}")

            triggered=int(r.triggered)
            if triggered:
                if rid not in alarm_map:
                    raise RuntimeError(f"E5-C says triggered but no E3-B alarm row: H={H}, {rid}")
                ar=alarm_map[rid]
                t0=int(ar.time)
                target,imps=choose_observable_target(ar)
                _,_,cf=simulate_explicit_target(rid,attack,seed,action,t0,[target])
                incurred=costs[action]
            else:
                t0=np.nan; target="none"
                imps={n:np.nan for n in ELIGIBLE_RESTORE}
                cf=base.copy(); incurred=0.0

            base_fail=int(base.failure.max())
            cf_fail=int(cf.failure.max())
            rows.append({
                "horizon":H,"run_id":rid,
                "triggered":triggered,"trigger_time":t0,
                "selected_action":action if triggered else "no_action",
                "observable_target":target,
                **{f"alarm_importance_{n}":imps[n] for n in ELIGIBLE_RESTORE},
                "baseline_failure":base_fail,
                "post_policy_failure":cf_fail,
                "failure_prevented":int(base_fail==1 and cf_fail==0),
                "failure_induced":int(base_fail==0 and cf_fail==1),
                "baseline_mean_effectiveness":float(base.mission_effectiveness.mean()),
                "post_policy_mean_effectiveness":float(cf.mission_effectiveness.mean()),
                "mission_effectiveness_gain":float(
                    cf.mission_effectiveness.mean()-base.mission_effectiveness.mean()
                ),
                "incurred_cost":incurred,
                # Evaluation-only metadata, never used by choose_observable_target.
                "attack_family_eval_only":attack,
                "attack_onset_eval_only":int(info.attack_onset),
                "simulated_onset_integrity":int(onset),
            })

    out=pd.DataFrame(rows)
    detail=RESULTS/"e7_observable_target_intervention_results.csv"
    out.to_csv(detail,index=False)

    summaries=[]
    for H,g in out.groupby("horizon"):
        bf=int(g.baseline_failure.sum()); pf=int(g.post_policy_failure.sum())
        summaries.append({
            "horizon":int(H),
            "test_runs":len(g),
            "triggered_runs":int(g.triggered.sum()),
            "baseline_failures":bf,
            "post_policy_failures":pf,
            "failures_prevented":int(g.failure_prevented.sum()),
            "failures_induced":int(g.failure_induced.sum()),
            "failure_rate_before":float(g.baseline_failure.mean()),
            "failure_rate_after":float(g.post_policy_failure.mean()),
            "net_failure_rate_reduction":float(g.baseline_failure.mean()-g.post_policy_failure.mean()),
            "prevention_rate_among_baseline_failures":float(g.failure_prevented.sum()/bf) if bf else np.nan,
            "mean_mission_effectiveness_gain":float(g.mission_effectiveness_gain.mean()),
            "mean_incurred_cost":float(g.incurred_cost.mean()),
        })
    summary=pd.DataFrame(summaries)
    sf=RESULTS/"e7_observable_target_intervention_summary.csv"
    summary.to_csv(sf,index=False)

    target_summary=(out[out.triggered==1]
        .groupby(["horizon","observable_target"],as_index=False)
        .agg(triggered_runs=("run_id","count"),
             baseline_failures=("baseline_failure","sum"),
             failures_prevented=("failure_prevented","sum"),
             failures_induced=("failure_induced","sum"),
             mean_effectiveness_gain=("mission_effectiveness_gain","mean")))
    tf=RESULTS/"e7_observable_target_summary.csv"
    target_summary.to_csv(tf,index=False)

    print("\nPRIMARY SUMMARY")
    print(summary.to_string(index=False))
    print("\nOBSERVABLE TARGET DISTRIBUTION/OUTCOMES")
    print(target_summary.to_string(index=False))
    print("\nSaved:")
    print(" ",detail); print(" ",sf); print(" ",tf)
    print("\nE7 COMPLETE")
    print("This is a development-stage integrity correction, not a pristine confirmatory test.")
    print("Freeze the observable target rule before any independent-cohort confirmation.")

if __name__=="__main__":
    main()
