import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r"D:\MissionShield")
SCRIPTS=ROOT/"scripts"
RESULTS=ROOT/"results"
RESULTS.mkdir(parents=True,exist_ok=True)

E0=SCRIPTS/"e0_missionshield_setup_v2.py"
PROTO=RESULTS/"e5a_intervention_protocol.json"

def load_module(path):
    spec=importlib.util.spec_from_file_location("e0v2",path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

e0=load_module(E0)

# Frozen E5-B intervention dynamics. These are explicit emulator mechanisms,
# not post-hoc edits of mission_effectiveness or failure labels.
#
# Each action changes pressure/damage dynamics from intervention_time onward.
# Mission outcomes are then regenerated from the resulting latent state.
#
# A0: no change
# A1 rate_limit: suppress network attack pressure, small utilization overhead
# A2 reroute: strongly suppress tactical-link pressure but adds path overhead
# A3 isolate: remove direct attack pressure at selected node, temporary service penalty
# A4 restore_service: reduce accumulated latent damage at selected service node
# A5 safe_mode: reduce propagation/direct pressure globally with mission-service penalty

def intervention_targets(action, attack):
    attacked=e0.target_nodes(attack)
    if action=="rate_limit": return ["tactical_link"]
    if action=="reroute": return ["tactical_link"]
    if action=="isolate":
        candidates=[n for n in attacked if n!="actuator"]
        return candidates[:1] if candidates else ["edge_processing"]
    if action=="restore_service":
        candidates=[n for n in attacked if n in ["edge_processing","fusion","command"]]
        return candidates[:1] if candidates else ["edge_processing"]
    if action=="safe_mode": return list(e0.BASE_NODES)
    return []

def update_damage_cf(previous,attack,attack_level,benign_level,rng,
                     action,target_nodes,active):
    new={}
    attacked=e0.target_nodes(attack)
    for idx,node in enumerate(e0.BASE_NODES):
        direct=attack_level if node in attacked else 0.0
        upstream=previous[e0.BASE_NODES[idx-1]] if idx>0 else 0.0
        service_penalty=0.0

        if active:
            if action=="rate_limit" and node=="tactical_link":
                direct*=0.45
                service_penalty=0.015
            elif action=="reroute" and node=="tactical_link":
                direct*=0.20
                service_penalty=0.025
            elif action=="isolate" and node in target_nodes:
                direct*=0.10
                upstream*=0.20
                service_penalty=0.055
            elif action=="restore_service" and node in target_nodes:
                # recovery is implemented on accumulated state, not outcome label
                previous_node=previous[node]*0.45
            else:
                previous_node=previous[node]

            if action=="safe_mode":
                direct*=0.55
                upstream*=0.65
                service_penalty+=0.025
        else:
            previous_node=previous[node]

        if not (active and action=="restore_service" and node in target_nodes):
            previous_node=previous[node]

        damage=(0.82*previous_node+0.12*direct+0.08*upstream)
        damage+=0.035*benign_level+service_penalty+rng.normal(0.0,0.012)
        if direct<0.05: damage-=0.025
        new[node]=float(np.clip(damage,0.0,1.0))
    return new

def simulate_counterfactual(run_id,attack,seed,action,intervention_time):
    rng=np.random.default_rng(seed)
    onset,severity=e0.generate_run_profile(attack,rng)
    latent={n:float(rng.uniform(0.0,0.03)) for n in e0.BASE_NODES}
    disturbance={"remaining":0,"strength":0.0}
    task_history=[]; timeline=[]
    targets=intervention_targets(action,attack)

    for t in range(e0.STEPS_PER_RUN):
        level=e0.attack_intensity(attack,t,onset,severity)
        benign=e0.benign_disturbance(t,rng,disturbance)
        active=(action!="no_action" and t>=intervention_time)
        latent=update_damage_cf(latent,attack,level,benign,rng,action,targets,active)

        # Consume telemetry RNG exactly as the original generator does.
        # This keeps stochastic stream structure comparable.
        for node in e0.BASE_NODES:
            e0.generate_telemetry(node,attack,level,benign,latent[node],rng)

        p=e0.mission_task_probability(latent,rng)

        # Explicit service-capacity penalty: interventions can protect the mission
        # while still imposing operational disruption.
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
    return onset,severity,targets,pd.DataFrame(timeline,columns=["time","mission_effectiveness","failure"])

def summarize_pair(run_id,attack,seed,action,cost,t0,base,cf,targets):
    pre=base[base.time<t0]
    post_b=base[base.time>=t0]
    post_c=cf[cf.time>=t0]
    base_fail=int(post_b.failure.max()) if len(post_b) else 0
    cf_fail=int(post_c.failure.max()) if len(post_c) else 0

    # Recovery: first post-intervention time at which a failing counterfactual
    # returns to >= threshold; NaN if never needed/never recovered.
    recovery=np.nan
    fail_times=post_c.loc[post_c.failure==1,"time"]
    if len(fail_times):
        first=int(fail_times.iloc[0])
        recovered=post_c[(post_c.time>first)&(post_c.failure==0)]
        if len(recovered): recovery=int(recovered.time.iloc[0]-first)

    return {
        "run_id":run_id,"attack_family":attack,"seed":seed,
        "action":action,"targets":";".join(targets),
        "intervention_time":t0,"intervention_cost":cost,
        "baseline_post_failure":base_fail,
        "post_intervention_failure":cf_fail,
        "failure_prevented":int(base_fail==1 and cf_fail==0),
        "mission_effectiveness_before":float(pre.mission_effectiveness.mean()) if len(pre) else np.nan,
        "baseline_post_effectiveness":float(post_b.mission_effectiveness.mean()),
        "mission_effectiveness_after":float(post_c.mission_effectiveness.mean()),
        "mission_effectiveness_gain":float(post_c.mission_effectiveness.mean()-post_b.mission_effectiveness.mean()),
        "recovery_time_steps":recovery,
    }

def main():
    with open(PROTO,encoding="utf-8") as f: proto=json.load(f)
    actions={x["action"]:float(x["cost"]) for x in proto["actions"]}

    original=pd.read_csv(e0.PROC_DIR/"missionshield_e0_v2_dataset.csv")
    run_info=(original[["run_id","seed","attack_family","attack_onset"]]
              .drop_duplicates("run_id").sort_values("run_id"))

    # E5-B is a mechanism-validation experiment, not planner evaluation.
    # Replay all non-benign runs. Intervention starts at attack onset so that
    # action efficacy can be characterized independently of forecaster timing.
    run_info=run_info[run_info.attack_family!="benign"].copy()

    print("="*96)
    print("MissionShield E5-B | COUNTERFACTUAL INTERVENTION EMULATOR")
    print("SAME SEED/SCENARIO | LATENT DYNAMICS RECOMPUTED | NO PLANNER TUNING")
    print("="*96)
    print("Runs:",len(run_info),"| Actions:",len(actions))

    rows=[]
    for i,r in enumerate(run_info.itertuples(),1):
        # Generate reference through the same CF engine with A0.
        onset,severity,_,base=simulate_counterfactual(
            r.run_id,r.attack_family,int(r.seed),"no_action",int(r.attack_onset))

        # Strong integrity check: A0 must reproduce original mission timeline.
        orig=(original[original.run_id==r.run_id][
            ["time","mission_effectiveness","failure"]].drop_duplicates("time")
            .sort_values("time").reset_index(drop=True))
        if not np.allclose(base.mission_effectiveness,orig.mission_effectiveness,atol=1e-12):
            raise RuntimeError(f"A0 mission-effectiveness replay mismatch: {r.run_id}")
        if not np.array_equal(base.failure.to_numpy(),orig.failure.to_numpy()):
            raise RuntimeError(f"A0 failure replay mismatch: {r.run_id}")

        t0=int(r.attack_onset)
        for action,cost in actions.items():
            if action=="no_action":
                cf=base.copy(); targets=[]
            else:
                _,_,targets,cf=simulate_counterfactual(
                    r.run_id,r.attack_family,int(r.seed),action,t0)
            rows.append(summarize_pair(
                r.run_id,r.attack_family,int(r.seed),action,cost,t0,base,cf,targets))

        if i%10==0 or i==len(run_info):
            print(f"Processed {i}/{len(run_info)} runs")

    out=pd.DataFrame(rows)
    detail=RESULTS/"e5b_counterfactual_intervention_results.csv"
    out.to_csv(detail,index=False)

    summary=(out.groupby(["attack_family","action"],as_index=False)
        .agg(runs=("run_id","nunique"),
             baseline_failure_rate=("baseline_post_failure","mean"),
             post_failure_rate=("post_intervention_failure","mean"),
             failure_prevention_rate=("failure_prevented","mean"),
             mean_effectiveness_gain=("mission_effectiveness_gain","mean"),
             intervention_cost=("intervention_cost","first"),
             median_recovery_steps=("recovery_time_steps","median")))
    summary["net_failure_rate_reduction"]=(
        summary["baseline_failure_rate"]-summary["post_failure_rate"])
    summary_file=RESULTS/"e5b_counterfactual_intervention_summary.csv"
    summary.to_csv(summary_file,index=False)

    print("\n"+"="*96)
    print("E5-B SUMMARY")
    print("="*96)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(" ",detail)
    print(" ",summary_file)
    print("\nE5-B COMPLETE")

if __name__=="__main__":
    main()
