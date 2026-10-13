from pathlib import Path
import importlib.util, gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT=Path(r"D:\MissionShield")
SCRIPTS=ROOT/"scripts"; RESULTS=ROOT/"results"; CHECKPOINTS=ROOT/"checkpoints"
E0=SCRIPTS/"e0_missionshield_setup_v2.py"
E1=SCRIPTS/"e1_missionshield_forecaster.py"
E5B=SCRIPTS/"e5b_counterfactual_intervention_emulator.py"

HORIZONS=[10,30,60]
N_RUNS_PER_FAMILY=30
CONFIRM_BASE_SEED=1_000_042
ELIGIBLE_RESTORE=["edge_processing","fusion","command"]
ACTION="restore_service"
ACTION_COST=0.25

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
e0=loadmod("e0_e8",E0); e1=loadmod("e1_e8",E1); e5b=loadmod("e5b_e8",E5B)

def temperature_apply(p,T):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    z=np.log(p/(1-p))
    return 1/(1+np.exp(-z/float(T)))

def make_confirmation_cohort():
    rows=[]
    for fi,attack in enumerate(e0.ATTACK_FAMILIES):
        for r in range(N_RUNS_PER_FAMILY):
            seed=CONFIRM_BASE_SEED+fi*10000+r
            run_id=f"e8_{attack}_{r:03d}"
            rows.extend(e0.simulate_run(run_id,attack,seed))
    df=pd.DataFrame(rows)
    return e0.add_future_labels(df)

def make_manifest(df,H):
    timeline=(df[["run_id","time","attack_family","failure",f"failure_within_{H}"]]
              .drop_duplicates(["run_id","time"]).sort_values(["run_id","time"]))
    rows=[]
    for rid,g in timeline.groupby("run_id",sort=False):
        g=g.sort_values("time").reset_index(drop=True)
        failures=g.failure.to_numpy(int)
        fam=str(g.attack_family.iloc[0])
        for i in range(len(g)):
            t=int(g.time.iloc[i])
            hs=t-e1.HISTORY+1
            if hs<0 or i+H>=len(g): continue
            # exact frozen eligibility: current history failure-free
            hist=g[(g.time>=hs)&(g.time<=t)]
            if len(hist)!=e1.HISTORY or int(hist.failure.max())!=0: continue
            future=failures[i+1:i+H+1]
            label=int(future.max())
            future_fail=np.where(future==1)[0]
            stf=int(future_fail[0]+1) if len(future_fail) else np.nan
            rows.append({
                "sample_id":f"E8_H{H}_{rid}_{t}",
                "run_id":rid,"time":t,"history_start_time":hs,
                "attack_family":fam,"horizon":H,"label":label,
                "steps_to_failure":stf,"split":"test"
            })
    return pd.DataFrame(rows)

def simulate_explicit_target(attack,seed,intervention_time,target):
    rng=np.random.default_rng(seed)
    onset,severity=e0.generate_run_profile(attack,rng)
    latent={n:float(rng.uniform(0.0,0.03)) for n in e0.BASE_NODES}
    disturbance={"remaining":0,"strength":0.0}; task_history=[]; timeline=[]
    for t in range(e0.STEPS_PER_RUN):
        level=e0.attack_intensity(attack,t,onset,severity)
        benign=e0.benign_disturbance(t,rng,disturbance)
        active=(t>=intervention_time)
        latent=e5b.update_damage_cf(
            latent,attack,level,benign,rng,ACTION,[target],active)
        for node in e0.BASE_NODES:
            e0.generate_telemetry(node,attack,level,benign,latent[node],rng)
        p=e0.mission_task_probability(latent,rng)
        if active: p-=0.005
        p=float(np.clip(p,0.02,0.995))
        tasks=rng.binomial(1,p,size=8); task_history.extend(tasks.tolist())
        me=float(np.mean(task_history[-e0.MISSION_WINDOW*8:]))
        timeline.append((t,me,int(me<e0.MISSION_CRIT)))
    return onset,pd.DataFrame(timeline,columns=["time","mission_effectiveness","failure"])

def main():
    print("="*108)
    print("MissionShield E8 | INDEPENDENT-COHORT CONFIRMATION")
    print("Fresh disjoint scenario seeds | frozen model/checkpoints | frozen calibration/FAR thresholds")
    print("Frozen policy: first fixed-FAR alarm -> highest eligible node importance -> restore_service")
    print("="*108)

    # Guard against accidental seed overlap with E0-v2.
    old=pd.read_csv(e0.PROC_DIR/"missionshield_e0_v2_dataset.csv",usecols=["seed"]).seed.unique()
    new_seeds=[CONFIRM_BASE_SEED+fi*10000+r
               for fi in range(len(e0.ATTACK_FAMILIES))
               for r in range(N_RUNS_PER_FAMILY)]
    if set(map(int,old)) & set(new_seeds):
        raise RuntimeError("Confirmation seeds overlap development cohort.")

    print("Generating fresh confirmation cohort...")
    df=make_confirmation_cohort()
    cohort_file=RESULTS/"e8_independent_cohort.csv"
    df.to_csv(cohort_file,index=False)
    print(f"Runs: {df.run_id.nunique()} | node rows: {len(df):,}")

    # Frozen training normalization is recomputed ONLY from original frozen training histories.
    old_df=pd.read_csv(e1.DATA)
    old_lookup=e1.build_telemetry_lookup(old_df)
    new_lookup=e1.build_telemetry_lookup(df)

    op=pd.read_csv(RESULTS/"e3b_operational_summary.csv")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    pred_all=[]; intervention_all=[]; forecast_summary=[]

    run_info=(df[["run_id","seed","attack_family","attack_onset"]]
              .drop_duplicates("run_id").set_index("run_id"))

    for H in HORIZONS:
        print("\n"+"-"*108); print("HORIZON",H); print("-"*108)
        old_manifest=pd.read_csv(RESULTS/f"e1_final_samples_H{H}.csv")
        old_manifest["run_id"]=old_manifest.run_id.astype(str)
        mean,std=e1.calculate_normalization(old_manifest,old_lookup)

        man=make_manifest(df,H)
        X,y,meta=e1.build_sequences_from_manifest(man,"test",new_lookup,mean,std)
        loader=DataLoader(e1.MissionDataset(X,y),batch_size=256,shuffle=False,num_workers=0)

        model=e1.MissionShieldForecaster(
            directed_mask=e1.build_directed_attention_mask()).to(device)
        ck=CHECKPOINTS/f"e1_missionshield_H{H}.pt"
        if not ck.exists(): raise FileNotFoundError(ck)
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True))
        ty,p,node_w,gates=e1.predict(model,loader,device,return_explanations=True)

        rr=op[(op.model=="MissionShield")&(op.horizon==H)]
        if len(rr)!=1: raise RuntimeError(f"Missing frozen E3-B operating point H={H}")
        T=float(rr.iloc[0].temperature)
        thr=float(rr.iloc[0].threshold_calibrated)
        pc=temperature_apply(p,T)
        alarm=(pc>=thr).astype(int)

        pred=pd.DataFrame(meta)
        pred["label"]=ty.astype(int); pred["probability"]=p
        pred["probability_calibrated"]=pc; pred["alarm_fixed_far"]=alarm
        pred["fusion_temporal_gate"]=gates
        for j,n in enumerate(e1.NODES): pred[f"importance_{n}"]=node_w[:,j]
        pred["horizon"]=H
        pred_all.append(pred)

        # Descriptive confirmation forecasting metrics.
        from sklearn.metrics import roc_auc_score,average_precision_score
        forecast_summary.append({
            "horizon":H,"samples":len(pred),"positive_rate":float(ty.mean()),
            "auroc":float(roc_auc_score(ty,p)),
            "auprc":float(average_precision_score(ty,p)),
            "alarm_rate":float(alarm.mean()),
            "temperature_frozen":T,"threshold_frozen":thr
        })

        first=(pred[pred.alarm_fixed_far==1].sort_values(["run_id","time"])
               .groupby("run_id",as_index=False).first())
        fmap={str(r.run_id):r for r in first.itertuples(index=False)}

        for rid,info in run_info.iterrows():
            attack=str(info.attack_family); seed=int(info.seed)
            base=(df[df.run_id==rid][["time","mission_effectiveness","failure"]]
                  .drop_duplicates("time").sort_values("time").reset_index(drop=True))
            base_fail=int(base.failure.max())

            if rid in fmap:
                ar=fmap[rid]; t0=int(ar.time)
                vals={n:float(getattr(ar,f"importance_{n}")) for n in ELIGIBLE_RESTORE}
                target=max(ELIGIBLE_RESTORE,
                           key=lambda n:(vals[n],-ELIGIBLE_RESTORE.index(n)))
                onset,cf=simulate_explicit_target(attack,seed,t0,target)
                # Verify generated profile onset is exactly the cohort's onset.
                if int(onset)!=int(info.attack_onset):
                    raise RuntimeError(f"Onset replay mismatch {rid}")
                triggered=1; cost=ACTION_COST
            else:
                t0=np.nan; target="none"; vals={n:np.nan for n in ELIGIBLE_RESTORE}
                cf=base.copy(); triggered=0; cost=0.0

            cf_fail=int(cf.failure.max())
            intervention_all.append({
                "horizon":H,"run_id":rid,"triggered":triggered,
                "trigger_time":t0,"observable_target":target,
                **{f"alarm_importance_{n}":vals[n] for n in ELIGIBLE_RESTORE},
                "baseline_failure":base_fail,"post_policy_failure":cf_fail,
                "failure_prevented":int(base_fail==1 and cf_fail==0),
                "failure_induced":int(base_fail==0 and cf_fail==1),
                "baseline_mean_effectiveness":float(base.mission_effectiveness.mean()),
                "post_policy_mean_effectiveness":float(cf.mission_effectiveness.mean()),
                "mission_effectiveness_gain":float(cf.mission_effectiveness.mean()
                                                  -base.mission_effectiveness.mean()),
                "incurred_cost":cost,
                "attack_family_eval_only":attack,
                "attack_onset_eval_only":int(info.attack_onset)
            })
        del X,y,loader,model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    preds=pd.concat(pred_all,ignore_index=True)
    detail=pd.DataFrame(intervention_all)
    fsum=pd.DataFrame(forecast_summary)

    summary=[]
    for H,g in detail.groupby("horizon"):
        bf=int(g.baseline_failure.sum()); pf=int(g.post_policy_failure.sum())
        summary.append({
            "horizon":int(H),"runs":len(g),"triggered_runs":int(g.triggered.sum()),
            "baseline_failures":bf,"post_policy_failures":pf,
            "failures_prevented":int(g.failure_prevented.sum()),
            "failures_induced":int(g.failure_induced.sum()),
            "failure_rate_before":float(g.baseline_failure.mean()),
            "failure_rate_after":float(g.post_policy_failure.mean()),
            "net_failure_rate_reduction":float(g.baseline_failure.mean()-g.post_policy_failure.mean()),
            "prevention_rate_among_baseline_failures":
                float(g.failure_prevented.sum()/bf) if bf else np.nan,
            "mean_effectiveness_gain":float(g.mission_effectiveness_gain.mean()),
            "mean_incurred_cost":float(g.incurred_cost.mean())
        })
    isum=pd.DataFrame(summary)

    preds.to_csv(RESULTS/"e8_confirmation_predictions.csv",index=False)
    detail.to_csv(RESULTS/"e8_confirmation_intervention_results.csv",index=False)
    fsum.to_csv(RESULTS/"e8_confirmation_forecasting_summary.csv",index=False)
    isum.to_csv(RESULTS/"e8_confirmation_intervention_summary.csv",index=False)

    print("\nCONFIRMATION FORECASTING")
    print(fsum.to_string(index=False))
    print("\nCONFIRMATION INTERVENTION")
    print(isum.to_string(index=False))
    print("\nTARGET DISTRIBUTION")
    print(detail[detail.triggered==1].groupby(
        ["horizon","observable_target"]).size().rename("runs").reset_index().to_string(index=False))
    print("\nE8 COMPLETE")
    print("No E8 outcome was used to alter model, calibration, threshold, action type, target rule, or dynamics.")

if __name__=="__main__":
    main()
