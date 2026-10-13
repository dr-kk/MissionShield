from pathlib import Path
import importlib.util, gc
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(r"D:\MissionShield"); SCRIPTS=ROOT/"scripts"; RESULTS=ROOT/"results"; CHECKPOINTS=ROOT/"checkpoints"
E0=SCRIPTS/"e0_missionshield_setup_v2.py"; E1=SCRIPTS/"e1_missionshield_forecaster.py"
HORIZONS=[10,30,60]; N_RUNS=30; BASE_SEED=2_000_042

# Same six observable nodes/features. Only latent dependency propagation changes.
# The trained MissionShield graph remains the ORIGINAL chain: this is zero-shot stress testing.
TOPOLOGIES={
 "chain_control": {
   "sensor":[], "tactical_link":["sensor"], "edge_processing":["tactical_link"],
   "fusion":["edge_processing"], "command":["fusion"], "actuator":["command"]},
 "bypass_redundancy": {
   "sensor":[], "tactical_link":["sensor"], "edge_processing":["tactical_link"],
   "fusion":["edge_processing","tactical_link"], "command":["fusion"], "actuator":["command"]},
 "dual_parent_command": {
   "sensor":[], "tactical_link":["sensor"], "edge_processing":["tactical_link"],
   "fusion":["edge_processing"], "command":["fusion","edge_processing"], "actuator":["command"]},
}

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
e0=loadmod("e0_e9",E0); e1=loadmod("e1_e9",E1)

def update_topology(previous,attack,attack_level,benign,rng,parents):
    attacked=e0.target_nodes(attack); new={}
    for node in e0.BASE_NODES:
        direct=attack_level if node in attacked else 0.0
        ps=parents[node]
        upstream=float(np.mean([previous[p] for p in ps])) if ps else 0.0
        damage=0.82*previous[node]+0.12*direct+0.08*upstream
        damage+=0.035*benign+rng.normal(0.0,0.012)
        if direct<0.05: damage-=0.025
        new[node]=float(np.clip(damage,0,1))
    return new

def simulate(run_id,attack,seed,parents):
    rng=np.random.default_rng(seed); onset,severity=e0.generate_run_profile(attack,rng)
    latent={n:float(rng.uniform(0,0.03)) for n in e0.BASE_NODES}
    disturbance={"remaining":0,"strength":0.0}; hist=[]; rows=[]
    for t in range(e0.STEPS_PER_RUN):
        level=e0.attack_intensity(attack,t,onset,severity)
        benign=e0.benign_disturbance(t,rng,disturbance)
        latent=update_topology(latent,attack,level,benign,rng,parents)
        telemetry={n:e0.generate_telemetry(n,attack,level,benign,latent[n],rng) for n in e0.BASE_NODES}
        p=e0.mission_task_probability(latent,rng)
        tasks=rng.binomial(1,p,size=8); hist.extend(tasks.tolist())
        me=float(np.mean(hist[-e0.MISSION_WINDOW*8:])); fail=int(me<e0.MISSION_CRIT)
        for n in e0.BASE_NODES:
            rows.append({"run_id":run_id,"seed":seed,"attack_family":attack,"attack_onset":onset,
                         "attack_severity":severity,"time":t,"node_id":n,**telemetry[n],
                         "mission_effectiveness":me,"failure":fail})
    return rows

def generate():
    all_frames=[]
    for ti,(topo,parents) in enumerate(TOPOLOGIES.items()):
        rows=[]
        for fi,attack in enumerate(e0.ATTACK_FAMILIES):
            for r in range(N_RUNS):
                seed=BASE_SEED+ti*100000+fi*10000+r
                rows.extend(simulate(f"e9_{topo}_{attack}_{r:03d}",attack,seed,parents))
        d=e0.add_future_labels(pd.DataFrame(rows)); d["topology"]=topo; all_frames.append(d)
        print(f"Generated {topo}: {d.run_id.nunique()} runs")
    return pd.concat(all_frames,ignore_index=True)

def manifest(df,H):
    tl=(df[["run_id","time","attack_family","failure",f"failure_within_{H}"]]
        .drop_duplicates(["run_id","time"]).sort_values(["run_id","time"]))
    rows=[]
    for rid,g in tl.groupby("run_id",sort=False):
        g=g.reset_index(drop=True); failures=g.failure.to_numpy(int); fam=str(g.attack_family.iloc[0])
        for i in range(len(g)):
            t=int(g.time.iloc[i]); hs=t-e1.HISTORY+1
            if hs<0 or i+H>=len(g): continue
            if int(g.iloc[i-e1.HISTORY+1:i+1].failure.max())!=0: continue
            fut=failures[i+1:i+H+1]; pos=np.where(fut==1)[0]
            rows.append({"sample_id":f"E9_H{H}_{rid}_{t}","run_id":rid,"time":t,
                         "history_start_time":hs,"attack_family":fam,"horizon":H,
                         "label":int(fut.max()),"steps_to_failure":int(pos[0]+1) if len(pos) else np.nan,
                         "split":"test"})
    return pd.DataFrame(rows)

def main():
    print("="*106)
    print("MissionShield E9 | ZERO-SHOT TOPOLOGY-SHIFT STRESS TEST")
    print("Frozen E1 checkpoints + original chain attention mask; NO retraining or topology adaptation")
    print("="*106)

    # Ensure new seeds do not overlap E0/E8.
    old=set(pd.read_csv(e0.PROC_DIR/"missionshield_e0_v2_dataset.csv",usecols=["seed"]).seed.astype(int))
    e8=RESULTS/"e8_independent_cohort.csv"
    if e8.exists(): old |= set(pd.read_csv(e8,usecols=["seed"]).seed.astype(int))
    proposed={BASE_SEED+ti*100000+fi*10000+r for ti in range(len(TOPOLOGIES))
              for fi in range(len(e0.ATTACK_FAMILIES)) for r in range(N_RUNS)}
    if old & proposed: raise RuntimeError("E9 seeds overlap prior cohorts.")

    df=generate(); df.to_csv(RESULTS/"e9_topology_shift_cohort.csv",index=False)
    olddf=pd.read_csv(e1.DATA); oldlookup=e1.build_telemetry_lookup(olddf)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:",device)

    rows=[]; family_rows=[]
    for topo in TOPOLOGIES:
        d=df[df.topology==topo].copy(); lookup=e1.build_telemetry_lookup(d)
        for H in HORIZONS:
            frozen=pd.read_csv(RESULTS/f"e1_final_samples_H{H}.csv"); frozen.run_id=frozen.run_id.astype(str)
            mean,std=e1.calculate_normalization(frozen,oldlookup)
            man=manifest(d,H); X,y,meta=e1.build_sequences_from_manifest(man,"test",lookup,mean,std)
            loader=DataLoader(e1.MissionDataset(X,y),batch_size=256,shuffle=False,num_workers=0)
            model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(device)
            model.load_state_dict(torch.load(CHECKPOINTS/f"e1_missionshield_H{H}.pt",
                                             map_location=device,weights_only=True))
            yy,p=e1.predict(model,loader,device)
            rows.append({"topology":topo,"horizon":H,"samples":len(yy),
                         "positive_rate":float(yy.mean()),"auroc":float(roc_auc_score(yy,p)),
                         "auprc":float(average_precision_score(yy,p))})
            md=pd.DataFrame(meta); md["y"]=yy; md["p"]=p
            for fam,g in md.groupby("attack_family"):
                if g.y.nunique()<2:
                    family_rows.append({"topology":topo,"horizon":H,"attack_family":fam,
                                        "samples":len(g),"positives":int(g.y.sum()),
                                        "auroc":np.nan,"auprc":np.nan})
                else:
                    family_rows.append({"topology":topo,"horizon":H,"attack_family":fam,
                                        "samples":len(g),"positives":int(g.y.sum()),
                                        "auroc":float(roc_auc_score(g.y,g.p)),
                                        "auprc":float(average_precision_score(g.y,g.p))})
            del X,y,loader,model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    res=pd.DataFrame(rows); fam=pd.DataFrame(family_rows)
    control=res[res.topology=="chain_control"][["horizon","auroc","auprc"]].rename(
        columns={"auroc":"control_auroc","auprc":"control_auprc"})
    comp=res.merge(control,on="horizon")
    comp["delta_auroc_vs_control"]=comp.auroc-comp.control_auroc
    comp["delta_auprc_vs_control"]=comp.auprc-comp.control_auprc
    res.to_csv(RESULTS/"e9_topology_shift_summary.csv",index=False)
    comp.to_csv(RESULTS/"e9_topology_shift_deltas.csv",index=False)
    fam.to_csv(RESULTS/"e9_topology_shift_by_attack.csv",index=False)

    print("\nTOPOLOGY-SHIFT SUMMARY")
    print(comp[["topology","horizon","samples","positive_rate","auroc","auprc",
                "delta_auroc_vs_control","delta_auprc_vs_control"]].to_string(index=False))
    print("\nE9 COMPLETE")
    print("Interpret as zero-shot synthetic topology stress testing, not external/emulator validation.")
    print("Do not tune the model or topology definitions from these outcomes.")

if __name__=="__main__":
    main()
