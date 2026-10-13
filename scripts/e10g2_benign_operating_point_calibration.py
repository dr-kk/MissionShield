#!/usr/bin/env python3
"""
MissionShield E10-G2 | Benign-only emulator operating-point calibration

Purpose:
  Adapt ONLY the alarm threshold to emulator-domain benign data.
  Model weights, E1 normalization, architecture, horizons, and E10-F outcomes remain untouched.

Calibration source:
  E10-E benign calibration telemetry (collected before E10-F/E10-G outcomes).

Rule:
  For each H, build eligible H-history samples from E10-E benign runs, obtain frozen
  MissionShield probabilities, apply the already-frozen E3-B temperature, and set
  threshold to the empirical 95th percentile of benign calibrated probabilities.
  This targets ~5% benign sample-level FAR in the calibration domain.

Evaluation:
  Re-score the already-frozen E10-G prediction file using ONLY the new threshold.
  No E10-F labels are used to select the threshold.
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

ROOT=Path(r"D:\MissionShield"); SCRIPTS=ROOT/"scripts"; RES=ROOT/"results"; CK=ROOT/"checkpoints"
RAW=RES/"e10e_benign_calibration_raw.csv"
E10G=RES/"e10g_zero_shot_predictions.csv"
HORIZONS=[10,30,60]; TARGET_FAR=.05

def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e1=loadmod("e1",SCRIPTS/"e1_missionshield_forecaster.py")

def temp_apply(p,T):
 p=np.clip(np.asarray(p,float),1e-7,1-1e-7);z=np.log(p/(1-p));return 1/(1+np.exp(-z/T))

def benign_manifest(df,H):
 z=[]
 for rid,g in df.groupby("run_id",sort=False):
  times=sorted(g.time.unique())
  for t in times:
   hs=t-e1.HISTORY+1
   if hs<0:continue
   z.append({"sample_id":f"E10G2_H{H}_{rid}_{t}","run_id":str(rid),"time":int(t),
    "history_start_time":int(hs),"attack_family":"benign_emulator","horizon":H,
    "label":0,"steps_to_failure":np.nan,"split":"calibration"})
 return pd.DataFrame(z)

def empirical_threshold(p,target_far):
 # Conservative finite-sample order statistic: choose smallest threshold with <= target FAR
 a=np.sort(np.asarray(p,float));n=len(a)
 k=int(np.ceil((1-target_far)*(n+1)))-1;k=min(max(k,0),n-1)
 return float(a[k])

def main():
 print("="*100);print("MissionShield E10-G2 | BENIGN-ONLY EMULATOR OPERATING-POINT CALIBRATION")
 print("Model weights unchanged | E1 normalization unchanged | NO E10-F labels used for threshold selection");print("="*100)
 d=pd.read_csv(RAW)
 d=d.rename(columns={"run":"run_id","interval":"time"})
 d["run_id"]=d.run_id.map(lambda x:f"e10e_benign_r{int(x):02d}")
 if len(d)!=1800 or d.run_id.nunique()!=5:raise RuntimeError("Expected frozen E10-E: 1800 rows / 5 runs")
 feats=["latency","packet_loss","utilization","availability","sensor_quality","detector_score"]
 if d[feats].isna().any().any():raise RuntimeError("NaN in E10-E calibration telemetry")
 old=pd.read_csv(e1.DATA);oldlk=e1.build_telemetry_lookup(old);newlk=e1.build_telemetry_lookup(d)
 op=pd.read_csv(RES/"e3b_operational_summary.csv");dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 predtest=pd.read_csv(E10G);out=[];calrows=[]
 for H in HORIZONS:
  om=pd.read_csv(RES/f"e1_final_samples_H{H}.csv");om["run_id"]=om.run_id.astype(str)
  mean,std=e1.calculate_normalization(om,oldlk);man=benign_manifest(d,H)
  # build_sequences expects split_name to match manifest
  X,y,meta=e1.build_sequences_from_manifest(man,"calibration",newlk,mean,std)
  dl=DataLoader(e1.MissionDataset(X,y),batch_size=256,shuffle=False,num_workers=0)
  model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(dev)
  model.load_state_dict(torch.load(CK/f"e1_missionshield_H{H}.pt",map_location=dev,weights_only=True))
  yy,p,nw,g=e1.predict(model,dl,dev,return_explanations=True)
  r=op[(op.model=="MissionShield")&(op.horizon==H)]
  if len(r)!=1:raise RuntimeError(f"Missing E3-B operating point H{H}")
  T=float(r.iloc[0].temperature);oldthr=float(r.iloc[0].threshold_calibrated)
  pc=temp_apply(p,T);thr=empirical_threshold(pc,TARGET_FAR);calfar=float(np.mean(pc>=thr))
  for i,m in enumerate(meta):
   calrows.append({"horizon":H,"run_id":m["run_id"],"time":m["time"],"probability":p[i],
    "probability_calibrated":pc[i],"threshold_benign_emulator":thr,"alarm":int(pc[i]>=thr)})
  q=predtest[predtest.horizon==H].copy()
  q["alarm_benign_calibrated"]=(q.probability_calibrated>=thr).astype(int)
  yq=q.label.to_numpy(int);aq=q.alarm_benign_calibrated.to_numpy(int);neg=yq==0;pos=yq==1
  row={"horizon":H,"benign_calibration_samples":len(pc),"temperature_frozen":T,
   "old_synthetic_threshold":oldthr,"new_benign_emulator_threshold":thr,
   "benign_calibration_far":calfar,"e10f_alarm_rate":float(aq.mean()),
   "e10f_far":float(aq[neg].mean()) if neg.any() else np.nan,
   "e10f_sensitivity":float(aq[pos].mean()) if pos.any() else np.nan}
  out.append(row);print(row)
 pd.DataFrame(calrows).to_csv(RES/"e10g2_benign_calibration_predictions.csv",index=False)
 pd.DataFrame(out).to_csv(RES/"e10g2_operating_point_summary.csv",index=False)

 # Apply new thresholds to frozen E10-G predictions and compute run-level pre-failure detection.
 th={x["horizon"]:x["new_benign_emulator_threshold"] for x in out}
 p=predtest.copy();p["threshold_benign_emulator"]=p.horizon.map(th)
 p["alarm_benign_calibrated"]=(p.probability_calibrated>=p.threshold_benign_emulator).astype(int)
 p.to_csv(RES/"e10g2_e10f_predictions_with_benign_threshold.csv",index=False)
 tl=pd.read_csv(RES/"e10f_emulator_telemetry.csv")[["run_id","time","condition","failure"]].drop_duplicates(["run_id","time"])
 ops=[]
 for H,q in p.groupby("horizon"):
  for rid,rp in q.groupby("run_id"):
   t=tl[tl.run_id==rid].sort_values("time");ft=t.loc[t.failure==1,"time"];at=rp.loc[rp.alarm_benign_calibrated==1,"time"]
   ff=int(ft.min()) if len(ft) else np.nan;fa=int(at.min()) if len(at) else np.nan
   lead=ff-fa if np.isfinite(ff) and np.isfinite(fa) and fa<ff else np.nan
   ops.append({"horizon":H,"run_id":rid,"condition":str(t.condition.iloc[0]),"ever_failed":int(len(ft)>0),
    "first_failure":ff,"ever_alarm":int(len(at)>0),"first_alarm":fa,
    "pre_failure_detected":int(np.isfinite(lead)),"lead_intervals":lead})
 O=pd.DataFrame(ops);O.to_csv(RES/"e10g2_operational_run_summary.csv",index=False)
 print("\nRUN-LEVEL OPERATIONAL SUMMARY")
 for H,q in O.groupby("horizon"):
  f=q[q.ever_failed==1];dd=f[f.pre_failure_detected==1]
  print({"horizon":int(H),"failed_runs":len(f),"prefailure_detected":len(dd),
   "event_detection_rate":len(dd)/len(f) if len(f) else np.nan,
   "median_lead_intervals":float(dd.lead_intervals.median()) if len(dd) else np.nan})
 print("\nOutputs:")
 print(RES/"e10g2_operating_point_summary.csv")
 print(RES/"e10g2_operational_run_summary.csv")
 print("E10-G2 COMPLETE | BENIGN-ONLY OPERATING-POINT CALIBRATION")
 print("MissionShield weights unchanged; no E10-F labels used for threshold selection.")
if __name__=="__main__":main()
