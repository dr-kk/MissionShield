#!/usr/bin/env python3
"""
MissionShield E10-G2 v2 | Benign-only emulator operating-point calibration

Uses the already-frozen E10-E SCORED benign calibration telemetry.
No imputation, no E10-F labels for threshold selection, no model retraining.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

ROOT=Path(r"D:\MissionShield"); SCRIPTS=ROOT/"scripts"; RES=ROOT/"results"; CK=ROOT/"checkpoints"
CAL=RES/"e10e_benign_calibration_scored.csv"
E10G=RES/"e10g_zero_shot_predictions.csv"
E10F=RES/"e10f_emulator_telemetry.csv"
HORIZONS=[10,30,60]; TARGET_FAR=.05

def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e1=loadmod("e1",SCRIPTS/"e1_missionshield_forecaster.py")

def temp_apply(p,T):
 p=np.clip(np.asarray(p,float),1e-7,1-1e-7);z=np.log(p/(1-p));return 1/(1+np.exp(-z/T))

def threshold_5pct(p):
 # conservative order statistic: threshold giving <=5% alarms on calibration samples
 a=np.sort(np.asarray(p,float)); n=len(a)
 candidates=np.unique(a)
 for t in candidates:
  if np.mean(a>=t)<=TARGET_FAR: return float(t)
 return float(np.nextafter(a[-1],np.inf))

def normalize_columns(d):
 # accept collector naming variants without modifying feature values
 ren={}
 if "interval" in d.columns and "time" not in d.columns: ren["interval"]="time"
 if "run" in d.columns and "run_id" not in d.columns: ren["run"]="run_id"
 d=d.rename(columns=ren)
 if "run_id" not in d.columns or "time" not in d.columns:
  raise RuntimeError(f"Calibration file lacks run_id/time. Columns={list(d.columns)}")
 d["run_id"]=d["run_id"].astype(str)
 return d

def manifest(d,H):
 z=[]
 for rid,g in d.groupby("run_id",sort=False):
  times=sorted(map(int,g.time.unique()))
  tset=set(times)
  for t in times:
   hs=t-e1.HISTORY+1
   if hs<0: continue
   if not all(x in tset for x in range(hs,t+1)): continue
   z.append({"sample_id":f"E10G2_H{H}_{rid}_{t}","run_id":rid,"time":t,
    "history_start_time":hs,"attack_family":"benign_emulator","horizon":H,
    "label":0,"steps_to_failure":np.nan,"split":"calibration"})
 return pd.DataFrame(z)

def main():
 print("="*104)
 print("MissionShield E10-G2 v2 | BENIGN-ONLY EMULATOR OPERATING-POINT CALIBRATION")
 print("Frozen E10-E scored telemetry | no imputation | no E10-F labels in threshold selection")
 print("="*104)
 if not CAL.exists(): raise FileNotFoundError(CAL)
 d=normalize_columns(pd.read_csv(CAL))
 feats=list(e1.FEATURES)
 print("Calibration file:",CAL)
 print("Rows:",len(d),"runs:",d.run_id.nunique(),"columns:",list(d.columns))
 missing=[x for x in feats if x not in d.columns]
 if missing: raise RuntimeError(f"Missing frozen E1 features in scored E10-E file: {missing}")
 bad=d[feats].isna().sum()
 if int(bad.sum())>0:
  raise RuntimeError("Frozen SCORED E10-E still contains NaNs; STOP. Counts="+str(bad[bad>0].to_dict()))
 # structural integrity: six nodes per run/time
 cnt=d.groupby(["run_id","time"]).size()
 if not (cnt==len(e1.NODES)).all(): raise RuntimeError("E10-E scored file does not have exactly six nodes per timestamp.")

 old=pd.read_csv(e1.DATA); oldlk=e1.build_telemetry_lookup(old); callk=e1.build_telemetry_lookup(d)
 op=pd.read_csv(RES/"e3b_operational_summary.csv"); test=pd.read_csv(E10G)
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:",dev)
 rows=[]; calout=[]
 for H in HORIZONS:
  om=pd.read_csv(RES/f"e1_final_samples_H{H}.csv");om["run_id"]=om.run_id.astype(str)
  mean,std=e1.calculate_normalization(om,oldlk)
  man=manifest(d,H)
  X,y,meta=e1.build_sequences_from_manifest(man,"calibration",callk,mean,std)
  dl=DataLoader(e1.MissionDataset(X,y),batch_size=256,shuffle=False,num_workers=0)
  model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(dev)
  model.load_state_dict(torch.load(CK/f"e1_missionshield_H{H}.pt",map_location=dev,weights_only=True))
  _,p,_,_=e1.predict(model,dl,dev,return_explanations=True)
  rr=op[(op.model=="MissionShield")&(op.horizon==H)]
  if len(rr)!=1: raise RuntimeError(f"Missing/nonunique frozen E3-B operating point H={H}")
  T=float(rr.iloc[0].temperature); oldthr=float(rr.iloc[0].threshold_calibrated)
  pc=temp_apply(p,T); thr=threshold_5pct(pc); cfar=float(np.mean(pc>=thr))
  for i,m in enumerate(meta):
   calout.append({"horizon":H,"run_id":m["run_id"],"time":m["time"],
    "probability":p[i],"probability_calibrated":pc[i],"threshold":thr,"alarm":int(pc[i]>=thr)})
  q=test[test.horizon==H].copy()
  a=(q.probability_calibrated.to_numpy(float)>=thr).astype(int); yy=q.label.to_numpy(int)
  neg=yy==0; pos=yy==1
  row={"horizon":H,"calibration_samples":len(pc),"old_synthetic_threshold":oldthr,
   "new_benign_threshold":thr,"calibration_far":cfar,"e10f_alarm_rate":float(a.mean()),
   "e10f_far":float(a[neg].mean()) if neg.any() else np.nan,
   "e10f_sensitivity":float(a[pos].mean()) if pos.any() else np.nan}
  rows.append(row); print(row)

 pd.DataFrame(calout).to_csv(RES/"e10g2_benign_calibration_predictions.csv",index=False)
 S=pd.DataFrame(rows);S.to_csv(RES/"e10g2_operating_point_summary.csv",index=False)
 th=dict(zip(S.horizon,S.new_benign_threshold))
 P=test.copy();P["threshold_benign_emulator"]=P.horizon.map(th)
 P["alarm_benign_calibrated"]=(P.probability_calibrated>=P.threshold_benign_emulator).astype(int)
 P.to_csv(RES/"e10g2_e10f_predictions_with_benign_threshold.csv",index=False)

 tl=pd.read_csv(E10F)[["run_id","time","condition","failure"]].drop_duplicates(["run_id","time"])
 tl["run_id"]=tl.run_id.astype(str); ops=[]
 for H,q in P.groupby("horizon"):
  for rid,rp in q.groupby("run_id"):
   rid=str(rid); t=tl[tl.run_id==rid].sort_values("time")
   ft=t.loc[t.failure==1,"time"];at=rp.loc[rp.alarm_benign_calibrated==1,"time"]
   ff=int(ft.min()) if len(ft) else np.nan;fa=int(at.min()) if len(at) else np.nan
   lead=ff-fa if np.isfinite(ff) and np.isfinite(fa) and fa<ff else np.nan
   ops.append({"horizon":H,"run_id":rid,"condition":str(t.condition.iloc[0]),
    "ever_failed":int(len(ft)>0),"first_failure":ff,"ever_alarm":int(len(at)>0),
    "first_alarm":fa,"pre_failure_detected":int(np.isfinite(lead)),"lead_intervals":lead})
 O=pd.DataFrame(ops);O.to_csv(RES/"e10g2_operational_run_summary.csv",index=False)
 print("\nRUN-LEVEL OPERATIONAL SUMMARY")
 for H,q in O.groupby("horizon"):
  f=q[q.ever_failed==1];dd=f[f.pre_failure_detected==1]
  print({"horizon":int(H),"failed_runs":len(f),"prefailure_detected":len(dd),
   "event_detection_rate":len(dd)/len(f) if len(f) else np.nan,
   "median_lead_intervals":float(dd.lead_intervals.median()) if len(dd) else np.nan})
 print("\nE10-G2 COMPLETE | BENIGN-ONLY OPERATING-POINT CALIBRATION")
 print("No imputation. MissionShield weights unchanged. No E10-F labels used to select thresholds.")
if __name__=="__main__":main()
