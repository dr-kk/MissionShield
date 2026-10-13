#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path(r"D:\MissionShield"); SCRIPTS=ROOT/"scripts"; RESULTS=ROOT/"results"; CK=ROOT/"checkpoints"
EMU=RESULTS/"e10f_emulator_telemetry.csv"; HORIZONS=[10,30,60]

def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e1=loadmod("e1",SCRIPTS/"e1_missionshield_forecaster.py")

def temp_apply(p,T):
 p=np.clip(np.asarray(p,float),1e-7,1-1e-7);z=np.log(p/(1-p));return 1/(1+np.exp(-z/T))
def ece(y,p,b=10):
 y=np.asarray(y);p=np.asarray(p);ed=np.linspace(0,1,b+1);v=0.
 for i in range(b):
  m=(p>=ed[i])&((p<ed[i+1]) if i<b-1 else (p<=ed[i+1]))
  if m.any():v+=m.mean()*abs(y[m].mean()-p[m].mean())
 return float(v)
def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
def ap(y,p):return float(average_precision_score(y,p)) if np.sum(y)>0 else np.nan

def manifest(df,H):
 tl=df[["run_id","time","condition","failure"]].drop_duplicates(["run_id","time"]).sort_values(["run_id","time"])
 z=[]
 for rid,g in tl.groupby("run_id",sort=False):
  g=g.sort_values("time").reset_index(drop=True);f=g.failure.to_numpy(int);c=str(g.condition.iloc[0])
  for i in range(len(g)):
   t=int(g.time.iloc[i]);hs=t-e1.HISTORY+1
   if hs<0 or i+H>=len(g):continue
   hist=g.iloc[i-e1.HISTORY+1:i+1]
   if len(hist)!=e1.HISTORY or int(hist.failure.max())!=0:continue
   fut=f[i+1:i+H+1];ii=np.where(fut==1)[0]
   z.append({"sample_id":f"E10G_H{H}_{rid}_{t}","run_id":str(rid),"time":t,"history_start_time":hs,
    "attack_family":c,"horizon":H,"label":int(fut.max()),"steps_to_failure":int(ii[0]+1) if len(ii) else np.nan,"split":"test"})
 return pd.DataFrame(z)

def main():
 print("="*98);print("MissionShield E10-G | FROZEN ZERO-SHOT MININET FORECASTING")
 print("NO emulator training/fine-tuning | H10/H30/H60");print("="*98)
 df=pd.read_csv(EMU);df["run_id"]=df.run_id.astype(str)
 feats=["latency","packet_loss","utilization","availability","sensor_quality","detector_score"]
 if df[feats].isna().any().any():raise RuntimeError("NaN telemetry present; no post-hoc imputation allowed.")
 if df.run_id.nunique()!=20:raise RuntimeError("Expected 20 frozen E10-F runs.")
 print(f"Rows={len(df):,} runs={df.run_id.nunique()} timestamps={len(df)//6:,}")
 old=pd.read_csv(e1.DATA);oldlk=e1.build_telemetry_lookup(old);newlk=e1.build_telemetry_lookup(df)
 op=pd.read_csv(RESULTS/"e3b_operational_summary.csv");dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print("Device:",dev);allp=[];summ=[];byc=[]
 for H in HORIZONS:
  om=pd.read_csv(RESULTS/f"e1_final_samples_H{H}.csv");om["run_id"]=om.run_id.astype(str)
  mean,std=e1.calculate_normalization(om,oldlk);man=manifest(df,H)
  X,y,meta=e1.build_sequences_from_manifest(man,"test",newlk,mean,std)
  dl=DataLoader(e1.MissionDataset(X,y),batch_size=256,shuffle=False,num_workers=0)
  model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(dev)
  model.load_state_dict(torch.load(CK/f"e1_missionshield_H{H}.pt",map_location=dev,weights_only=True))
  yy,p,nw,g=e1.predict(model,dl,dev,return_explanations=True)
  r=op[(op.model=="MissionShield")&(op.horizon==H)]
  if len(r)!=1:raise RuntimeError(f"Frozen E3-B operating point missing/nonunique H={H}")
  T=float(r.iloc[0].temperature);thr=float(r.iloc[0].threshold_calibrated);pc=temp_apply(p,T);alarm=(pc>=thr).astype(int)
  pr=pd.DataFrame(meta);pr["label"]=yy.astype(int);pr["probability"]=p;pr["probability_calibrated"]=pc
  pr["alarm_fixed_far"]=alarm;pr["fusion_temporal_gate"]=g;pr["condition"]=pr["attack_family"];pr["horizon"]=H
  for j,n in enumerate(e1.NODES):pr[f"importance_{n}"]=nw[:,j]
  allp.append(pr);neg=yy==0;pos=yy==1
  row={"horizon":H,"samples":len(yy),"positives":int(pos.sum()),"positive_rate":float(yy.mean()),
   "auroc":auc(yy,p),"auprc":ap(yy,p),"ece_raw":ece(yy,p),"brier_raw":float(brier_score_loss(yy,p)),
   "ece_calibrated":ece(yy,pc),"brier_calibrated":float(brier_score_loss(yy,pc)),
   "temperature_frozen":T,"threshold_frozen":thr,"alarm_rate":float(alarm.mean()),
   "far_test":float(alarm[neg].mean()) if neg.any() else np.nan,
   "sensitivity":float(alarm[pos].mean()) if pos.any() else np.nan}
  summ.append(row);print("\n",row)
  for c,q in pr.groupby("condition"):
   a=q.label.to_numpy(int);b=q.probability.to_numpy(float);al=q.alarm_fixed_far.to_numpy(int);nn=a==0;pp=a==1
   byc.append({"horizon":H,"condition":c,"samples":len(q),"positives":int(pp.sum()),"positive_rate":float(a.mean()),
    "auroc":auc(a,b),"auprc":ap(a,b),"alarm_rate":float(al.mean()),
    "far":float(al[nn].mean()) if nn.any() else np.nan,"sensitivity":float(al[pp].mean()) if pp.any() else np.nan})
 P=pd.concat(allp,ignore_index=True);P.to_csv(RESULTS/"e10g_zero_shot_predictions.csv",index=False)
 pd.DataFrame(summ).to_csv(RESULTS/"e10g_zero_shot_summary.csv",index=False);pd.DataFrame(byc).to_csv(RESULTS/"e10g_zero_shot_by_condition.csv",index=False)
 tl=df[["run_id","time","condition","failure"]].drop_duplicates(["run_id","time"]);ops=[]
 for H,q in P.groupby("horizon"):
  for rid,rp in q.groupby("run_id"):
   t=tl[tl.run_id==rid].sort_values("time");ft=t.loc[t.failure==1,"time"];at=rp.loc[rp.alarm_fixed_far==1,"time"]
   ff=int(ft.min()) if len(ft) else np.nan;fa=int(at.min()) if len(at) else np.nan
   lead=ff-fa if np.isfinite(ff) and np.isfinite(fa) and fa<ff else np.nan
   ops.append({"horizon":H,"run_id":rid,"condition":str(t.condition.iloc[0]),"ever_failed":int(len(ft)>0),
    "first_failure":ff,"ever_alarm":int(len(at)>0),"first_alarm":fa,"pre_failure_detected":int(np.isfinite(lead)),"lead_intervals":lead})
 O=pd.DataFrame(ops);O.to_csv(RESULTS/"e10g_operational_run_summary.csv",index=False)
 print("\nE10-G OVERALL SUMMARY");print(pd.DataFrame(summ).to_string(index=False))
 print("\nRUN-LEVEL OPERATIONAL SUMMARY")
 for H,q in O.groupby("horizon"):
  f=q[q.ever_failed==1];d=f[f.pre_failure_detected==1]
  print({"horizon":int(H),"failed_runs":len(f),"prefailure_detected":len(d),
   "event_detection_rate":len(d)/len(f) if len(f) else np.nan,"median_lead_intervals":d.lead_intervals.median() if len(d) else np.nan})
 print("\nE10-G COMPLETE | FROZEN ZERO-SHOT EMULATOR EVALUATION")
 print("No emulator training or fine-tuning performed.")
if __name__=="__main__":main()
