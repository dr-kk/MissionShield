#!/usr/bin/env python3
# MissionShield E10-H Windows inference worker.
# Run in Windows PowerShell; watches request JSON written by WSL and returns frozen-model inference.
import json,time,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(r"D:\MissionShield"); RES=ROOT/"results"; SCRIPTS=ROOT/"scripts"; CK=ROOT/"checkpoints"
IPC=RES/"e10h_ipc"; IPC.mkdir(parents=True,exist_ok=True)
REQ=IPC/"request.json"; RESP=IPC/"response.json"; STOP=IPC/"STOP"

def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e1=loadmod("e1",SCRIPTS/"e1_missionshield_forecaster.py")
op=pd.read_csv(RES/"e3b_operational_summary.csv")
g2=pd.read_csv(RES/"e10g2_operating_point_summary.csv")
old=pd.read_csv(e1.DATA); oldlk=e1.build_telemetry_lookup(old)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
models={}; norm={}; temps={}; th={}
for H in [10,30,60]:
 m=pd.read_csv(RES/f"e1_final_samples_H{H}.csv");m["run_id"]=m.run_id.astype(str)
 norm[H]=e1.calculate_normalization(m,oldlk)
 model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(device)
 model.load_state_dict(torch.load(CK/f"e1_missionshield_H{H}.pt",map_location=device,weights_only=True));model.eval();models[H]=model
 r=op[(op.model=="MissionShield")&(op.horizon==H)].iloc[0];temps[H]=float(r.temperature)
 th[H]=float(g2[g2.horizon==H].iloc[0].new_benign_threshold)

def temp_apply(p,T):
 p=float(np.clip(p,1e-7,1-1e-7));z=np.log(p/(1-p));return float(1/(1+np.exp(-z/T)))

print("E10-H WINDOWS INFERENCE WORKER READY | device=",device)
print("thresholds=",th)
last=None
while not STOP.exists():
 if REQ.exists():
  try:
   q=json.loads(REQ.read_text())
   token=q["token"]
   if token!=last:
    H=int(q["horizon"]); X=np.asarray(q["history"],dtype=np.float32)
    if X.shape!=(10,6,6): raise RuntimeError(f"bad history shape {X.shape}")
    mean,std=norm[H]; X=(X-mean)/std
    xt=torch.tensor(X[None,...],dtype=torch.float32,device=device)
    with torch.no_grad():
     logits,nw,gate=models[H](xt)
     raw=float(torch.sigmoid(logits)[0].cpu()); weights=nw[0,-1,:].cpu().numpy().tolist()
    cal=temp_apply(raw,temps[H])
    out={"token":token,"horizon":H,"raw_probability":raw,"calibrated_probability":cal,
         "threshold":th[H],"alarm":bool(cal>=th[H]),"node_importance":weights}
    tmp=IPC/"response.tmp";tmp.write_text(json.dumps(out));tmp.replace(RESP);last=token
  except Exception as e:
   try:
    token=json.loads(REQ.read_text()).get("token","unknown")
    tmp=IPC/"response.tmp";tmp.write_text(json.dumps({"token":token,"error":repr(e)}));tmp.replace(RESP);last=token
   except: pass
 time.sleep(.02)
print("E10-H WINDOWS INFERENCE WORKER STOPPED")
