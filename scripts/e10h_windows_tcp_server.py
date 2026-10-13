#!/usr/bin/env python3
# MissionShield E10-H TCP inference server (Windows).
import json, socketserver, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT=Path(r"D:\MissionShield"); RES=ROOT/"results"; SCRIPTS=ROOT/"scripts"; CK=ROOT/"checkpoints"
HOST="0.0.0.0"; PORT=50555

def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
e1=loadmod("e1",SCRIPTS/"e1_missionshield_forecaster.py")
op=pd.read_csv(RES/"e3b_operational_summary.csv")
g2=pd.read_csv(RES/"e10g2_operating_point_summary.csv")
old=pd.read_csv(e1.DATA); oldlk=e1.build_telemetry_lookup(old)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
models={}; norm={}; temps={}; thresholds={}
for H in (10,30,60):
    man=pd.read_csv(RES/f"e1_final_samples_H{H}.csv"); man["run_id"]=man.run_id.astype(str)
    norm[H]=e1.calculate_normalization(man,oldlk)
    model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(device)
    model.load_state_dict(torch.load(CK/f"e1_missionshield_H{H}.pt",map_location=device,weights_only=True))
    model.eval(); models[H]=model
    r=op[(op.model=="MissionShield")&(op.horizon==H)].iloc[0]
    temps[H]=float(r.temperature)
    thresholds[H]=float(g2[g2.horizon==H].iloc[0].new_benign_threshold)

def temp_apply(p,T):
    p=float(np.clip(p,1e-7,1-1e-7)); z=np.log(p/(1-p)); return float(1/(1+np.exp(-z/T)))

def infer(q):
    H=int(q["horizon"]); X=np.asarray(q["history"],dtype=np.float32)
    if H not in models: raise ValueError("horizon must be 10,30,60")
    if X.shape!=(10,6,6): raise ValueError(f"history shape must be (10,6,6), got {X.shape}")
    mean,std=norm[H]; X=(X-mean)/std
    xt=torch.tensor(X[None,...],dtype=torch.float32,device=device)
    with torch.no_grad():
        logits,nw,gate=models[H](xt)
        raw=float(torch.sigmoid(logits)[0].cpu())
        weights=nw[0,-1,:].cpu().numpy().tolist()
    cal=temp_apply(raw,temps[H])
    return {"token":q.get("token"),"horizon":H,"raw_probability":raw,
            "calibrated_probability":cal,"threshold":thresholds[H],
            "alarm":bool(cal>=thresholds[H]),"node_importance":weights}

class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            q=json.loads(self.rfile.readline().decode("utf-8"))
            if q.get("cmd")=="ping":
                out={"ok":True,"device":str(device),"thresholds":thresholds}
            else: out=infer(q)
        except Exception as e:
            out={"error":repr(e)}
        self.wfile.write((json.dumps(out)+"\n").encode("utf-8"))

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True

print(f"E10-H TCP SERVER READY | {HOST}:{PORT} | device={device}")
print("thresholds=",thresholds)
with Server((HOST,PORT),Handler) as srv: srv.serve_forever()
