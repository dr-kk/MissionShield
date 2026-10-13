#!/usr/bin/env python3
"""
MissionShield E10-H | Paired Mininet intervention replay

Runs intervention replays only. Frozen E10-F is the paired baseline.
For each horizon H10/H30/H60 and each of the 20 frozen condition/run IDs:
  - identical topology, traffic, onset, severity, task payloads, mission definition
  - online frozen MissionShield inference
  - frozen E10-G2 benign-only threshold for that horizon
  - FIRST qualifying alarm only
  - target = argmax MissionShield node importance among edge_processing/fusion/command
  - action = restore_service on interfaces incident to selected node
No attack-family truth or damaged-node oracle is used for target selection.

IMPORTANT: E10-H is confirmatory intervention evaluation, not zero-shot forecasting.
"""
import csv,json,math,time,hashlib,sys,importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

ROOT=Path("/mnt/d/MissionShield"); RES=ROOT/"results"; SCRIPTS=ROOT/"scripts"; CK=ROOT/"checkpoints"
PROTOCOL=RES/"e10f_campaign_protocol.json"; CAL=RES/"e10e_detector_calibration.json"
BASELINE=RES/"e10f_emulator_telemetry.csv"; G2=RES/"e10g2_operating_point_summary.csv"
RUNDIR=RES/"e10h_runs"; RUNDIR.mkdir(parents=True,exist_ok=True)
OUT=RES/"e10h_intervention_telemetry.csv"; SUM=RES/"e10h_intervention_summary.csv"
HORIZONS=[10,30,60]; ELIGIBLE=["edge_processing","fusion","command"]
N=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
S=["sen","tlink","edge","fus","cmd","act"]; FEATURES=["latency","packet_loss","utilization","availability"]
PORT=7100; PREFLIGHT_ATTEMPTS=5; HISTORY=10

RELAY=r"""
import json,socket,sys,time
li=sys.argv[1];port=int(sys.argv[2]);si=sys.argv[3];ni=sys.argv[4];out=sys.argv[5]
rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind((li,port))
tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);tx.bind((si,0))
with open(out,"a",buffering=1) as f:
 while True:
  d,a=rx.recvfrom(65535);now=time.time_ns();m=json.loads(d.decode())
  f.write(json.dumps({"task_id":m["task_id"],"interval":m["interval"],"truth":m["truth"],"value":m["value"],
   "send_ns":m["send_ns"],"rx_ns":now},separators=(",",":"))+"\n")
  tx.sendto(d,(ni,port))
"""
ACT=r"""
import json,socket,sys,time
ip=sys.argv[1];port=int(sys.argv[2]);out=sys.argv[3]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((ip,port))
with open(out,"a",buffering=1) as f:
 while True:
  d,a=s.recvfrom(65535);now=time.time_ns();m=json.loads(d.decode())
  f.write(json.dumps({"task_id":m["task_id"],"interval":m["interval"],"truth":m["truth"],"value":m["value"],
   "send_ns":m["send_ns"],"rx_ns":now},separators=(",",":"))+"\n")
"""
TX=r"""
import socket,sys
src=sys.argv[1];dst=sys.argv[2];port=int(sys.argv[3]);fn=sys.argv[4]
d=open(fn,"rb").read();s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((src,0));s.sendto(d,(dst,port))
"""

def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def readj(h,fn):
 z=[]
 for line in h.cmd(f"cat {fn} 2>/dev/null || true").splitlines():
  try:z.append(json.loads(line))
  except:pass
 return z
def ibytes(h,intf):
 try:return sum(int(x) for x in h.cmd(f"cat /sys/class/net/{intf}/statistics/tx_bytes /sys/class/net/{intf}/statistics/rx_bytes").split()[:2])
 except:return 0
def score(x,vals,tail):
 if not np.isfinite(x): return 1.0
 n=len(vals);ext=sum(v>=x for v in vals) if tail=="high" else sum(v<=x for v in vals)
 return 1-(1+ext)/(n+1)
def build(cfg):
 net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True);h={n:net.addHost(s) for n,s in zip(N,S)};links=[]
 for i in range(5):
  sw=net.addSwitch(f"s{i+1}")
  l=net.addLink(h[N[i]],sw,cls=TCLink,intfName1=f"{S[i]}e{i}",intfName2=f"s{i+1}a",bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  r=net.addLink(sw,h[N[i+1]],cls=TCLink,intfName1=f"s{i+1}b",intfName2=f"{S[i+1]}e{i}",bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  links.append((l,r))
 net.start();pair={};nodeints={n:[] for n in N}
 for i,(a,b) in enumerate(zip(N[:-1],N[1:])):
  l,r=links[i];ia=l.intf1 if l.intf1.node==h[a] else l.intf2;ib=r.intf1 if r.intf1.node==h[b] else r.intf2
  A=f"10.{i+1}.0.1";B=f"10.{i+1}.0.2"
  h[a].cmd(f"ip addr flush dev {ia};ip addr add {A}/24 dev {ia};ip link set {ia} up")
  h[b].cmd(f"ip addr flush dev {ib};ip addr add {B}/24 dev {ib};ip link set {ib} up")
  pair[(a,b)]=(A,B);nodeints[a].append(str(ia));nodeints[b].append(str(ib))
 return net,h,pair,nodeints
def reconfigure(net,cfg,only_interfaces=None):
 for link in net.links:
  for intf in (link.intf1,link.intf2):
   if only_interfaces is not None and str(intf) not in only_interfaces: continue
   intf.config(bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
def temp_apply(p,T):
 p=float(np.clip(p,1e-7,1-1e-7));z=math.log(p/(1-p));return 1/(1+math.exp(-z/T))

def main():
 # Linux imports Windows-trained architecture via /mnt/d.
 e1=loadmod("e1",SCRIPTS/"e1_missionshield_forecaster.py")
 P=json.loads(PROTOCOL.read_text());C=json.loads(CAL.read_text());camp=P["campaign"];mission=P["mission"]
 assert camp["runs_per_condition"]==5 and camp["intervals_per_run"]==150 and camp["impairment_onset_interval"]==40
 TASKS=camp["tasks_per_interval"];DEADLINE=mission["deadline_ms"];TOL=mission["payload_tolerance_abs"];WINDOW=mission["mission_window_intervals"];FAIL=mission["failure_threshold"]
 base={"bw":20,"delay":"2ms","loss":0}
 conds={"latency":{"bw":20,"delay":"15ms","loss":0},"loss":{"bw":20,"delay":"2ms","loss":5},
        "bandwidth":{"bw":5,"delay":"2ms","loss":0},"combined":{"bw":5,"delay":"15ms","loss":5}}
 g2=pd.read_csv(G2);thresholds=dict(zip(g2.horizon.astype(int),g2.new_benign_threshold.astype(float)))
 op=pd.read_csv(RES/"e3b_operational_summary.csv");temps={}
 for H in HORIZONS:
  r=op[(op.model=="MissionShield")&(op.horizon==H)]
  if len(r)!=1:raise RuntimeError(f"Missing frozen E3-B temperature H={H}")
  temps[H]=float(r.iloc[0].temperature)
 # frozen E1 normalization
 old=pd.read_csv(e1.DATA);oldlk=e1.build_telemetry_lookup(old);norm={}
 for H in HORIZONS:
  m=pd.read_csv(RES/f"e1_final_samples_H{H}.csv");m["run_id"]=m.run_id.astype(str);norm[H]=e1.calculate_normalization(m,oldlk)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 models={}
 for H in HORIZONS:
  model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(device)
  model.load_state_dict(torch.load(CK/f"e1_missionshield_H{H}.pt",map_location=device,weights_only=True));model.eval();models[H]=model
 print("Device:",device);print("Thresholds:",thresholds)
 d=Path("/tmp/missionshield_e10h");d.mkdir(exist_ok=True);(d/"relay.py").write_text(RELAY);(d/"act.py").write_text(ACT);(d/"tx.py").write_text(TX)
 print("="*104);print("MissionShield E10-H | PAIRED MININET INTERVENTION REPLAY")
 print("60 intervention replays = 20 frozen runs x H10/H30/H60 | baseline = frozen E10-F");print("="*104)

 for H in HORIZONS:
  mean,std=norm[H]
  for condition,target in conds.items():
   for rr in range(1,6):
    baseid=f"{condition}_r{rr:02d}";rid=f"H{H}_{baseid}";rf=RUNDIR/f"{rid}.csv";dn=RUNDIR/f"{rid}.done.json"
    if rf.exists() and dn.exists():print("SKIP",rid);continue
    net=h=None;rows=[];hist=[];tele_hist=[];intervened=False;alarm_t=None;chosen=None;alarm_prob=None
    try:
     net,h,pair,nodeints=build(base);files={n:f"/tmp/e10h_{rid}_{S[i]}.jsonl" for i,n in enumerate(N)}
     for n in N:h[n].cmd(f"rm -f {files[n]}")
     h["actuator"].cmd(f"python3 {d/'act.py'} {pair[('command','actuator')][1]} {PORT} {files['actuator']} >/tmp/e10h_act.out 2>&1 &")
     for i in range(4,0,-1):
      n=N[i];pr=N[i-1];nx=N[i+1]
      h[n].cmd(f"python3 {d/'relay.py'} {pair[(pr,n)][1]} {PORT} {pair[(n,nx)][0]} {pair[(n,nx)][1]} {files[n]} >/tmp/e10h_{S[i]}.out 2>&1 &")
     time.sleep(.6);src=pair[("sensor","tactical_link")][0];dst=pair[("sensor","tactical_link")][1];mf=f"/tmp/e10h_{rid}.json"
     ok=False
     for pa in range(PREFLIGHT_ATTEMPTS):
      m={"task_id":f"__pf_{pa}__","truth":0.,"value":0.,"send_ns":time.time_ns(),"interval":-1};Path(mf).write_text(json.dumps(m,separators=(",",":")))
      h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {mf}");time.sleep(.3)
      if any(x["task_id"]==m["task_id"] for x in readj(h["actuator"],files["actuator"])):ok=True;break
     if not ok:raise RuntimeError(f"{rid} preflight failed")
     print(f"\nSTART {rid} | preflight PASS");prev={n:{it:ibytes(h[n],it) for it in nodeints[n]} for n in N}
     for t in range(150):
      if t==40:
       reconfigure(net,target);time.sleep(.2);prev={n:{it:ibytes(h[n],it) for it in nodeints[n]} for n in N};print(f" {rid}: impairment ON")
      t0=time.time();sent={}
      for k in range(TASKS):
       tid=f"{baseid}_{t:03d}_{k:02d}";truth=math.sin((t*TASKS+k)/10.);m={"task_id":tid,"truth":truth,"value":truth,"send_ns":time.time_ns(),"interval":t};sent[tid]=m
       Path(mf).write_text(json.dumps(m,separators=(",",":")));h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {mf}");time.sleep(.02)
      if time.time()-t0<1:time.sleep(1-(time.time()-t0))
      nodeobs={}
      for n in N:
       us=[]
       for it in nodeints[n]:
        now=ibytes(h[n],it);delta=max(0,now-prev[n][it]);prev[n][it]=now;den=(target["bw"] if t>=40 and not intervened else 20)*1e6/8.;us.append(min(1.,delta/den))
       util=max(us) if us else 0.
       if n=="sensor":lat=0.;loss=0.;avail=1.;qual=1.
       else:
        allr={x["task_id"]:x for x in readj(h[n],files[n]) if x.get("interval")==t};rec=[allr[x] for x in sent if x in allr]
        lats=[(x["rx_ns"]-sent[x["task_id"]]["send_ns"])/1e6 for x in rec];lat=sum(lats)/len(lats) if lats else float("nan")
        ontime=sum(1 for x in rec if (x["rx_ns"]-sent[x["task_id"]]["send_ns"])/1e6<=DEADLINE)
        loss=1-ontime/TASKS;avail=len(rec)/TASKS;qual=sum(abs(x["value"]-x["truth"])<=TOL for x in rec)/len(rec) if rec else 0.
       fs={f:score({"latency":lat,"packet_loss":loss,"utilization":util,"availability":avail}[f],C["calibration_values"][n][f],C["tail_direction"][f]) for f in FEATURES}
       nodeobs[n]=(lat,loss,util,avail,qual,max(fs.values()))
      a=nodeobs["actuator"];success=1-a[1];hist.append(success);me=sum(hist[-WINDOW:])/len(hist[-WINDOW:]);failure=int(me<FAIL)
      mat=np.array([[*nodeobs[n][:5],nodeobs[n][5]] for n in N],dtype=np.float32);tele_hist.append(mat)
      rawp=calp=np.nan
      # Exact E1 eligibility: current 10-step history failure-free.
      if len(tele_hist)>=HISTORY and len(hist)>=HISTORY and max(int((sum(hist[max(0,j-WINDOW+1):j+1])/len(hist[max(0,j-WINDOW+1):j+1]))<FAIL) for j in range(t-HISTORY+1,t+1))==0:
       X=np.stack(tele_hist[-HISTORY:]);X=(X-mean)/std;xt=torch.tensor(X[None,...],dtype=torch.float32,device=device)
       with torch.no_grad():
        logits,nw,gate=models[H](xt);rawp=float(torch.sigmoid(logits)[0].cpu());weights=nw[0,-1,:].cpu().numpy()
       calp=temp_apply(rawp,temps[H])
       if (not intervened) and calp>=thresholds[H]:
        chosen=max(ELIGIBLE,key=lambda n:(float(weights[N.index(n)]),-ELIGIBLE.index(n)));alarm_t=t;alarm_prob=calp;intervened=True
        # restore_service: restore both interfaces incident to selected node to benign link service
        reconfigure(net,base,set(nodeints[chosen]));time.sleep(.2)
        prev={n:{it:ibytes(h[n],it) for it in nodeints[n]} for n in N}
        print(f" {rid}: INTERVENE t={t} target={chosen} p={calp:.6f}")
      for n in N:
       x=nodeobs[n];rows.append({"run_id":rid,"baseline_run_id":baseid,"horizon":H,"condition":condition,"run":rr,"time":t,
        "phase":"post" if t>=40 else "pre","node_id":n,"latency":x[0],"packet_loss":x[1],"utilization":x[2],"availability":x[3],
        "sensor_quality":x[4],"detector_score":x[5],"mission_effectiveness":me,"failure":failure,
        "raw_probability":rawp,"calibrated_probability":calp,"intervened":int(intervened),"intervention_time":alarm_t if alarm_t is not None else "",
        "intervention_target":chosen or ""})
      if t%10==9:print(f" {rid}: {t-9:03d}-{t:03d} | ME={me:.3f} failure={failure}")
     with rf.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
     sha=hashlib.sha256(rf.read_bytes()).hexdigest();dn.write_text(json.dumps({"run_id":rid,"rows":len(rows),"sha256":sha,"complete":True,
      "intervened":intervened,"intervention_time":alarm_t,"target":chosen,"alarm_probability":alarm_prob},indent=2))
     print(f"COMPLETE {rid} | intervention={intervened} t={alarm_t} target={chosen}")
    finally:
     if h:
      for n in N:h[n].cmd("pkill -f '/tmp/missionshield_e10h/' >/dev/null 2>&1 || true")
     if net:net.stop()

 # consolidate + paired outcomes
 base_df=pd.read_csv(BASELINE);allrows=[];summary=[]
 for H in HORIZONS:
  for condition in conds:
   for rr in range(1,6):
    bid=f"{condition}_r{rr:02d}";rid=f"H{H}_{bid}";rf=RUNDIR/f"{rid}.csv";dn=RUNDIR/f"{rid}.done.json"
    if not(rf.exists() and dn.exists()):continue
    g=pd.read_csv(rf);allrows.append(g);act=g[g.node_id=="actuator"].sort_values("time");ba=base_df[(base_df.run_id==bid)&(base_df.node_id=="actuator")].sort_values("time")
    meta=json.loads(dn.read_text());bf=int(ba.failure.max());pf=int(act.failure.max())
    summary.append({"horizon":H,"baseline_run_id":bid,"condition":condition,"intervened":int(meta["intervened"]),
     "intervention_time":meta["intervention_time"],"target":meta["target"],"baseline_ever_failure":bf,"post_ever_failure":pf,
     "prevented":int(bf==1 and pf==0),"induced":int(bf==0 and pf==1),
     "baseline_failure_intervals":int(ba.failure.sum()),"post_failure_intervals":int(act.failure.sum()),
     "baseline_mean_me":float(ba.mission_effectiveness.mean()),"post_mean_me":float(act.mission_effectiveness.mean()),
     "mean_me_gain":float(act.mission_effectiveness.mean()-ba.mission_effectiveness.mean())})
 if allrows:pd.concat(allrows,ignore_index=True).to_csv(OUT,index=False)
 SDF=pd.DataFrame(summary);SDF.to_csv(SUM,index=False)
 print("\nE10-H COLLECTION STATUS:",len(summary),"/ 60 intervention replays complete")
 if len(summary)==60:
  print("\nE10-H PAIRED SUMMARY")
  for H,g in SDF.groupby("horizon"):
   print({"horizon":int(H),"runs":len(g),"interventions":int(g.intervened.sum()),"baseline_failed_runs":int(g.baseline_ever_failure.sum()),
    "post_failed_runs":int(g.post_ever_failure.sum()),"prevented":int(g.prevented.sum()),"induced":int(g.induced.sum()),
    "failure_interval_reduction":int(g.baseline_failure_intervals.sum()-g.post_failure_intervals.sum()),
    "mean_me_gain":float(g.mean_me_gain.mean()),
    "median_intervention_time":float(g.loc[g.intervened==1,"intervention_time"].median()) if g.intervened.any() else np.nan})
  print("E10-H COMPLETE | PAIRED MININET INTERVENTION VALIDATION")
 else:print("Re-run same command to resume.")
if __name__=="__main__":
 setLogLevel("warning");main()
