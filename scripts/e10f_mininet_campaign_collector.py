#!/usr/bin/env python3
"""
MissionShield E10-F | Resume-safe onset-based Mininet campaign collector

Protocol is read from results/e10f_campaign_protocol.json.
Collects six-node telemetry only. No MissionShield inference.
Detector score is computed strictly from the frozen E10-E benign calibration.
Completed runs are checkpointed individually and skipped on restart.
"""
import csv,json,math,time,hashlib
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

ROOT=Path("/mnt/d/MissionShield"); RES=ROOT/"results"
PROTOCOL=RES/"e10f_campaign_protocol.json"
CAL=RES/"e10e_detector_calibration.json"
RUNDIR=RES/"e10f_runs"; RUNDIR.mkdir(parents=True,exist_ok=True)
OUT=RES/"e10f_emulator_telemetry.csv"; RUNSUM=RES/"e10f_run_summary.csv"

N=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
S=["sen","tlink","edge","fus","cmd","act"]
FEATURES=["latency","packet_loss","utilization","availability"]
PORT=7000; PREFLIGHT_ATTEMPTS=5

RELAY=r"""
import json,socket,sys,time
li=sys.argv[1];port=int(sys.argv[2]);si=sys.argv[3];ni=sys.argv[4];node=sys.argv[5];out=sys.argv[6]
rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind((li,port))
tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);tx.bind((si,0))
with open(out,"a",buffering=1) as f:
 while True:
  d,a=rx.recvfrom(65535);now=time.time_ns();m=json.loads(d.decode())
  f.write(json.dumps({"task_id":m["task_id"],"interval":m["interval"],"truth":m["truth"],
   "value":m["value"],"send_ns":m["send_ns"],"rx_ns":now},separators=(",",":"))+"\n")
  tx.sendto(d,(ni,port))
"""
ACT=r"""
import json,socket,sys,time
ip=sys.argv[1];port=int(sys.argv[2]);out=sys.argv[3]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((ip,port))
with open(out,"a",buffering=1) as f:
 while True:
  d,a=s.recvfrom(65535);now=time.time_ns();m=json.loads(d.decode())
  f.write(json.dumps({"task_id":m["task_id"],"interval":m["interval"],"truth":m["truth"],
   "value":m["value"],"send_ns":m["send_ns"],"rx_ns":now},separators=(",",":"))+"\n")
"""
TX=r"""
import socket,sys
src=sys.argv[1];dst=sys.argv[2];port=int(sys.argv[3]);fn=sys.argv[4]
d=open(fn,"rb").read();s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((src,0));s.sendto(d,(dst,port))
"""

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
 n=len(vals)
 extreme=sum(v>=x for v in vals) if tail=="high" else sum(v<=x for v in vals)
 return 1-(1+extreme)/(n+1)

def build(cfg):
 net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
 h={n:net.addHost(s) for n,s in zip(N,S)};links=[]
 for i in range(5):
  sw=net.addSwitch(f"s{i+1}")
  l=net.addLink(h[N[i]],sw,cls=TCLink,intfName1=f"{S[i]}e{i}",intfName2=f"s{i+1}a",
   bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  r=net.addLink(sw,h[N[i+1]],cls=TCLink,intfName1=f"s{i+1}b",intfName2=f"{S[i+1]}e{i}",
   bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  links.append((l,r))
 net.start();pair={};nodeints={n:[] for n in N}
 for i,(a,b) in enumerate(zip(N[:-1],N[1:])):
  l,r=links[i];ia=l.intf1 if l.intf1.node==h[a] else l.intf2;ib=r.intf1 if r.intf1.node==h[b] else r.intf2
  A=f"10.{i+1}.0.1";B=f"10.{i+1}.0.2"
  h[a].cmd(f"ip addr flush dev {ia};ip addr add {A}/24 dev {ia};ip link set {ia} up")
  h[b].cmd(f"ip addr flush dev {ib};ip addr add {B}/24 dev {ib};ip link set {ib} up")
  pair[(a,b)]=(A,B);nodeints[a].append(str(ia));nodeints[b].append(str(ib))
 return net,h,pair,nodeints

def reconfigure(net,cfg):
 for link in net.links:
  for intf in (link.intf1,link.intf2):
   # Mininet TCLink config updates existing qdisc; protocol severity is unchanged.
   try:intf.config(bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
   except Exception as e:raise RuntimeError(f"TC reconfiguration failed on {intf}: {e}")

def main():
 P=json.loads(PROTOCOL.read_text()); C=json.loads(CAL.read_text())
 assert C["source_sha256"]=="d422d318b6f0a1dcfddee240cb7edfed35ca4b8dfdd5ad3ffbefba78ebdf0e2e"
 camp=P["campaign"]; mission=P["mission"]
 assert camp["runs_per_condition"]==5 and camp["intervals_per_run"]==150 and camp["impairment_onset_interval"]==40
 TASKS=camp["tasks_per_interval"]; DEADLINE=mission["deadline_ms"]; TOL=mission["payload_tolerance_abs"]
 WINDOW=mission["mission_window_intervals"]; FAIL=mission["failure_threshold"]
 base={"bw":20,"delay":"2ms","loss":0}
 conds={
  "latency":{"bw":20,"delay":"15ms","loss":0},
  "loss":{"bw":20,"delay":"2ms","loss":5},
  "bandwidth":{"bw":5,"delay":"2ms","loss":0},
  "combined":{"bw":5,"delay":"15ms","loss":5}}
 d=Path("/tmp/missionshield_e10f");d.mkdir(exist_ok=True)
 (d/"relay.py").write_text(RELAY);(d/"act.py").write_text(ACT);(d/"tx.py").write_text(TX)
 print("="*94);print("MissionShield E10-F | RESUME-SAFE ONSET-BASED MININET CAMPAIGN")
 print("20 runs | 150 intervals/run | onset=40 | NO MissionShield inference");print("="*94)
 total=0
 for condition,target in conds.items():
  for rr in range(1,6):
   runid=f"{condition}_r{rr:02d}";runfile=RUNDIR/f"{runid}.csv";done=RUNDIR/f"{runid}.done.json"
   if done.exists() and runfile.exists():
    print(f"SKIP completed {runid}");continue
   net=h=None; rows=[]; hist=[]
   try:
    net,h,pair,nodeints=build(base); files={n:f"/tmp/e10f_{runid}_{S[i]}.jsonl" for i,n in enumerate(N)}
    for n in N:h[n].cmd(f"rm -f {files[n]}")
    h["actuator"].cmd(f"python3 {d/'act.py'} {pair[('command','actuator')][1]} {PORT} {files['actuator']} >/tmp/e10f_act.out 2>&1 &")
    for i in range(4,0,-1):
     n=N[i];pr=N[i-1];nx=N[i+1]
     h[n].cmd(f"python3 {d/'relay.py'} {pair[(pr,n)][1]} {PORT} {pair[(n,nx)][0]} {pair[(n,nx)][1]} {n} {files[n]} >/tmp/e10f_{S[i]}.out 2>&1 &")
    time.sleep(.6);src=pair[("sensor","tactical_link")][0];dst=pair[("sensor","tactical_link")][1];mf=f"/tmp/e10f_{runid}.json"
    ok=False
    for pa in range(PREFLIGHT_ATTEMPTS):
     m={"task_id":f"__pf_{pa}__","truth":0.,"value":0.,"send_ns":time.time_ns(),"interval":-1}
     Path(mf).write_text(json.dumps(m,separators=(",",":")));h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {mf}");time.sleep(.3)
     if any(x["task_id"]==m["task_id"] for x in readj(h["actuator"],files["actuator"])):ok=True;break
    if not ok:raise RuntimeError(f"{runid} benign preflight failed")
    print(f"\nSTART {runid} | preflight PASS")
    prev={n:{it:ibytes(h[n],it) for it in nodeints[n]} for n in N}
    for t in range(150):
     if t==40:
      reconfigure(net,target);time.sleep(.2)
      # reset byte counters baseline after qdisc update/setup traffic
      prev={n:{it:ibytes(h[n],it) for it in nodeints[n]} for n in N}
      print(f" {runid}: impairment ON at interval 40 -> {target}")
     t0=time.time();sent={}
     for k in range(TASKS):
      tid=f"{runid}_{t:03d}_{k:02d}";truth=math.sin((t*TASKS+k)/10.)
      m={"task_id":tid,"truth":truth,"value":truth,"send_ns":time.time_ns(),"interval":t};sent[tid]=m
      Path(mf).write_text(json.dumps(m,separators=(",",":")));h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {mf}");time.sleep(.02)
     if time.time()-t0<1:time.sleep(1-(time.time()-t0))
     nodeobs={}
     for n in N:
      us=[]
      for it in nodeints[n]:
       now=ibytes(h[n],it);delta=max(0,now-prev[n][it]);prev[n][it]=now;us.append(min(1.,delta/((target["bw"] if t>=40 else 20)*1e6/8.)))
      util=max(us) if us else 0.
      if n=="sensor":lat=0.;loss=0.;avail=1.;qual=1.
      else:
       allr={x["task_id"]:x for x in readj(h[n],files[n]) if x.get("interval")==t};rec=[allr[x] for x in sent if x in allr]
       lats=[(x["rx_ns"]-sent[x["task_id"]]["send_ns"])/1e6 for x in rec];lat=sum(lats)/len(lats) if lats else float("nan")
       ontime=sum(1 for x in rec if (x["rx_ns"]-sent[x["task_id"]]["send_ns"])/1e6<=DEADLINE)
       loss=1-ontime/TASKS;avail=len(rec)/TASKS;qual=sum(abs(x["value"]-x["truth"])<=TOL for x in rec)/len(rec) if rec else 0.
      fs={f:score({"latency":lat,"packet_loss":loss,"utilization":util,"availability":avail}[f],C["calibration_values"][n][f],C["tail_direction"][f]) for f in FEATURES}
      ds=max(fs.values());nodeobs[n]=(lat,loss,util,avail,qual,ds)
     # Mission outcome is independently defined at actuator.
     a=nodeobs["actuator"];success=1-a[1];hist.append(success);me=sum(hist[-WINDOW:])/len(hist[-WINDOW:]);failure=int(me<FAIL)
     for n in N:
      x=nodeobs[n];rows.append({"run_id":runid,"condition":condition,"run":rr,"time":t,"phase":"post" if t>=40 else "pre",
       "node_id":n,"latency":x[0],"packet_loss":x[1],"utilization":x[2],"availability":x[3],"sensor_quality":x[4],
       "detector_score":x[5],"mission_effectiveness":me,"failure":failure})
     if t%10==9:print(f" {runid}: {t-9:03d}-{t:03d} | ME={me:.3f} failure={failure}")
    with runfile.open("w",newline="") as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sha=hashlib.sha256(runfile.read_bytes()).hexdigest()
    done.write_text(json.dumps({"run_id":runid,"rows":len(rows),"sha256":sha,"complete":True},indent=2))
    print(f"COMPLETE {runid} | rows={len(rows)} | failures={sum(r['failure'] for r in rows if r['node_id']=='actuator')}")
   finally:
    if h:
     for n in N:h[n].cmd("pkill -f '/tmp/missionshield_e10f/' >/dev/null 2>&1 || true")
    if net:net.stop()

 # consolidate only after all 20 done
 allrows=[]; sums=[]
 for condition in conds:
  for rr in range(1,6):
   rid=f"{condition}_r{rr:02d}";rf=RUNDIR/f"{rid}.csv";dn=RUNDIR/f"{rid}.done.json"
   if not (rf.exists() and dn.exists()):continue
   with rf.open(newline="") as f:g=list(csv.DictReader(f));allrows.extend(g)
   act=[x for x in g if x["node_id"]=="actuator"];post=[x for x in act if x["phase"]=="post"]
   sums.append({"run_id":rid,"condition":condition,"rows":len(g),
    "failure_intervals":sum(int(x["failure"]) for x in act),
    "post_failure_intervals":sum(int(x["failure"]) for x in post),
    "final_mission_effectiveness":float(act[-1]["mission_effectiveness"])})
 if allrows:
  with OUT.open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 if sums:
  with RUNSUM.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(sums[0]));w.writeheader();w.writerows(sums)
 print("\nE10-F COLLECTION STATUS:",len(sums),"/ 20 runs complete")
 print("Consolidated:",OUT);print("Run summary:",RUNSUM)
 if len(sums)==20:print("E10-F TELEMETRY COLLECTION COMPLETE")
 else:print("Re-run the same command to resume incomplete runs.")
 print("NO MissionShield inference performed.")
if __name__=="__main__":
 setLogLevel("warning");main()
