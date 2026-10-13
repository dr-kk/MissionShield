#!/usr/bin/env python3
"""
MissionShield E10-E | Benign-only six-node calibration collector
Frozen contract: 5 runs x 60 intervals, 8 tasks/interval, benign 20 Mbps/2 ms/0%.
Collects node-level latency/loss/utilization/availability/sensor_quality.
detector_score is NOT computed here.
No MissionShield inference and no non-benign condition.
"""
import csv,json,math,time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

ROOT=Path("/mnt/d/MissionShield"); RES=ROOT/"results"
CONTRACT=RES/"e10e_telemetry_detector_contract.json"
OUT=RES/"e10e_benign_calibration_raw.csv"
SUMMARY=RES/"e10e_benign_calibration_summary.csv"
N=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
S=["sen","tlink","edge","fus","cmd","act"]
RUNS=5; INTERVALS=60; TASKS=8; PORT=7000; DEADLINE=500.; TOL=.05
BW=20; DELAY="2ms"; LOSS=0; PREFLIGHT_ATTEMPTS=5

RELAY=r"""
import json,socket,sys,time
li=sys.argv[1];port=int(sys.argv[2]);si=sys.argv[3];ni=sys.argv[4];node=sys.argv[5];out=sys.argv[6]
rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind((li,port))
tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);tx.bind((si,0))
with open(out,"a",buffering=1) as f:
 while True:
  d,a=rx.recvfrom(65535); now=time.time_ns()
  m=json.loads(d.decode()); m.setdefault("stage_rx_ns",{})[node]=now
  f.write(json.dumps({"task_id":m["task_id"],"interval":m["interval"],"truth":m["truth"],
                      "value":m["value"],"send_ns":m["send_ns"],"rx_ns":now},
                     separators=(",",":"))+"\n")
  tx.sendto(json.dumps(m,separators=(",",":")).encode(),(ni,port))
"""
ACT=r"""
import json,socket,sys,time
ip=sys.argv[1];port=int(sys.argv[2]);out=sys.argv[3]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((ip,port))
with open(out,"a",buffering=1) as f:
 while True:
  d,a=s.recvfrom(65535); now=time.time_ns();m=json.loads(d.decode())
  f.write(json.dumps({"task_id":m["task_id"],"interval":m["interval"],"truth":m["truth"],
                      "value":m["value"],"send_ns":m["send_ns"],"rx_ns":now},
                     separators=(",",":"))+"\n")
"""
TX=r"""
import socket,sys
src=sys.argv[1];dst=sys.argv[2];port=int(sys.argv[3]);fn=sys.argv[4]
d=open(fn,"rb").read();s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((src,0));s.sendto(d,(dst,port))
"""

def ibytes(h,intf):
 vals=[]
 for x in h.cmd(f"cat /sys/class/net/{intf}/statistics/tx_bytes /sys/class/net/{intf}/statistics/rx_bytes").split():
  try: vals.append(int(x))
  except: pass
 return sum(vals[:2])

def readj(h,fn):
 z=[]
 for line in h.cmd(f"cat {fn} 2>/dev/null || true").splitlines():
  try:z.append(json.loads(line))
  except:pass
 return z

def build():
 net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
 h={n:net.addHost(s) for n,s in zip(N,S)}; links=[]
 for i in range(5):
  sw=net.addSwitch(f"s{i+1}")
  l=net.addLink(h[N[i]],sw,cls=TCLink,intfName1=f"{S[i]}e{i}",intfName2=f"s{i+1}a",bw=BW,delay=DELAY,loss=LOSS)
  r=net.addLink(sw,h[N[i+1]],cls=TCLink,intfName1=f"s{i+1}b",intfName2=f"{S[i+1]}e{i}",bw=BW,delay=DELAY,loss=LOSS)
  links.append((l,r))
 net.start();pair={};nodeints={n:[] for n in N}
 for i,(a,b) in enumerate(zip(N[:-1],N[1:])):
  l,r=links[i];ia=l.intf1 if l.intf1.node==h[a] else l.intf2;ib=r.intf1 if r.intf1.node==h[b] else r.intf2
  A=f"10.{i+1}.0.1";B=f"10.{i+1}.0.2"
  h[a].cmd(f"ip addr flush dev {ia};ip addr add {A}/24 dev {ia};ip link set {ia} up")
  h[b].cmd(f"ip addr flush dev {ib};ip addr add {B}/24 dev {ib};ip link set {ib} up")
  pair[(a,b)]=(A,B);nodeints[a].append(str(ia));nodeints[b].append(str(ib))
 return net,h,pair,nodeints

def main():
 c=json.loads(CONTRACT.read_text())
 assert c["benign_calibration"]["runs"]==RUNS and c["benign_calibration"]["intervals_per_run"]==INTERVALS
 d=Path("/tmp/missionshield_e10e");d.mkdir(exist_ok=True)
 (d/"relay.py").write_text(RELAY);(d/"act.py").write_text(ACT);(d/"tx.py").write_text(TX)
 rows=[]
 print("="*92);print("MissionShield E10-E | BENIGN-ONLY SIX-NODE CALIBRATION COLLECTION")
 print("5 runs x 60 intervals | detector_score NOT fitted yet | NO MissionShield inference");print("="*92)
 for run in range(1,RUNS+1):
  net=h=None
  try:
   net,h,pair,nodeints=build()
   files={n:f"/tmp/e10e_r{run}_{S[i]}.jsonl" for i,n in enumerate(N)}
   for n in N:h[n].cmd(f"rm -f {files[n]}")
   h["actuator"].cmd(f"python3 {d/'act.py'} {pair[('command','actuator')][1]} {PORT} {files['actuator']} >/tmp/e10e_act.out 2>&1 &")
   for i in range(4,0,-1):
    n=N[i];pr=N[i-1];nx=N[i+1]
    h[n].cmd(f"python3 {d/'relay.py'} {pair[(pr,n)][1]} {PORT} {pair[(n,nx)][0]} {pair[(n,nx)][1]} {n} {files[n]} >/tmp/e10e_{S[i]}.out 2>&1 &")
   time.sleep(.6);src=pair[("sensor","tactical_link")][0];dst=pair[("sensor","tactical_link")][1];mf=f"/tmp/e10e_msg_r{run}.json"
   # benign engineering preflight, excluded
   ok=False
   for pa in range(PREFLIGHT_ATTEMPTS):
    m={"task_id":f"__pf_{run}_{pa}__","truth":0.,"value":0.,"send_ns":time.time_ns(),"interval":-1}
    Path(mf).write_text(json.dumps(m,separators=(",",":")));h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {mf}");time.sleep(.3)
    if any(x["task_id"]==m["task_id"] for x in readj(h["actuator"],files["actuator"])):ok=True;break
   if not ok:raise RuntimeError(f"run {run} preflight failed")
   print(f"\nRUN {run}/{RUNS} PREFLIGHT PASS")
   prev={n:{i:ibytes(h[n],i) for i in nodeints[n]} for n in N}
   for t in range(INTERVALS):
    t0=time.time();sent={}
    for k in range(TASKS):
     tid=f"r{run}_{t:03d}_{k:02d}";truth=math.sin(((run-1)*INTERVALS*TASKS+t*TASKS+k)/10.)
     m={"task_id":tid,"truth":truth,"value":truth,"send_ns":time.time_ns(),"interval":t};sent[tid]=m
     Path(mf).write_text(json.dumps(m,separators=(",",":")));h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {mf}");time.sleep(.02)
    if time.time()-t0<1:time.sleep(1-(time.time()-t0))
    for n in N:
     utils=[]
     for it in nodeints[n]:
      now=ibytes(h[n],it);delta=max(0,now-prev[n][it]);prev[n][it]=now
      utils.append(min(1.,delta/(BW*1e6/8.)))
     util=max(utils) if utils else 0.
     if n=="sensor":
      rec=list(sent.values());lat=0.;ontime=TASKS;avail=1.;qual=1.;loss=0.
     else:
      allr={x["task_id"]:x for x in readj(h[n],files[n]) if x.get("interval")==t}
      rec=[allr[x] for x in sent if x in allr]; lats=[(x["rx_ns"]-sent[x["task_id"]]["send_ns"])/1e6 for x in rec]
      lat=sum(lats)/len(lats) if lats else float("nan")
      ontime=sum(1 for x in rec if (x["rx_ns"]-sent[x["task_id"]]["send_ns"])/1e6<=DEADLINE)
      avail=len(rec)/TASKS;loss=1-ontime/TASKS
      qual=sum(abs(float(x["value"])-float(x["truth"]))<=TOL for x in rec)/len(rec) if rec else 0.
     rows.append({"run":run,"interval":t,"node_id":n,"latency":lat,"packet_loss":loss,
                  "utilization":util,"availability":avail,"sensor_quality":qual,"detector_score":""})
    if t%10==9:print(f" run={run} intervals={t-9:02d}-{t:02d} collected")
  finally:
   if h:
    for n in N:h[n].cmd("pkill -f '/tmp/missionshield_e10e/' >/dev/null 2>&1 || true")
   if net:net.stop()
 with OUT.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for n in N:
  g=[x for x in rows if x["node_id"]==n];ls=[x["latency"] for x in g if math.isfinite(x["latency"])]
  summary.append({"node_id":n,"n":len(g),"mean_latency":sum(ls)/len(ls) if ls else float("nan"),
                  "mean_packet_loss":sum(x["packet_loss"] for x in g)/len(g),
                  "mean_utilization":sum(x["utilization"] for x in g)/len(g),
                  "mean_availability":sum(x["availability"] for x in g)/len(g),
                  "mean_sensor_quality":sum(x["sensor_quality"] for x in g)/len(g)})
 with SUMMARY.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 print("\nE10-E BENIGN CALIBRATION COLLECTION SUMMARY")
 for x in summary:print(x)
 print(f"\nRows: {len(rows)} (expected {RUNS*INTERVALS*len(N)})")
 print(f"Raw: {OUT}\nSummary: {SUMMARY}")
 print("E10-E COLLECTION COMPLETE")
 print("No detector score fitted yet; no MissionShield inference performed.")
if __name__=="__main__":
 setLogLevel("warning");main()
