#!/usr/bin/env python3
"""
MissionShield E10-D v7 | Corrected mission-application smoke campaign

Uses the sender/relay implementation proven by E10-D v5.
Frozen E10-C scientific settings are unchanged.
Conditions are the already-defined benign and combined smoke conditions.
No MissionShield inference. detector_score remains blank.
"""
import csv,json,math,time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

ROOT=Path("/mnt/d/MissionShield")
CONTRACT=ROOT/"results"/"e10c_feature_contract.json"
OUT=ROOT/"results"/"e10d_v7_smoke_telemetry.csv"
SUMMARY=ROOT/"results"/"e10d_v7_smoke_summary.csv"

N=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
S=["sen","tlink","edge","fus","cmd","act"]
PORT=7000; TASKS=8; INTERVALS=12; DEADLINE=500.; TOL=.05; WINDOW=10; FAIL=.70
PREFLIGHT_ATTEMPTS=5
CONDS={"benign":{"bw":20,"delay":"2ms","loss":0},
       "combined":{"bw":5,"delay":"15ms","loss":5}}

RELAY=r"""
import json,socket,sys,time
li=sys.argv[1];port=int(sys.argv[2]);si=sys.argv[3];ni=sys.argv[4];node=sys.argv[5]
rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind((li,port))
tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);tx.bind((si,0))
while True:
 d,a=rx.recvfrom(65535)
 m=json.loads(d.decode());m.setdefault("hops",[]).append({"node":node,"t_ns":time.time_ns()})
 tx.sendto(json.dumps(m,separators=(",",":")).encode(),(ni,port))
"""
ACT=r"""
import json,socket,sys,time
ip=sys.argv[1];port=int(sys.argv[2]);out=sys.argv[3]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((ip,port))
with open(out,"a",buffering=1) as f:
 while True:
  d,a=s.recvfrom(65535);m=json.loads(d.decode());m["recv_ns"]=time.time_ns()
  f.write(json.dumps(m,separators=(",",":"))+"\n")
"""
TX=r"""
import socket,sys
src=sys.argv[1];dst=sys.argv[2];port=int(sys.argv[3]);fn=sys.argv[4]
data=open(fn,"rb").read();s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((src,0))
s.sendto(data,(dst,port))
"""

def ibytes(host,intf):
 vals=[]
 for x in host.cmd(f"cat /sys/class/net/{intf}/statistics/tx_bytes /sys/class/net/{intf}/statistics/rx_bytes").split():
  try: vals.append(int(x))
  except: pass
 return sum(vals[:2])

def build(cfg):
 net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
 h={n:net.addHost(s) for n,s in zip(N,S)}; links=[]
 for i in range(5):
  sw=net.addSwitch(f"s{i+1}")
  l=net.addLink(h[N[i]],sw,cls=TCLink,intfName1=f"{S[i]}e{i}",intfName2=f"s{i+1}a",
                bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  r=net.addLink(sw,h[N[i+1]],cls=TCLink,intfName1=f"s{i+1}b",intfName2=f"{S[i+1]}e{i}",
                bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  links.append((l,r))
 net.start();pair={};ints={}
 for i,(a,b) in enumerate(zip(N[:-1],N[1:])):
  l,r=links[i];ia=l.intf1 if l.intf1.node==h[a] else l.intf2;ib=r.intf1 if r.intf1.node==h[b] else r.intf2
  A=f"10.{i+1}.0.1";B=f"10.{i+1}.0.2"
  h[a].cmd(f"ip addr flush dev {ia};ip addr add {A}/24 dev {ia};ip link set {ia} up")
  h[b].cmd(f"ip addr flush dev {ib};ip addr add {B}/24 dev {ib};ip link set {ib} up")
  pair[(a,b)]=(A,B);ints[(a,b)]=(str(ia),str(ib))
 return net,h,pair,ints

def read_recv(h,fn):
 out=[]
 for line in h["actuator"].cmd(f"cat {fn} 2>/dev/null || true").splitlines():
  try: out.append(json.loads(line))
  except: pass
 return out

def main():
 c=json.loads(CONTRACT.read_text())
 assert c["sampling"]["tasks_per_interval"]==TASKS
 assert c["sampling"]["mission_window_intervals"]==WINDOW
 assert c["sampling"]["mission_failure_threshold"]==FAIL
 assert c["application"]["deadline_ms"]==DEADLINE
 assert c["application"]["payload_tolerance_abs"]==TOL

 d=Path("/tmp/missionshield_e10dv7");d.mkdir(exist_ok=True)
 (d/"relay.py").write_text(RELAY);(d/"act.py").write_text(ACT);(d/"tx.py").write_text(TX)
 rows=[]
 print("="*90);print("MissionShield E10-D v7 | CORRECTED SMOKE CAMPAIGN")
 print("Frozen E10-C settings unchanged | NO MissionShield inference");print("="*90)

 for cond,cfg in CONDS.items():
  print(f"\n{cond.upper()} {cfg}");net=h=None
  recv=f"/tmp/e10dv7_{cond}.jsonl";msgfile=f"/tmp/e10dv7_msg_{cond}.json"
  try:
   net,h,pair,ints=build(cfg)
   for a,b in zip(N[:-1],N[1:]):
    if "0% packet loss" not in h[a].cmd(f"LC_ALL=C ping -n -c 1 -W 1 {pair[(a,b)][1]}"):
     raise RuntimeError(f"adjacency failed {a}->{b}")
   h["actuator"].cmd(f"rm -f {recv}")
   h["actuator"].cmd(f"python3 {d/'act.py'} {pair[('command','actuator')][1]} {PORT} {recv} >/tmp/e10dv7_act.out 2>&1 &")
   for i in range(4,0,-1):
    node=N[i];prev=N[i-1];nxt=N[i+1]
    h[node].cmd(f"python3 {d/'relay.py'} {pair[(prev,node)][1]} {PORT} {pair[(node,nxt)][0]} {pair[(node,nxt)][1]} {node} >/tmp/e10dv7_{S[i]}.out 2>&1 &")
   time.sleep(.6)

   # proven-style end-to-end preflight
   src=pair[("sensor","tactical_link")][0];dst=pair[("sensor","tactical_link")][1]
   # Engineering connectivity guard only. Under a lossy condition, one probe may
   # legitimately be dropped. Retry up to a fixed five probes; these probes are
   # excluded from all mission telemetry/outcomes.
   prs=[]
   for pa in range(PREFLIGHT_ATTEMPTS):
    pre={"task_id":f"__preflight_{pa}__","truth":0.,"value":0.,"send_ns":time.time_ns(),"interval":-1}
    Path(msgfile).write_text(json.dumps(pre,separators=(",",":")))
    h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {msgfile}")
    time.sleep(.5)
    prs=[x for x in read_recv(h,recv) if x.get("task_id")==pre["task_id"]]
    if prs:
     break
   if not prs: raise RuntimeError(f"{cond}: end-to-end preflight failed after {PREFLIGHT_ATTEMPTS} engineering probes")
   hops=[x["node"] for x in prs[-1].get("hops",[])]
   if hops!=["tactical_link","edge_processing","fusion","command"]: raise RuntimeError(f"bad hops {hops}")
   print(f" PREFLIGHT PASS attempt={pa+1}/{PREFLIGHT_ATTEMPTS} latency={(prs[-1]['recv_ns']-pre['send_ns'])/1e6:.2f} ms")

   prevbytes={}
   for a,b in zip(N[:-1],N[1:]):
    ia,ib=ints[(a,b)];prevbytes[(a,b)]=ibytes(h[a],ia)+ibytes(h[b],ib)
   hist=[]
   for t in range(INTERVALS):
    start=time.time();sent={}
    for k in range(TASKS):
     tid=f"{cond}_{t:03d}_{k:02d}";truth=math.sin((t*TASKS+k)/10.)
     m={"task_id":tid,"truth":truth,"value":truth,"send_ns":time.time_ns(),"interval":t}
     sent[tid]=m;Path(msgfile).write_text(json.dumps(m,separators=(",",":")))
     h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {msgfile}")
     time.sleep(.02)
    if time.time()-start<1:time.sleep(1-(time.time()-start))
    rec={x["task_id"]:x for x in read_recv(h,recv) if x.get("interval")==t}
    delivered=correct=ontime=0;lats=[]
    for tid,sm in sent.items():
     rm=rec.get(tid)
     if rm is None:continue
     delivered+=1;lat=(rm["recv_ns"]-sm["send_ns"])/1e6;lats.append(lat)
     ok=abs(float(rm["value"])-float(sm["truth"]))<=TOL
     correct+=int(ok);ontime+=int(ok and lat<=DEADLINE)
    hist.append(ontime/TASKS);me=sum(hist[-WINDOW:])/len(hist[-WINDOW:])
    utils=[]
    for a,b in zip(N[:-1],N[1:]):
     ia,ib=ints[(a,b)];now=ibytes(h[a],ia)+ibytes(h[b],ib);delta=max(0,now-prevbytes[(a,b)]);prevbytes[(a,b)]=now
     utils.append(min(1.,delta/(cfg["bw"]*1e6/8.)))
    row={"condition":cond,"interval":t,"expected_tasks":TASKS,"delivered_tasks":delivered,
         "on_time_correct_tasks":ontime,"latency_ms":sum(lats)/len(lats) if lats else float("nan"),
         "packet_loss":1-ontime/TASKS,"utilization":sum(utils)/len(utils),
         "availability":delivered/TASKS,"sensor_quality":correct/delivered if delivered else 0.,
         "mission_effectiveness":me,"failure":int(me<FAIL),"detector_score":""}
    rows.append(row)
    lt=f"{row['latency_ms']:.2f}" if math.isfinite(row["latency_ms"]) else "nan"
    print(f" interval={t:02d} delivered={delivered}/8 ontime={ontime}/8 lat={lt}ms ME={me:.3f} failure={row['failure']}")
  finally:
   if h:
    for x in N:h[x].cmd("pkill -f '/tmp/missionshield_e10dv7/' >/dev/null 2>&1 || true")
   if net:net.stop()

 with OUT.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summ=[]
 for cond in CONDS:
  g=[r for r in rows if r["condition"]==cond];lat=[r["latency_ms"] for r in g if math.isfinite(r["latency_ms"])]
  summ.append({"condition":cond,"intervals":len(g),"mean_latency_ms":sum(lat)/len(lat) if lat else float("nan"),
   "mean_packet_loss":sum(r["packet_loss"] for r in g)/len(g),"mean_utilization":sum(r["utilization"] for r in g)/len(g),
   "mean_availability":sum(r["availability"] for r in g)/len(g),"mean_sensor_quality":sum(r["sensor_quality"] for r in g)/len(g),
   "final_mission_effectiveness":g[-1]["mission_effectiveness"],"failure_intervals":sum(r["failure"] for r in g)})
 with SUMMARY.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(summ[0]));w.writeheader();w.writerows(summ)
 print("\nE10-D v7 SUMMARY")
 for x in summ:print(x)
 print(f"\nRaw: {OUT}\nSummary: {SUMMARY}")
 print("E10-D v7 COMPLETE")
 print("detector_score intentionally blank; NO MissionShield inference performed.")
if __name__=="__main__":
 setLogLevel("warning");main()
