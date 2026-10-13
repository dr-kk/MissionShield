#!/usr/bin/env python3
import csv,json,math,time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

ROOT=Path("/mnt/d/MissionShield")
OUT=ROOT/"results"/"e10d_mission_application_telemetry.csv"
SUMMARY=ROOT/"results"/"e10d_mission_application_summary.csv"
CONTRACT=ROOT/"results"/"e10c_feature_contract.json"
NODES=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
SHORT=["sen","tlink","edge","fus","cmd","act"]
TASKS=8; INTERVALS=12; DEADLINE=500.; TOL=.05; WINDOW=10; FAIL=.70
CONDS={"benign":{"bw":20,"delay":"2ms","loss":0},
       "combined":{"bw":5,"delay":"15ms","loss":5}}

RELAY='''import socket,sys,json,time
ip=sys.argv[1]; port=int(sys.argv[2]); nip=sys.argv[3]; name=sys.argv[4]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind((ip,port))
o=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
while True:
 d,a=s.recvfrom(65535)
 try:
  m=json.loads(d.decode()); m.setdefault("hops",[]).append({"node":name,"t":time.time_ns()})
  d=json.dumps(m,separators=(",",":")).encode()
 except: pass
 o.sendto(d,(nip,port))
'''
ACT='''import socket,sys,json,time
ip=sys.argv[1]; port=int(sys.argv[2]); fn=sys.argv[3]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind((ip,port))
with open(fn,"a",buffering=1) as f:
 while True:
  d,a=s.recvfrom(65535); now=time.time_ns()
  try:
   m=json.loads(d.decode()); m["recv_ns"]=now
   f.write(json.dumps(m,separators=(",",":"))+"\\n")
  except: pass
'''

def helpers():
 d=Path("/tmp/missionshield_e10d"); d.mkdir(exist_ok=True)
 (d/"relay.py").write_text(RELAY); (d/"actuator.py").write_text(ACT); return d

def ibytes(host,intf):
 x=host.cmd(f"cat /sys/class/net/{intf}/statistics/tx_bytes /sys/class/net/{intf}/statistics/rx_bytes")
 v=[int(z) for z in x.split() if z.isdigit()]
 return sum(v[:2]) if len(v)>=2 else 0

def network(cfg):
 net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
 h={n:net.addHost(s) for n,s in zip(NODES,SHORT)}; links=[]
 for i in range(5):
  sw=net.addSwitch(f"s{i+1}")
  l=net.addLink(h[NODES[i]],sw,cls=TCLink,intfName1=f"{SHORT[i]}e{i}",
    intfName2=f"s{i+1}a",bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  r=net.addLink(sw,h[NODES[i+1]],cls=TCLink,intfName1=f"s{i+1}b",
    intfName2=f"{SHORT[i+1]}e{i}",bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"])
  links.append((l,r))
 net.start(); ips={}; ints={}
 for i,(a,b) in enumerate(zip(NODES[:-1],NODES[1:])):
  l,r=links[i]; ia=l.intf1 if l.intf1.node==h[a] else l.intf2
  ib=r.intf1 if r.intf1.node==h[b] else r.intf2
  A=f"10.{i+1}.0.1"; B=f"10.{i+1}.0.2"
  h[a].cmd(f"ip addr flush dev {ia}; ip addr add {A}/24 dev {ia}")
  h[b].cmd(f"ip addr flush dev {ib}; ip addr add {B}/24 dev {ib}")
  ips[b]=B; ints[(a,b)]=(str(ia),str(ib))
 return net,h,ips,ints

def main():
 c=json.loads(CONTRACT.read_text())
 assert c["status"]=="FROZEN_BEFORE_MODEL_INFERENCE"
 assert c["application"]["deadline_ms"]==500.0
 d=helpers(); OUT.parent.mkdir(parents=True,exist_ok=True); rows=[]
 print("="*88); print("MissionShield E10-D | SIX-STAGE UDP MISSION APPLICATION")
 print("NO MissionShield inference | detector_score intentionally absent"); print("="*88)
 for cond,cfg in CONDS.items():
  print(f"\\n{cond.upper()} {cfg}"); net=h=None
  fn=f"/tmp/e10d_{cond}.jsonl"
  try:
   net,h,ips,ints=network(cfg); h["actuator"].cmd(f"rm -f {fn}"); port=7000
   h["actuator"].cmd(f"python3 {d/'actuator.py'} {ips['actuator']} {port} {fn} >/tmp/e10d_act.log 2>&1 &")
   for i in range(4,0,-1):
    node=NODES[i]; nxt=NODES[i+1]
    h[node].cmd(f"python3 {d/'relay.py'} {ips[node]} {port} {ips[nxt]} {node} >/tmp/e10d_{SHORT[i]}.log 2>&1 &")
   time.sleep(.5)
   prev={}
   for a,b in zip(NODES[:-1],NODES[1:]):
    ia,ib=ints[(a,b)]; prev[(a,b)]=ibytes(h[a],ia)+ibytes(h[b],ib)
   hist=[]
   for t in range(INTERVALS):
    t0=time.time(); sent={}
    for k in range(TASKS):
     tid=f"{cond}_{t:03d}_{k:02d}"; truth=math.sin((t*TASKS+k)/10.)
     m={"task_id":tid,"truth":truth,"value":truth,"send_ns":time.time_ns(),"interval":t}
     sent[tid]=m; payload=json.dumps(m,separators=(",",":"))
     h["sensor"].cmd("python3 -c "+repr("import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.sendto("+repr(payload.encode())+",("+repr(ips["tactical_link"])+","+str(port)+"))"))
     time.sleep(.02)
    if time.time()-t0<1: time.sleep(1-(time.time()-t0))
    rec={}
    for line in h["actuator"].cmd(f"cat {fn} 2>/dev/null || true").splitlines():
     try:
      m=json.loads(line)
      if m.get("interval")==t: rec[m["task_id"]]=m
     except: pass
    delivered=correct=ontime=0; lats=[]
    for tid,sm in sent.items():
     rm=rec.get(tid)
     if not rm: continue
     delivered+=1; lat=(rm["recv_ns"]-sm["send_ns"])/1e6; lats.append(lat)
     ok=abs(float(rm["value"])-float(sm["truth"]))<=TOL
     correct+=int(ok); ontime+=int(ok and lat<=DEADLINE)
    success=ontime/TASKS; hist.append(success); me=sum(hist[-WINDOW:])/len(hist[-WINDOW:])
    us=[]
    for a,b in zip(NODES[:-1],NODES[1:]):
     ia,ib=ints[(a,b)]; now=ibytes(h[a],ia)+ibytes(h[b],ib)
     delta=max(0,now-prev[(a,b)]); prev[(a,b)]=now
     us.append(min(1.,delta/(cfg["bw"]*1e6/8.)))
    row={"condition":cond,"interval":t,"expected_tasks":TASKS,"delivered_tasks":delivered,
      "on_time_correct_tasks":ontime,"latency_ms":sum(lats)/len(lats) if lats else float("nan"),
      "packet_loss":1-ontime/TASKS,"utilization":sum(us)/len(us),
      "availability":delivered/TASKS,"sensor_quality":correct/delivered if delivered else 0.,
      "mission_effectiveness":me,"failure":int(me<FAIL),"detector_score":""}
    rows.append(row); print(f" interval={t:02d} delivered={delivered}/8 ontime={ontime}/8 ME={me:.3f} failure={row['failure']}")
  finally:
   if h:
    for n in NODES: h[n].cmd("pkill -f '/tmp/missionshield_e10d/' >/dev/null 2>&1 || true")
   if net: net.stop()
 with OUT.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 summ=[]
 for cond in CONDS:
  g=[r for r in rows if r["condition"]==cond]; lat=[r["latency_ms"] for r in g if math.isfinite(r["latency_ms"])]
  summ.append({"condition":cond,"intervals":len(g),"mean_latency_ms":sum(lat)/len(lat) if lat else float("nan"),
   "mean_packet_loss":sum(r["packet_loss"] for r in g)/len(g),"mean_utilization":sum(r["utilization"] for r in g)/len(g),
   "mean_availability":sum(r["availability"] for r in g)/len(g),"mean_sensor_quality":sum(r["sensor_quality"] for r in g)/len(g),
   "final_mission_effectiveness":g[-1]["mission_effectiveness"],"failure_intervals":sum(r["failure"] for r in g)})
 with SUMMARY.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=summ[0]); w.writeheader(); w.writerows(summ)
 print("\\nE10-D SUMMARY")
 for x in summ: print(x)
 print(f"\\nRaw: {OUT}\\nSummary: {SUMMARY}")
 print("E10-D COMPLETE"); print("detector_score intentionally blank; NO MissionShield inference performed.")
if __name__=="__main__":
 setLogLevel("warning"); main()
