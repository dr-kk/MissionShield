#!/usr/bin/env python3
"""
MissionShield E10-D v5 | corrected full-chain preflight.
Engineering validation only. No scientific outcome / no MissionShield inference.
Uses standalone sender helper; frozen E10-C parameters are unchanged.
"""
import json,time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

N=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
S=["sen","tlink","edge","fus","cmd","act"]
PORT=7000

RELAY=r"""
import json,socket,sys,time
li=sys.argv[1]; port=int(sys.argv[2]); si=sys.argv[3]; ni=sys.argv[4]; node=sys.argv[5]; tr=sys.argv[6]
def log(x):
 with open(tr,"a",buffering=1) as f:f.write(x+"\n")
rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind((li,port))
tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);tx.bind((si,0))
log("READY")
while True:
 d,a=rx.recvfrom(65535);log("RX")
 m=json.loads(d.decode());m.setdefault("hops",[]).append({"node":node,"t_ns":time.time_ns()})
 d=json.dumps(m,separators=(",",":")).encode();tx.sendto(d,(ni,port));log("TX")
"""
ACT=r"""
import json,socket,sys,time
ip=sys.argv[1];port=int(sys.argv[2]);out=sys.argv[3];tr=sys.argv[4]
def log(x):
 with open(tr,"a",buffering=1) as f:f.write(x+"\n")
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((ip,port));log("READY")
with open(out,"a",buffering=1) as f:
 while True:
  d,a=s.recvfrom(65535);log("RX");m=json.loads(d.decode());m["recv_ns"]=time.time_ns()
  f.write(json.dumps(m,separators=(",",":"))+"\n")
"""
TX=r"""
import socket,sys
src=sys.argv[1];dst=sys.argv[2];port=int(sys.argv[3]);fn=sys.argv[4]
data=open(fn,"rb").read()
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind((src,0))
print("TX_OK",s.sendto(data,(dst,port)))
"""

def main():
 d=Path("/tmp/missionshield_e10dv5");d.mkdir(exist_ok=True)
 (d/"relay.py").write_text(RELAY);(d/"act.py").write_text(ACT);(d/"tx.py").write_text(TX)
 net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
 h={n:net.addHost(s) for n,s in zip(N,S)};links=[]
 for i in range(5):
  sw=net.addSwitch(f"s{i+1}")
  l=net.addLink(h[N[i]],sw,cls=TCLink,intfName1=f"{S[i]}e{i}",intfName2=f"s{i+1}a",bw=20,delay="2ms",loss=0)
  r=net.addLink(sw,h[N[i+1]],cls=TCLink,intfName1=f"s{i+1}b",intfName2=f"{S[i+1]}e{i}",bw=20,delay="2ms",loss=0)
  links.append((l,r))
 net.start()
 try:
  pair={}
  for i,(a,b) in enumerate(zip(N[:-1],N[1:])):
   l,r=links[i];ia=l.intf1 if l.intf1.node==h[a] else l.intf2;ib=r.intf1 if r.intf1.node==h[b] else r.intf2
   A=f"10.{i+1}.0.1";B=f"10.{i+1}.0.2"
   h[a].cmd(f"ip addr flush dev {ia};ip addr add {A}/24 dev {ia}")
   h[b].cmd(f"ip addr flush dev {ib};ip addr add {B}/24 dev {ib}")
   pair[(a,b)]=(A,B)
  print("="*84);print("MissionShield E10-D v5 | CORRECTED END-TO-END PREFLIGHT");print("="*84)
  for a,b in zip(N[:-1],N[1:]):
   assert "0% packet loss" in h[a].cmd(f"LC_ALL=C ping -n -c 1 -W 1 {pair[(a,b)][1]}")
  print("ADJACENCY CHECK: PASS")
  out="/tmp/e10dv5_received.jsonl";h["actuator"].cmd(f"rm -f {out} /tmp/e10dv5_*.trace")
  actip=pair[("command","actuator")][1]
  h["actuator"].cmd(f"python3 {d/'act.py'} {actip} {PORT} {out} /tmp/e10dv5_act.trace >/tmp/e10dv5_act.out 2>&1 &")
  for i in range(4,0,-1):
   node=N[i];prev=N[i-1];nxt=N[i+1]
   li=pair[(prev,node)][1];si=pair[(node,nxt)][0];ni=pair[(node,nxt)][1]
   h[node].cmd(f"python3 {d/'relay.py'} {li} {PORT} {si} {ni} {node} /tmp/e10dv5_{S[i]}.trace >/tmp/e10dv5_{S[i]}.out 2>&1 &")
  time.sleep(.6)
  msg={"task_id":"__preflight_v5__","truth":0.0,"value":0.0,"send_ns":time.time_ns(),"interval":-1}
  mf="/tmp/e10dv5_message.json";Path(mf).write_text(json.dumps(msg,separators=(",",":")))
  src=pair[("sensor","tactical_link")][0];dst=pair[("sensor","tactical_link")][1]
  print("SENDER:",h["sensor"].cmd(f"python3 {d/'tx.py'} {src} {dst} {PORT} {mf}").strip())
  time.sleep(1)
  print("\nTRACE COUNTS")
  for i in range(1,5):
   node=N[i];tr=h[node].cmd(f"cat /tmp/e10dv5_{S[i]}.trace 2>/dev/null || true").strip()
   print(node,repr(tr))
  print("actuator",repr(h["actuator"].cmd("cat /tmp/e10dv5_act.trace 2>/dev/null || true").strip()))
  raw=h["actuator"].cmd(f"cat {out} 2>/dev/null || true").strip()
  if not raw: raise RuntimeError("END-TO-END PREFLIGHT FAILED")
  m=json.loads(raw.splitlines()[-1]);hops=[x["node"] for x in m["hops"]]
  exp=["tactical_link","edge_processing","fusion","command"]
  if hops!=exp: raise RuntimeError(f"Unexpected hops {hops}")
  lat=(m["recv_ns"]-msg["send_ns"])/1e6
  print(f"\nEND-TO-END PREFLIGHT: PASS | latency={lat:.3f} ms")
  print("HOPS: "+" -> ".join(hops))
  print("E10-D v5 COMPLETE")
  print("No scientific E10-D outcome recorded.")
 finally:
  if h:
   for x in N:h[x].cmd("pkill -f '/tmp/missionshield_e10dv5/' >/dev/null 2>&1 || true")
  net.stop()
if __name__=="__main__":
 setLogLevel("warning");main()
