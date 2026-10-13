#!/usr/bin/env python3
"""
MissionShield E10-D v4 | sender-to-first-relay diagnostic.
No scientific outcome, no MissionShield inference, no E10-C change.
"""
import time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

PORT=7000

RX=r"""
import socket,sys,time
ip=sys.argv[1]; port=int(sys.argv[2]); out=sys.argv[3]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((ip,port))
with open(out,"a",buffering=1) as f:
 f.write("READY\n")
 d,a=s.recvfrom(65535)
 f.write("RX %d %s %d\n"%(len(d),a[0],a[1]))
 f.write(d.decode(errors="replace")+"\n")
"""

TX=r"""
import socket,sys
src=sys.argv[1]; dst=sys.argv[2]; port=int(sys.argv[3]); msg=sys.argv[4]
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((src,0))
n=s.sendto(msg.encode(),(dst,port))
print("TX_OK bytes=%d src=%s dst=%s:%d"%(n,src,dst,port))
"""

def main():
 d=Path("/tmp/missionshield_e10dv4"); d.mkdir(exist_ok=True)
 (d/"rx.py").write_text(RX); (d/"tx.py").write_text(TX)

 net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
 sensor=net.addHost("sen"); tactical=net.addHost("tlink"); sw=net.addSwitch("s1")
 l=net.addLink(sensor,sw,cls=TCLink,intfName1="sene0",intfName2="s1a",
               bw=20,delay="2ms",loss=0)
 r=net.addLink(sw,tactical,cls=TCLink,intfName1="s1b",intfName2="tlinke0",
               bw=20,delay="2ms",loss=0)
 net.start()
 try:
  ia=l.intf1 if l.intf1.node==sensor else l.intf2
  ib=r.intf1 if r.intf1.node==tactical else r.intf2
  sensor.cmd(f"ip addr flush dev {ia}; ip addr add 10.1.0.1/24 dev {ia}")
  tactical.cmd(f"ip addr flush dev {ib}; ip addr add 10.1.0.2/24 dev {ib}")

  print("="*80)
  print("MissionShield E10-D v4 | SENSOR -> FIRST RELAY PREFLIGHT")
  print("Engineering diagnostic only")
  print("="*80)
  ping=sensor.cmd("LC_ALL=C ping -n -c 1 -W 1 10.1.0.2")
  print("PING:", "PASS" if "0% packet loss" in ping else "FAIL")

  trace="/tmp/e10dv4_rx.trace"
  tactical.cmd(f"rm -f {trace}")
  tactical.cmd(f"python3 {d/'rx.py'} 10.1.0.2 {PORT} {trace} >/tmp/e10dv4_rx.stdout 2>&1 &")
  time.sleep(.4)
  print("RECEIVER BEFORE SEND:", tactical.cmd(f"cat {trace} 2>/dev/null || true").strip() or "<EMPTY>")

  # Crucial change: use a helper script with ordinary argv; no nested python -c quoting.
  txout=sensor.cmd(f"python3 {d/'tx.py'} 10.1.0.1 10.1.0.2 {PORT} E10DV4")
  print("SENDER OUTPUT:", txout.strip() or "<EMPTY>")
  time.sleep(.5)

  final=tactical.cmd(f"cat {trace} 2>/dev/null || true").strip()
  print("RECEIVER AFTER SEND:")
  print(final or "<EMPTY>")
  if "RX " in final and "E10DV4" in final:
   print("\nSENDER PREFLIGHT: PASS")
   print("Root cause isolated: previous inline python -c sender invocation was faulty.")
  else:
   print("\nSENDER PREFLIGHT: FAIL")
   print("Sender stdout:", txout)
   print("Receiver stdout:", tactical.cmd("cat /tmp/e10dv4_rx.stdout 2>/dev/null || true"))
  print("No scientific E10-D outcome recorded.")
 finally:
  sensor.cmd("pkill -f '/tmp/missionshield_e10dv4/' >/dev/null 2>&1 || true")
  tactical.cmd("pkill -f '/tmp/missionshield_e10dv4/' >/dev/null 2>&1 || true")
  net.stop()

if __name__=="__main__":
 setLogLevel("warning"); main()
