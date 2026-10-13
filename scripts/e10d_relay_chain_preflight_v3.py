#!/usr/bin/env python3
"""
MissionShield E10-D v3 | Instrumented end-to-end relay-chain preflight

Diagnostic engineering stage only.
- Uses the already-validated five Mininet adjacencies.
- Starts the same four UDP application relays plus actuator.
- Sends exactly one preflight packet.
- Each relay writes an RX/TX trace before/after forwarding.
- No E10-D scientific outcome is recorded.
- No MissionShield inference.
- No frozen E10-C scientific parameter is changed.
"""
import json, time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

NODES=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
SHORT=["sen","tlink","edge","fus","cmd","act"]
PORT=7000

RELAY = r"""
import json,socket,sys,time,traceback
listen_ip=sys.argv[1]; port=int(sys.argv[2]); send_ip=sys.argv[3]
next_ip=sys.argv[4]; node=sys.argv[5]; trace=sys.argv[6]
def log(x):
    with open(trace,"a",buffering=1) as f: f.write(x+"\n")
try:
    rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    rx.bind((listen_ip,port))
    tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    tx.bind((send_ip,0))
    log("READY node=%s listen=%s:%d send=%s next=%s:%d" %
        (node,listen_ip,port,send_ip,next_ip,port))
    while True:
        data,addr=rx.recvfrom(65535)
        log("RX node=%s bytes=%d from=%s:%d" % (node,len(data),addr[0],addr[1]))
        try:
            m=json.loads(data.decode())
            m.setdefault("hops",[]).append({"node":node,"t_ns":time.time_ns()})
            data=json.dumps(m,separators=(",",":")).encode()
        except Exception:
            log("JSON_ERROR "+traceback.format_exc().replace("\n"," | "))
        n=tx.sendto(data,(next_ip,port))
        log("TX node=%s bytes=%d to=%s:%d" % (node,n,next_ip,port))
except Exception:
    log("FATAL "+traceback.format_exc().replace("\n"," | "))
    raise
"""

ACTUATOR = r"""
import json,socket,sys,time,traceback
ip=sys.argv[1]; port=int(sys.argv[2]); outfile=sys.argv[3]; trace=sys.argv[4]
def log(x):
    with open(trace,"a",buffering=1) as f: f.write(x+"\n")
try:
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    s.bind((ip,port))
    log("READY actuator listen=%s:%d" % (ip,port))
    with open(outfile,"a",buffering=1) as f:
        while True:
            data,addr=s.recvfrom(65535)
            log("RX actuator bytes=%d from=%s:%d" % (len(data),addr[0],addr[1]))
            m=json.loads(data.decode())
            m["recv_ns"]=time.time_ns()
            f.write(json.dumps(m,separators=(",",":"))+"\n")
except Exception:
    log("FATAL "+traceback.format_exc().replace("\n"," | "))
    raise
"""

def helpers():
    d=Path("/tmp/missionshield_e10dv3")
    d.mkdir(parents=True,exist_ok=True)
    (d/"relay.py").write_text(RELAY,encoding="utf-8")
    (d/"actuator.py").write_text(ACTUATOR,encoding="utf-8")
    return d

def build():
    net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
    h={n:net.addHost(s) for n,s in zip(NODES,SHORT)}
    links=[]
    for i in range(5):
        sw=net.addSwitch(f"s{i+1}")
        l=net.addLink(h[NODES[i]],sw,cls=TCLink,
            intfName1=f"{SHORT[i]}e{i}",intfName2=f"s{i+1}a",
            bw=20,delay="2ms",loss=0)
        r=net.addLink(sw,h[NODES[i+1]],cls=TCLink,
            intfName1=f"s{i+1}b",intfName2=f"{SHORT[i+1]}e{i}",
            bw=20,delay="2ms",loss=0)
        links.append((l,r))
    net.start()
    pair={}
    for i,(a,b) in enumerate(zip(NODES[:-1],NODES[1:])):
        l,r=links[i]
        ia=l.intf1 if l.intf1.node==h[a] else l.intf2
        ib=r.intf1 if r.intf1.node==h[b] else r.intf2
        A=f"10.{i+1}.0.1"; B=f"10.{i+1}.0.2"
        h[a].cmd(f"ip addr flush dev {ia}; ip addr add {A}/24 dev {ia}; ip link set {ia} up")
        h[b].cmd(f"ip addr flush dev {ib}; ip addr add {B}/24 dev {ib}; ip link set {ib} up")
        pair[(a,b)]=(A,B)
    return net,h,pair

def main():
    d=helpers()
    net=h=None
    traces={}
    recv="/tmp/e10dv3_received.jsonl"
    try:
        net,h,pair=build()
        print("="*88)
        print("MissionShield E10-D v3 | INSTRUMENTED RELAY-CHAIN PREFLIGHT")
        print("Engineering diagnostic only | NO scientific outcome | NO MissionShield inference")
        print("="*88)

        # Confirm physical adjacency again.
        for a,b in zip(NODES[:-1],NODES[1:]):
            B=pair[(a,b)][1]
            out=h[a].cmd(f"LC_ALL=C ping -n -c 1 -W 1 {B}")
            if "0% packet loss" not in out:
                raise RuntimeError(f"Adjacency failed: {a}->{b}")
        print("ADJACENCY CHECK: PASS")

        # Clean diagnostic files before processes start.
        h["actuator"].cmd(f"rm -f {recv} /tmp/e10dv3_*.trace")
        for node in NODES[1:-1]:
            traces[node]=f"/tmp/e10dv3_{SHORT[NODES.index(node)]}.trace"
        traces["actuator"]="/tmp/e10dv3_act.trace"

        act_ip=pair[("command","actuator")][1]
        h["actuator"].cmd(
            f"python3 {d/'actuator.py'} {act_ip} {PORT} {recv} {traces['actuator']} "
            f">/tmp/e10dv3_act.stdout 2>&1 &"
        )

        for i in range(4,0,-1):
            node=NODES[i]; prev=NODES[i-1]; nxt=NODES[i+1]
            listen=pair[(prev,node)][1]
            send=pair[(node,nxt)][0]
            dest=pair[(node,nxt)][1]
            h[node].cmd(
                f"python3 {d/'relay.py'} {listen} {PORT} {send} {dest} {node} {traces[node]} "
                f">/tmp/e10dv3_{SHORT[i]}.stdout 2>&1 &"
            )
        time.sleep(0.75)

        print("\n=== PROCESS READY TRACES ===")
        ready=True
        for node in NODES[1:-1]:
            t=h[node].cmd(f"cat {traces[node]} 2>/dev/null || true").strip()
            print(f"[{node}] {t or '<EMPTY>'}")
            ready &= ("READY" in t)
        t=h["actuator"].cmd(f"cat {traces['actuator']} 2>/dev/null || true").strip()
        print(f"[actuator] {t or '<EMPTY>'}")
        ready &= ("READY" in t)
        if not ready:
            raise RuntimeError("One or more application processes failed to become READY.")

        src=pair[("sensor","tactical_link")][0]
        dst=pair[("sensor","tactical_link")][1]
        msg={"task_id":"__preflight_v3__","truth":0.0,"value":0.0,
             "send_ns":time.time_ns(),"interval":-1}
        raw=json.dumps(msg,separators=(",",":")).encode()
        py=("import socket;"
            "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
            f"s.bind(({src!r},0));"
            f"s.sendto({raw!r},({dst!r},{PORT}))")
        h["sensor"].cmd("python3 -c "+repr(py))
        time.sleep(1.0)

        print("\n=== AFTER ONE PREFLIGHT PACKET ===")
        for node in NODES[1:-1]:
            t=h[node].cmd(f"cat {traces[node]} 2>/dev/null || true").strip()
            print(f"\n[{node}]\n{t or '<EMPTY>'}")
        t=h["actuator"].cmd(f"cat {traces['actuator']} 2>/dev/null || true").strip()
        print(f"\n[actuator]\n{t or '<EMPTY>'}")

        rawrecv=h["actuator"].cmd(f"cat {recv} 2>/dev/null || true").strip()
        if not rawrecv:
            print("\nEND-TO-END PREFLIGHT: FAIL")
            raise RuntimeError("Packet did not reach actuator. Trace above identifies stopping hop.")

        last=json.loads(rawrecv.splitlines()[-1])
        hops=[x["node"] for x in last.get("hops",[])]
        expected=["tactical_link","edge_processing","fusion","command"]
        if hops != expected:
            print("\nEND-TO-END PREFLIGHT: FAIL")
            raise RuntimeError(f"Unexpected hop trace: {hops}")

        latency=(last["recv_ns"]-msg["send_ns"])/1e6
        print(f"\nEND-TO-END PREFLIGHT: PASS | latency={latency:.3f} ms")
        print("HOPS:", " -> ".join(hops))
        print("E10-D v3 DIAGNOSTIC COMPLETE")
        print("No scientific E10-D outcome recorded.")

    finally:
        if h:
            for n in NODES:
                h[n].cmd("pkill -f '/tmp/missionshield_e10dv3/' >/dev/null 2>&1 || true")
        if net:
            net.stop()

if __name__=="__main__":
    setLogLevel("warning")
    main()
