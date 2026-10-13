#!/usr/bin/env python3
"""
MissionShield E10-A v2
Mininet smoke/measurement experiment.

Linux interface names are deliberately short (<=15 chars).
The scientific/logical node names remain unchanged in output.
"""
import csv, re, time, json
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

OUT=Path("/mnt/d/MissionShield/results/e10a_mininet_link_measurements.csv")

LOGICAL=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
SHORT=["sen","tlink","edge","fus","cmd","act"]

def ping_stats(src,dst_ip,count=10):
    x=src.cmd(f"ping -c {count} -i 0.05 -W 1 {dst_ip}")
    loss=re.search(r"([\\d.]+)% packet loss",x)
    rtt=re.search(r"= ([\\d.]+)/([\\d.]+)/([\\d.]+)/([\\d.]+) ms",x)
    return (
        float(rtt.group(2)) if rtt else float("nan"),
        float(loss.group(1))/100.0 if loss else float("nan")
    )

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)

    net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
    h={logical:net.addHost(short) for logical,short in zip(LOGICAL,SHORT)}
    links=[]

    # Five independent L2 segments implementing the six-stage logical chain.
    for i in range(5):
        s=net.addSwitch(f"s{i+1}")
        left=net.addLink(
            h[LOGICAL[i]],s,cls=TCLink,
            intfName1=f"{SHORT[i]}e{i}",
            intfName2=f"s{i+1}a",
            bw=100,delay="2ms",loss=0
        )
        right=net.addLink(
            s,h[LOGICAL[i+1]],cls=TCLink,
            intfName1=f"s{i+1}b",
            intfName2=f"{SHORT[i+1]}e{i}",
            bw=100,delay="2ms",loss=0
        )
        links.append((left,right))

    net.start()
    rows=[]

    for i,(a,b) in enumerate(zip(LOGICAL[:-1],LOGICAL[1:])):
        left,right=links[i]
        ia=left.intf1 if left.intf1.node==h[a] else left.intf2
        ib=right.intf1 if right.intf1.node==h[b] else right.intf2

        ipa=f"10.{i+1}.0.1/24"
        ipb=f"10.{i+1}.0.2/24"
        h[a].cmd(f"ip addr flush dev {ia}; ip addr add {ipa} dev {ia}; ip link set {ia} up")
        h[b].cmd(f"ip addr flush dev {ib}; ip addr add {ipb} dev {ib}; ip link set {ib} up")

        lat,loss=ping_stats(h[a],ipb.split("/")[0],10)

        port=5201+i
        h[b].cmd(f"iperf3 -s -D -1 -p {port}")
        time.sleep(0.15)
        raw=h[a].cmd(f"iperf3 -c {ipb.split('/')[0]} -p {port} -t 1 -J")
        try:
            mbps=float(json.loads(raw)["end"]["sum_received"]["bits_per_second"])/1e6
        except Exception:
            mbps=float("nan")

        rows.append({
            "src":a,"dst":b,
            "latency_ms":lat,
            "packet_loss":loss,
            "throughput_mbps":mbps
        })

    with OUT.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print("\nE10-A ADJACENT-LINK MEASUREMENTS")
    for r in rows:
        print(r)

    print(f"\nSaved: {OUT}")
    print("E10-A COMPLETE")
    print("No MissionShield inference performed; no missing semantic feature was synthesized.")
    net.stop()

if __name__=="__main__":
    setLogLevel("warning")
    main()
