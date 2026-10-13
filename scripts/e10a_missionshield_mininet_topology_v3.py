#!/usr/bin/env python3
"""
MissionShield E10-A v3
Real Mininet six-stage adjacent-link measurement smoke test.

Fixes:
1) robust iputils ping parsing (supports "packet loss" and "packet loss, time ...")
2) finite-value integrity checks
3) uses 20 Mbps TCLink to avoid noisy HTB quantum warnings from 100 Mbps
No MissionShield inference and no synthetic semantic features.
"""
import csv, re, time, json, math
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

OUT=Path("/mnt/d/MissionShield/results/e10a_mininet_link_measurements.csv")
LOGICAL=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
SHORT=["sen","tlink","edge","fus","cmd","act"]

def ping_stats(src,dst_ip,count=10):
    x=src.cmd(f"LC_ALL=C ping -n -c {count} -i 0.05 -W 1 {dst_ip} 2>&1")
    loss=re.search(r"([0-9]+(?:\.[0-9]+)?)%\s+packet loss",x)
    rtt=re.search(
        r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev)\s*=\s*"
        r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\s*ms",x
    )
    if not loss or not rtt:
        raise RuntimeError(f"Could not parse ping output to {dst_ip}:\n{x}")
    return float(rtt.group(2)), float(loss.group(1))/100.0

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
    h={logical:net.addHost(short) for logical,short in zip(LOGICAL,SHORT)}
    links=[]

    for i in range(5):
        s=net.addSwitch(f"s{i+1}")
        left=net.addLink(
            h[LOGICAL[i]],s,cls=TCLink,
            intfName1=f"{SHORT[i]}e{i}",intfName2=f"s{i+1}a",
            bw=20,delay="2ms",loss=0,use_htb=True
        )
        right=net.addLink(
            s,h[LOGICAL[i+1]],cls=TCLink,
            intfName1=f"s{i+1}b",intfName2=f"{SHORT[i+1]}e{i}",
            bw=20,delay="2ms",loss=0,use_htb=True
        )
        links.append((left,right))

    net.start()
    rows=[]

    for i,(a,b) in enumerate(zip(LOGICAL[:-1],LOGICAL[1:])):
        left,right=links[i]
        ia=left.intf1 if left.intf1.node==h[a] else left.intf2
        ib=right.intf1 if right.intf1.node==h[b] else right.intf2

        ipa=f"10.{i+1}.0.1/24"; ipb=f"10.{i+1}.0.2/24"; dst=ipb.split("/")[0]
        h[a].cmd(f"ip addr flush dev {ia}; ip addr add {ipa} dev {ia}; ip link set {ia} up")
        h[b].cmd(f"ip addr flush dev {ib}; ip addr add {ipb} dev {ib}; ip link set {ib} up")

        lat,loss=ping_stats(h[a],dst,10)

        port=5201+i
        h[b].cmd(f"pkill -f 'iperf3 -s.*-p {port}' >/dev/null 2>&1 || true")
        h[b].cmd(f"iperf3 -s -D -1 -p {port}")
        time.sleep(0.15)
        raw=h[a].cmd(f"iperf3 -c {dst} -p {port} -t 1 -J 2>&1")
        try:
            j=json.loads(raw)
            mbps=float(j["end"]["sum_received"]["bits_per_second"])/1e6
        except Exception as e:
            raise RuntimeError(f"Could not parse iperf3 result for {a}->{b}:\n{raw}") from e

        vals=[lat,loss,mbps]
        if not all(math.isfinite(v) for v in vals):
            raise RuntimeError(f"Non-finite measurement for {a}->{b}: {vals}")

        rows.append({
            "src":a,"dst":b,
            "latency_ms":lat,
            "packet_loss":loss,
            "throughput_mbps":mbps
        })

    with OUT.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    print("\nE10-A ADJACENT-LINK MEASUREMENTS")
    for r in rows: print(r)
    print(f"\nSaved: {OUT}")
    print("FINITE CHECK: PASS")
    print("E10-A COMPLETE")
    print("No MissionShield inference performed; no missing semantic feature was synthesized.")
    net.stop()

if __name__=="__main__":
    setLogLevel("warning")
    main()
