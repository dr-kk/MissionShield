#!/usr/bin/env python3
"""
MissionShield E10-B | Controlled Mininet impairment campaigns
PRE-SPECIFIED before MissionShield inference.

Purpose:
  Collect emulator-native time-series telemetry only.
  No MissionShield inference.
  No sensor_quality/detector_score synthesis.
  No outcome-dependent tuning.

Logical chain:
 sensor -> tactical_link -> edge_processing -> fusion -> command -> actuator

Campaign conditions (fixed):
 benign:          20 Mbps,  2 ms/link-side, 0% loss
 latency:         20 Mbps, 15 ms/link-side, 0% loss
 loss:            20 Mbps,  2 ms/link-side, 5% loss
 bandwidth:        5 Mbps,  2 ms/link-side, 0% loss
 combined:         5 Mbps, 15 ms/link-side, 5% loss

Each condition: 5 independent emulator runs.
Each run: every adjacent dependency measured with 20-packet ping + 1 s iperf3.
"""
import csv, json, math, re, time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

ROOT=Path("/mnt/d/MissionShield")
OUT=ROOT/"results"/"e10b_mininet_impairment_campaigns.csv"
SUMMARY=ROOT/"results"/"e10b_mininet_impairment_summary.csv"

LOGICAL=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
SHORT=["sen","tlink","edge","fus","cmd","act"]

CONDITIONS={
    "benign":    {"bw":20, "delay":"2ms",  "loss":0},
    "latency":   {"bw":20, "delay":"15ms", "loss":0},
    "loss":      {"bw":20, "delay":"2ms",  "loss":5},
    "bandwidth": {"bw":5,  "delay":"2ms",  "loss":0},
    "combined":  {"bw":5,  "delay":"15ms", "loss":5},
}
RUNS_PER_CONDITION=5
PING_COUNT=20

def ping_stats(src,dst_ip):
    x=src.cmd(f"LC_ALL=C ping -n -c {PING_COUNT} -i 0.05 -W 1 {dst_ip} 2>&1")
    loss=re.search(r"([0-9]+(?:\.[0-9]+)?)%\s+packet loss",x)
    rtt=re.search(
        r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev)\s*=\s*"
        r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\s*ms",x)
    if not loss:
        raise RuntimeError(f"Could not parse ping loss:\n{x}")
    loss_v=float(loss.group(1))/100.0
    # If 100% loss, RTT is legitimately unavailable.
    lat=float(rtt.group(2)) if rtt else float("nan")
    return lat,loss_v

def one_run(condition,run_index,cfg):
    net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
    h={logical:net.addHost(short) for logical,short in zip(LOGICAL,SHORT)}
    links=[]
    for i in range(5):
        s=net.addSwitch(f"s{i+1}")
        left=net.addLink(
            h[LOGICAL[i]],s,cls=TCLink,
            intfName1=f"{SHORT[i]}e{i}",intfName2=f"s{i+1}a",
            bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"],use_htb=True)
        right=net.addLink(
            s,h[LOGICAL[i+1]],cls=TCLink,
            intfName1=f"s{i+1}b",intfName2=f"{SHORT[i+1]}e{i}",
            bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"],use_htb=True)
        links.append((left,right))
    net.start()
    rows=[]
    try:
        for i,(a,b) in enumerate(zip(LOGICAL[:-1],LOGICAL[1:])):
            left,right=links[i]
            ia=left.intf1 if left.intf1.node==h[a] else left.intf2
            ib=right.intf1 if right.intf1.node==h[b] else right.intf2
            ipa=f"10.{i+1}.0.1/24"; ipb=f"10.{i+1}.0.2/24"; dst=ipb.split("/")[0]
            h[a].cmd(f"ip addr flush dev {ia}; ip addr add {ipa} dev {ia}; ip link set {ia} up")
            h[b].cmd(f"ip addr flush dev {ib}; ip addr add {ipb} dev {ib}; ip link set {ib} up")

            lat,loss=ping_stats(h[a],dst)
            port=5201+i
            h[b].cmd(f"pkill -f 'iperf3 -s.*-p {port}' >/dev/null 2>&1 || true")
            h[b].cmd(f"iperf3 -s -D -1 -p {port}")
            time.sleep(.12)
            raw=h[a].cmd(f"iperf3 -c {dst} -p {port} -t 1 -J 2>&1")
            try:
                j=json.loads(raw)
                mbps=float(j["end"]["sum_received"]["bits_per_second"])/1e6
            except Exception as e:
                raise RuntimeError(f"iperf parse failed {condition} run {run_index} {a}->{b}:\n{raw}") from e

            if not math.isfinite(loss) or not math.isfinite(mbps):
                raise RuntimeError("Non-finite loss/throughput.")
            if loss < 1.0 and not math.isfinite(lat):
                raise RuntimeError("RTT missing despite successful packets.")

            rows.append({
                "condition":condition,
                "run":run_index,
                "src":a,"dst":b,
                "configured_bw_mbps":cfg["bw"],
                "configured_delay_each_side_ms":float(cfg["delay"].replace("ms","")),
                "configured_loss_each_side_pct":cfg["loss"],
                "latency_ms":lat,
                "packet_loss":loss,
                "throughput_mbps":mbps,
            })
    finally:
        net.stop()
    return rows

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    print("="*100)
    print("MissionShield E10-B | CONTROLLED MININET IMPAIRMENT CAMPAIGNS")
    print("Fixed conditions | emulator-native measurements only | NO MissionShield inference")
    print("="*100)

    rows=[]
    for condition,cfg in CONDITIONS.items():
        print(f"\n{condition.upper()} | {cfg}")
        for r in range(1,RUNS_PER_CONDITION+1):
            print(f"  run {r}/{RUNS_PER_CONDITION} ...",end="",flush=True)
            rr=one_run(condition,r,cfg)
            rows.extend(rr)
            print(" PASS")

    fields=list(rows[0].keys())
    with OUT.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    # Dependency-free summary (stdlib only).
    summary=[]
    for cond in CONDITIONS:
        g=[x for x in rows if x["condition"]==cond]
        lat=[x["latency_ms"] for x in g if math.isfinite(x["latency_ms"])]
        loss=[x["packet_loss"] for x in g]
        tp=[x["throughput_mbps"] for x in g]
        summary.append({
            "condition":cond,
            "measurements":len(g),
            "mean_latency_ms":sum(lat)/len(lat) if lat else float("nan"),
            "mean_packet_loss":sum(loss)/len(loss),
            "mean_throughput_mbps":sum(tp)/len(tp),
        })
    with SUMMARY.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)

    print("\nE10-B SUMMARY")
    for x in summary: print(x)
    print(f"\nRaw:     {OUT}")
    print(f"Summary: {SUMMARY}")
    print("E10-B COMPLETE")
    print("No MissionShield inference performed and no semantic feature was synthesized.")

if __name__=="__main__":
    setLogLevel("warning")
    main()
