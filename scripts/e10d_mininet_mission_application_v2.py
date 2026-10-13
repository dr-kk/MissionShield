#!/usr/bin/env python3
"""
MissionShield E10-D v2
Engineering correction of the six-stage UDP mission application.

Scientific protocol is unchanged from frozen E10-C:
- 1 s interval
- 8 tasks/interval
- 10-interval mission window
- 500 ms deadline
- payload tolerance 0.05
- failure iff rolling mission effectiveness < 0.70
- no MissionShield inference
- detector_score remains blank

Engineering fix:
Each multi-homed relay explicitly receives on its upstream IP and sends from
its downstream IP. A one-packet end-to-end preflight is required before data
collection. Failed preflight aborts the condition instead of recording false
mission failures.
"""
import csv, json, math, time
from pathlib import Path
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel

ROOT = Path("/mnt/d/MissionShield")
CONTRACT = ROOT / "results" / "e10c_feature_contract.json"
OUT = ROOT / "results" / "e10d_mission_application_telemetry.csv"
SUMMARY = ROOT / "results" / "e10d_mission_application_summary.csv"

NODES = ["sensor","tactical_link","edge_processing","fusion","command","actuator"]
SHORT = ["sen","tlink","edge","fus","cmd","act"]
TASKS = 8
INTERVALS = 12
DEADLINE_MS = 500.0
TOL = 0.05
WINDOW = 10
FAIL_THRESHOLD = 0.70
PORT = 7000

CONDITIONS = {
    "benign":   {"bw":20, "delay":"2ms",  "loss":0},
    "combined": {"bw":5,  "delay":"15ms", "loss":5},
}

RELAY_PROGRAM = r"""
import json, socket, sys, time
listen_ip=sys.argv[1]
port=int(sys.argv[2])
send_ip=sys.argv[3]
next_ip=sys.argv[4]
node=sys.argv[5]

rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
rx.bind((listen_ip,port))
tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
tx.bind((send_ip,0))

while True:
    data,_=rx.recvfrom(65535)
    try:
        m=json.loads(data.decode())
        m.setdefault("hops",[]).append({"node":node,"t_ns":time.time_ns()})
        data=json.dumps(m,separators=(",",":")).encode()
    except Exception:
        pass
    tx.sendto(data,(next_ip,port))
"""

ACTUATOR_PROGRAM = r"""
import json, socket, sys, time
ip=sys.argv[1]
port=int(sys.argv[2])
outfile=sys.argv[3]

s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((ip,port))
with open(outfile,"a",buffering=1) as f:
    while True:
        data,_=s.recvfrom(65535)
        try:
            m=json.loads(data.decode())
            m["recv_ns"]=time.time_ns()
            f.write(json.dumps(m,separators=(",",":"))+"\n")
        except Exception:
            pass
"""

def make_helpers():
    d=Path("/tmp/missionshield_e10d")
    d.mkdir(parents=True,exist_ok=True)
    (d/"relay.py").write_text(RELAY_PROGRAM,encoding="utf-8")
    (d/"actuator.py").write_text(ACTUATOR_PROGRAM,encoding="utf-8")
    return d

def interface_bytes(host, intf):
    out=host.cmd(
        f"cat /sys/class/net/{intf}/statistics/tx_bytes "
        f"/sys/class/net/{intf}/statistics/rx_bytes"
    )
    vals=[]
    for x in out.split():
        try: vals.append(int(x))
        except ValueError: pass
    if len(vals)<2:
        raise RuntimeError(f"Could not read counters for {host.name}:{intf}: {out}")
    return vals[0]+vals[1]

def build_network(cfg):
    net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
    hosts={n:net.addHost(s) for n,s in zip(NODES,SHORT)}
    links=[]
    for i in range(5):
        sw=net.addSwitch(f"s{i+1}")
        left=net.addLink(
            hosts[NODES[i]],sw,cls=TCLink,
            intfName1=f"{SHORT[i]}e{i}",intfName2=f"s{i+1}a",
            bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"],use_htb=True
        )
        right=net.addLink(
            sw,hosts[NODES[i+1]],cls=TCLink,
            intfName1=f"s{i+1}b",intfName2=f"{SHORT[i+1]}e{i}",
            bw=cfg["bw"],delay=cfg["delay"],loss=cfg["loss"],use_htb=True
        )
        links.append((left,right))
    net.start()

    pair_ips={}
    pair_intfs={}
    for i,(a,b) in enumerate(zip(NODES[:-1],NODES[1:])):
        left,right=links[i]
        ia=left.intf1 if left.intf1.node==hosts[a] else left.intf2
        ib=right.intf1 if right.intf1.node==hosts[b] else right.intf2
        ip_a=f"10.{i+1}.0.1"
        ip_b=f"10.{i+1}.0.2"
        hosts[a].cmd(f"ip addr flush dev {ia}; ip addr add {ip_a}/24 dev {ia}; ip link set {ia} up")
        hosts[b].cmd(f"ip addr flush dev {ib}; ip addr add {ip_b}/24 dev {ib}; ip link set {ib} up")
        pair_ips[(a,b)]=(ip_a,ip_b)
        pair_intfs[(a,b)]=(str(ia),str(ib))

    return net,hosts,pair_ips,pair_intfs

def start_apps(hosts,pair_ips,helper,recv_file):
    hosts["actuator"].cmd(f"rm -f {recv_file}")
    actuator_ip=pair_ips[("command","actuator")][1]
    hosts["actuator"].cmd(
        f"python3 {helper/'actuator.py'} {actuator_ip} {PORT} {recv_file} "
        f">/tmp/e10d_actuator.log 2>&1 &"
    )

    # downstream-to-upstream startup
    for i in range(4,0,-1):
        node=NODES[i]
        prev_node=NODES[i-1]
        next_node=NODES[i+1]
        listen_ip=pair_ips[(prev_node,node)][1]
        send_ip=pair_ips[(node,next_node)][0]
        next_ip=pair_ips[(node,next_node)][1]
        hosts[node].cmd(
            f"python3 {helper/'relay.py'} {listen_ip} {PORT} {send_ip} {next_ip} {node} "
            f">/tmp/e10d_{SHORT[i]}.log 2>&1 &"
        )
    time.sleep(0.5)

    for i in range(1,5):
        node=NODES[i]
        if not hosts[node].cmd(f"ss -lun | grep ':{PORT} ' || true").strip():
            log=hosts[node].cmd(f"cat /tmp/e10d_{SHORT[i]}.log 2>/dev/null || true")
            raise RuntimeError(f"{node} relay is not listening.\n{log}")
    if not hosts["actuator"].cmd(f"ss -lun | grep ':{PORT} ' || true").strip():
        log=hosts["actuator"].cmd("cat /tmp/e10d_actuator.log 2>/dev/null || true")
        raise RuntimeError(f"Actuator is not listening.\n{log}")

def send_message(sensor,pair_ips,message):
    src_ip=pair_ips[("sensor","tactical_link")][0]
    dst_ip=pair_ips[("sensor","tactical_link")][1]
    raw=json.dumps(message,separators=(",",":")).encode()
    py=(
        "import socket;"
        "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
        f"s.bind(({src_ip!r},0));"
        f"s.sendto({raw!r},({dst_ip!r},{PORT}))"
    )
    sensor.cmd("python3 -c "+repr(py))

def read_messages(actuator, recv_file):
    messages=[]
    raw=actuator.cmd(f"cat {recv_file} 2>/dev/null || true")
    for line in raw.splitlines():
        try: messages.append(json.loads(line))
        except Exception: pass
    return messages

def preflight(hosts,pair_ips,recv_file):
    m={"task_id":"__preflight__","truth":0.0,"value":0.0,
       "send_ns":time.time_ns(),"interval":-1}
    send_message(hosts["sensor"],pair_ips,m)
    deadline=time.time()+2.0
    while time.time()<deadline:
        for x in read_messages(hosts["actuator"],recv_file):
            if x.get("task_id")=="__preflight__":
                hops=[h.get("node") for h in x.get("hops",[])]
                expected=["tactical_link","edge_processing","fusion","command"]
                if hops!=expected:
                    raise RuntimeError(f"Preflight path mismatch: {hops}, expected {expected}")
                lat=(x["recv_ns"]-m["send_ns"])/1e6
                return lat,hops
        time.sleep(0.05)
    logs=[]
    for i in range(1,5):
        logs.append(f"{NODES[i]}: "+hosts[NODES[i]].cmd(
            f"cat /tmp/e10d_{SHORT[i]}.log 2>/dev/null || true"))
    logs.append("actuator: "+hosts["actuator"].cmd(
        "cat /tmp/e10d_actuator.log 2>/dev/null || true"))
    raise RuntimeError("END-TO-END PREFLIGHT FAILED.\n"+"\n".join(logs))

def main():
    contract=json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"]=="FROZEN_BEFORE_MODEL_INFERENCE"
    assert contract["sampling"]["interval_seconds"]==1.0
    assert contract["sampling"]["tasks_per_interval"]==TASKS
    assert contract["sampling"]["mission_window_intervals"]==WINDOW
    assert contract["sampling"]["mission_failure_threshold"]==FAIL_THRESHOLD
    assert contract["application"]["deadline_ms"]==DEADLINE_MS
    assert contract["application"]["payload_tolerance_abs"]==TOL

    helper=make_helpers()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    rows=[]

    print("="*92)
    print("MissionShield E10-D v2 | SIX-STAGE UDP MISSION APPLICATION")
    print("Engineering routing correction only; frozen E10-C protocol unchanged")
    print("NO MissionShield inference | detector_score intentionally blank")
    print("="*92)

    for condition,cfg in CONDITIONS.items():
        print(f"\n{condition.upper()} {cfg}")
        net=None; hosts=None
        recv_file=f"/tmp/e10d_{condition}_received.jsonl"
        try:
            net,hosts,pair_ips,pair_intfs=build_network(cfg)
            start_apps(hosts,pair_ips,helper,recv_file)

            pf_lat,pf_hops=preflight(hosts,pair_ips,recv_file)
            print(f" PREFLIGHT PASS | latency={pf_lat:.2f} ms | hops={' -> '.join(pf_hops)}")

            previous={}
            for a,b in zip(NODES[:-1],NODES[1:]):
                ia,ib=pair_intfs[(a,b)]
                previous[(a,b)]=interface_bytes(hosts[a],ia)+interface_bytes(hosts[b],ib)

            history=[]
            for interval in range(INTERVALS):
                t0=time.time()
                sent={}
                for k in range(TASKS):
                    tid=f"{condition}_{interval:03d}_{k:02d}"
                    truth=math.sin((interval*TASKS+k)/10.0)
                    m={"task_id":tid,"truth":truth,"value":truth,
                       "send_ns":time.time_ns(),"interval":interval}
                    sent[tid]=m
                    send_message(hosts["sensor"],pair_ips,m)
                    time.sleep(0.02)

                elapsed=time.time()-t0
                if elapsed<1.0:
                    time.sleep(1.0-elapsed)

                received={}
                for m in read_messages(hosts["actuator"],recv_file):
                    if m.get("interval")==interval:
                        received[m["task_id"]]=m

                delivered=correct=ontime=0
                latencies=[]
                for tid,sm in sent.items():
                    rm=received.get(tid)
                    if rm is None: continue
                    delivered+=1
                    lat=(rm["recv_ns"]-sm["send_ns"])/1e6
                    latencies.append(lat)
                    ok=abs(float(rm["value"])-float(sm["truth"]))<=TOL
                    correct+=int(ok)
                    ontime+=int(ok and lat<=DEADLINE_MS)

                interval_success=ontime/TASKS
                history.append(interval_success)
                rolling=history[-WINDOW:]
                me=sum(rolling)/len(rolling)

                utils=[]
                for a,b in zip(NODES[:-1],NODES[1:]):
                    ia,ib=pair_intfs[(a,b)]
                    now=interface_bytes(hosts[a],ia)+interface_bytes(hosts[b],ib)
                    delta=max(0,now-previous[(a,b)])
                    previous[(a,b)]=now
                    capacity_bytes=cfg["bw"]*1e6/8.0
                    utils.append(min(1.0,delta/capacity_bytes))

                row={
                    "condition":condition,
                    "interval":interval,
                    "expected_tasks":TASKS,
                    "delivered_tasks":delivered,
                    "on_time_correct_tasks":ontime,
                    "latency_ms":sum(latencies)/len(latencies) if latencies else float("nan"),
                    "packet_loss":1.0-ontime/TASKS,
                    "utilization":sum(utils)/len(utils),
                    "availability":delivered/TASKS,
                    "sensor_quality":correct/delivered if delivered else 0.0,
                    "mission_effectiveness":me,
                    "failure":int(me<FAIL_THRESHOLD),
                    "detector_score":""
                }
                rows.append(row)
                lat_text=f"{row['latency_ms']:.2f}" if math.isfinite(row["latency_ms"]) else "nan"
                print(f" interval={interval:02d} delivered={delivered}/8 "
                      f"ontime={ontime}/8 lat={lat_text}ms ME={me:.3f} "
                      f"failure={row['failure']}")

        finally:
            if hosts:
                for n in NODES:
                    hosts[n].cmd("pkill -f '/tmp/missionshield_e10d/' >/dev/null 2>&1 || true")
            if net:
                net.stop()

    with OUT.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary=[]
    for condition in CONDITIONS:
        g=[r for r in rows if r["condition"]==condition]
        finite_lat=[r["latency_ms"] for r in g if math.isfinite(r["latency_ms"])]
        summary.append({
            "condition":condition,
            "intervals":len(g),
            "mean_latency_ms":sum(finite_lat)/len(finite_lat) if finite_lat else float("nan"),
            "mean_packet_loss":sum(r["packet_loss"] for r in g)/len(g),
            "mean_utilization":sum(r["utilization"] for r in g)/len(g),
            "mean_availability":sum(r["availability"] for r in g)/len(g),
            "mean_sensor_quality":sum(r["sensor_quality"] for r in g)/len(g),
            "final_mission_effectiveness":g[-1]["mission_effectiveness"],
            "failure_intervals":sum(r["failure"] for r in g),
        })

    with SUMMARY.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    print("\nE10-D SUMMARY")
    for x in summary: print(x)
    print(f"\nRaw: {OUT}")
    print(f"Summary: {SUMMARY}")
    print("E10-D COMPLETE")
    print("detector_score intentionally blank; NO MissionShield inference performed.")

if __name__=="__main__":
    setLogLevel("warning")
    main()
