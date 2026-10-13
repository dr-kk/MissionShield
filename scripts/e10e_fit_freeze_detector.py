#!/usr/bin/env python3
"""
MissionShield E10-E | Fit/freeze benign empirical detector calibration.

Input: e10e_benign_calibration_raw.csv (5 x 60 x 6 = 1800 rows)
Uses only benign latency, packet_loss, utilization, availability.
No attack labels, mission outcomes, sensor_quality, or MissionShield predictions.
No MissionShield inference.

Frozen empirical tail rule:
 high-tail features: p=(1 + # calibration values >= x)/(n+1); anomaly=1-p
 low-tail feature:   p=(1 + # calibration values <= x)/(n+1); anomaly=1-p
 node detector_score = max anomaly across the four base features.
"""
import csv, json, math, hashlib
from pathlib import Path
from collections import defaultdict

ROOT=Path("/mnt/d/MissionShield"); RES=ROOT/"results"
RAW=RES/"e10e_benign_calibration_raw.csv"
CONTRACT=RES/"e10e_telemetry_detector_contract.json"
CAL=RES/"e10e_detector_calibration.json"
SCORED=RES/"e10e_benign_calibration_scored.csv"
SUMMARY=RES/"e10e_detector_calibration_summary.csv"

NODES=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
FEATURES=["latency","packet_loss","utilization","availability"]
TAIL={"latency":"high","packet_loss":"high","utilization":"high","availability":"low"}

def score(x, vals, tail):
    n=len(vals)
    if tail=="high": extreme=sum(v>=x for v in vals)
    else: extreme=sum(v<=x for v in vals)
    p=(1+extreme)/(n+1)
    return 1-p

def main():
    contract=json.loads(CONTRACT.read_text())
    rows=[]
    with RAW.open(newline="") as f:
        for r in csv.DictReader(f):
            q=dict(r)
            q["run"]=int(q["run"]); q["interval"]=int(q["interval"])
            for k in ["latency","packet_loss","utilization","availability","sensor_quality"]:
                q[k]=float(q[k])
            rows.append(q)

    assert len(rows)==1800, f"Expected 1800 rows, got {len(rows)}"
    for n in NODES:
        assert sum(r["node_id"]==n for r in rows)==300, f"{n}: expected 300"

    vals=defaultdict(dict)
    for n in NODES:
        g=[r for r in rows if r["node_id"]==n]
        for f in FEATURES:
            a=sorted(r[f] for r in g if math.isfinite(r[f]))
            assert len(a)==300, f"{n}/{f}: expected 300 finite values"
            vals[n][f]=a

    raw_sha=hashlib.sha256(RAW.read_bytes()).hexdigest()
    cal={
      "stage":"E10-E",
      "status":"FROZEN_BENIGN_EMPIRICAL_CALIBRATION",
      "source_file":str(RAW),
      "source_sha256":raw_sha,
      "n_per_node":300,
      "features":FEATURES,
      "tail_direction":TAIL,
      "formula":{
        "high":"p=(1 + count(calibration >= x))/(n+1); anomaly=1-p",
        "low":"p=(1 + count(calibration <= x))/(n+1); anomaly=1-p",
        "node_detector_score":"max(feature anomaly scores)"
      },
      "uses_attack_labels":False,
      "uses_mission_failure":False,
      "uses_sensor_quality":False,
      "uses_model_predictions":False,
      "calibration_values":{n:{f:vals[n][f] for f in FEATURES} for n in NODES}
    }
    CAL.write_text(json.dumps(cal,indent=2))

    scored=[]
    for r in rows:
        q=dict(r); fs={}
        for f in FEATURES:
            fs[f]=score(r[f],vals[r["node_id"]][f],TAIL[f])
            q["anomaly_"+f]=fs[f]
        q["detector_score"]=max(fs.values())
        scored.append(q)

    fields=["run","interval","node_id","latency","packet_loss","utilization","availability",
            "sensor_quality","anomaly_latency","anomaly_packet_loss","anomaly_utilization",
            "anomaly_availability","detector_score"]
    with SCORED.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in scored:w.writerow({k:r[k] for k in fields})

    summary=[]
    for n in NODES:
        a=sorted(r["detector_score"] for r in scored if r["node_id"]==n)
        def q(p): return a[min(len(a)-1,max(0,math.ceil(p*len(a))-1))]
        summary.append({"node_id":n,"n":len(a),"mean_detector_score":sum(a)/len(a),
                        "p95":q(.95),"p99":q(.99),"max":max(a)})
    with SUMMARY.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

    print("="*88)
    print("MissionShield E10-E | BENIGN EMPIRICAL DETECTOR CALIBRATION")
    print("="*88)
    print("Integrity: PASS")
    print("Rows:",len(rows),"| 300 per node")
    print("Raw SHA256:",raw_sha)
    print("\nBENIGN SCORE SUMMARY")
    for x in summary:print(x)
    print("\nCalibration:",CAL)
    print("Scored benign:",SCORED)
    print("Summary:",SUMMARY)
    print("E10-E DETECTOR CALIBRATION FROZEN")
    print("No attack/failure labels or MissionShield predictions were used.")
    print("No MissionShield inference performed.")

if __name__=="__main__":
    main()
