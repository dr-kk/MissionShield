#!/usr/bin/env python3
"""
MissionShield E10-C | Mission instrumentation + feature-contract freeze

PURPOSE
-------
Freeze the emulator-to-MissionShield measurement contract BEFORE frozen-model
inference. This is a protocol-definition stage, not a performance experiment.

Measured/derived feature contract
---------------------------------
latency:
    application task end-to-end latency in milliseconds.
packet_loss:
    fraction of application tasks not received by the deadline in the interval.
utilization:
    measured interface byte-rate / configured link capacity, clipped [0,1].
availability:
    fraction of expected application heartbeats/tasks successfully received
    within the interval.
sensor_quality:
    fraction of sensor payloads whose received value is within fixed tolerance
    of the deterministic ground-truth signal.
detector_score:
    fixed benign-calibrated anomaly score based ONLY on observable network
    telemetry (latency, packet_loss, utilization, availability). Calibration
    parameters must be learned from benign emulator calibration runs only,
    then frozen before attack/degradation evaluation.

Mission outcome (independent of detector/model):
    Each emitted sensor task succeeds only if the actuator receives the correct
    task within DEADLINE_MS. Mission effectiveness is the fraction of successful
    tasks over the last MISSION_WINDOW intervals. Mission failure = ME < 0.70.

This script writes the frozen contract; it intentionally does not infer with
MissionShield and does not inspect E10-B outcomes to tune thresholds.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path("/mnt/d/MissionShield")
OUT=ROOT/"results"/"e10c_feature_contract.json"

contract={
    "experiment":"E10-C",
    "status":"FROZEN_BEFORE_MODEL_INFERENCE",
    "logical_nodes":[
        "sensor","tactical_link","edge_processing","fusion","command","actuator"
    ],
    "sampling":{
        "interval_seconds":1.0,
        "history_intervals":10,
        "tasks_per_interval":8,
        "mission_window_intervals":10,
        "mission_failure_threshold":0.70
    },
    "application":{
        "task_transport":"UDP application messages through six logical stages",
        "task_id_required":True,
        "ground_truth_signal":"deterministic sequence known to evaluator",
        "payload_tolerance_abs":0.05,
        "deadline_ms":500.0,
        "success_definition":
            "actuator receives matching task_id, payload within tolerance, before deadline"
    },
    "features":{
        "latency":{
            "unit":"ms",
            "definition":"application end-to-end latency for delivered tasks; interval mean"
        },
        "packet_loss":{
            "range":[0,1],
            "definition":"1 - on-time delivered task count / expected task count"
        },
        "utilization":{
            "range":[0,1],
            "definition":"measured interface bytes per second divided by configured link bytes per second, clipped to [0,1]"
        },
        "availability":{
            "range":[0,1],
            "definition":"successful application heartbeat/task receipts divided by expected receipts"
        },
        "sensor_quality":{
            "range":[0,1],
            "definition":"fraction of delivered sensor payloads within absolute tolerance of evaluator ground truth"
        },
        "detector_score":{
            "range":[0,1],
            "definition":"benign-calibrated anomaly score using observable latency, packet_loss, utilization, availability only",
            "calibration":"benign emulator calibration runs only",
            "evaluation_outcomes_used":False
        }
    },
    "mission_outcome":{
        "mission_effectiveness":
            "successful tasks / expected tasks over rolling 10-interval window",
        "failure":"mission_effectiveness < 0.70",
        "independent_of_missionshield_prediction":True,
        "independent_of_detector_score":True
    },
    "model_transfer":{
        "checkpoint":"original frozen E1 MissionShield checkpoint",
        "training_on_emulator_data":False,
        "fine_tuning_on_emulator_data":False,
        "synthetic_training_normalization_retained":True,
        "horizons_to_evaluate":[10,30,60],
        "reason":"avoid post-hoc selection of a single horizon"
    },
    "integrity":{
        "e10b_impairment_levels_changed_after_results":False,
        "feature_mapping_tuned_on_missionshield_predictions":False,
        "mission_threshold_tuned_on_model_predictions":False
    }
}

OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(contract,indent=2),encoding="utf-8")

print("="*88)
print("MissionShield E10-C | FEATURE-CONTRACT FREEZE")
print("="*88)
print(json.dumps(contract,indent=2))
print(f"\nSaved: {OUT}")
print("E10-C CONTRACT FREEZE COMPLETE")
print("Next stage: implement application-chain telemetry under this frozen contract.")
