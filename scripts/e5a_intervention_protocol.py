from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"D:\MissionShield")
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# MissionShield E5-A
# INTERVENTION PROTOCOL FREEZE
#
# Purpose:
#   Freeze the candidate defensive action space BEFORE
#   counterfactual intervention experiments.
#
# IMPORTANT:
#   - No model training.
#   - No test-result tuning.
#   - Costs below are normalized experimental costs, NOT
#     measured real-world monetary/operational costs.
#   - E5-B must implement these actions without changing this
#     table after observing intervention outcomes.
# ============================================================

PROTOCOL_VERSION = "E5-A-v1"

NODES = [
    "sensor",
    "tactical_link",
    "edge_processing",
    "fusion",
    "command",
    "actuator",
]

# Frozen action definitions.
#
# intervention_cost is a normalized [0,1] experimental penalty.
# It represents increasing operational disruption, not money.
#
# target_scope tells E5-B whether the action is applied to a
# selected node/path or globally.
ACTIONS = [
    {
        "action_id": "A0",
        "action": "no_action",
        "target_scope": "none",
        "cost": 0.00,
        "description": "Continue mission operation without defensive intervention.",
    },
    {
        "action_id": "A1",
        "action": "rate_limit",
        "target_scope": "selected_path",
        "cost": 0.10,
        "description": "Throttle suspicious traffic while preserving connectivity.",
    },
    {
        "action_id": "A2",
        "action": "reroute",
        "target_scope": "selected_path",
        "cost": 0.20,
        "description": "Shift mission traffic away from a degraded or suspicious path.",
    },
    {
        "action_id": "A3",
        "action": "isolate",
        "target_scope": "selected_node",
        "cost": 0.35,
        "description": "Temporarily isolate a suspected compromised/degraded node.",
    },
    {
        "action_id": "A4",
        "action": "restore_service",
        "target_scope": "selected_node",
        "cost": 0.25,
        "description": "Restore/reinitialize a degraded service or processing component.",
    },
    {
        "action_id": "A5",
        "action": "safe_mode",
        "target_scope": "mission",
        "cost": 0.30,
        "description": "Enter degraded-safe operation to preserve essential mission functionality.",
    },
]

# Applicability is intentionally broad at protocol-freeze stage.
# E5-B will implement physical/telemetry effects consistently.
APPLICABILITY = {
    "A0": NODES,
    "A1": ["tactical_link"],
    "A2": ["tactical_link"],
    "A3": ["sensor", "tactical_link", "edge_processing", "fusion", "command"],
    "A4": ["edge_processing", "fusion", "command"],
    "A5": NODES,
}

# Frozen evaluation quantities for E5-B onward.
METRICS = [
    "failure_prevented",
    "post_intervention_failure",
    "mission_effectiveness_before",
    "mission_effectiveness_after",
    "mission_effectiveness_gain",
    "recovery_time_steps",
    "intervention_cost",
]

# Planner objective is frozen conceptually here.
# Exact score is evaluated in E5-C after counterfactual outcomes exist:
#
# utility = mission_benefit - lambda_cost * intervention_cost
#
# lambda_cost MUST be chosen using TRAIN/VALIDATION only.
PLANNER_OBJECTIVE = {
    "benefit_term": "mission_effectiveness_gain",
    "failure_term": "failure_prevented",
    "cost_term": "intervention_cost",
    "cost_weight_selection": "validation_only",
    "test_tuning_allowed": False,
}

def validate():
    ids = [a["action_id"] for a in ACTIONS]
    names = [a["action"] for a in ACTIONS]

    assert len(ids) == len(set(ids)), "Duplicate action_id."
    assert len(names) == len(set(names)), "Duplicate action name."
    assert ids[0] == "A0" and names[0] == "no_action"

    for a in ACTIONS:
        c = float(a["cost"])
        assert 0.0 <= c <= 1.0, f"Invalid cost for {a['action_id']}"
        assert a["action_id"] in APPLICABILITY

    for aid, nodes in APPLICABILITY.items():
        assert aid in ids
        for node in nodes:
            assert node in NODES, f"Unknown node {node} for {aid}"

def main():
    validate()

    print("=" * 92)
    print("MissionShield E5-A | INTERVENTION PROTOCOL FREEZE")
    print("NO TRAINING | NO TEST TUNING | ACTION SPACE ONLY")
    print("=" * 92)
    print(f"Protocol version : {PROTOCOL_VERSION}")
    print(f"Mission nodes    : {len(NODES)}")
    print(f"Candidate actions: {len(ACTIONS)}")

    rows = []
    for a in ACTIONS:
        applicable = ",".join(APPLICABILITY[a["action_id"]])
        row = dict(a)
        row["applicable_nodes"] = applicable
        rows.append(row)

        print(
            f"{a['action_id']} | {a['action']:<16} | "
            f"scope={a['target_scope']:<13} | cost={a['cost']:.2f}"
        )
        print(f"     applicable: {applicable}")
        print(f"     {a['description']}")

    action_file = RESULTS / "e5a_intervention_action_space.csv"
    pd.DataFrame(rows).to_csv(action_file, index=False)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_AFTER_E5A",
        "nodes": NODES,
        "actions": ACTIONS,
        "applicability": APPLICABILITY,
        "metrics": METRICS,
        "planner_objective": PLANNER_OBJECTIVE,
        "scientific_constraints": [
            "Do not change intervention definitions after observing E5-B test outcomes.",
            "Do not choose intervention cost weights using test outcomes.",
            "Use identical underlying scenario/run when comparing no-action and intervention counterfactuals.",
            "Report intervention benefit together with intervention cost.",
            "Preserve no-action counterfactual as the reference condition.",
        ],
    }

    json_file = RESULTS / "e5a_intervention_protocol.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    print("\nFrozen evaluation metrics:")
    for m in METRICS:
        print(" -", m)

    print("\nPlanner objective:")
    print(" utility = mission benefit - lambda_cost * intervention cost")
    print(" lambda_cost selection = TRAIN/VALIDATION ONLY")

    print("\nSaved:")
    print(" ", action_file)
    print(" ", json_file)
    print("\nE5-A COMPLETE")

if __name__ == "__main__":
    main()
