import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# MissionShield E0
# Mission CPS Emulator + Dataset Generation + Verification
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(r"D:\MissionShield")
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Fixed E0 configuration
# ------------------------------------------------------------

BASE_NODES = [
    "sensor",
    "tactical_link",
    "edge_processing",
    "fusion",
    "command",
    "actuator",
]

ATTACK_FAMILIES = [
    "benign",
    "ddos",
    "reconnaissance",
    "spoofing",
    "service_degradation",
    "multi_stage",
]

N_RUNS_PER_FAMILY = 20
STEPS_PER_RUN = 120

MISSION_WINDOW = 10
MISSION_CRIT = 0.70

FORECAST_HORIZONS = [10, 30, 60]

FEATURES = [
    "latency",
    "packet_loss",
    "utilization",
    "availability",
    "sensor_quality",
    "detector_score",
]


def build_graph():
    """Six-function mission dependency graph."""
    G = nx.DiGraph()

    G.add_nodes_from(BASE_NODES)

    for u, v in zip(BASE_NODES[:-1], BASE_NODES[1:]):
        G.add_edge(u, v)

    return G


def attack_intensity(attack, t):
    """Return normalized attack intensity in [0,1]."""

    # First 25 steps are clean for every run
    if t < 25:
        return 0.0

    x = (t - 25) / max(1, STEPS_PER_RUN - 25)

    if attack == "benign":
        return 0.0

    if attack == "ddos":
        return min(1.0, 0.25 + 0.85 * x)

    if attack == "reconnaissance":
        return min(0.55, 0.10 + 0.45 * x)

    if attack == "spoofing":
        return min(0.85, 0.15 + 0.70 * x)

    if attack == "service_degradation":
        return min(1.0, 0.05 + 0.95 * x)

    if attack == "multi_stage":
        if t < 50:
            return 0.25
        elif t < 80:
            return 0.55
        else:
            return 0.95

    return 0.0


def target_nodes(attack):
    """Mission functions primarily affected by each attack family."""

    mapping = {
        "benign": [],
        "ddos": ["tactical_link", "edge_processing"],
        "reconnaissance": ["sensor", "tactical_link"],
        "spoofing": ["sensor", "fusion"],
        "service_degradation": ["edge_processing", "command"],
        "multi_stage": [
            "sensor",
            "tactical_link",
            "edge_processing",
            "fusion",
            "command",
        ],
    }

    return mapping[attack]


def generate_node_telemetry(node, attack, intensity, rng):
    """Generate telemetry independently from mission labels."""

    affected = node in target_nodes(attack)
    a = intensity if affected else 0.15 * intensity

    latency = max(
        1.0,
        rng.normal(20.0 + 100.0 * a, 3.0)
    )

    packet_loss = np.clip(
        rng.normal(0.01 + 0.45 * a, 0.015),
        0.0,
        1.0,
    )

    utilization = np.clip(
        rng.normal(0.35 + 0.55 * a, 0.05),
        0.0,
        1.0,
    )

    availability = np.clip(
        rng.normal(0.995 - 0.55 * a, 0.02),
        0.0,
        1.0,
    )

    sensor_quality = np.clip(
        rng.normal(0.97 - 0.60 * a, 0.03),
        0.0,
        1.0,
    )

    # Detector score is telemetry only; it does NOT define mission failure.
    detector_score = np.clip(
        rng.normal(0.05 + 0.80 * a, 0.05),
        0.0,
        1.0,
    )

    return {
        "latency": latency,
        "packet_loss": packet_loss,
        "utilization": utilization,
        "availability": availability,
        "sensor_quality": sensor_quality,
        "detector_score": detector_score,
    }


def task_success_probability(node_records):
    """
    Independent mission-performance model.

    Mission success is derived from task execution quality,
    NOT from detector score and NOT from the future ML model.
    """

    mean_latency = np.mean(
        [x["latency"] for x in node_records]
    )

    mean_loss = np.mean(
        [x["packet_loss"] for x in node_records]
    )

    mean_availability = np.mean(
        [x["availability"] for x in node_records]
    )

    mean_quality = np.mean(
        [x["sensor_quality"] for x in node_records]
    )

    latency_penalty = np.clip(
        (mean_latency - 25.0) / 100.0,
        0.0,
        1.0,
    )

    p = (
        1.0
        - 0.35 * latency_penalty
        - 0.35 * mean_loss
        - 0.20 * (1.0 - mean_availability)
        - 0.20 * (1.0 - mean_quality)
    )

    return float(np.clip(p, 0.0, 1.0))


def simulate_run(run_id, attack, seed):
    rng = np.random.default_rng(seed)

    rows = []
    task_history = []

    for t in range(STEPS_PER_RUN):

        intensity = attack_intensity(attack, t)

        telemetry = {}

        for node in BASE_NODES:
            telemetry[node] = generate_node_telemetry(
                node,
                attack,
                intensity,
                rng,
            )

        # ----------------------------------------------------
        # Mission ground truth
        # ----------------------------------------------------
        p_success = task_success_probability(
            list(telemetry.values())
        )

        # Ten mission tasks per time step
        tasks = rng.binomial(
            1,
            p_success,
            size=10,
        )

        task_history.extend(tasks.tolist())

        recent = task_history[
            -MISSION_WINDOW * 10:
        ]

        mission_effectiveness = float(np.mean(recent))

        failure = int(
            mission_effectiveness < MISSION_CRIT
        )

        # ----------------------------------------------------
        # One row per node per time step
        # ----------------------------------------------------
        for node in BASE_NODES:

            x = telemetry[node]

            rows.append({
                "run_id": run_id,
                "seed": seed,
                "attack_family": attack,
                "time": t,
                "node_id": node,
                "latency": x["latency"],
                "packet_loss": x["packet_loss"],
                "utilization": x["utilization"],
                "availability": x["availability"],
                "sensor_quality": x["sensor_quality"],
                "detector_score": x["detector_score"],
                "mission_effectiveness":
                    mission_effectiveness,
                "failure": failure,
            })

    return rows


def add_future_failure_labels(df):
    """
    Label whether mission failure occurs within H future steps.
    Required later by E1.
    """

    run_time = (
        df[
            ["run_id", "time", "failure"]
        ]
        .drop_duplicates()
        .sort_values(["run_id", "time"])
    )

    labels = []

    for run_id, g in run_time.groupby("run_id"):

        g = g.sort_values("time")
        failures = g["failure"].to_numpy()
        times = g["time"].to_numpy()

        temp = pd.DataFrame({
            "run_id": run_id,
            "time": times,
        })

        for H in FORECAST_HORIZONS:

            y = np.zeros(len(g), dtype=np.int64)

            for i in range(len(g)):
                end = min(len(g), i + H + 1)

                # Future only: t+1 ... t+H
                y[i] = int(
                    failures[i + 1:end].max()
                    if i + 1 < end
                    else 0
                )

            temp[f"failure_within_{H}"] = y

        labels.append(temp)

    labels = pd.concat(
        labels,
        ignore_index=True,
    )

    return df.merge(
        labels,
        on=["run_id", "time"],
        how="left",
    )


def verify_dataset(df):
    print("\n" + "=" * 72)
    print("MissionShield E0 | Dataset Verification")
    print("=" * 72)

    print(f"Rows                : {len(df):,}")
    print(f"Runs                : {df.run_id.nunique()}")
    print(f"Nodes               : {df.node_id.nunique()}")
    print(
        f"Attack families     : "
        f"{df.attack_family.nunique()}"
    )

    print(
        f"Mission M_crit      : "
        f"{MISSION_CRIT:.2f}"
    )

    print(
        f"Mission effectiveness: "
        f"{df.mission_effectiveness.min():.3f} "
        f"to "
        f"{df.mission_effectiveness.max():.3f}"
    )

    print(
        f"Failure row rate    : "
        f"{df.failure.mean():.4f}"
    )

    print("\nRuns per attack family:")
    print(
        df[
            ["run_id", "attack_family"]
        ]
        .drop_duplicates()
        .groupby("attack_family")
        .size()
    )

    print("\nFuture failure label rates:")

    for H in FORECAST_HORIZONS:
        col = f"failure_within_{H}"

        unique = (
            df[
                ["run_id", "time", col]
            ]
            .drop_duplicates()
        )

        print(
            f"H={H:>2}: "
            f"{unique[col].mean():.4f}"
        )

    # Hard checks
    assert df.run_id.nunique() == (
        len(ATTACK_FAMILIES)
        * N_RUNS_PER_FAMILY
    )

    assert df.node_id.nunique() == 6

    assert not df.isnull().any().any()

    assert (
        df.mission_effectiveness
        .between(0, 1)
        .all()
    )

    print("\n[PASS] E0 dataset integrity checks passed.")


def main():

    print("=" * 72)
    print("MissionShield E0")
    print("Mission CPS Emulator + Dataset Generation")
    print("=" * 72)

    G = build_graph()

    print("\nMission graph:")
    for u, v in G.edges():
        print(f"  {u} -> {v}")

    all_rows = []

    run_counter = 0

    for attack_idx, attack in enumerate(ATTACK_FAMILIES):

        print(f"\nGenerating: {attack}")

        for r in range(N_RUNS_PER_FAMILY):

            run_counter += 1

            seed = (
                SEED
                + attack_idx * 1000
                + r
            )

            run_id = (
                f"{attack}_"
                f"{r:03d}"
            )

            rows = simulate_run(
                run_id,
                attack,
                seed,
            )

            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    df = add_future_failure_labels(df)

    output_file = (
        PROC_DIR /
        "missionshield_e0_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    config = {
        "seed": SEED,
        "nodes": BASE_NODES,
        "attack_families": ATTACK_FAMILIES,
        "runs_per_family": N_RUNS_PER_FAMILY,
        "steps_per_run": STEPS_PER_RUN,
        "mission_window": MISSION_WINDOW,
        "mission_crit": MISSION_CRIT,
        "forecast_horizons":
            FORECAST_HORIZONS,
        "features": FEATURES,
    }

    config_file = (
        RESULT_DIR /
        "e0_config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    verify_dataset(df)

    print("\nSaved:")
    print(f"  {output_file}")
    print(f"  {config_file}")

    print("\nE0 COMPLETE")


if __name__ == "__main__":
    main()