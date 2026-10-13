import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# MissionShield E0-v2
# Harder Mission CPS Benchmark
#
# Design principles:
# 1. Benign conditions can look anomalous.
# 2. Attacks do not always cause mission failure.
# 3. Attack/benign telemetry distributions overlap.
# 4. Degradation propagates with delay through dependencies.
# 5. Mission failure depends on accumulated functional state.
# 6. Detector score is noisy and is NOT mission ground truth.
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(r"D:\MissionShield")
PROC_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results"

PROC_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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

N_RUNS_PER_FAMILY = 30
STEPS_PER_RUN = 150

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
    G = nx.DiGraph()

    G.add_nodes_from(BASE_NODES)

    for u, v in zip(BASE_NODES[:-1], BASE_NODES[1:]):
        G.add_edge(u, v)

    return G


def target_nodes(attack):
    return {
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
    }[attack]


def generate_run_profile(attack, rng):
    """
    Randomize attack severity/onset.

    Some attacks remain weak enough not to cause mission failure.
    """

    onset = int(rng.integers(25, 60))

    if attack == "benign":
        severity = 0.0
    elif attack == "reconnaissance":
        severity = rng.uniform(0.15, 0.45)
    elif attack == "ddos":
        severity = rng.uniform(0.35, 0.85)
    elif attack == "spoofing":
        severity = rng.uniform(0.30, 0.80)
    elif attack == "service_degradation":
        severity = rng.uniform(0.30, 0.90)
    else:
        severity = rng.uniform(0.40, 0.95)

    return onset, severity


def attack_intensity(attack, t, onset, severity):
    if attack == "benign" or t < onset:
        return 0.0

    progress = (
        (t - onset) /
        max(1, STEPS_PER_RUN - onset)
    )

    if attack == "reconnaissance":
        value = severity * (
            0.45 + 0.15 * np.sin(t / 6.0)
        )

    elif attack == "ddos":
        # Bursty rather than monotonically obvious
        value = severity * (
            0.40
            + 0.30 * progress
            + 0.20 * np.sin(t / 4.0)
        )

    elif attack == "spoofing":
        value = severity * (
            0.25 + 0.60 * progress
        )

    elif attack == "service_degradation":
        value = severity * (
            0.20 + 0.70 * progress
        )

    elif attack == "multi_stage":
        if progress < 0.30:
            value = 0.30 * severity
        elif progress < 0.65:
            value = 0.55 * severity
        else:
            value = 0.85 * severity

    else:
        value = 0.0

    return float(np.clip(value, 0.0, 1.0))


def benign_disturbance(t, rng, state):
    """
    Benign congestion / workload spikes.

    These produce attack-like telemetry without malicious activity.
    """

    if state["remaining"] <= 0:
        if rng.random() < 0.025:
            state["remaining"] = int(
                rng.integers(4, 15)
            )
            state["strength"] = float(
                rng.uniform(0.20, 0.60)
            )
        else:
            return 0.0

    state["remaining"] -= 1

    return state["strength"]


def update_latent_damage(
    previous,
    attack,
    attack_level,
    benign_level,
    rng,
):
    """
    Hidden functional degradation propagates through the mission chain.

    This state is deliberately NOT exposed directly to the learner.
    """

    new = {}

    attacked = target_nodes(attack)

    for idx, node in enumerate(BASE_NODES):

        direct = (
            attack_level
            if node in attacked
            else 0.0
        )

        upstream = 0.0

        if idx > 0:
            parent = BASE_NODES[idx - 1]
            upstream = previous[parent]

        # Persistence + direct damage + delayed upstream propagation
        damage = (
            0.82 * previous[node]
            + 0.12 * direct
            + 0.08 * upstream
        )

        # Benign workload can temporarily degrade functionality.
        damage += 0.035 * benign_level

        damage += rng.normal(0.0, 0.012)

        # Slow recovery when direct pressure disappears.
        if direct < 0.05:
            damage -= 0.025

        new[node] = float(
            np.clip(damage, 0.0, 1.0)
        )

    return new


def generate_telemetry(
    node,
    attack,
    attack_level,
    benign_level,
    latent_damage,
    rng,
):
    attacked = node in target_nodes(attack)

    direct = (
        attack_level
        if attacked
        else 0.10 * attack_level
    )

    # Telemetry is an imperfect view of hidden functional damage.
    signal = (
        0.35 * direct
        + 0.40 * latent_damage
        + 0.40 * benign_level
    )

    latency = rng.normal(
        24.0 + 55.0 * signal,
        8.0,
    )

    packet_loss = rng.normal(
        0.025 + 0.22 * signal,
        0.035,
    )

    utilization = rng.normal(
        0.42 + 0.32 * signal,
        0.10,
    )

    availability = rng.normal(
        0.975 - 0.25 * latent_damage,
        0.045,
    )

    # Spoofing affects information quality more than network load.
    spoof_extra = (
        0.22 * attack_level
        if attack == "spoofing"
        and node in ["sensor", "fusion"]
        else 0.0
    )

    sensor_quality = rng.normal(
        0.94
        - 0.30 * latent_damage
        - spoof_extra,
        0.065,
    )

    # Imperfect detector: attacks and benign disturbances overlap.
    detector_signal = (
        0.35 * direct
        + 0.18 * latent_damage
        + 0.25 * benign_level
    )

    detector_score = rng.normal(
        0.12 + detector_signal,
        0.12,
    )

    return {
        "latency": float(max(1.0, latency)),
        "packet_loss": float(
            np.clip(packet_loss, 0.0, 1.0)
        ),
        "utilization": float(
            np.clip(utilization, 0.0, 1.0)
        ),
        "availability": float(
            np.clip(availability, 0.0, 1.0)
        ),
        "sensor_quality": float(
            np.clip(sensor_quality, 0.0, 1.0)
        ),
        "detector_score": float(
            np.clip(detector_score, 0.0, 1.0)
        ),
    }


def mission_task_probability(latent, rng):
    """
    Mission outcome depends on hidden accumulated functional state.

    Bottlenecks and downstream propagation matter more than
    instantaneous observed telemetry.
    """

    d = np.array(
        [latent[n] for n in BASE_NODES]
    )

    # Downstream functions have somewhat greater mission importance.
    weights = np.array(
        [0.10, 0.15, 0.17, 0.18, 0.20, 0.20]
    )

    weighted_damage = float(
        np.sum(weights * d)
    )

    bottleneck = float(np.max(d))

    p = (
        0.985
        - 0.52 * weighted_damage
        - 0.27 * bottleneck
    )

    # Unmodelled mission variability
    p += rng.normal(0.0, 0.025)

    return float(
        np.clip(p, 0.02, 0.995)
    )


def simulate_run(run_id, attack, seed):

    rng = np.random.default_rng(seed)

    onset, severity = generate_run_profile(
        attack,
        rng,
    )

    latent = {
        n: float(rng.uniform(0.0, 0.03))
        for n in BASE_NODES
    }

    disturbance_state = {
        "remaining": 0,
        "strength": 0.0,
    }

    task_history = []
    rows = []

    for t in range(STEPS_PER_RUN):

        attack_level = attack_intensity(
            attack,
            t,
            onset,
            severity,
        )

        benign_level = benign_disturbance(
            t,
            rng,
            disturbance_state,
        )

        latent = update_latent_damage(
            latent,
            attack,
            attack_level,
            benign_level,
            rng,
        )

        telemetry = {}

        for node in BASE_NODES:
            telemetry[node] = generate_telemetry(
                node,
                attack,
                attack_level,
                benign_level,
                latent[node],
                rng,
            )

        # ----------------------------------------------------
        # Independent mission ground truth
        # ----------------------------------------------------

        p_success = mission_task_probability(
            latent,
            rng,
        )

        # 8 mission tasks per second/time step
        tasks = rng.binomial(
            1,
            p_success,
            size=8,
        )

        task_history.extend(
            tasks.tolist()
        )

        recent = task_history[
            -MISSION_WINDOW * 8:
        ]

        mission_effectiveness = float(
            np.mean(recent)
        )

        failure = int(
            mission_effectiveness < MISSION_CRIT
        )

        for node in BASE_NODES:

            x = telemetry[node]

            rows.append({
                "run_id": run_id,
                "seed": seed,
                "attack_family": attack,
                "attack_onset": onset,
                "attack_severity": severity,
                "time": t,
                "node_id": node,
                **x,
                "mission_effectiveness":
                    mission_effectiveness,
                "failure": failure,
            })

    return rows


def add_future_labels(df):

    timeline = (
        df[
            ["run_id", "time", "failure"]
        ]
        .drop_duplicates()
        .sort_values(["run_id", "time"])
    )

    outputs = []

    for run_id, g in timeline.groupby(
        "run_id"
    ):

        g = g.sort_values("time")

        failures = (
            g["failure"]
            .to_numpy(dtype=int)
        )

        times = g["time"].to_numpy()

        temp = pd.DataFrame({
            "run_id": run_id,
            "time": times,
        })

        for H in FORECAST_HORIZONS:

            y = np.zeros(
                len(g),
                dtype=np.int64,
            )

            for i in range(len(g)):

                start = i + 1
                end = min(
                    len(g),
                    i + H + 1,
                )

                if start < end:
                    y[i] = int(
                        failures[
                            start:end
                        ].max()
                    )

            temp[
                f"failure_within_{H}"
            ] = y

        outputs.append(temp)

    labels = pd.concat(
        outputs,
        ignore_index=True,
    )

    return df.merge(
        labels,
        on=["run_id", "time"],
        how="left",
    )


def verify(df):

    print("\n" + "=" * 72)
    print("MissionShield E0-v2 | Verification")
    print("=" * 72)

    timeline = (
        df[
            [
                "run_id",
                "attack_family",
                "time",
                "mission_effectiveness",
                "failure",
            ]
        ]
        .drop_duplicates(
            ["run_id", "time"]
        )
    )

    run_summary = (
        timeline.groupby(
            ["run_id", "attack_family"]
        )["failure"]
        .max()
        .reset_index()
    )

    print(
        f"Rows                 : "
        f"{len(df):,}"
    )

    print(
        f"Runs                 : "
        f"{df.run_id.nunique()}"
    )

    print(
        f"Nodes                : "
        f"{df.node_id.nunique()}"
    )

    print(
        f"Attack families      : "
        f"{df.attack_family.nunique()}"
    )

    print(
        f"Mission effectiveness: "
        f"{df.mission_effectiveness.min():.3f}"
        f" to "
        f"{df.mission_effectiveness.max():.3f}"
    )

    print(
        f"Failure row rate     : "
        f"{timeline.failure.mean():.4f}"
    )

    print(
        f"Runs with >=1 failure: "
        f"{run_summary.failure.mean():.4f}"
    )

    print("\nFailure-producing runs by family:")

    family = (
        run_summary
        .groupby("attack_family")
        ["failure"]
        .agg(["sum", "count", "mean"])
    )

    print(family)

    print("\nFuture failure label rates:")

    for H in FORECAST_HORIZONS:

        col = f"failure_within_{H}"

        temp = (
            df[
                ["run_id", "time", col]
            ]
            .drop_duplicates()
        )

        print(
            f"H={H:>2}: "
            f"{temp[col].mean():.4f}"
        )

    # Important sanity check:
    # benign runs should occasionally look anomalous,
    # but ideally should not systematically fail.
    benign = run_summary[
        run_summary.attack_family
        == "benign"
    ]

    print(
        "\nBenign runs with mission failure: "
        f"{int(benign.failure.sum())}/"
        f"{len(benign)}"
    )

    assert not df.isnull().any().any()
    assert df.node_id.nunique() == 6
    assert df.run_id.nunique() == (
        len(ATTACK_FAMILIES)
        * N_RUNS_PER_FAMILY
    )

    assert (
        df.mission_effectiveness
        .between(0, 1)
        .all()
    )

    print(
        "\n[PASS] E0-v2 integrity checks passed."
    )


def main():

    print("=" * 72)
    print("MissionShield E0-v2")
    print("Harder Temporal Mission CPS Benchmark")
    print("=" * 72)

    G = build_graph()

    print("\nMission dependency graph:")

    for u, v in G.edges():
        print(f"  {u} -> {v}")

    all_rows = []

    for family_idx, attack in enumerate(
        ATTACK_FAMILIES
    ):

        print(
            f"\nGenerating {attack} "
            f"({N_RUNS_PER_FAMILY} runs)"
        )

        for r in range(
            N_RUNS_PER_FAMILY
        ):

            seed = (
                SEED
                + family_idx * 10000
                + r
            )

            run_id = (
                f"v2_{attack}_{r:03d}"
            )

            all_rows.extend(
                simulate_run(
                    run_id,
                    attack,
                    seed,
                )
            )

    df = pd.DataFrame(all_rows)

    df = add_future_labels(df)

    output = (
        PROC_DIR /
        "missionshield_e0_v2_dataset.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    config = {
        "version": "E0-v2",
        "seed": SEED,
        "nodes": BASE_NODES,
        "attack_families":
            ATTACK_FAMILIES,
        "runs_per_family":
            N_RUNS_PER_FAMILY,
        "steps_per_run":
            STEPS_PER_RUN,
        "mission_window":
            MISSION_WINDOW,
        "mission_crit":
            MISSION_CRIT,
        "forecast_horizons":
            FORECAST_HORIZONS,
        "features": FEATURES,
        "design": [
            "overlapping attack/benign telemetry",
            "benign workload disturbances",
            "variable attack onset and severity",
            "non-fatal attack runs",
            "latent temporal degradation",
            "dependency propagation",
        ],
    }

    config_file = (
        RESULT_DIR /
        "e0_v2_config.json"
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

    verify(df)

    print("\nSaved:")
    print(f"  {output}")
    print(f"  {config_file}")

    print("\nE0-v2 COMPLETE")


if __name__ == "__main__":
    main()