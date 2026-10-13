from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(r"D:\MissionShield")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

e0 = load_module(
    "e0_replay",
    SCRIPTS / "e0_missionshield_setup_v2.py"
)

e5 = load_module(
    "e5_replay",
    SCRIPTS / "e5b_counterfactual_intervention_emulator.py"
)

cohort = pd.read_csv(
    RESULTS / "e8_independent_cohort.csv"
)

runs = (
    cohort[["run_id", "seed", "attack_family"]]
    .drop_duplicates("run_id")
)

assert len(runs) == 180

checked = 0
mismatched = []

for row in runs.itertuples(index=False):

    run_id = str(row.run_id)
    seed = int(row.seed)
    attack = str(row.attack_family)

    original = (
        cohort[cohort.run_id.eq(run_id)]
        [["time", "mission_effectiveness", "failure"]]
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Use the same explicit-target replay logic as E8.
    rng = np.random.default_rng(seed)

    onset, severity = e0.generate_run_profile(
        attack, rng
    )

    latent = {
        n: float(rng.uniform(0.0, 0.03))
        for n in e0.BASE_NODES
    }

    disturbance = {
        "remaining": 0,
        "strength": 0.0
    }

    task_history = []
    timeline = []

    for t in range(e0.STEPS_PER_RUN):

        level = e0.attack_intensity(
            attack, t, onset, severity
        )

        benign = e0.benign_disturbance(
            t, rng, disturbance
        )

        latent = e5.update_damage_cf(
            latent,
            attack,
            level,
            benign,
            rng,
            "restore_service",
            ["command"],
            False
        )

        for node in e0.BASE_NODES:
            e0.generate_telemetry(
                node,
                attack,
                level,
                benign,
                latent[node],
                rng
            )

        p = e0.mission_task_probability(
            latent, rng
        )

        p = float(np.clip(p, 0.02, 0.995))

        tasks = rng.binomial(
            1, p, size=8
        )

        task_history.extend(tasks.tolist())

        me = float(np.mean(
            task_history[-e0.MISSION_WINDOW * 8:]
        ))

        timeline.append(
            (t, me, int(me < e0.MISSION_CRIT))
        )

    replay = pd.DataFrame(
        timeline,
        columns=[
            "time",
            "mission_effectiveness",
            "failure"
        ]
    )

    same_time = np.array_equal(
        original.time.to_numpy(),
        replay.time.to_numpy()
    )

    same_failure = np.array_equal(
        original.failure.to_numpy(),
        replay.failure.to_numpy()
    )

    same_effectiveness = np.allclose(
        original.mission_effectiveness.to_numpy(),
        replay.mission_effectiveness.to_numpy(),
        rtol=0,
        atol=1e-12
    )

    if not (
        same_time and
        same_failure and
        same_effectiveness
    ):
        mismatched.append({
            "run_id": run_id,
            "time_match": same_time,
            "failure_match": same_failure,
            "effectiveness_match": same_effectiveness
        })

    checked += 1

print("\n=== E18-D3 ZERO-ACTION REPLAY ===")
print("Missions checked:", checked)
print("Mismatched missions:", len(mismatched))

if mismatched:
    print(pd.DataFrame(mismatched).to_string(index=False))
    raise SystemExit("FAIL: Replay differs from stored cohort.")

print("PASS: All 180 no-action trajectories reproduced.")
print("Existing experiment files unchanged.")
