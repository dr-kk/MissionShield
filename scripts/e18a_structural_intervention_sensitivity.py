"""E18-A | Independent development-cohort restoration sensitivity.

Diagnostic only. Never trains on or reads the E8 confirmation cohort.
Preserves E0/E5B simulator dynamics and the sustained restore_service action.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(r'D:\MissionShield')
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'results' / 'e18_development'
OUT.mkdir(parents=True, exist_ok=True)

N_PER_FAMILY = 12
BASE_SEED = 2_000_042  # disjoint from E8 confirmation seed range
OFFSETS = (0, 15, 30)
TARGETS = ('edge_processing', 'fusion', 'command')
ACTION = 'restore_service'
COST = 0.25


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e0 = load('e18a_e0', SCRIPTS / 'e0_missionshield_setup_v2.py')
e5b = load('e18a_e5b', SCRIPTS / 'e5b_counterfactual_intervention_emulator.py')


def simulate(attack, seed, action_time=None, target=None):
    rng = np.random.default_rng(seed)
    onset, severity = e0.generate_run_profile(attack, rng)
    latent = {n: float(rng.uniform(0.0, 0.03)) for n in e0.BASE_NODES}
    disturbance = {'remaining': 0, 'strength': 0.0}
    task_history, timeline = [], []
    for t in range(e0.STEPS_PER_RUN):
        level = e0.attack_intensity(attack, t, onset, severity)
        benign = e0.benign_disturbance(t, rng, disturbance)
        active = action_time is not None and t >= action_time
        latent = e5b.update_damage_cf(
            latent, attack, level, benign, rng,
            ACTION if active else 'no_action', [target] if active else [], active
        )
        # Exactly preserve E0's telemetry RNG consumption/order.
        for node in e0.BASE_NODES:
            e0.generate_telemetry(node, attack, level, benign, latent[node], rng)
        p = e0.mission_task_probability(latent, rng)
        if active:
            p -= 0.005
        p = float(np.clip(p, 0.02, 0.995))
        tasks = rng.binomial(1, p, size=8)
        task_history.extend(tasks.tolist())
        me = float(np.mean(task_history[-e0.MISSION_WINDOW * 8:]))
        timeline.append((t, me, int(me < e0.MISSION_CRIT)))
    return onset, pd.DataFrame(timeline, columns=['time', 'mission_effectiveness', 'failure'])


def main():
    print('E18-A | STRUCTURAL INTERVENTION SENSITIVITY | DEVELOPMENT ONLY', flush=True)
    print(f'Families={len(e0.ATTACK_FAMILIES)}, seeds/family={N_PER_FAMILY}, offsets={OFFSETS}', flush=True)
    records = []
    for fi, family in enumerate(e0.ATTACK_FAMILIES):
        for r in range(N_PER_FAMILY):
            seed = BASE_SEED + fi * 10_000 + r
            run_id = f'e18dev_{family}_{r:03d}'
            onset, baseline = simulate(family, seed)
            # Exact replay gate against the independent E0 implementation.
            reference = pd.DataFrame(e0.simulate_run(run_id, family, seed))
            ref = reference[['time', 'mission_effectiveness', 'failure']].drop_duplicates('time').sort_values('time')
            if not np.allclose(ref.mission_effectiveness.to_numpy(), baseline.mission_effectiveness.to_numpy(), atol=1e-12, rtol=0):
                raise RuntimeError(f'Baseline E0 replay mismatch: {run_id}')
            if not np.array_equal(ref.failure.to_numpy(), baseline.failure.to_numpy()):
                raise RuntimeError(f'Baseline failure replay mismatch: {run_id}')
            base_any = int(baseline.failure.max())
            for offset in OFFSETS:
                t0 = int(onset + offset)
                base_post = int(baseline.loc[baseline.time >= t0, 'failure'].max())
                pre_failure = int(baseline.loc[baseline.time < t0, 'failure'].max())
                base_me = float(baseline.mission_effectiveness.mean())
                for target in TARGETS:
                    _, cf = simulate(family, seed, t0, target)
                    # A genuine intervention cannot affect its pre-action prefix.
                    before = baseline.time < t0
                    if not np.array_equal(cf.loc[before, 'failure'].to_numpy(), baseline.loc[before, 'failure'].to_numpy()):
                        raise RuntimeError(f'Pre-action failure changed: {run_id} {target} {t0}')
                    if not np.allclose(cf.loc[before, 'mission_effectiveness'].to_numpy(), baseline.loc[before, 'mission_effectiveness'].to_numpy(), atol=1e-12, rtol=0):
                        raise RuntimeError(f'Pre-action effectiveness changed: {run_id} {target} {t0}')
                    cf_any = int(cf.failure.max())
                    cf_post = int(cf.loc[cf.time >= t0, 'failure'].max())
                    records.append(dict(
                        run_id=run_id, attack_family=family, seed=seed,
                        onset=onset, offset=offset, action_time=t0, target=target,
                        baseline_failure=base_any, post_failure=cf_any,
                        baseline_post_failure=base_post, cf_post_failure=cf_post,
                        prior_failure=pre_failure,
                        prevented=int(base_any == 1 and cf_any == 0),
                        induced=int(base_any == 0 and cf_any == 1),
                        net_prevented=base_any - cf_any,
                        effectiveness_gain=float(cf.mission_effectiveness.mean() - base_me),
                        action_cost=COST,
                    ))
        print(f'  Completed family: {family}', flush=True)
    df = pd.DataFrame(records)
    expected = len(e0.ATTACK_FAMILIES) * N_PER_FAMILY * len(OFFSETS) * len(TARGETS)
    assert len(df) == expected
    assert not df.duplicated(['run_id', 'offset', 'target']).any()
    assert df.groupby(['run_id', 'offset']).target.nunique().eq(3).all()
    detail = OUT / 'e18a_target_sensitivity_detail.csv'
    df.to_csv(detail, index=False)
    summary = df.groupby(['offset', 'target'], as_index=False).agg(
        missions=('run_id', 'nunique'), baseline_failures=('baseline_failure', 'sum'),
        post_failures=('post_failure', 'sum'), prevented=('prevented', 'sum'),
        induced=('induced', 'sum'), net_prevented=('net_prevented', 'sum'),
        prior_failure=('prior_failure', 'sum'),
        mean_effectiveness_gain=('effectiveness_gain', 'mean'),
    )
    summary['failure_reduction_pct'] = 100 * summary.net_prevented / summary.baseline_failures.replace(0, np.nan)
    summary.to_csv(OUT / 'e18a_target_sensitivity_summary.csv', index=False)
    family_summary = df.groupby(['attack_family', 'offset', 'target'], as_index=False).agg(
        missions=('run_id', 'nunique'), baseline_failures=('baseline_failure', 'sum'),
        post_failures=('post_failure', 'sum'), prevented=('prevented', 'sum'),
        induced=('induced', 'sum'), net_prevented=('net_prevented', 'sum'),
        mean_effectiveness_gain=('effectiveness_gain', 'mean'),
    )
    family_summary.to_csv(OUT / 'e18a_target_by_attack_family.csv', index=False)
    metadata = dict(experiment='E18-A', role='development diagnostic, not confirmation',
                    base_seed=BASE_SEED, runs_per_family=N_PER_FAMILY,
                    offsets=list(OFFSETS), targets=list(TARGETS),
                    action='sustained restore_service', cost=COST,
                    simulations='E0/E5B mechanisms unchanged',
                    caveat='Single stochastic scenario set; repeated intervention from t0 onward; no learned policy.')
    (OUT / 'e18a_protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nSUMMARY (mission-level ever-failure; lower post_failures is better):')
    print(summary.to_string(index=False))
    print('\n[PASS] E0 baseline replay, pre-action invariance, complete matched targets')
    print('Saved to:', OUT)


if __name__ == '__main__':
    main()
