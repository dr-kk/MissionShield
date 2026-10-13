"""MissionShield E18-C | Leakage-safe intervention learning dataset.

Run: python scripts/e18c_intervention_learning_dataset.py
Requires existing scripts/e0_missionshield_setup_v2.py and
scripts/e5b_counterfactual_intervention_emulator.py.
Does NOT fit a model or touch the E8 confirmation cohort.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r'D:\MissionShield')
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'results' / 'e18_learning'
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ('edge_processing', 'fusion', 'command')
DECISION_TIMES = (60, 75)  # fixed clock times; NOT derived from hidden attack onset
TRAIN_PER_FAMILY = 20
VAL_PER_FAMILY = 10
SEED_BASE = 2_300_000  # disjoint from E8 confirmation seed region
FEATURE_WINDOW = 12

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {filename}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

e0 = load('e0_for_e18c', 'e0_missionshield_setup_v2.py')
e5 = load('e5_for_e18c', 'e5b_counterfactual_intervention_emulator.py')

def explicit_restore(attack, seed, t0, target):
    """Same RNG stream as E0; target explicitly selected, never inferred from attack."""
    rng = np.random.default_rng(seed)
    onset, severity = e0.generate_run_profile(attack, rng)
    latent = {n: float(rng.uniform(0., .03)) for n in e0.BASE_NODES}
    disturbance = {'remaining': 0, 'strength': 0.0}
    history = []
    timeline = []
    for t in range(e0.STEPS_PER_RUN):
        level = e0.attack_intensity(attack, t, onset, severity)
        benign = e0.benign_disturbance(t, rng, disturbance)
        active = t >= t0
        latent = e5.update_damage_cf(latent, attack, level, benign, rng,
                                     'restore_service', [target], active)
        for node in e0.BASE_NODES:
            e0.generate_telemetry(node, attack, level, benign, latent[node], rng)
        p = e0.mission_task_probability(latent, rng)
        if active:
            p -= .005
        p = float(np.clip(p, .02, .995))
        tasks = rng.binomial(1, p, size=8)
        history.extend(tasks.tolist())
        me = float(np.mean(history[-e0.MISSION_WINDOW * 8:]))
        timeline.append((t, me, int(me < e0.MISSION_CRIT)))
    return pd.DataFrame(timeline, columns=['time', 'mission_effectiveness', 'failure'])

def features_at(baseline, t0):
    """Only observed telemetry/mission effectiveness strictly before decision time."""
    pre = baseline[baseline.time < t0]
    window = pre[pre.time >= t0 - FEATURE_WINDOW]
    result = {'decision_time': t0}
    for node in e0.BASE_NODES:
        g = window[window.node_id == node].sort_values('time')
        if len(g) != FEATURE_WINDOW:
            raise AssertionError('Missing historical node telemetry')
        for feat in e0.FEATURES:
            vals = g[feat].to_numpy(dtype=float)
            key = f'{node}__{feat}'
            result[key + '__mean'] = float(vals.mean())
            result[key + '__last'] = float(vals[-1])
            result[key + '__slope'] = float(vals[-1] - vals[0])
    timeline = pre.drop_duplicates('time').sort_values('time')
    me = timeline.mission_effectiveness.to_numpy(dtype=float)
    result['observed_me_last'] = float(me[-1])
    result['observed_me_mean12'] = float(me[-FEATURE_WINDOW:].mean())
    result['observed_me_delta12'] = float(me[-1] - me[-FEATURE_WINDOW])
    return result

def main():
    print('E18-C | LEAKAGE-SAFE INTERVENTION DATA | NO MODEL FITTING')
    print(f'Train={TRAIN_PER_FAMILY}/family, validation={VAL_PER_FAMILY}/family, times={DECISION_TIMES}')
    records, audits = [], []
    for fi, attack in enumerate(e0.ATTACK_FAMILIES):
        for k in range(TRAIN_PER_FAMILY + VAL_PER_FAMILY):
            split = 'train' if k < TRAIN_PER_FAMILY else 'validation'
            seed = SEED_BASE + fi * 10000 + k
            run_id = f'e18c_{fi:02d}_{k:03d}'
            raw = pd.DataFrame(e0.simulate_run(run_id, attack, seed))
            base = raw[['time', 'mission_effectiveness', 'failure']].drop_duplicates('time').sort_values('time').reset_index(drop=True)
            assert len(base) == e0.STEPS_PER_RUN
            for t0 in DECISION_TIMES:
                # Exclude missions already failed; pre-action failures cannot be prevented.
                eligible = not bool(base.loc[base.time < t0, 'failure'].max())
                if not eligible:
                    audits.append({'run_id': run_id, 'split': split, 'decision_time': t0,
                                   'eligible': False, 'reason': 'predecision_failure'})
                    continue
                feat = features_at(raw, t0)
                bpost = base.loc[base.time >= t0]
                base_fail = int(bpost.failure.max())
                base_mean = float(bpost.mission_effectiveness.mean())
                for target in TARGETS:
                    cf = explicit_restore(attack, seed, t0, target)
                    # Counterfactual must be bitwise-identical until action begins.
                    pre = cf.loc[cf.time < t0]
                    orig_pre = base.loc[base.time < t0]
                    if not np.array_equal(pre.failure.to_numpy(), orig_pre.failure.to_numpy()) or not np.allclose(
                            pre.mission_effectiveness.to_numpy(), orig_pre.mission_effectiveness.to_numpy(), atol=0, rtol=0):
                        raise RuntimeError(f'Pre-action RNG mismatch: {run_id}, {t0}, {target}')
                    cpost = cf.loc[cf.time >= t0]
                    post_fail = int(cpost.failure.max())
                    records.append({**feat, 'split': split, 'run_id': run_id,
                                    'target': target, 'action_cost': .25,
                                    'baseline_post_failure': base_fail,
                                    'post_failure': post_fail,
                                    'net_failure_benefit': base_fail - post_fail,
                                    'effectiveness_gain': float(cpost.mission_effectiveness.mean() - base_mean)})
                audits.append({'run_id': run_id, 'split': split, 'decision_time': t0,
                               'eligible': True, 'reason': 'eligible'})
        print(f'  Completed {attack}', flush=True)
    df = pd.DataFrame(records)
    audit = pd.DataFrame(audits)
    if df.empty:
        raise RuntimeError('No eligible training examples')
    keys = ['split', 'run_id', 'decision_time']
    assert not df.duplicated(keys + ['target']).any()
    assert (df.groupby(keys).target.nunique() == len(TARGETS)).all()
    assert set(df[df.split == 'train'].run_id).isdisjoint(set(df[df.split == 'validation'].run_id))
    forbidden = ('attack_family', 'seed', 'attack_onset', 'attack_severity', 'latent_damage', 'future_failure')
    assert all(not any(f in c for f in forbidden) for c in df.columns)
    # Split by mission, never by target/decision row. Never expose outcome columns to the feature pipeline.
    features = ['decision_time'] + [c for c in df.columns if c.endswith(('__mean', '__last', '__slope'))] + [
        'observed_me_last', 'observed_me_mean12', 'observed_me_delta12']
    assert not df[features].isna().any().any()
    out_csv = OUT / 'e18c_intervention_learning_long.csv'
    df.to_csv(out_csv, index=False)
    audit.to_csv(OUT / 'e18c_eligibility_audit.csv', index=False)
    proto = {'experiment': 'E18-C', 'seed_base': SEED_BASE, 'train_per_family': TRAIN_PER_FAMILY,
             'validation_per_family': VAL_PER_FAMILY, 'decision_times': DECISION_TIMES,
             'feature_window': FEATURE_WINDOW, 'targets': TARGETS,
             'features': features, 'outcome_columns': ['baseline_post_failure', 'post_failure',
             'net_failure_benefit', 'effectiveness_gain'],
             'policy_note': 'Fixed decision times; hidden attack onset never used for features or timing.',
             'split_note': 'Disjoint mission seeds; same mission may appear at two times within one split.',
             'restoration_note': 'Original E5-B sustained restoration, unchanged.',
             'evaluation_note': 'Exploratory simulator development only; no E8 confirmation outcomes used.'}
    (OUT / 'e18c_protocol.json').write_text(json.dumps(proto, indent=2), encoding='utf-8')
    print('\nDATA SUMMARY')
    print(df.groupby(['split', 'decision_time', 'target']).agg(
        rows=('run_id', 'size'), baseline_failures=('baseline_post_failure', 'sum'),
        post_failures=('post_failure', 'sum'), mean_gain=('effectiveness_gain', 'mean')).to_string())
    print('\nELIGIBILITY')
    print(audit.groupby(['split', 'decision_time', 'eligible']).size().to_string())
    print(f'\n[PASS] Replay invariance, pairing, split disjointness and feature checks.\nSaved: {OUT}')
    print('CAUTION: These are simulated counterfactual labels, not empirical causal estimates.')

if __name__ == '__main__':
    main()
