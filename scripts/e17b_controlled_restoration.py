"""E17-B controlled restoration benchmark for MissionShield E8.

Copy to D:\\MissionShield\\scripts\\e17b_controlled_restoration.py and run there.
Requires original E0, E1, E5B modules and E8 generated CSVs.
No original result file is modified.

Caution: random-time sampling from all E8 eligible windows is a retrospective
negative control, NOT a deployable online policy. Model comparisons require
separately generated baseline-forecaster predictions.
"""
from pathlib import Path
import importlib.util
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
RESULTS = ROOT / 'results'
OUT = RESULTS / 'e17_controls'
TARGETS = ('edge_processing', 'fusion', 'command')
ACTION_COST = 0.25
RANDOM_SEED = 170042
HORIZONS = (10, 30, 60)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_rng(*parts):
    digest = hashlib.sha256('|'.join(map(str, (RANDOM_SEED,) + parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def require_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f'{name}: missing columns {missing}')


def main():
    e0 = load_module('e0_e17b', SCRIPTS / 'e0_missionshield_setup_v2.py')
    e5b = load_module('e5b_e17b', SCRIPTS / 'e5b_counterfactual_intervention_emulator.py')
    e8 = load_module('e8_e17b', SCRIPTS / 'e8_independent_cohort_confirmation.py')
    cohort = pd.read_csv(RESULTS / 'e8_independent_cohort.csv')
    details = pd.read_csv(RESULTS / 'e8_confirmation_intervention_results.csv')
    predictions = pd.read_csv(RESULTS / 'e8_confirmation_predictions.csv')
    require_columns(cohort, ['run_id', 'seed', 'attack_family', 'attack_onset', 'time', 'failure', 'mission_effectiveness'], 'cohort')
    require_columns(details, ['run_id', 'horizon', 'triggered', 'trigger_time', 'observable_target', 'baseline_failure', 'post_policy_failure', 'incurred_cost'], 'details')
    require_columns(predictions, ['run_id', 'horizon', 'time', 'alarm_fixed_far'], 'predictions')
    run_info = cohort[['run_id', 'seed', 'attack_family', 'attack_onset']].drop_duplicates()
    if run_info.run_id.duplicated().any() or len(run_info) != 180:
        raise RuntimeError(f'Expected 180 unique missions, got {len(run_info)}')
    info_map = run_info.set_index('run_id').to_dict('index')
    base = cohort[['run_id', 'time', 'failure', 'mission_effectiveness']].drop_duplicates(['run_id', 'time'])
    baseline = base.groupby('run_id').agg(baseline_failure=('failure', 'max'), baseline_mean_effectiveness=('mission_effectiveness', 'mean'))
    if int(baseline.baseline_failure.sum()) != 48:
        raise RuntimeError('Baseline failure count is not 48; stop to investigate')
    if len(details) != 540 or details.duplicated(['horizon', 'run_id']).any():
        raise RuntimeError('E8 intervention records are incomplete or duplicated')
    if not set(details.run_id) == set(run_info.run_id):
        raise RuntimeError('E8 intervention mission IDs do not match cohort')
    for row in details.itertuples(index=False):
        if int(row.baseline_failure) != int(baseline.loc[row.run_id, 'baseline_failure']):
            raise RuntimeError(f'Baseline mismatch: {row.run_id} H{row.horizon}')
    # Source-equivalence check: E8 action dynamics must match the reference implementation.
    if e8.ACTION != 'restore_service' or float(e8.ACTION_COST) != ACTION_COST or tuple(e8.ELIGIBLE_RESTORE) != TARGETS:
        raise RuntimeError('E8 restoration action/target definitions changed')
    if e8.e0.STEPS_PER_RUN != e0.STEPS_PER_RUN:
        raise RuntimeError('E8 simulator and E0 disagree on step count')

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cache = {}

    def evaluate(rid, time, target):
        key = (rid, int(time), target)
        if key not in cache:
            inf = info_map[rid]
            onset, cf = e8.simulate_explicit_target(str(inf['attack_family']), int(inf['seed']), int(time), target)
            if int(onset) != int(inf['attack_onset']):
                raise RuntimeError(f'Onset mismatch: {rid}')
            cache[key] = (int(cf.failure.max()), float(cf.mission_effectiveness.mean()))
        return cache[key]

    def append(H, rid, policy, time=None, target=None, reference=None):
        b = baseline.loc[rid]
        if time is None:
            pf = int(b.baseline_failure)
            eff = float(b.baseline_mean_effectiveness)
            action = 0
        elif reference is not None:
            pf = int(reference.post_policy_failure)
            eff = float(reference.post_policy_mean_effectiveness)
            action = 1
        else:
            pf, eff = evaluate(rid, time, target)
            action = 1
        rows.append(dict(horizon=H, run_id=rid, attack_family=str(info_map[rid]['attack_family']),
                         policy=policy, baseline_failure=int(b.baseline_failure),
                         post_policy_failure=pf, failure_prevented=int(int(b.baseline_failure)==1 and pf==0),
                         failure_induced=int(int(b.baseline_failure)==0 and pf==1),
                         action_executed=action, action_time=time if action else np.nan,
                         restored_node=target if action else 'none',
                         incurred_cost=ACTION_COST * action,
                         baseline_mean_effectiveness=float(b.baseline_mean_effectiveness),
                         post_policy_mean_effectiveness=eff,
                         mission_effectiveness_gain=eff-float(b.baseline_mean_effectiveness)))

    for H in HORIZONS:
        d = details[details.horizon == H].set_index('run_id')
        p = predictions[predictions.horizon == H]
        valid = p.groupby('run_id').time.apply(lambda x: sorted(set(map(int, x)))).to_dict()
        triggered_ids = sorted(d.index[d.triggered.astype(int) == 1].tolist())
        K = len(triggered_ids)
        if K not in (99, 105, 117):
            raise RuntimeError(f'Unexpected trigger count H{H}: {K}')
        for rid in sorted(info_map):
            ref = d.loc[rid]
            append(H, rid, 'no_action')
            if int(ref.triggered) == 1:
                t = int(ref.trigger_time)
                target = str(ref.observable_target)
                if target not in TARGETS:
                    raise RuntimeError(f'Invalid E8 target: {rid}: {target}')
                append(H, rid, 'missionshield_reference', t, target, reference=ref)
                append(H, rid, 'same_alarm_fixed_edge', t, 'edge_processing')
                append(H, rid, 'same_alarm_fixed_fusion', t, 'fusion')
                append(H, rid, 'same_alarm_fixed_command', t, 'command')
                rng = deterministic_rng(H, rid, 'random_node')
                append(H, rid, 'same_alarm_random_node', t, str(rng.choice(TARGETS)))
                # Retrospective timing control on exactly the same alarmed missions.
                eligible_times = valid.get(rid, [])
                if not eligible_times:
                    raise RuntimeError(f'No eligible windows for alarmed mission {rid} H{H}')
                rng_t = deterministic_rng(H, rid, 'random_time')
                append(H, rid, 'same_missions_random_eligible_time', int(rng_t.choice(eligible_times)), target)
            else:
                for policy in ('missionshield_reference', 'same_alarm_fixed_edge', 'same_alarm_fixed_fusion',
                               'same_alarm_fixed_command', 'same_alarm_random_node',
                               'same_missions_random_eligible_time'):
                    append(H, rid, policy)

        # Independent no-forecast allocation, same action count, no labels used to choose missions.
        candidates = sorted(rid for rid in info_map if valid.get(rid))
        if len(candidates) < K:
            raise RuntimeError(f'Insufficient eligible missions H{H}: {len(candidates)} < {K}')
        alloc_rng = deterministic_rng(H, 'independent_random_allocation')
        selected = set(map(str, alloc_rng.choice(candidates, size=K, replace=False)))
        for rid in sorted(info_map):
            if rid in selected:
                # Earliest observable eligible time; never use attack onset or future failure label.
                append(H, rid, 'no_forecast_random_mission_earliest_time', valid[rid][0], 'edge_processing')
            else:
                append(H, rid, 'no_forecast_random_mission_earliest_time')

    out = pd.DataFrame(rows)
    summary = out.groupby(['horizon', 'policy'], as_index=False).agg(
        missions=('run_id', 'nunique'), actions=('action_executed', 'sum'),
        baseline_failures=('baseline_failure', 'sum'), post_failures=('post_policy_failure', 'sum'),
        prevented=('failure_prevented', 'sum'), induced=('failure_induced', 'sum'),
        mean_cost=('incurred_cost', 'mean'), mean_effectiveness_gain=('mission_effectiveness_gain', 'mean'))
    summary['net_prevented'] = summary.prevented - summary.induced
    summary['failure_reduction_pct'] = 100 * summary.net_prevented / summary.baseline_failures
    ref_actions = summary[summary.policy == 'missionshield_reference'].set_index('horizon').actions.to_dict()
    summary['matched_action_budget'] = summary.apply(
        lambda r: int(r.actions) == int(ref_actions[int(r.horizon)]), axis=1)
    for H in HORIZONS:
        s = summary[(summary.horizon == H) & (summary.policy == 'missionshield_reference')].iloc[0]
        expected = {10:(99,42), 30:(105,31), 60:(117,36)}[H]
        if (int(s.actions), int(s.post_failures)) != expected:
            raise RuntimeError(f'E8 reference mismatch H{H}: {s.actions}, {s.post_failures}')
    out.to_csv(OUT / 'e17b_policy_results.csv', index=False)
    summary.to_csv(OUT / 'e17b_policy_summary.csv', index=False)
    print('\nE17-B COMPLETE. Output:', OUT)
    print(summary.to_string(index=False))
    print('\nCAVEATS: retrospective random-time is not an online policy; independent allocation uses a')
    print('random fixed-size mission set (not a real-time controller). No baseline-forecaster controls yet.')
    print('This experiment does not establish causality or external validity.')


if __name__ == '__main__':
    main()
