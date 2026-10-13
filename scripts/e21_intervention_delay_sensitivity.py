"""E21 exploratory intervention-delay sensitivity using frozen E20-D first alarms.
No model training, alarm refitting, or outcome-based action selection.
Run: python D:\\MissionShield\\scripts\\e21_intervention_delay_sensitivity.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import importlib.util

ROOT = Path(r'D:\MissionShield')
S = ROOT / 'scripts'
SOURCE = ROOT / 'results' / 'e20d_fresh'
OUT = ROOT / 'results' / 'e21_delay_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
DELAYS = (0, 5, 10, 20)
TARGET = 'edge_processing'

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, S / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    e8 = load('e8_e21', 'e8_independent_cohort_confirmation.py')
    e0 = e8.e0
    actions = pd.read_csv(SOURCE / 'e20d_actions.csv')
    protocol = json.loads((SOURCE / 'e20d_protocol.json').read_text())
    assert protocol['action'] == 'restore_service' and protocol['target'] == TARGET
    required = {'run_id', 'horizon', 'model', 'alarm', 'alarm_time', 'baseline_failure', 'post_failure'}
    if not required.issubset(actions.columns):
        raise RuntimeError('E20-D action schema mismatch: ' + str(required - set(actions.columns)))
    if actions.duplicated(['run_id', 'horizon', 'model']).any():
        raise RuntimeError('Duplicate E20-D action rows')
    seed_base = int(protocol['seed_base'])
    n_per_family = int(protocol['missions_per_family'])
    family_map = {f'e20d_{fi:02d}_{j:03d}': (family, seed_base + fi * 10000 + j)
                  for fi, family in enumerate(e0.ATTACK_FAMILIES)
                  for j in range(n_per_family)}
    if set(actions.run_id.astype(str)) != set(family_map):
        raise RuntimeError('E20-D run IDs do not match the frozen cohort protocol')

    print('E21 | FROZEN E20-D FIRST ALARMS + DELAYED RESTORATION', flush=True)
    print('Delays:', DELAYS, '| Missions:', len(family_map), flush=True)
    # Reproduce the baseline cohort exactly once per mission; do not re-infer models.
    baselines = {}
    for rid, (family, seed) in family_map.items():
        raw = pd.DataFrame(e0.simulate_run(rid, family, seed))
        base = raw[['time', 'mission_effectiveness', 'failure', 'attack_onset']].drop_duplicates('time').sort_values('time')
        if base.time.duplicated().any() or len(base) != 150:
            raise RuntimeError(f'Unexpected baseline timeline: {rid}')
        baselines[rid] = base
    print('Recreated baseline timelines for', len(baselines), 'missions', flush=True)

    cache = {}
    results = []
    for r in actions.itertuples(index=False):
        rid = str(r.run_id)
        family, seed = family_map[rid]
        base = baselines[rid]
        bf = int(base.failure.max())
        if bf != int(r.baseline_failure):
            raise RuntimeError(f'Baseline failure mismatch: {rid}')
        has_alarm = int(r.alarm) == 1
        if has_alarm and pd.isna(r.alarm_time):
            raise RuntimeError(f'Missing alarm time: {rid}')
        t_alarm = int(r.alarm_time) if has_alarm else None
        failure_times = base.loc[base.failure.astype(bool), 'time']
        first_failure = int(failure_times.min()) if len(failure_times) else None
        for delay in DELAYS:
            t_action = t_alarm + delay if has_alarm else None
            # An action scheduled after the simulated timeline is not executed.
            executed = t_action is not None and t_action <= int(base.time.max())
            if executed:
                key = (rid, t_action)
                if key not in cache:
                    onset, cf = e8.simulate_explicit_target(family, seed, t_action, TARGET)
                    if int(onset) != int(base.attack_onset.iloc[0]):
                        raise RuntimeError(f'Attack onset mismatch: {rid}')
                    joined = base.merge(cf, on='time', suffixes=('_base', '_cf'), validate='one_to_one')
                    if len(joined) != len(base):
                        raise RuntimeError(f'Replay length mismatch: {rid}')
                    pre = joined[joined.time < t_action]
                    if not np.allclose(pre.mission_effectiveness_base,
                                       pre.mission_effectiveness_cf, atol=1e-12, rtol=0):
                        raise RuntimeError(f'Pre-action replay mismatch: {rid}, t={t_action}')
                    cache[key] = cf
                cf = cache[key]
            else:
                cf = base
            pf = int(cf.failure.max())
            if delay == 0 and pf != int(r.post_failure):
                raise RuntimeError(f'E20-D delay-zero outcome mismatch: {rid}, H={r.horizon}, {r.model}')
            too_late = bool(has_alarm and first_failure is not None and
                            (not executed or t_action >= first_failure))
            results.append({
                'horizon': int(r.horizon), 'model': str(r.model), 'run_id': rid,
                'delay': delay, 'alarm': int(has_alarm), 'alarm_time': t_alarm,
                'scheduled_action_time': t_action, 'action_executed': int(executed),
                'first_baseline_failure_time': first_failure,
                'action_at_or_after_first_failure': int(too_late),
                'baseline_failure': bf, 'post_failure': pf,
                'prevented': int(bf == 1 and pf == 0),
                'induced': int(bf == 0 and pf == 1),
                'mean_baseline_effectiveness': float(base.mission_effectiveness.mean()),
                'mean_post_effectiveness': float(cf.mission_effectiveness.mean()),
                'mean_gain': float(cf.mission_effectiveness.mean() - base.mission_effectiveness.mean()),
                'action_cost': .25 if executed else 0.0,
            })
        # Print only at completed model-horizon boundaries.
        if len(results) % (len(DELAYS) * len(family_map)) == 0:
            print('Completed', r.model, 'H', r.horizon, flush=True)

    detail = pd.DataFrame(results)
    detail.to_csv(OUT / 'e21_delay_actions.csv', index=False)
    summary = detail.groupby(['horizon', 'model', 'delay'], as_index=False).agg(
        missions=('run_id', 'size'), alarms=('alarm', 'sum'),
        executed=('action_executed', 'sum'),
        actions_at_or_after_failure=('action_at_or_after_first_failure', 'sum'),
        baseline_failures=('baseline_failure', 'sum'),
        post_failures=('post_failure', 'sum'), prevented=('prevented', 'sum'),
        induced=('induced', 'sum'), mean_gain=('mean_gain', 'mean'),
        mean_post_effectiveness=('mean_post_effectiveness', 'mean'),
        mean_cost=('action_cost', 'mean'))
    summary.to_csv(OUT / 'e21_delay_summary.csv', index=False)
    comparisons = []
    for (H, model), group in detail.groupby(['horizon', 'model']):
        ref = group[group.delay == 0].set_index('run_id')
        for delay in DELAYS[1:]:
            comp = group[group.delay == delay].set_index('run_id').reindex(ref.index)
            a = int(((comp.post_failure == 1) & (ref.post_failure == 0)).sum())
            b = int(((comp.post_failure == 0) & (ref.post_failure == 1)).sum())
            n = a + b
            comparisons.append({'horizon': H, 'model': model, 'delay': delay,
                                'delayed_only_failures': a, 'immediate_only_failures': b,
                                'extra_failures_vs_immediate': int(comp.post_failure.sum()-ref.post_failure.sum()),
                                'exact_mcnemar_p_unadjusted': float(binomtest(min(a,b), n, .5).pvalue) if n else 1.0})
    pd.DataFrame(comparisons).to_csv(OUT / 'e21_delay_paired.csv', index=False)
    (OUT / 'e21_protocol.json').write_text(json.dumps({
        'source': str(SOURCE), 'delays': DELAYS, 'target': TARGET,
        'action': 'restore_service', 'policy': 'frozen first alarm + fixed delay',
        'independent_cohort': False, 'exploratory': True,
        'failure_definition': 'any failure over full mission timeline',
        'too_late_definition': 'action scheduled at/after first baseline failure, or outside timeline',
        'note': 'Replays are counterfactual simulator outputs; no online alarm recomputation after intervention.'
    }, indent=2))
    print('\nE21 RESULTS\n', summary.to_string(index=False))
    print('\n[PASS] Baseline invariance and delay-zero agreement with E20-D')
    print('CAUTION: Exploratory repeated evaluation of E20-D cohort; no independent confirmation.')
    print('Saved:', OUT)

if __name__ == '__main__':
    main()
