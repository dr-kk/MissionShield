"""E17-D read-only structural and mission-level intervention audit.
Run: python scripts/e17d_restoration_mechanism_audit.py
Does not modify simulator, checkpoints, or E8/E17 results.
"""
from pathlib import Path
import ast
import pandas as pd
import numpy as np

ROOT = Path(r'D:\MissionShield')
OUT = ROOT / 'results' / 'e17_controls'
OUT.mkdir(parents=True, exist_ok=True)
SCRIPTS = ROOT / 'scripts'
E0 = SCRIPTS / 'e0_missionshield_setup_v2.py'
E5 = SCRIPTS / 'e5b_counterfactual_intervention_emulator.py'
DETAIL = OUT / 'e17b_policy_results.csv'
COHORT = ROOT / 'results' / 'e8_independent_cohort.csv'

for path in (E0, E5, DETAIL, COHORT):
    if not path.exists():
        raise FileNotFoundError(path)

for path in (E0, E5):
    ast.parse(path.read_text(encoding='utf-8'))

source = E5.read_text(encoding='utf-8')
tree = ast.parse(source)
fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'update_damage_cf'), None)
if fn is None:
    raise RuntimeError('Missing update_damage_cf: inspect simulator before proceeding')
body = ast.get_source_segment(source, fn) or ''
print('E17-D | READ-ONLY MECHANISM AUDIT')
print('update_damage_cf contains previous[node]*0.45:', 'previous[node]*0.45' in body.replace(' ', ''))
print('NOTE: source inspection alone does not establish empirical causal contribution.')

cols = ['horizon','run_id','attack_family','policy','baseline_failure','post_policy_failure','action_executed','action_time','restored_node','incurred_cost']
d = pd.read_csv(DETAIL, usecols=cols)
if d.duplicated(['horizon','run_id','policy']).any():
    raise RuntimeError('Duplicate mission-policy records')
required = {'missionshield_reference','same_alarm_fixed_edge','no_action'}
if not required.issubset(set(d.policy)):
    raise RuntimeError(f'Missing policies: {required - set(d.policy)}')

# Derive pre-action failures from the original baseline timeline, never from post-action trajectories.
t = pd.read_csv(COHORT, usecols=['run_id','time','failure']).drop_duplicates(['run_id','time'])
if t.duplicated(['run_id','time']).any():
    raise RuntimeError('Duplicate baseline timestamps')
first_fail = t.loc[t.failure == 1].groupby('run_id').time.min()
all_fail = t.groupby('run_id').failure.max()
records = []
for r in d.itertuples(index=False):
    baseline = int(all_fail.loc[r.run_id])
    if baseline != int(r.baseline_failure):
        raise RuntimeError(f'Baseline mismatch: {r.run_id} H{r.horizon} {r.policy}')
    first = first_fail.get(r.run_id, np.nan)
    action = int(r.action_executed)
    action_time = float(r.action_time) if action else np.nan
    prior = int(action and pd.notna(first) and first < action_time)
    at_or_prior = int(action and pd.notna(first) and first <= action_time)
    records.append({'horizon':int(r.horizon),'run_id':r.run_id,'attack_family':r.attack_family,
                    'policy':r.policy,'action_executed':action,'action_time':action_time,
                    'restored_node':r.restored_node,'baseline_failure':baseline,
                    'post_policy_failure':int(r.post_policy_failure),'first_baseline_failure_time':first,
                    'baseline_failure_before_action':prior,
                    'baseline_failure_at_or_before_action':at_or_prior,
                    'prevented':int(baseline == 1 and int(r.post_policy_failure) == 0),
                    'induced':int(baseline == 0 and int(r.post_policy_failure) == 1)})
a = pd.DataFrame(records)

# Timing check: a policy should not be credited with preventing a failure that occurred before action.
invalid = a[(a.prevented == 1) & (a.baseline_failure_before_action == 1)]
print('Prevented cases with baseline failure before action:', len(invalid))
if len(invalid):
    print('CAUTION: Baseline failures before action are not automatically invalid counterfactuals; investigate per-mission timelines.')

summary = (a.groupby(['horizon','policy'],as_index=False)
    .agg(missions=('run_id','nunique'),actions=('action_executed','sum'),
         baseline_failures=('baseline_failure','sum'),post_failures=('post_policy_failure','sum'),
         prevented=('prevented','sum'),induced=('induced','sum'),
         actions_after_first_baseline_failure=('baseline_failure_before_action','sum'),
         actions_at_or_after_first_baseline_failure=('baseline_failure_at_or_before_action','sum')))
targets = (a[a.action_executed == 1].groupby(['horizon','policy','restored_node'],as_index=False)
           .agg(actions=('run_id','size'),prevented=('prevented','sum'),induced=('induced','sum')))

# Same alarm means identical action-time and action-selection across the two target rules.
ref = a[a.policy == 'missionshield_reference'].set_index(['horizon','run_id'])
edge = a[a.policy == 'same_alarm_fixed_edge'].set_index(['horizon','run_id'])
if not ref.index.equals(edge.index):
    ref, edge = ref.align(edge, join='inner', axis=0)
    if len(ref) != 540:
        raise RuntimeError('Reference/fixed-edge pairing incomplete')
if not np.array_equal(ref.action_executed.to_numpy(), edge.action_executed.to_numpy()):
    raise RuntimeError('Action allocation differs for fixed-edge comparison')
if not np.allclose(ref.action_time.to_numpy(dtype=float), edge.action_time.to_numpy(dtype=float), equal_nan=True):
    raise RuntimeError('Action timing differs for fixed-edge comparison')

summary.to_csv(OUT / 'e17d_mechanism_audit_summary.csv', index=False)
targets.to_csv(OUT / 'e17d_target_distribution.csv', index=False)
a.to_csv(OUT / 'e17d_timing_audit.csv', index=False)
print('\nSUMMARY\n', summary.to_string(index=False))
print('\nTARGETS (MissionShield and fixed edge)\n', targets[targets.policy.isin(['missionshield_reference','same_alarm_fixed_edge'])].to_string(index=False))
print('\n[PASS] Mission IDs, baseline labels and fixed-edge action timing checked.')
print('Outputs:', OUT)
