"""E17-C: paired mission-level inference for MissionShield E17-B.

Run: python scripts/e17c_paired_statistics.py
Reads E17-B mission-level CSV; writes new E17-C outputs without modifying E8/E17-B.
Inference: exact two-sided McNemar; paired mission bootstrap percentile CIs.
Randomized E17-B policies are single draws, not replicated randomization studies.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / 'results' / 'e17_controls' / 'e17b_policy_results.csv'
OUTDIR = ROOT / 'results' / 'e17_controls'
REF = 'missionshield_reference'
POLICIES = ['no_action', 'same_alarm_fixed_edge', 'same_alarm_fixed_fusion',
            'same_alarm_fixed_command', 'same_alarm_random_node',
            'same_missions_random_eligible_time',
            'no_forecast_random_mission_earliest_time']
B = 20000
SEED = 170043


def exact_mcnemar(a, b):
    # a,b are 0/1 post-policy failure indicators for the same missions.
    ref_only = int(((a == 1) & (b == 0)).sum())
    other_only = int(((a == 0) & (b == 1)).sum())
    discordant = ref_only + other_only
    p = float(binomtest(ref_only, discordant, 0.5).pvalue) if discordant else 1.0
    return ref_only, other_only, discordant, p


def boot_ci(d, rng):
    # d = per-mission (other failure - reference failure), so positive favors reference.
    n = len(d)
    draws = np.empty(B)
    for k in range(B):
        draws[k] = d[rng.integers(0, n, size=n)].mean()
    return np.quantile(draws, [.025, .975])


def main():
    df = pd.read_csv(INFILE)
    cols = ['horizon', 'run_id', 'policy', 'baseline_failure',
            'post_policy_failure', 'action_executed', 'incurred_cost']
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df.duplicated(['horizon', 'run_id', 'policy']).any():
        raise ValueError('Duplicate mission/horizon/policy records')
    if len(df) != 4320:
        raise ValueError(f'Expected 4320 rows, found {len(df)}')
    rng = np.random.default_rng(SEED)
    rows = []
    for h in [10, 30, 60]:
        sub = df[df.horizon == h]
        wide = sub.pivot(index='run_id', columns='policy', values='post_policy_failure')
        actions = sub.pivot(index='run_id', columns='policy', values='action_executed')
        costs = sub.pivot(index='run_id', columns='policy', values='incurred_cost')
        baselines = sub.pivot(index='run_id', columns='policy', values='baseline_failure')
        if len(wide) != 180 or wide.isna().any().any():
            raise ValueError(f'H{h}: incomplete paired missions')
        if not (baselines.nunique(axis=1) == 1).all():
            raise ValueError(f'H{h}: baseline differs between policies')
        if not np.isin(wide.to_numpy(), [0, 1]).all():
            raise ValueError(f'H{h}: invalid binary outcomes')
        a = wide[REF].to_numpy(dtype=int)
        for pol in POLICIES:
            b = wide[pol].to_numpy(dtype=int)
            n10, n01, nd, p = exact_mcnemar(a, b)
            d = b - a
            low, high = boot_ci(d, rng)
            rows.append(dict(horizon=h, reference=REF, comparator=pol,
                             missions=len(a), ref_actions=int(actions[REF].sum()),
                             comparator_actions=int(actions[pol].sum()),
                             ref_failures=int(a.sum()), comparator_failures=int(b.sum()),
                             ref_only_failures=n10, comparator_only_failures=n01,
                             discordant_missions=nd, mcnemar_exact_p_two_sided=p,
                             failure_rate_advantage_ref_pp=100*d.mean(),
                             advantage_95ci_low_pp=100*low,
                             advantage_95ci_high_pp=100*high,
                             mean_cost_ref=float(costs[REF].mean()),
                             mean_cost_comparator=float(costs[pol].mean())))
    result = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dest = OUTDIR / 'e17c_paired_policy_statistics.csv'
    result.to_csv(dest, index=False)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 230)
    print(result[['horizon','comparator','ref_failures','comparator_failures',
                  'ref_only_failures','comparator_only_failures',
                  'mcnemar_exact_p_two_sided','failure_rate_advantage_ref_pp',
                  'advantage_95ci_low_pp','advantage_95ci_high_pp']].to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('\nSaved:', dest)
    print('CAUTION: unadjusted p-values; exploratory multiple comparisons.')
    print('CAUTION: bootstrap CIs sample missions, not simulator parameter uncertainty.')
    print('CAUTION: random-policy results reflect one randomization draw each.')


if __name__ == '__main__':
    main()
