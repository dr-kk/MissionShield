"""E18-B: Paired statistical analysis of E18-A development results.
No resimulation, model fitting, or E8 confirmation data access.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(r'D:\MissionShield')
OUT = ROOT / 'results' / 'e18_development'
INPUT = OUT / 'e18a_target_sensitivity_detail.csv'
TARGETS = ('edge_processing', 'fusion', 'command')
OFFSETS = (0, 15, 30)
RNG_SEED = 180042
BOOTSTRAPS = 10000


def ci_paired(diff, seed):
    """Percentile paired bootstrap CI for mean(B-A), at mission level."""
    a = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        means[i] = a[rng.integers(0, len(a), len(a))].mean()
    return np.percentile(means, [2.5, 97.5])


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    result = np.empty(len(p))
    prev = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * p[idx]))
        result[idx] = prev
    return result


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing E18-A detail file: {INPUT}')
    df = pd.read_csv(INPUT)
    required = {'run_id', 'attack_family', 'offset', 'target', 'baseline_failure',
                'post_failure', 'effectiveness_gain', 'prior_failure'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.duplicated(['run_id', 'offset', 'target']).any():
        raise ValueError('Duplicate mission-offset-target rows')
    if set(df.offset.unique()) != set(OFFSETS) or set(df.target.unique()) != set(TARGETS):
        raise ValueError('Unexpected offsets or restoration targets')
    if df.run_id.nunique() != 72 or len(df) != 648:
        raise ValueError(f'Expected 72 missions and 648 rows; got {df.run_id.nunique()} / {len(df)}')
    if not df.groupby(['run_id', 'offset']).target.nunique().eq(3).all():
        raise ValueError('Incomplete paired target outcomes')
    if not df.groupby('run_id').offset.nunique().eq(3).all():
        raise ValueError('Incomplete offset outcomes')
    if not df.groupby('run_id').baseline_failure.nunique().eq(1).all():
        raise ValueError('Baseline failure inconsistent within mission')
    if int(df.prior_failure.sum()) != 0:
        print('CAUTION: prior failures exist; inspect timing before interpreting prevention.')

    records = []
    for offset in OFFSETS:
        subset = df.loc[df.offset == offset]
        failures = subset.pivot(index='run_id', columns='target', values='post_failure')
        gains = subset.pivot(index='run_id', columns='target', values='effectiveness_gain')
        for j, (a, b) in enumerate(itertools.combinations(TARGETS, 2)):
            # delta > 0: A has fewer failures than B (A is better)
            fa = failures[a].to_numpy(dtype=int)
            fb = failures[b].to_numpy(dtype=int)
            a_only = int(((fa == 1) & (fb == 0)).sum())
            b_only = int(((fa == 0) & (fb == 1)).sum())
            n_disc = a_only + b_only
            p = float(binomtest(min(a_only, b_only), n_disc, .5).pvalue) if n_disc else 1.0
            d = fb - fa
            low, high = ci_paired(d, RNG_SEED + offset * 10 + j)
            gain_diff = gains[a].to_numpy() - gains[b].to_numpy()
            gain_low, gain_high = ci_paired(gain_diff, RNG_SEED + 1000 + offset * 10 + j)
            records.append(dict(offset=offset, target_a=a, target_b=b,
                                missions=len(fa), a_failures=int(fa.sum()), b_failures=int(fb.sum()),
                                a_only_failures=a_only, b_only_failures=b_only,
                                failure_rate_advantage_a_pp=100 * d.mean(),
                                advantage_a_ci_low_pp=100 * low, advantage_a_ci_high_pp=100 * high,
                                exact_mcnemar_p=p, mean_effectiveness_advantage_a=float(gain_diff.mean()),
                                effectiveness_ci_low=float(gain_low), effectiveness_ci_high=float(gain_high)))
    result = pd.DataFrame(records)
    result['holm_p_across_9'] = holm(result.exact_mcnemar_p.to_numpy())
    result.to_csv(OUT / 'e18b_paired_target_statistics.csv', index=False)

    family = df.groupby(['attack_family', 'offset', 'target'], as_index=False).agg(
        missions=('run_id', 'nunique'), baseline_failures=('baseline_failure', 'sum'),
        post_failures=('post_failure', 'sum'), mean_effectiveness_gain=('effectiveness_gain', 'mean'))
    family.to_csv(OUT / 'e18b_attack_target_diagnostics.csv', index=False)

    # Retrospective best outcome is an oracle bound, NOT a usable online policy.
    wide = df.pivot_table(index=['run_id', 'offset'], columns='target', values='post_failure', aggfunc='first')
    oracle = wide.min(axis=1).groupby(level='offset').sum().astype(int)
    print('E18-B | PAIRED TARGET ANALYSIS | DEVELOPMENT ONLY')
    print('72 unique missions; comparisons paired within each offset.')
    print('\nPAIRED TESTS (positive advantage = target A has fewer failures):')
    print(result[['offset','target_a','target_b','a_failures','b_failures',
                  'a_only_failures','b_only_failures','failure_rate_advantage_a_pp',
                  'exact_mcnemar_p','holm_p_across_9']].to_string(index=False))
    print('\nRETROSPECTIVE ORACLE LOWER BOUND (not a deployable policy):')
    print(oracle.to_string())
    print('\nNOTE: same 72 missions recur at each offset; 9 tests corrected together.')
    print('NOTE: bootstrap CIs are descriptive for this simulated development cohort.')
    print('NOTE: this analysis cannot establish causal policy learning or external validity.')
    print('\n[PASS] Pairing and completeness verified. Saved to:', OUT)

if __name__ == '__main__':
    main()
