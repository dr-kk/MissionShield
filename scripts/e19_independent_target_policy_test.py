"""E19 independent fixed-clock target-selection test. No E8 access.
Reconstructs frozen E18-D Ridge(alpha=100) models from E18-C train rows.
Uses independent simulator seeds, with matched target counterfactuals.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import binomtest

ROOT = Path(r'D:\MissionShield')
SCRIPTS = ROOT / 'scripts'
LEARN = ROOT / 'results' / 'e18_learning'
OUT = ROOT / 'results' / 'e19_independent'
TARGETS = ('edge_processing', 'fusion', 'command')
TIMES = (60, 75)
SEED_BASE = 3_500_000
PER_FAMILY = 20
BOOTSTRAPS = 10000
META = {'split','run_id','target','action_cost','baseline_post_failure','post_failure','net_failure_benefit','effectiveness_gain'}

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def paired_stats(a, b, seed):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    n = a_only + b_only
    p = float(binomtest(min(a_only, b_only), n, 0.5).pvalue) if n else 1.0
    rng = np.random.default_rng(seed)
    dif = b - a  # positive = policy a has fewer failures
    draws = rng.integers(0, len(dif), size=(BOOTSTRAPS, len(dif)))
    lo, hi = np.quantile(dif[draws].mean(axis=1) * 100, [0.025, 0.975])
    return a_only, b_only, p, float(dif.mean() * 100), float(lo), float(hi)

def main():
    e18c = load('e18c_for_e19', SCRIPTS / 'e18c_intervention_learning_dataset.py')
    e0 = e18c.e0
    df = pd.read_csv(LEARN / 'e18c_intervention_learning_long.csv')
    proto = json.loads((LEARN / 'e18d_protocol.json').read_text(encoding='utf-8'))
    if proto.get('model') != 'target-specific StandardScaler + Ridge(alpha=100)':
        raise ValueError('E18-D protocol differs from frozen expected model')
    if proto.get('target') != 'post_failure':
        raise ValueError('Unexpected E18-D training target')
    features = list(proto['features'])
    if set(features) != set(df.columns) - META:
        raise ValueError('Feature schema differs from frozen E18-D protocol')
    train = df[df.split == 'train'].copy()
    if train.run_id.nunique() != 120:
        raise ValueError('Expected 120 training missions')
    if train.duplicated(['run_id','decision_time','target']).any():
        raise ValueError('Duplicate training rows')
    models = {}
    for target in TARGETS:
        sub = train[train.target == target]
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100.0))
        model.fit(sub[features], sub.post_failure.astype(float))
        models[target] = model
    # No validation data are read to fit or select a model.
    old_seeds = set(range(e18c.SEED_BASE, e18c.SEED_BASE + 100))
    old_seeds |= {e18c.SEED_BASE + fi*10000+k for fi in range(6) for k in range(30)}
    if any(SEED_BASE + fi*10000+k in old_seeds for fi in range(6) for k in range(PER_FAMILY)):
        raise ValueError('Overlapping development seeds')
    print('E19 | INDEPENDENT FIXED-CLOCK TARGET POLICY TEST')
    print(f'Frozen Ridge alpha=100; {PER_FAMILY} fresh missions/family; times={TIMES}')
    decisions, eligibility = [], []
    for fi, attack in enumerate(e0.ATTACK_FAMILIES):
        for k in range(PER_FAMILY):
            seed = SEED_BASE + fi*10000 + k
            rid = f'e19_{fi:02d}_{k:03d}'
            raw = pd.DataFrame(e0.simulate_run(rid, attack, seed))
            base = (raw[['time','mission_effectiveness','failure']]
                    .drop_duplicates('time').sort_values('time').reset_index(drop=True))
            for t in TIMES:
                eligible = not bool(base.loc[base.time < t, 'failure'].max())
                eligibility.append({'run_id':rid,'attack_family':attack,'decision_time':t,'eligible':eligible})
                if not eligible:
                    continue
                feats = e18c.features_at(raw, t)
                x = pd.DataFrame([feats])[features]
                risks = {target:float(np.clip(models[target].predict(x)[0],0,1)) for target in TARGETS}
                selected = min(TARGETS, key=lambda z:(risks[z], TARGETS.index(z)))
                bpost = base.loc[base.time >= t]
                b_fail = int(bpost.failure.max())
                b_mean = float(bpost.mission_effectiveness.mean())
                outcome = {}
                gain = {}
                for target in TARGETS:
                    cf = e18c.explicit_restore(attack, seed, t, target)
                    pre = cf.loc[cf.time < t]
                    original_pre = base.loc[base.time < t]
                    if not np.array_equal(pre.mission_effectiveness.to_numpy(), original_pre.mission_effectiveness.to_numpy()):
                        raise RuntimeError(f'Pre-action replay mismatch: {rid} {t} {target}')
                    after = cf.loc[cf.time >= t]
                    outcome[target] = int(after.failure.max())
                    gain[target] = float(after.mission_effectiveness.mean() - b_mean)
                decisions.append({
                    'run_id':rid, 'attack_family':attack, 'seed':seed, 'decision_time':t,
                    'baseline_failure':b_fail, 'selected_target':selected,
                    'learned_failure':outcome[selected],
                    'edge_failure':outcome['edge_processing'],
                    'fusion_failure':outcome['fusion'],
                    'command_failure':outcome['command'],
                    'oracle_failure':min(outcome.values()),
                    'learned_gain':gain[selected], 'edge_gain':gain['edge_processing'],
                    **{f'predicted_{target}_risk':risks[target] for target in TARGETS}
                })
        print(f'  Completed {attack}', flush=True)
    out = pd.DataFrame(decisions)
    elig = pd.DataFrame(eligibility)
    if out.empty: raise RuntimeError('No eligible missions')
    if out.duplicated(['run_id','decision_time']).any(): raise RuntimeError('Duplicate decisions')
    if set(out.run_id) & set(train.run_id): raise RuntimeError('Run-ID overlap')
    summary = []
    for t,g in out.groupby('decision_time'):
        for competitor, col in [('fixed_edge','edge_failure'),('fixed_fusion','fusion_failure'),
                                ('fixed_command','command_failure'),('no_action','baseline_failure')]:
            aonly, bonly, p, advantage, lo, hi = paired_stats(g.learned_failure, g[col], 2026+t)
            summary.append({'decision_time':int(t),'comparison':'learned_vs_'+competitor,
                'missions':len(g),'baseline_failures':int(g.baseline_failure.sum()),
                'learned_failures':int(g.learned_failure.sum()),
                'comparator_failures':int(g[col].sum()),
                'learned_only_failures':aonly,'comparator_only_failures':bonly,
                'exact_mcnemar_p':p,'learned_advantage_pp':advantage,
                'paired_bootstrap_95ci_low_pp':lo,'paired_bootstrap_95ci_high_pp':hi,
                'learned_mean_gain':float(g.learned_gain.mean()),
                'edge_mean_gain':float(g.edge_gain.mean()),
                'selected_edge':int((g.selected_target=='edge_processing').sum()),
                'selected_fusion':int((g.selected_target=='fusion').sum()),
                'selected_command':int((g.selected_target=='command').sum()),
                'retrospective_oracle_failures':int(g.oracle_failure.sum())})
    summary = pd.DataFrame(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'e19_independent_decisions.csv',index=False)
    elig.to_csv(OUT / 'e19_eligibility.csv',index=False)
    summary.to_csv(OUT / 'e19_independent_summary.csv',index=False)
    (OUT / 'e19_protocol.json').write_text(json.dumps({
        'experiment':'E19','seed_base':SEED_BASE,'per_family':PER_FAMILY,
        'decision_times':TIMES,'frozen_model':'E18-D target-specific Ridge alpha=100',
        'training_data':'E18-C train only','action':'sustained restore_service',
        'action_cost':0.25,'features':features,'bootstrap_draws':BOOTSTRAPS,
        'scope':'fixed-clock target selection, not end-to-end alarm policy',
        'no_action':'reference only, different action budget',
        'limitation':'simulated outcomes; no external validity or real-world causal effect established'
    },indent=2),encoding='utf-8')
    print('\nINDEPENDENT RESULTS (positive advantage means learned policy is better):')
    print(summary.to_string(index=False))
    print('\n[PASS] Fresh seeds, pre-action replay, matched target actions, fixed model.')
    print('NOTE: Same missions appear at both decision times; do not pool as independent.')
    print('NOTE: Original attention/alarm policy is NOT tested in E19; requires a separate matched alarm protocol.')
    print('Saved:',OUT)

if __name__ == '__main__':
    main()
