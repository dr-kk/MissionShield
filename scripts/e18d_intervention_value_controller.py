"""E18-D: leakage-safe intervention value estimation, development validation only.
Input: results/e18_learning/e18c_intervention_learning_long.csv
No simulator reruns, no use of E8 cohort, no tuning on validation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import binomtest

ROOT = Path(r'D:\MissionShield')
DIR = ROOT / 'results' / 'e18_learning'
INPUT = DIR / 'e18c_intervention_learning_long.csv'
TARGETS = ['edge_processing', 'fusion', 'command']
TIMES = [60, 75]
META = {'split','run_id','target','action_cost','baseline_post_failure','post_failure','net_failure_benefit','effectiveness_gain'}


def exact_mcnemar(a, b):
    a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n = a_only + b_only
    return a_only, b_only, float(binomtest(min(a_only,b_only),n,0.5).pvalue) if n else 1.0


def main():
    df = pd.read_csv(INPUT)
    required = META | {'decision_time'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.duplicated(['split','run_id','decision_time','target']).any():
        raise ValueError('Duplicate mission/time/target rows')
    if set(df.target.unique()) != set(TARGETS): raise ValueError('Unexpected targets')
    if set(df.split.unique()) != {'train','validation'}: raise ValueError('Unexpected splits')
    if set(df.decision_time.unique()) != set(TIMES): raise ValueError('Unexpected decision times')
    tr = df[df.split == 'train'].copy()
    va = df[df.split == 'validation'].copy()
    if set(tr.run_id) & set(va.run_id): raise ValueError('Mission leakage across splits')
    for part in [tr,va]:
        sizes = part.groupby(['run_id','decision_time']).target.nunique()
        if not sizes.eq(3).all(): raise ValueError('Incomplete matched targets')
        for col in ['baseline_post_failure','post_failure']:
            if not part[col].isin([0,1]).all(): raise ValueError(f'Invalid binary label: {col}')
        for _,g in part.groupby(['run_id','decision_time']):
            if g.baseline_post_failure.nunique() != 1: raise ValueError('Baseline mismatch')
            if g.action_cost.nunique() != 1: raise ValueError('Cost mismatch')
            if not np.allclose(g.net_failure_benefit, g.baseline_post_failure - g.post_failure):
                raise ValueError('Inconsistent benefit labels')

    features = [c for c in df.columns if c not in META]
    if len(features) != 112:  # 108 telemetry + 3 observed mission features + decision time
        raise ValueError(f'Unexpected feature count: {len(features)}')
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in features):
        raise ValueError('Non-numeric input features')
    if not np.isfinite(df[features].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite features')
    # Features must be identical for all candidate targets at each decision.
    for part in [tr,va]:
        for _,g in part.groupby(['run_id','decision_time']):
            if not np.allclose(g[features].to_numpy(),g[features].iloc[0].to_numpy(dtype=float),atol=1e-10,rtol=1e-10):
                raise ValueError('Candidate-specific feature leakage')

    # Fixed, prespecified regularization. No validation-driven hyperparameter search.
    # Predict probability-like post-failure score per target; regression can exceed [0,1],
    # so clip predictions. Equal action costs -> select minimum predicted failure risk.
    models = {}
    for target in TARGETS:
        sub = tr[tr.target == target]
        model = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100.0))
        model.fit(sub[features],sub.post_failure.astype(float))
        models[target] = model

    decision_rows = []
    for (run_id,t), g in va.groupby(['run_id','decision_time'],sort=True):
        row = g.iloc[0]
        x = row[features].to_frame().T
        risk = {target: float(np.clip(models[target].predict(x)[0],0,1)) for target in TARGETS}
        # Deterministic tie-breaking prefers the strongest established baseline (edge).
        selected = min(TARGETS,key=lambda target: (risk[target],TARGETS.index(target)))
        actual = {r.target: int(r.post_failure) for r in g.itertuples()}
        baseline = int(row.baseline_post_failure)
        decision_rows.append({
            'run_id':run_id,'decision_time':int(t),'baseline_failure':baseline,
            'selected_target':selected,'learned_failure':actual[selected],
            'edge_failure':actual['edge_processing'],
            'fusion_failure':actual['fusion'],'command_failure':actual['command'],
            'oracle_failure':min(actual.values()),
            'predicted_edge_risk':risk['edge_processing'],
            'predicted_fusion_risk':risk['fusion'],
            'predicted_command_risk':risk['command'],
            'selected_effectiveness_gain':float(g.set_index('target').loc[selected,'effectiveness_gain']),
            'edge_effectiveness_gain':float(g.set_index('target').loc['edge_processing','effectiveness_gain']),
        })
    decisions = pd.DataFrame(decision_rows)
    summary = []
    for t,g in decisions.groupby('decision_time'):
        base = int(g.baseline_failure.sum())
        summary.append({'decision_time':int(t),'missions':len(g),'baseline_failures':base,
            'learned_failures':int(g.learned_failure.sum()),'edge_failures':int(g.edge_failure.sum()),
            'fusion_failures':int(g.fusion_failure.sum()),'command_failures':int(g.command_failure.sum()),
            'retrospective_oracle_failures':int(g.oracle_failure.sum()),
            'learned_minus_edge_failures':int(g.learned_failure.sum()-g.edge_failure.sum()),
            'learned_mean_effectiveness_gain':float(g.selected_effectiveness_gain.mean()),
            'edge_mean_effectiveness_gain':float(g.edge_effectiveness_gain.mean()),
            'selected_edge':int((g.selected_target=='edge_processing').sum()),
            'selected_fusion':int((g.selected_target=='fusion').sum()),
            'selected_command':int((g.selected_target=='command').sum()),
            'learned_only_failures':exact_mcnemar(g.learned_failure,g.edge_failure)[0],
            'edge_only_failures':exact_mcnemar(g.learned_failure,g.edge_failure)[1],
            'mcnemar_p_vs_edge_unadjusted':exact_mcnemar(g.learned_failure,g.edge_failure)[2]})
    out = pd.DataFrame(summary)
    DIR.mkdir(parents=True,exist_ok=True)
    decisions.to_csv(DIR/'e18d_validation_decisions.csv',index=False)
    out.to_csv(DIR/'e18d_validation_summary.csv',index=False)
    protocol = {'experiment':'E18-D','model':'target-specific StandardScaler + Ridge(alpha=100)',
        'training_split':'train only','validation_split':'validation only','target':'post_failure',
        'decision':'minimum predicted post-failure risk, tie-break edge/fusion/command',
        'features':features,'action_cost':'identical across targets',
        'note':'Development validation, not an independent confirmation test; retrospective oracle is not deployable.'}
    (DIR/'e18d_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('E18-D | INTERVENTION-VALUE CONTROLLER | DEVELOPMENT VALIDATION')
    print(f'Train missions={tr.run_id.nunique()}, validation missions={va.run_id.nunique()}, features={len(features)}')
    print('\nVALIDATION SUMMARY (lower failures better):')
    print(out.to_string(index=False))
    print('\n[PASS] Disjoint splits, matched counterfactuals, feature invariance, no future labels in X.')
    print('CAUTION: validation is reused across decision times; do not pool rows as independent.')
    print('CAUTION: no tuning performed on validation; model is not proven causally optimal.')
    print('Saved to:',DIR)

if __name__ == '__main__': main()
