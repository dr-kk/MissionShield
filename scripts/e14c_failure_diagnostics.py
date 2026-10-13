"""E14-C: read-only service-degradation H60 diagnostics for frozen E12-E predictions.
Usage: python scripts/e14c_failure_diagnostics.py
Optional: --input-dir PATH --output-dir PATH --threshold 0.5
The threshold is diagnostic only, not an optimized decision rule.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = [101,202,303,404,505,606,707,808,909,1010]
NODES = ['importance_sensor','importance_tactical_link','importance_edge_processing',
         'importance_fusion','importance_command','importance_actuator']
REQUIRED = ['sample_id','run_id','time','attack_family','horizon','steps_to_failure',
            'label','probability','fusion_temporal_gate'] + NODES

def safe_ap(y,p):
    return float(average_precision_score(y,p)) if len(set(y)) == 2 else np.nan

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(set(y)) == 2 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default=r'D:\MissionShield\results\e12\e12e_predictions')
    ap.add_argument('--output-dir', default=r'D:\MissionShield\results\e12\e14c')
    ap.add_argument('--threshold',type=float,default=0.5)
    args=ap.parse_args()
    inp,out=Path(args.input_dir),Path(args.output_dir)
    if not 0 < args.threshold < 1: raise ValueError('Threshold must be between zero and one')
    out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for seed in SEEDS:
        path=inp/f'e12e_missionshield_service_degradation_H60_seed{seed}_predictions.csv'
        if not path.is_file(): raise FileNotFoundError(f'Missing frozen prediction file: {path}')
        d=pd.read_csv(path)
        missing=set(REQUIRED)-set(d.columns)
        if missing: raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
        if d.sample_id.duplicated().any(): raise ValueError(f'{path.name}: duplicate sample IDs')
        if not (d.horizon.eq(60).all() and d.attack_family.eq('service_degradation').all()):
            raise ValueError(f'{path.name}: wrong attack family or horizon')
        if not d.label.isin([0,1]).all() or not d.probability.between(0,1).all():
            raise ValueError(f'{path.name}: invalid labels or probabilities')
        if d[REQUIRED[6:]].drop(columns=['steps_to_failure'],errors='ignore').isna().any().any():
            raise ValueError(f'{path.name}: unexpected missing numeric diagnostic')
        d['seed']=seed
        frames.append(d)
    base=frames[0].set_index('sample_id')[['run_id','time','label','horizon','attack_family']].sort_index()
    for d in frames[1:]:
        other=d.set_index('sample_id')[base.columns].sort_index()
        if not base.equals(other): raise ValueError('Frozen sample metadata differs across seeds')
    all_d=pd.concat(frames,ignore_index=True)
    all_d['predicted_label']=(all_d.probability >= args.threshold).astype(int)
    all_d['outcome']=np.select(
        [(all_d.label.eq(1)&all_d.predicted_label.eq(1)),
         (all_d.label.eq(0)&all_d.predicted_label.eq(1)),
         (all_d.label.eq(0)&all_d.predicted_label.eq(0)),
         (all_d.label.eq(1)&all_d.predicted_label.eq(0))],
        ['TP','FP','TN','FN'],default='ERROR')
    all_d['time_phase']=pd.cut(all_d.time, bins=[-np.inf,29,59,np.inf],labels=['early_t_le29','middle_t30_59','late_t_ge60'])
    per_run=[]
    for (seed,run),g in all_d.groupby(['seed','run_id'],sort=True):
        per_run.append({'seed':seed,'run_id':run,'samples':len(g),'positives':int(g.label.sum()),
            'auprc':safe_ap(g.label,g.probability),'auroc':safe_auc(g.label,g.probability),
            'TP':int((g.outcome=='TP').sum()),'FP':int((g.outcome=='FP').sum()),
            'TN':int((g.outcome=='TN').sum()),'FN':int((g.outcome=='FN').sum()),
            'mean_positive_probability':g.loc[g.label.eq(1),'probability'].mean(),
            'mean_negative_probability':g.loc[g.label.eq(0),'probability'].mean()})
    pd.DataFrame(per_run).to_csv(out/'e14c_per_seed_run_metrics.csv',index=False)
    grouped=all_d.groupby(['seed','outcome'],observed=True)
    diagnostics=grouped[['probability','fusion_temporal_gate']+NODES].agg(['mean','std','count'])
    diagnostics.columns=['_'.join(c) for c in diagnostics.columns]
    diagnostics.reset_index().to_csv(out/'e14c_outcome_gate_node_statistics.csv',index=False)
    all_d.groupby(['seed','run_id','time_phase','outcome'],observed=True).agg(
        count=('sample_id','size'),mean_probability=('probability','mean'),
        mean_gate=('fusion_temporal_gate','mean')).reset_index().to_csv(
            out/'e14c_time_phase_errors.csv',index=False)
    # High-confidence false positives and low-confidence missed failures, per seed.
    errors=all_d.loc[all_d.outcome.isin(['FP','FN'])].copy()
    errors['error_severity']=np.where(errors.outcome.eq('FP'),errors.probability,1-errors.probability)
    errors.sort_values(['seed','error_severity'],ascending=[True,False]).groupby('seed',sort=True).head(20).to_csv(
        out/'e14c_top_errors_per_seed.csv',index=False)
    # Same sample across seeds: measures repeatability without treating seeds as independent missions.
    consensus=all_d.groupby(['sample_id','run_id','time','label'],as_index=False).agg(
        mean_probability=('probability','mean'),std_probability=('probability','std'),
        mean_gate=('fusion_temporal_gate','mean'),fraction_predicted_positive=('predicted_label','mean'))
    consensus.to_csv(out/'e14c_sample_seed_consensus.csv',index=False)
    # Descriptive metrics across all samples, for each seed.
    summary=[]
    for seed,g in all_d.groupby('seed'):
        summary.append({'seed':seed,'samples':len(g),'positives':int(g.label.sum()),
            'auprc':safe_ap(g.label,g.probability),'auroc':safe_auc(g.label,g.probability),
            **{name:int((g.outcome==name).sum()) for name in ['TP','FP','TN','FN']}})
    pd.DataFrame(summary).to_csv(out/'e14c_seed_summary.csv',index=False)
    print(f'E14-C PASS: {len(SEEDS)} seeds, {len(base)} identical samples/seed, {base.run_id.nunique()} missions')
    print(f'Diagnostic threshold: {args.threshold:.3f} (not optimized)')
    print('Output:',out)
    for p in sorted(out.glob('e14c_*.csv')): print(' ',p.name)
    print('Note: importance/gates are observational diagnostics, not causal explanations.')
    print('Note: only five held-out missions, two containing positive labels.')

if __name__=='__main__': main()
