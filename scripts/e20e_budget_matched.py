"""E20-E: retrospective budget-controlled replay of E20-D, no outcome-based selection.
Budgets are shared per 180-mission batch and per horizon. This is an offline,
synchronous-batch allocation analysis, NOT an online sequential deployment test.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(r'D:\MissionShield')
SRC=ROOT/'results'/'e20d_fresh'
OUT=ROOT/'results'/'e20e_budget_matched'
MODELS=['MissionShield','GraphTemporal','LSTM','Transformer']
HORIZONS=[10,30,60]
BUDGETS=[40,60,80]  # pre-specified; each is below the minimum E20-D alarm count (86)

def tie_key(run_id):
    return hashlib.sha256(('e20e-fixed-tie-v1|'+str(run_id)).encode()).hexdigest()

def exact_p(a,b):
    return float(binomtest(min(a,b),a+b,0.5).pvalue) if a+b else 1.0

def holm(pvals):
    n=len(pvals); order=np.argsort(pvals,kind='stable'); adjusted=np.empty(n); running=0.
    for j,i in enumerate(order):
        running=max(running,min(1.,float(pvals[i])*(n-j)))
        adjusted[i]=running
    return adjusted

def main():
    for fn in ['e20d_actions.csv','e20d_protocol.json']:
        if not (SRC/fn).exists(): raise FileNotFoundError(SRC/fn)
    protocol=json.loads((SRC/'e20d_protocol.json').read_text())
    assert protocol['seed_base']==4200042 and protocol['missions_per_family']==30
    df=pd.read_csv(SRC/'e20d_actions.csv')
    required={'horizon','model','run_id','alarm','alarm_time','baseline_failure','post_failure','mean_gain','action_cost'}
    assert required.issubset(df.columns), sorted(required-set(df.columns))
    assert len(df)==3*4*180, f'Unexpected rows: {len(df)}'
    assert not df.duplicated(['horizon','model','run_id']).any()
    assert set(df.model)==set(MODELS) and set(df.horizon)==set(HORIZONS)
    assert df.groupby(['horizon','model']).run_id.nunique().eq(180).all()
    assert df.groupby(['horizon','model']).alarm.sum().min()>=max(BUDGETS)
    assert ((df.alarm==1)==df.alarm_time.notna()).all()
    # Outcomes are looked up only AFTER eligibility is chosen from alarm/time/run ID.
    all_details=[]; summaries=[]; comparisons=[]
    for H in HORIZONS:
        hdf=df[df.horizon==H]
        base=hdf[hdf.model==MODELS[0]].set_index('run_id').baseline_failure.sort_index()
        for name in MODELS[1:]:
            other=hdf[hdf.model==name].set_index('run_id').baseline_failure.sort_index()
            assert other.equals(base), f'Baseline mismatch H{H} {name}'
        for budget in BUDGETS:
            selected_by_model={}
            for name in MODELS:
                sub=hdf[hdf.model==name].copy()
                eligible=sub[sub.alarm==1][['run_id','alarm_time']].copy()
                eligible['tie']=eligible.run_id.map(tie_key)
                # Chronological first-alarm priority across synchronous independent missions;
                # no use of baseline or intervention outcomes in selection.
                selected=set(eligible.sort_values(['alarm_time','tie','run_id'],kind='stable').head(budget).run_id)
                assert len(selected)==budget
                selected_by_model[name]=selected
                sub['budget_action']=sub.run_id.isin(selected).astype(int)
                sub['budget_post_failure']=np.where(sub.budget_action==1,sub.post_failure,sub.baseline_failure).astype(int)
                sub['budget_gain']=np.where(sub.budget_action==1,sub.mean_gain,0.)
                sub['budget_cost']=sub.budget_action*.25
                sub['budget']=budget
                assert int(sub.budget_action.sum())==budget
                all_details.append(sub)
                summaries.append({'horizon':H,'budget':budget,'model':name,'missions':len(sub),
                    'available_alarms':int(sub.alarm.sum()),'actions':budget,
                    'baseline_failures':int(sub.baseline_failure.sum()),
                    'post_failures':int(sub.budget_post_failure.sum()),
                    'prevented':int(((sub.baseline_failure==1)&(sub.budget_post_failure==0)).sum()),
                    'induced':int(((sub.baseline_failure==0)&(sub.budget_post_failure==1)).sum()),
                    'mean_gain':float(sub.budget_gain.mean()),'mean_cost':float(sub.budget_cost.mean()),
                    'median_selected_alarm_time':float(sub.loc[sub.budget_action==1,'alarm_time'].median())})
            group=pd.concat(all_details[-len(MODELS):])
            ms=group[group.model=='MissionShield'].set_index('run_id').sort_index()
            for other in MODELS[1:]:
                co=group[group.model==other].set_index('run_id').reindex(ms.index)
                a=int(((ms.budget_post_failure==1)&(co.budget_post_failure==0)).sum())
                b=int(((ms.budget_post_failure==0)&(co.budget_post_failure==1)).sum())
                comparisons.append({'horizon':H,'budget':budget,'comparison':'MissionShield vs '+other,
                    'ms_only_failures':a,'comparator_only_failures':b,
                    'failure_advantage_pp':100*(co.budget_post_failure.mean()-ms.budget_post_failure.mean()),
                    'p_unadjusted':exact_p(a,b)})
    detail=pd.concat(all_details,ignore_index=True)
    summary=pd.DataFrame(summaries)
    paired=pd.DataFrame(comparisons)
    # Family-wise correction across 3 model comparisons at each horizon/budget.
    paired['p_holm_within_horizon_budget']=np.nan
    for _,idx in paired.groupby(['horizon','budget']).groups.items():
        paired.loc[idx,'p_holm_within_horizon_budget']=holm(paired.loc[idx,'p_unadjusted'].to_numpy())
    OUT.mkdir(parents=True,exist_ok=True)
    detail.to_csv(OUT/'e20e_actions.csv',index=False)
    summary.to_csv(OUT/'e20e_summary.csv',index=False)
    paired.to_csv(OUT/'e20e_paired.csv',index=False)
    (OUT/'e20e_protocol.json').write_text(json.dumps({'source':'e20d_fresh/e20d_actions.csv',
        'design':'retrospective synchronous-batch allocation using frozen first alarms',
        'budgets':BUDGETS,'tie_break':'sha256(e20e-fixed-tie-v1|run_id)',
        'selection':'earliest alarm time, deterministic hash tie-break; outcome-blind',
        'counterfactual':'E20-D fixed-edge replay for selected alarms; baseline otherwise',
        'limitations':['Same 180 missions reused for horizons and budgets',
          'Not a deployable streaming/global scheduling simulation',
          'Alarm-based prioritization can favor earlier, not necessarily better, warnings',
          'Holm adjustment covers only 3 comparators within each horizon/budget',
          'Exploratory follow-up to already inspected E20-D outcomes']},indent=2))
    print('E20-E | RETROSPECTIVE MATCHED ACTION BUDGETS')
    print(summary.to_string(index=False))
    print('\nPAIRED COMPARISONS\n',paired.to_string(index=False))
    print('\n[PASS] Outcome-blind selection, identical action counts, paired baseline consistency')
    print('CAUTION: Exploratory post-E20-D analysis; do not claim independent confirmation.')
    print('Saved:',OUT)

if __name__=='__main__': main()
