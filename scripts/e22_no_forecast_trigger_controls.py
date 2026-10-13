"""E22: exploratory no-forecast trigger controls on the frozen E20-D cohort.
Run: python D:\\MissionShield\\scripts\\e22_no_forecast_trigger_controls.py
No fitting or outcome-informed selection. Fixed-edge restore_service for all policies.
"""
from pathlib import Path
import importlib.util, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(r'D:\MissionShield')
S=ROOT/'scripts'; SRC=ROOT/'results'/'e20d_fresh'; OUT=ROOT/'results'/'e22_no_forecast'
OUT.mkdir(parents=True,exist_ok=True)
TARGET='edge_processing'; CLOCK=60; DETECTOR_THRESHOLD=0.5; RANDOM_LO=9; RANDOM_HI=129

def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,S/filename)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def p_exact(a,b):
    n=a+b
    return float(binomtest(min(a,b),n,0.5).pvalue) if n else 1.0

def main():
    e8=load('e8_e22','e8_independent_cohort_confirmation.py'); e0=e8.e0
    actions=pd.read_csv(SRC/'e20d_actions.csv'); protocol=json.loads((SRC/'e20d_protocol.json').read_text())
    assert protocol['target']==TARGET and protocol['action']=='restore_service'
    required={'run_id','horizon','model','baseline_failure','post_failure'}
    if not required.issubset(actions.columns): raise RuntimeError('E20-D schema mismatch')
    family_map={f'e20d_{fi:02d}_{j:03d}':(family,int(protocol['seed_base'])+fi*10000+j)
                for fi,family in enumerate(e0.ATTACK_FAMILIES)
                for j in range(int(protocol['missions_per_family']))}
    if set(actions.run_id.astype(str))!=set(family_map): raise RuntimeError('Cohort IDs mismatch')
    # Original telemetry, without looking at the future failure label.
    cohort=[]
    for rid,(family,seed) in family_map.items(): cohort.extend(e0.simulate_run(rid,family,seed))
    df=pd.DataFrame(cohort)
    required_cols={'run_id','time','node_id','detector_score','failure','mission_effectiveness','attack_onset'}
    if 'node_id' not in df.columns:
        node_col=next((x for x in ('node','node_name') if x in df.columns),None)
        if node_col is None: raise RuntimeError('Cannot resolve node column')
        df=df.rename(columns={node_col:'node_id'})
    if not required_cols.issubset(df.columns): raise RuntimeError(f'Missing simulator columns {required_cols-set(df.columns)}')
    result=[]; replay_cache={}
    for rid,(family,seed) in family_map.items():
        sub=df[df.run_id.astype(str)==rid].copy()
        base=sub[['time','failure','mission_effectiveness','attack_onset']].drop_duplicates('time').sort_values('time')
        if len(base)!=e0.STEPS_PER_RUN: raise RuntimeError(f'Timeline length mismatch {rid}')
        base_failure=int(base.failure.max())
        original=actions[actions.run_id.astype(str)==rid]
        if not (original.baseline_failure.astype(int)==base_failure).all(): raise RuntimeError(f'Baseline mismatch {rid}')
        # Deterministic random timing; seed derived solely from the mission seed.
        random_time=int(np.random.default_rng(seed+7_777_777).integers(RANDOM_LO,RANDOM_HI+1))
        # First rise above a PRE-SPECIFIED detector threshold, after history warmup.
        # Aggregate by max detector score across the six nodes (observable at time t).
        detector=sub.groupby('time').detector_score.max().sort_index()
        detector=detector[detector.index>=9]
        previous=detector.shift(1,fill_value=0.0)
        rises=detector.index[(detector>=DETECTOR_THRESHOLD)&(previous<DETECTOR_THRESHOLD)]
        detector_time=int(rises[0]) if len(rises) else None
        triggers={'No action':None,'Fixed clock t=60':CLOCK,'Random clock':random_time,
                  'Detector rise 0.5':detector_time,'Immediate t=0':0}
        for policy,t in triggers.items():
            if t is None:
                cf=base
            else:
                key=(rid,t)
                if key not in replay_cache:
                    onset,trial=e8.simulate_explicit_target(family,seed,t,TARGET)
                    if int(onset)!=int(base.attack_onset.iloc[0]): raise RuntimeError(f'Onset mismatch {rid}')
                    joined=base.merge(trial,on='time',suffixes=('_base','_cf'),validate='one_to_one')
                    pre=joined[joined.time<t]
                    if not np.allclose(pre.mission_effectiveness_base,pre.mission_effectiveness_cf,atol=1e-12,rtol=0):
                        raise RuntimeError(f'Pre-action invariance failed {rid}, t={t}')
                    replay_cache[key]=trial
                cf=replay_cache[key]
            pf=int(cf.failure.max())
            result.append({'run_id':rid,'attack_family':family,'policy':policy,'action_time':t,
                           'action':int(t is not None),'baseline_failure':base_failure,'post_failure':pf,
                           'prevented':int(base_failure==1 and pf==0),'induced':int(base_failure==0 and pf==1),
                           'mean_gain':float(cf.mission_effectiveness.mean()-base.mission_effectiveness.mean()),
                           'action_cost':0.25 if t is not None else 0.0})
    detail=pd.DataFrame(result)
    detail.to_csv(OUT/'e22_actions.csv',index=False)
    summary=detail.groupby('policy',as_index=False).agg(missions=('run_id','size'),actions=('action','sum'),
        baseline_failures=('baseline_failure','sum'),post_failures=('post_failure','sum'),
        prevented=('prevented','sum'),induced=('induced','sum'),mean_gain=('mean_gain','mean'),mean_cost=('action_cost','mean'))
    summary.to_csv(OUT/'e22_summary.csv',index=False)
    paired=[]
    for H in sorted(actions.horizon.unique()):
        ms=actions[(actions.horizon==H)&(actions.model=='MissionShield')].set_index('run_id')
        for policy,g in detail.groupby('policy'):
            comp=g.set_index('run_id').reindex(ms.index)
            if comp.post_failure.isna().any(): raise RuntimeError('Pairing failed')
            a=int(((ms.post_failure==1)&(comp.post_failure==0)).sum())
            b=int(((ms.post_failure==0)&(comp.post_failure==1)).sum())
            paired.append({'horizon':int(H),'comparison':'MissionShield vs '+policy,
                           'missions':len(ms),'missionshield_only_failures':a,'control_only_failures':b,
                           'difference_pp_control_minus_ms':100*(comp.post_failure.mean()-ms.post_failure.mean()),
                           'exact_mcnemar_p_unadjusted':p_exact(a,b),
                           'ms_actions':int(ms.alarm.sum()),'control_actions':int(comp.action.sum())})
    paired=pd.DataFrame(paired); paired.to_csv(OUT/'e22_paired.csv',index=False)
    (OUT/'e22_protocol.json').write_text(json.dumps({'source':'E20-D cohort','exploratory':True,
       'fixed_clock':CLOCK,'random_range_inclusive':[RANDOM_LO,RANDOM_HI],
       'random_seed_rule':'mission_seed + 7777777','detector_threshold':DETECTOR_THRESHOLD,
       'detector_rule':'first rising crossing of max six-node detector_score at t>=9',
       'target':TARGET,'action':'restore_service','one_action_per_mission':True,
       'note':'Not budget matched. Immediate t=0 is an idealized early-action upper-control; do not call it operationally feasible. Detector threshold not calibrated for matched FAR.'},indent=2))
    print('E22 | NO-FORECAST TRIGGER CONTROLS')
    print(summary.to_string(index=False))
    print('\nPAIRED (unadjusted; different action counts)\n',paired.to_string(index=False))
    print('\n[PASS] Frozen cohort, pre-action replay checks, no model training or outcome-based trigger tuning')
    print('CAUTION: Exploratory; alarm budgets differ; t=0 is idealized; no generalization claim')
    print('Saved:',OUT)

if __name__=='__main__': main()
