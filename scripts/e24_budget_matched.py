"""E24-A: outcome-blind budget-matched fixed-clock comparison on frozen E20-D.

Run: python D:\\MissionShield\\scripts\\e24_budget_matched.py
Requires existing scripts/e8_independent_cohort_confirmation.py and
results/e20d_fresh/{e20d_actions.csv,e20d_protocol.json}.
No model fitting; independent random assignment before outcome evaluation.
"""
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(r'D:\MissionShield')
SRC=ROOT/'results'/'e20d_fresh'
OUT=ROOT/'results'/'e24_budget_matched'
CLOCK=60
REPS=1000
RANDOM_SEED=20261013
TARGET='edge_processing'

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def exact_mcnemar(b,c):
    return float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    e8=load(ROOT/'scripts'/'e8_independent_cohort_confirmation.py','e8_e24')
    e0=e8.e0
    protocol=json.loads((SRC/'e20d_protocol.json').read_text())
    assert protocol['target']==TARGET and protocol['action']=='restore_service'
    actions=pd.read_csv(SRC/'e20d_actions.csv')
    assert len(actions)==2160, f'Unexpected E20-D rows: {len(actions)}'
    assert actions.groupby(['model','horizon']).run_id.nunique().eq(180).all()
    family_map={f'e20d_{fi:02d}_{j:03d}':(family,int(protocol['seed_base'])+fi*10000+j)
                for fi,family in enumerate(e0.ATTACK_FAMILIES)
                for j in range(int(protocol['missions_per_family']))}
    ids=sorted(family_map)
    assert len(ids)==180 and set(ids)==set(actions.run_id)
    # Replay one fixed-clock intervention per mission, irrespective of forecast.
    fixed={}
    for rid in ids:
        family,seed=family_map[rid]
        onset,trial=e8.simulate_explicit_target(family,seed,CLOCK,TARGET)
        assert len(trial)==e0.STEPS_PER_RUN, (rid,len(trial))
        fixed[rid]=int(trial.failure.max())
    baseline=actions[(actions.model=='MissionShield')&(actions.horizon==10)].set_index('run_id').baseline_failure.astype(int)
    assert set(baseline.index)==set(ids)
    fixed_arr=np.array([fixed[rid] for rid in ids],dtype=int)
    base_arr=baseline.reindex(ids).to_numpy()
    assert int(base_arr.sum())==44, 'Cohort baseline changed'
    assert int(fixed_arr.sum())==33, 'Fixed clock E22 outcome mismatch; investigate before reporting'
    rng=np.random.default_rng(RANDOM_SEED)
    rows=[]; paired=[]
    for h in (10,30,60):
        for model in ('MissionShield','GraphTemporal','LSTM','Transformer'):
            ms=actions[(actions.horizon==h)&(actions.model==model)].set_index('run_id').reindex(ids)
            k=int(ms.alarm.sum())
            obs=ms.post_failure.astype(int).to_numpy()
            assert len(obs)==180 and 0<=k<=180
            # Sample exactly k mission IDs without replacement, blinded to labels.
            samples=np.empty((REPS,180),dtype=bool)
            for r in range(REPS):
                samples[r]=False
                samples[r,rng.choice(180,size=k,replace=False)]=True
            controls=np.where(samples,fixed_arr[None,:],base_arr[None,:])
            post=controls.sum(axis=1)
            b=((obs[None,:]==1)&(controls==0)).sum(axis=1)
            c=((obs[None,:]==0)&(controls==1)).sum(axis=1)
            rows.append({'horizon':h,'model':model,'missions':180,'budget_actions':k,
                         'model_post_failures':int(obs.sum()),
                         'clock_mean_post_failures':float(post.mean()),
                         'clock_p025_post_failures':float(np.quantile(post,.025)),
                         'clock_p975_post_failures':float(np.quantile(post,.975)),
                         'clock_prob_fewer_failures':float(np.mean(post<obs.sum())),
                         'clock_prob_equal_failures':float(np.mean(post==obs.sum())),
                         'clock_prob_more_failures':float(np.mean(post>obs.sum())),
                         'mean_clock_minus_model_failures':float(np.mean(post-obs.sum()))})
            paired.append({'horizon':h,'model':model,'mean_model_only_failures_b':float(b.mean()),
                           'mean_clock_only_failures_c':float(c.mean()),
                           'fraction_exact_mcnemar_p_lt_005':float(np.mean([exact_mcnemar(int(x),int(y))<.05 for x,y in zip(b,c)]))})
    pd.DataFrame(rows).to_csv(OUT/'e24_budget_matched_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'e24_budget_matched_paired.csv',index=False)
    (OUT/'e24_budget_matched_protocol.json').write_text(json.dumps({
        'cohort':'E20-D frozen 180 missions','fixed_clock':CLOCK,'target':TARGET,
        'action':'restore_service','randomizations':REPS,'seed':RANDOM_SEED,
        'assignment':'Uniformly choose exactly k distinct mission IDs, blinded to labels, for each repetition',
        'comparison':'Model natural-alarm outcomes vs randomized fixed-clock subsets at same action count',
        'inference':'Exploratory randomization distribution; 2.5%-97.5% are policy-assignment quantiles, not model uncertainty CIs',
        'restoration_strength':'Not varied in this script; requires simulator source to change operator correctly.'
    },indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print('Saved:',OUT)
    print('NOTE: E24-B restoration-strength sensitivity requires the simulator source; not silently simulated.')

if __name__=='__main__':
    main()
