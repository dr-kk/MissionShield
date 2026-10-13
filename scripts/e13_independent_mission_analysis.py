"""E13 frozen prediction analysis. No model training. Run from D:\\MissionShield."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(r'D:\MissionShield')
BASE = ROOT / 'results' / 'e12'
SEEDS = [101,202,303,404,505,606,707,808,909,1010]
FAMILIES = ['ddos','multi_stage','service_degradation']
MODELS = ['MissionShield','lstm','patch','graphwave']


def scores(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if len(np.unique(y))<2:
        return np.nan,np.nan
    return float(average_precision_score(y,p)),float(roc_auc_score(y,p))


def load_predictions(model,family,horizon,seed):
    if model=='MissionShield':
        f=BASE/'e12e_predictions'/f'e12e_missionshield_{family}_H{horizon}_seed{seed}_predictions.csv'
    else:
        f=BASE/'predictions'/f'e12_{model}_{family}_H{horizon}_seed{seed}_predictions.csv'
    if not f.exists():
        raise FileNotFoundError(f'Prediction file missing: {f}')
    d=pd.read_csv(f)
    needed={'sample_id','run_id','label','probability'}
    if not needed.issubset(d): raise ValueError(f'Missing columns {needed-set(d)} in {f}')
    if d.sample_id.duplicated().any(): raise ValueError(f'Duplicate sample_id in {f}')
    if d[['label','probability']].isna().any().any() or not d.probability.between(0,1).all():
        raise ValueError(f'Invalid labels/probabilities in {f}')
    d['sample_id']=d.sample_id.astype(str);d['run_id']=d.run_id.astype(str)
    return d[['sample_id','run_id','label','probability']].set_index('sample_id').sort_index()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=13013)
    args=ap.parse_args()
    if args.bootstrap<100: raise ValueError('Use at least 100 bootstrap replicates')
    out=BASE/'e13';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    overall=[]; per_run=[]; loo=[]; comparisons=[]; boot=[]
    for family in FAMILIES:
      for h in [10,30,60]:
        predictions={}; reference=None
        for model in MODELS:
          runs=[]
          for seed in SEEDS:
            d=load_predictions(model,family,h,seed)
            if reference is None: reference=d[['run_id','label']]
            if not d[['run_id','label']].equals(reference):
                raise ValueError(f'Sample/label/run alignment failure: {model}/{family}/H{h}/seed{seed}')
            runs.append(d.probability.rename(str(seed)))
          probs=pd.concat(runs,axis=1)
          if probs.isna().any().any(): raise ValueError('Missing prediction values')
          predictions[model]=probs
          for seed in SEEDS:
            p=probs[str(seed)].to_numpy();y=reference.label.to_numpy()
            a,u=scores(y,p)
            overall.append(dict(model=model,family=family,horizon=h,seed=seed,auprc=a,auroc=u,n_samples=len(y),n_runs=reference.run_id.nunique()))
            for rid in reference.run_id.unique():
              mask=reference.run_id.to_numpy()==rid
              ar,ur=scores(y[mask],p[mask])
              per_run.append(dict(model=model,family=family,horizon=h,seed=seed,run_id=rid,n_samples=int(mask.sum()),n_positive=int(y[mask].sum()),auprc=ar,auroc=ur,defined=(len(np.unique(y[mask]))==2)))
          meanp=probs.mean(axis=1).to_numpy(); y=reference.label.to_numpy(); rid=reference.run_id.to_numpy()
          for excluded in np.unique(rid):
            m=rid!=excluded
            ar,ur=scores(y[m],meanp[m])
            loo.append(dict(model=model,family=family,horizon=h,excluded_run=excluded,auprc=ar,auroc=ur,n_remaining=int(m.sum())))
        # Paired run-cluster bootstrap on seed-averaged probabilities. Same sampled run IDs for all models.
        unique=np.unique(reference.run_id.to_numpy())
        y=reference.label.to_numpy(); r=reference.run_id.to_numpy()
        means={m:predictions[m].mean(axis=1).to_numpy() for m in MODELS}
        orig={m:scores(y,means[m])[0] for m in MODELS}
        for b in range(args.bootstrap):
          sampled=rng.choice(unique,size=len(unique),replace=True)
          idx=np.concatenate([np.flatnonzero(r==rr) for rr in sampled])
          if len(np.unique(y[idx]))<2: continue
          vals={m:scores(y[idx],means[m][idx])[0] for m in MODELS}
          for m in MODELS:
            boot.append(dict(family=family,horizon=h,replicate=b,model=m,auprc=vals[m]))
          for other in MODELS[1:]:
            comparisons.append(dict(family=family,horizon=h,replicate=b,comparator=other,delta_auprc=vals['MissionShield']-vals[other]))
        print(f'E13 processed {family} H{h}: {len(reference)} samples, {len(unique)} independent test runs',flush=True)
    pd.DataFrame(overall).to_csv(out/'e13_seed_metrics.csv',index=False)
    pd.DataFrame(per_run).to_csv(out/'e13_per_run_metrics.csv',index=False)
    pd.DataFrame(loo).to_csv(out/'e13_leave_one_run_out.csv',index=False)
    bs=pd.DataFrame(boot);bs.to_csv(out/'e13_run_cluster_bootstrap.csv',index=False)
    cb=pd.DataFrame(comparisons)
    rows=[]
    for (f,h,c),d in cb.groupby(['family','horizon','comparator']):
      a=d.delta_auprc.to_numpy()
      rows.append(dict(family=f,horizon=h,comparator=c,valid_replicates=len(a),
                       mean_bootstrap_delta=float(np.mean(a)),ci_low=float(np.quantile(a,.025)),
                       ci_high=float(np.quantile(a,.975)),
                       proportion_delta_positive=float(np.mean(a>0)),
                       ci_excludes_zero=bool(np.quantile(a,.025)>0 or np.quantile(a,.975)<0)))
    pd.DataFrame(rows).to_csv(out/'e13_cluster_bootstrap_comparisons.csv',index=False)
    print('Saved E13 outputs to:',out)
    print('NOTE: only 5 held-out runs/condition; bootstrap CIs are exploratory, not independent external validation.')

if __name__=='__main__':main()
