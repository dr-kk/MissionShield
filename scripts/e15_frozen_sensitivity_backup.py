"""E15 frozen MissionShield inference reproduction and input sensitivity.
Run from D:\\MissionShield: python scripts/e15_frozen_sensitivity.py
Outputs only results/e15; never retrains or overwrites E1-E14 artifacts.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
RESULTS = ROOT / 'results'
SEEDS = [101,202,303,404,505,606,707,808,909,1010]

def load_module():
    path = SCRIPTS / 'e1_missionshield_forecaster.py'
    spec = importlib.util.spec_from_file_location('e15_ms', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def infer(model, X, device, batch_size=256):
    model.eval()
    probs=[]
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(np.ascontiguousarray(X[start:start+batch_size]),dtype=torch.float32,device=device)
            logits, _, _ = model(x)
            probs.append(torch.sigmoid(logits).reshape(-1).detach().cpu().numpy())
    return np.concatenate(probs)

def metrics(y, p):
    return {'auprc':float(average_precision_score(y,p)),
            'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'mean_probability':float(np.mean(p))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    ap.add_argument('--tolerance',type=float,default=2e-5)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    if len(set(args.seeds))!=len(args.seeds): raise ValueError('Duplicate seeds')
    ms=load_module()
    held='service_degradation'; H=60
    df=pd.read_csv(ms.DATA)
    lookup=ms.build_telemetry_lookup(df)
    mask=ms.build_directed_attention_mask()
    base=pd.read_csv(RESULTS / f'e1_final_samples_H{H}.csv')
    tr=base[(base['split']=='train')&(base['attack_family']!=held)].copy()
    va=base[(base['split']=='validation')&(base['attack_family']!=held)].copy()
    te=base[(base['split']=='test')&(base['attack_family']==held)].copy()
    manifest=pd.concat([tr,va,te],ignore_index=True)
    if any(a.empty for a in (tr,va,te)): raise RuntimeError('Empty split')
    if (set(tr.run_id)&set(va.run_id)) or (set(tr.run_id)&set(te.run_id)) or (set(va.run_id)&set(te.run_id)):
        raise RuntimeError('Run leakage')
    mean,std=ms.calculate_normalization(manifest,lookup)
    Xt,yt,meta=ms.build_sequences_from_manifest(manifest,'test',lookup,mean,std)
    y=np.asarray(yt).astype(int)
    ids=[m['sample_id'] for m in meta]
    if len(set(ids))!=len(ids): raise RuntimeError('Duplicate sample IDs')
    nodes=list(ms.NODES)
    if Xt.ndim!=4 or Xt.shape[2]!=len(nodes): raise RuntimeError(f'Unexpected X shape {Xt.shape}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models=[]; refs=[]; verification=[]
    print(f'E15 device={device} samples={len(y)} nodes={nodes}',flush=True)
    # Strictly verify ALL requested seeds before ANY intervention.
    for seed in args.seeds:
        name=f'e12e_missionshield_{held}_H{H}_seed{seed}'
        ckpt=RESULTS/'e12'/'e12e_checkpoints'/f'{name}.pt'
        pred=RESULTS/'e12'/'e12e_predictions'/f'{name}_predictions.csv'
        if not ckpt.is_file() or not pred.is_file(): raise FileNotFoundError(f'Missing checkpoint or predictions for {name}')
        ref=pd.read_csv(pred)
        if ref['sample_id'].tolist()!=ids: raise RuntimeError(f'Sample order mismatch: seed {seed}')
        if not np.array_equal(ref['label'].to_numpy(dtype=int),y): raise RuntimeError(f'Label mismatch: seed {seed}')
        model=ms.MissionShieldForecaster(mask).to(device)
        model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
        model.eval()
        p=infer(model,Xt,device)
        err=np.abs(p-ref['probability'].to_numpy(dtype=float))
        maxerr=float(err.max())
        verification.append({'seed':seed,'max_abs_probability_error':maxerr,'mean_abs_probability_error':float(err.mean())})
        print(f'E15-A seed={seed} max_probability_error={maxerr:.8g}',flush=True)
        if maxerr>args.tolerance: raise RuntimeError(f'REPRODUCTION FAILED seed={seed} max_error={maxerr} tolerance={args.tolerance}; NO interventions run')
        models.append(model);refs.append(p)
    print('E15-A PASS: all requested checkpoints reproduce saved probabilities',flush=True)
    out=RESULTS/'e15';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(verification).to_csv(out/'e15_reproduction_verification.csv',index=False)
    if args.verify_only: return
    rows=[];sample_rows=[]
    for seed,model,original in zip(args.seeds,models,refs):
        baseline=metrics(y,original)
        conditions=[('original',None)]
        conditions += [('node_zero_'+node,('node',j)) for j,node in enumerate(nodes)]
        conditions += [('history_zero_first_half',('time',slice(0,Xt.shape[1]//2))),
                       ('history_zero_second_half',('time',slice(Xt.shape[1]//2,None))),
                       ('history_reverse',('reverse',None))]
        for name,action in conditions:
            if action is None: p=original
            else:
                x=Xt.copy()
                kind,arg=action
                if kind=='node': x[:,:,arg,:]=0.0  # training-normalized mean replacement
                elif kind=='time': x[:,arg,:,:]=0.0
                elif kind=='reverse': x=x[:,::-1,:,:].copy()
                p=infer(model,x,device)
            met=metrics(y,p)
            rows.append({'seed':seed,'condition':name,**met,
                         'delta_auprc':met['auprc']-baseline['auprc'],
                         'mean_abs_probability_shift':float(np.mean(np.abs(p-original))),
                         'positive_mean_abs_shift':float(np.mean(np.abs(p[y==1]-original[y==1]))),
                         'negative_mean_abs_shift':float(np.mean(np.abs(p[y==0]-original[y==0])))})
            for i,m in enumerate(meta):
                sample_rows.append({'seed':seed,'condition':name,'sample_id':ids[i],
                                    'run_id':m['run_id'],'time':m['time'],'label':int(y[i]),
                                    'original_probability':float(original[i]),'perturbed_probability':float(p[i]),
                                    'delta_probability':float(p[i]-original[i])})
        print(f'E15 sensitivity completed seed={seed}',flush=True)
    pd.DataFrame(rows).to_csv(out/'e15_sensitivity_metrics.csv',index=False)
    pd.DataFrame(sample_rows).to_csv(out/'e15_sample_probability_shifts.csv',index=False)
    summary=pd.DataFrame(rows).groupby('condition',as_index=False).agg(
        mean_delta_auprc=('delta_auprc','mean'),sd_delta_auprc=('delta_auprc','std'),
        mean_abs_probability_shift=('mean_abs_probability_shift','mean'))
    summary.to_csv(out/'e15_sensitivity_summary.csv',index=False)
    print(f'E15 COMPLETE: {out}',flush=True)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__': main()

