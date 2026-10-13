"""E20-D: fresh-cohort frozen forecasting and fixed-edge intervention comparison.
No training, no threshold fitting, no action-budget matching by test outcomes.
"""
from pathlib import Path
import gc, json, hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import binomtest

ROOT=Path(r'D:\MissionShield'); S=ROOT/'scripts'; R=ROOT/'results'; C=ROOT/'checkpoints'
OUT=R/'e20d_fresh'; OUT.mkdir(parents=True,exist_ok=True)
import importlib.util

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,S/file)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

e8=load('e8_e20d','e8_independent_cohort_confirmation.py')
e2=load('e2_e20d','e2_frozen_baselines.py')
e1=e8.e1; e0=e8.e0
lstm=load('lstm_e20d','e1_lstm_prefailure.py')
trans=load('trans_e20d','e1_transformer_prefailure.py')
graph=load('graph_e20d','e1_graph_temporal_prefailure.py')
MODELS=['MissionShield','GraphTemporal','LSTM','Transformer']
HORIZONS=[10,30,60]
SEED_BASE=4_200_042
N_PER_FAMILY=30

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def cohort():
    rows=[]; seen=set()
    for fi,fam in enumerate(e0.ATTACK_FAMILIES):
        for j in range(N_PER_FAMILY):
            seed=SEED_BASE+fi*10000+j
            seen.add(seed)
            rows.extend(e0.simulate_run(f'e20d_{fi:02d}_{j:03d}',fam,seed))
        print('  Generated',fam,flush=True)
    old=pd.read_csv(e0.PROC_DIR/'missionshield_e0_v2_dataset.csv',usecols=['seed']).seed.dropna().astype(int)
    if seen.intersection(set(old)): raise RuntimeError('Seed collision with original cohort')
    return e0.add_future_labels(pd.DataFrame(rows))

def temp_apply(p,T):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return 1/(1+np.exp(-np.log(p/(1-p))/float(T)))

def predict_one(name,H,manifest,old_manifest,old_lookup,new_lookup,device):
    if name=='MissionShield':
        mean,std=e1.calculate_normalization(old_manifest,old_lookup)
        X,y,meta=e1.build_sequences_from_manifest(manifest,'test',new_lookup,mean,std)
        model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(device)
        ck=C/f'e1_missionshield_H{H}.pt'
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True),strict=True)
        loader=DataLoader(e1.MissionDataset(X,y),batch_size=256,shuffle=False,num_workers=0)
        _,p,_,_=e1.predict(model,loader,device,return_explanations=True)
    else:
        old_nodes=e2.build_node_lookup_cached if hasattr(e2,'build_node_lookup_cached') else None
        # Use E2's exact original training-only normalization.
        if name=='GraphTemporal':
            mean,std=e2.graph_normalization(old_manifest,old_lookup)
            X,y,meta=e2.build_graph_sequences(manifest,'test',new_lookup,mean,std)
            model=graph.GraphTemporalForecaster(adjacency=graph.build_adjacency(),graph_hidden=graph.GRAPH_HIDDEN,temporal_dim=graph.TEMPORAL_DIM,nhead=graph.NHEAD,temporal_layers=graph.TEMPORAL_LAYERS,ff_dim=graph.FF_DIM,dropout=graph.DROPOUT,max_len=max(64,e2.HISTORY))
            key='graphtemporal'; is_graph=True
        else:
            mean,std=e2.vector_normalization(old_manifest,old_lookup)
            X,y,meta=e2.build_vector_sequences(manifest,'test',new_lookup,mean,std)
            if name=='LSTM':
                model=lstm.LSTMForecaster(input_dim=36,hidden=lstm.HIDDEN,layers=lstm.LAYERS,dropout=lstm.DROPOUT)
                key='lstm'
            else:
                model=trans.TransformerForecaster(input_dim=36,d_model=trans.D_MODEL,nhead=trans.NHEAD,layers=trans.LAYERS,dropout=trans.DROPOUT,ff_dim=trans.FF_DIM,max_len=max(64,e2.HISTORY))
                key='transformer'
            is_graph=False
        model=model.to(device)
        ck=C/f'e2_frozen_{key}_H{H}.pt'
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True),strict=True)
        loader=DataLoader(e2.SequenceDataset(X,y),batch_size=256,shuffle=False,num_workers=0)
        _,p=e2.predict_neural(model,loader,device,graph_model=is_graph)
    if len(meta)!=len(p): raise RuntimeError('Prediction/manifest length mismatch')
    del X,y,loader,model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pd.DataFrame(meta),np.asarray(p,float),ck

def exact_p(a,b):
    n=int(a+b)
    return float(binomtest(min(int(a),int(b)),n,p=.5).pvalue) if n else 1.0

def main():
    print('E20-D | FROZEN FORECAST -> FIRST ALARM -> FIXED EDGE RESTORATION',flush=True)
    print('Fresh seeds; no training; no test-set threshold fitting; 30 missions/family',flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    op=pd.read_csv(R/'e3b_operational_summary.csv')
    assert set(MODELS).issubset(set(op.model))
    df=cohort()
    info=df[['run_id','seed','attack_family','attack_onset']].drop_duplicates('run_id').set_index('run_id')
    old=pd.read_csv(e1.DATA)
    old_ms=e1.build_telemetry_lookup(old); new_ms=e1.build_telemetry_lookup(df)
    old_e2=e2.build_node_lookup(old); new_e2=e2.build_node_lookup(df)
    predictions=[]; actions=[]; hashes={}
    for H in HORIZONS:
        man=e8.make_manifest(df,H)
        man['split']='test'
        frozen=pd.read_csv(R/f'e1_final_samples_H{H}.csv')
        frozen['run_id']=frozen.run_id.astype(str)
        if len(man)==0: raise RuntimeError('No eligible samples')
        expected=set(zip(man.run_id.astype(str),man.time.astype(int)))
        for name in MODELS:
            print('Infer',name,'H',H,flush=True)
            m,p,ck=predict_one(name,H,man,frozen,old_ms if name=='MissionShield' else old_e2,new_ms if name=='MissionShield' else new_e2,device)
            keys=list(zip(m.run_id.astype(str),m.time.astype(int)))
            if len(keys)!=len(expected) or set(keys)!=expected or len(set(keys))!=len(keys):
                raise RuntimeError(f'Coverage mismatch {name} H{H}')
            row=op[(op.model==name)&(op.horizon==H)]
            if len(row)!=1: raise RuntimeError('Missing unique operating point')
            T=float(row.iloc[0].temperature); threshold=float(row.iloc[0].threshold_calibrated)
            pc=temp_apply(p,T)
            m['model']=name; m['horizon']=H; m['probability']=p
            m['probability_calibrated']=pc; m['alarm_fixed_far']=(pc>=threshold).astype(int)
            predictions.append(m)
            hashes[f'{name}_H{H}']={'checkpoint':str(ck.relative_to(ROOT)),'sha256':digest(ck),'temperature':T,'threshold':threshold}
            first=(m[m.alarm_fixed_far==1].sort_values(['run_id','time']).drop_duplicates('run_id').set_index('run_id'))
            for rid,record in info.iterrows():
                t=int(first.loc[rid,'time']) if rid in first.index else None
                base=(df[df.run_id==rid][['time','mission_effectiveness','failure']].drop_duplicates('time').sort_values('time'))
                if t is not None:
                    onset,cf=e8.simulate_explicit_target(str(record.attack_family),int(record.seed),t,'edge_processing')
                    if int(onset)!=int(record.attack_onset): raise RuntimeError('Onset mismatch')
                    # Matched random streams must give identical pre-action outcomes.
                    joined=base.merge(cf,on='time',suffixes=('_base','_cf'),validate='one_to_one')
                    pre=joined[joined.time<t]
                    if not np.allclose(pre.mission_effectiveness_base,pre.mission_effectiveness_cf,atol=1e-12,rtol=0):
                        raise RuntimeError(f'Pre-action replay mismatch {rid}')
                else: cf=base
                bf=int(base.failure.max()); pf=int(cf.failure.max())
                actions.append({'horizon':H,'model':name,'run_id':rid,'attack_family_eval_only':str(record.attack_family),'alarm':int(t is not None),'alarm_time':t,'baseline_failure':bf,'post_failure':pf,'prevented':int(bf==1 and pf==0),'induced':int(bf==0 and pf==1),'mean_gain':float(cf.mission_effectiveness.mean()-base.mission_effectiveness.mean()),'action_cost':.25 if t is not None else 0.})
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'e20d_predictions.csv',index=False)
    detail=pd.DataFrame(actions); detail.to_csv(OUT/'e20d_actions.csv',index=False)
    summary=detail.groupby(['horizon','model'],as_index=False).agg(missions=('run_id','size'),alarms=('alarm','sum'),baseline_failures=('baseline_failure','sum'),post_failures=('post_failure','sum'),prevented=('prevented','sum'),induced=('induced','sum'),mean_gain=('mean_gain','mean'),mean_cost=('action_cost','mean'))
    summary.to_csv(OUT/'e20d_summary.csv',index=False)
    paired=[]
    for H in HORIZONS:
        sub=detail[detail.horizon==H]
        ref=sub[sub.model=='MissionShield'].set_index('run_id')
        for other in MODELS[1:]:
            comp=sub[sub.model==other].set_index('run_id').reindex(ref.index)
            a=int(((ref.post_failure==1)&(comp.post_failure==0)).sum())
            b=int(((ref.post_failure==0)&(comp.post_failure==1)).sum())
            paired.append({'horizon':H,'comparison':'MissionShield vs '+other,'missions':len(ref),'missionshield_only_failures':a,'comparator_only_failures':b,'exact_mcnemar_p_unadjusted':exact_p(a,b),'failure_advantage_pp':100*(comp.post_failure.mean()-ref.post_failure.mean()),'action_budget_difference':int(ref.alarm.sum()-comp.alarm.sum())})
    paired=pd.DataFrame(paired); paired.to_csv(OUT/'e20d_paired.csv',index=False)
    (OUT/'e20d_protocol.json').write_text(json.dumps({'seed_base':SEED_BASE,'missions_per_family':N_PER_FAMILY,'horizons':HORIZONS,'models':MODELS,'action':'restore_service','target':'edge_processing','policy':'first frozen-threshold alarm; at most one intervention per mission','checkpoints':hashes,'note':'Natural policy comparison; different alarm counts mean action budgets are NOT matched. E8 cohort is not reused.'},indent=2))
    print('\nNATURAL ALARM POLICY RESULTS\n',summary.to_string(index=False))
    print('\nPAIRED COMPARISONS (unadjusted)\n',paired.to_string(index=False))
    print('\n[PASS] Fresh cohort, frozen checkpoints and thresholds, complete prediction coverage, pre-action replay invariance')
    print('CAUTION: Alarm budgets differ; do not claim action-budget-matched superiority.')
    print('CAUTION: Same missions reused across horizons; no cross-horizon independence.')
    print('Saved:',OUT)

if __name__=='__main__': main()
