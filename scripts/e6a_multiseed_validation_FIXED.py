from pathlib import Path
import importlib.util, random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path(r"D:\MissionShield")
SCRIPTS=ROOT/"scripts"; RESULTS=ROOT/"results"; CHECKPOINTS=ROOT/"checkpoints"
RESULTS.mkdir(exist_ok=True); CHECKPOINTS.mkdir(exist_ok=True)

E1_FILE=SCRIPTS/"e1_missionshield_forecaster.py"
E2_FILE=SCRIPTS/"e2_frozen_baselines.py"
HORIZONS=[10,30,60]
# Ten predeclared replicate seeds. Same seed is paired across neural models.
SEEDS=[101,202,303,404,505,606,707,808,909,1010]

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

e1=loadmod("e1",E1_FILE)
e2=loadmod("e2",E2_FILE)
lstm_mod=e2.load_module("e6_lstm","e1_lstm_prefailure.py")
tr_mod=e2.load_module("e6_transformer","e1_transformer_prefailure.py")
graph_mod=e2.load_module("e6_graph","e1_graph_temporal_prefailure.py")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def calc_metrics(y,p):
    return dict(
        auroc=float(roc_auc_score(y,p)),
        auprc=float(average_precision_score(y,p)),
        ece=float(e1.expected_calibration_error(y,p)),
        brier=float(brier_score_loss(y,p)),
    )

def loaders(ds_train,ds_val,ds_test,seed,device):
    g=torch.Generator(); g.manual_seed(seed)
    kw=dict(batch_size=256,num_workers=0,pin_memory=(device.type=="cuda"))
    return (
        DataLoader(ds_train,shuffle=True,generator=g,**kw),
        DataLoader(ds_val,shuffle=False,**kw),
        DataLoader(ds_test,shuffle=False,**kw),
    )

def train_missionshield(H,seed,manifest,lookup,device):
    mean,std=e1.calculate_normalization(manifest,lookup)
    Xtr,ytr,_=e1.build_sequences_from_manifest(manifest,"train",lookup,mean,std)
    Xv,yv,_=e1.build_sequences_from_manifest(manifest,"validation",lookup,mean,std)
    Xt,yt,meta=e1.build_sequences_from_manifest(manifest,"test",lookup,mean,std)
    tr=e1.MissionDataset(Xtr,ytr); va=e1.MissionDataset(Xv,yv); te=e1.MissionDataset(Xt,yt)
    tl,vl,el=loaders(tr,va,te,seed,device)

    seed_all(seed)
    model=e1.MissionShieldForecaster(directed_mask=e1.build_directed_attention_mask()).to(device)
    pos=float(ytr.sum()); neg=float(len(ytr)-pos)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/max(pos,1.),device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=e1.LR,weight_decay=e1.WEIGHT_DECAY)
    ck=CHECKPOINTS/f"e6a_missionshield_H{H}_seed{seed}.pt"
    best=-np.inf; bestep=0; bad=0
    for ep in range(1,e1.EPOCHS+1):
        model.train()
        for xb,yb in tl:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            z,_,_=model(xb); loss=crit(z,yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        vy,vp=e1.predict(model,vl,device)
        ap=average_precision_score(vy,vp)
        if ap>best+1e-4:
            best=ap; bestep=ep; bad=0; torch.save(model.state_dict(),ck)
        else:
            bad+=1
            if bad>=e1.PATIENCE: break
    model.load_state_dict(torch.load(ck,map_location=device,weights_only=True))
    ty,tp=e1.predict(model,el,device)
    m=calc_metrics(ty,tp)
    pred=pd.DataFrame(meta); pred["label"]=ty.astype(int); pred["probability"]=tp
    pred["seed"]=seed
    pred.to_csv(RESULTS/f"e6a_missionshield_H{H}_seed{seed}_predictions.csv",index=False)
    del model,Xtr,Xv,Xt,tl,vl,el,tr,va,te
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"model":"MissionShield","horizon":H,"seed":seed,"best_epoch":bestep,
            "best_validation_auprc":best,**m}

def train_baseline(name,H,seed,model,Xtr,ytr,Xv,yv,Xt,yt,meta,device,graph=False):
    tr=e2.SequenceDataset(Xtr,ytr); va=e2.SequenceDataset(Xv,yv); te=e2.SequenceDataset(Xt,yt)
    tl,vl,el=loaders(tr,va,te,seed,device)
    seed_all(seed); model=model.to(device)
    pos=float(ytr.sum()); neg=float(len(ytr)-pos)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/max(pos,1.),device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=e2.LR,weight_decay=e2.WEIGHT_DECAY)
    ck=CHECKPOINTS/f"e6a_{name.lower()}_H{H}_seed{seed}.pt"
    best=-np.inf; bestep=0; bad=0
    for ep in range(1,e2.EPOCHS+1):
        model.train()
        for xb,yb in tl:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            o=model(xb); z=o[0] if graph else o
            loss=crit(z,yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        vy,vp=e2.predict_neural(model,vl,device,graph_model=graph)
        ap=average_precision_score(vy,vp)
        if ap>best+1e-4:
            best=ap; bestep=ep; bad=0; torch.save(model.state_dict(),ck)
        else:
            bad+=1
            if bad>=e2.PATIENCE: break
    model.load_state_dict(torch.load(ck,map_location=device,weights_only=True))
    ty,tp=e2.predict_neural(model,el,device,graph_model=graph)
    m=calc_metrics(ty,tp)
    pred=pd.DataFrame(meta); pred["label"]=ty.astype(int); pred["probability"]=tp; pred["seed"]=seed
    pred.to_csv(RESULTS/f"e6a_{name.lower()}_H{H}_seed{seed}_predictions.csv",index=False)
    del model,tl,vl,el,tr,va,te
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"model":name,"horizon":H,"seed":seed,"best_epoch":bestep,
            "best_validation_auprc":best,**m}

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*100)
    print("MissionShield E6-A | 10-SEED FROZEN-PROTOCOL VALIDATION")
    print("Same manifests/hyperparameters | validation-AUPRC checkpointing | no test tuning")
    print("="*100)
    print("Device:",device,("| "+torch.cuda.get_device_name(0) if device.type=="cuda" else ""))
    print("Seeds:",SEEDS)

    df=pd.read_csv(e1.DATA)
    lookup=e1.build_telemetry_lookup(df)
    agg=e2.aggregate_current_step(df)
    gbdt_features=[c for c in agg.columns if any(c.startswith(f+"_") for f in e2.FEATURES)]
    rows=[]

    for H in HORIZONS:
        print("\n"+"#"*100); print("HORIZON",H); print("#"*100)
        man=pd.read_csv(RESULTS/f"e1_final_samples_H{H}.csv"); man["run_id"]=man["run_id"].astype(str)

        # GBDT is deterministic for its frozen random_state=42. Run once and record
        # as a deterministic reference; do NOT manufacture ten duplicate "seeds".
        Xg,yg,_=e2.gbdt_dataset(man,"train",agg,gbdt_features)
        Xgt,ygt,mg=e2.gbdt_dataset(man,"test",agg,gbdt_features)
        gm=HistGradientBoostingClassifier(learning_rate=.05,max_iter=200,max_leaf_nodes=15,
             l2_regularization=1.0,random_state=42).fit(Xg,yg)
        gp=gm.predict_proba(Xgt)[:,1]; mm=calc_metrics(ygt,gp)
        rows.append({"model":"GBDT","horizon":H,"seed":42,"best_epoch":np.nan,
                     "best_validation_auprc":np.nan,**mm,"deterministic_reference":1})
        pd.DataFrame(mg).assign(label=ygt.astype(int),probability=gp,seed=42).to_csv(
            RESULTS/f"e6a_gbdt_H{H}_seed42_predictions.csv",index=False)

        vm,vs=e2.vector_normalization(man,lookup)
        VXtr,Vytr,_=e2.build_vector_sequences(man,"train",lookup,vm,vs)
        VXv,Vyv,_=e2.build_vector_sequences(man,"validation",lookup,vm,vs)
        VXt,Vyt,Vmeta=e2.build_vector_sequences(man,"test",lookup,vm,vs)
        gm_,gs=e2.graph_normalization(man,lookup)
        GXtr,Gytr,_=e2.build_graph_sequences(man,"train",lookup,gm_,gs)
        GXv,Gyv,_=e2.build_graph_sequences(man,"validation",lookup,gm_,gs)
        GXt,Gyt,Gmeta=e2.build_graph_sequences(man,"test",lookup,gm_,gs)

        for seed in SEEDS:
            print(f"\nH={H} | seed={seed}")
            r=train_missionshield(H,seed,man,lookup,device); r["deterministic_reference"]=0
            rows.append(r); print(" MissionShield",r["auroc"],r["auprc"])

            seed_all(seed)
            model=lstm_mod.LSTMForecaster(input_dim=e2.NUM_NODES*e2.NUM_FEATURES,
                hidden=lstm_mod.HIDDEN,layers=lstm_mod.LAYERS,dropout=lstm_mod.DROPOUT)
            r=train_baseline("LSTM",H,seed,model,VXtr,Vytr,VXv,Vyv,VXt,Vyt,Vmeta,device)
            r["deterministic_reference"]=0; rows.append(r); print(" LSTM",r["auroc"],r["auprc"])

            seed_all(seed)
            model=tr_mod.TransformerForecaster(input_dim=e2.NUM_NODES*e2.NUM_FEATURES,
                d_model=tr_mod.D_MODEL,nhead=tr_mod.NHEAD,layers=tr_mod.LAYERS,
                dropout=tr_mod.DROPOUT,ff_dim=tr_mod.FF_DIM,max_len=max(64,e2.HISTORY))
            r=train_baseline("Transformer",H,seed,model,VXtr,Vytr,VXv,Vyv,VXt,Vyt,Vmeta,device)
            r["deterministic_reference"]=0; rows.append(r); print(" Transformer",r["auroc"],r["auprc"])

            seed_all(seed)
            adjacency = graph_mod.build_adjacency()
            model=graph_mod.GraphTemporalForecaster(
                adjacency=adjacency,
                graph_hidden=graph_mod.GRAPH_HIDDEN,
                temporal_dim=graph_mod.TEMPORAL_DIM,
                nhead=graph_mod.NHEAD,
                temporal_layers=graph_mod.TEMPORAL_LAYERS,
                ff_dim=graph_mod.FF_DIM,
                dropout=graph_mod.DROPOUT,
                max_len=max(64,e2.HISTORY),
            )
            r=train_baseline("GraphTemporal",H,seed,model,GXtr,Gytr,GXv,Gyv,GXt,Gyt,Gmeta,device,graph=True)
            r["deterministic_reference"]=0; rows.append(r); print(" GraphTemporal",r["auroc"],r["auprc"])

            pd.DataFrame(rows).to_csv(RESULTS/"e6a_multiseed_results.csv",index=False)

        del VXtr,VXv,VXt,GXtr,GXv,GXt
        gc.collect()

    res=pd.DataFrame(rows)
    res.to_csv(RESULTS/"e6a_multiseed_results.csv",index=False)
    print("\n"+"="*100)
    print("E6-A COMPLETE")
    print("="*100)
    print(res.groupby(["model","horizon"])[["auroc","auprc"]].agg(["count","mean","std"]).to_string())
    print("\nSaved:",RESULTS/"e6a_multiseed_results.csv")
    print("GBDT is intentionally one deterministic reference per horizon; significance tests must not treat duplicates as independent seeds.")

if __name__=="__main__":
    main()
