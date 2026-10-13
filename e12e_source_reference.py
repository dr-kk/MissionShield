from pathlib import Path
import importlib.util
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(r"D:\MissionShield")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
HORIZONS = [10, 30, 60]
HELD_OUT = ["ddos", "multi_stage", "service_degradation"]
SEED = 42

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ms = load_module("ms", SCRIPTS / "e1_missionshield_forecaster.py")

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else np.nan

def safe_ap(y,p):
    return float(average_precision_score(y,p)) if np.sum(y) > 0 else np.nan

def make_manifest(base, train_m, val_m, test_m):
    # E1 helpers expect one manifest plus split labels.
    z = pd.concat([train_m, val_m, test_m], ignore_index=True)
    z["run_id"] = z["run_id"].astype(str)
    return z

def main():
    ms.seed_everything(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*96)
    print("MissionShield E4-B | LEAVE-ONE-ATTACK-FAMILY-OUT")
    print("MISSIONShield ONLY | FROZEN ARCHITECTURE/HYPERPARAMETERS | NO TEST TUNING")
    print("="*96)
    print("Device :",device)
    if torch.cuda.is_available(): print("GPU    :",torch.cuda.get_device_name(0))

    # Exact symbols from the frozen E1 implementation.
    df=pd.read_csv(ms.DATA)
    print(f"Node rows : {len(df):,}")
    print(f"Runs      : {df.run_id.nunique()}")
    lookup=ms.build_telemetry_lookup(df)
    print(f"Telemetry timestamps : {len(lookup):,}")
    directed_mask=ms.build_directed_attention_mask()

    rows=[]
    for H in HORIZONS:
        base=pd.read_csv(RESULTS/f"e1_final_samples_H{H}.csv")
        base["run_id"]=base["run_id"].astype(str)
        base["attack_family"]=base["attack_family"].astype(str)
        base["split"]=base["split"].astype(str)
        base["label"]=base["label"].astype(int)

        for held in HELD_OUT:
            print("\n"+"="*96)
            print(f"H={H} | HELD-OUT FAMILY={held}")
            print("="*96)

            tr=base[(base.split=="train")&(base.attack_family!=held)].copy()
            va=base[(base.split=="validation")&(base.attack_family!=held)].copy()
            te=base[(base.split=="test")&(base.attack_family==held)].copy()

            print(f"Train n={len(tr):,} pos={int(tr.label.sum()):,} | "
                  f"Val n={len(va):,} pos={int(va.label.sum()):,} | "
                  f"LOAO test n={len(te):,} pos={int(te.label.sum()):,}")

            loao_manifest=make_manifest(base,tr,va,te)

            # E1 normalization uses only rows tagged train.
            mean,std=ms.calculate_normalization(loao_manifest,lookup)

            Xtr,ytr,meta_tr=ms.build_sequences_from_manifest(
                loao_manifest,"train",lookup,mean,std)
            Xv,yv,meta_v=ms.build_sequences_from_manifest(
                loao_manifest,"validation",lookup,mean,std)
            Xt,yt,meta_t=ms.build_sequences_from_manifest(
                loao_manifest,"test",lookup,mean,std)

            assert len(ytr)==len(tr) and len(yv)==len(va) and len(yt)==len(te)
            if len(np.unique(ytr))<2 or len(np.unique(yv))<2:
                raise RuntimeError("LOAO train/validation became single-class.")

            g=torch.Generator()
            g.manual_seed(SEED+H)
            train_loader=DataLoader(ms.MissionDataset(Xtr,ytr),
                batch_size=ms.BATCH_SIZE,shuffle=True,num_workers=0,
                pin_memory=torch.cuda.is_available(),generator=g)
            val_loader=DataLoader(ms.MissionDataset(Xv,yv),
                batch_size=ms.BATCH_SIZE,shuffle=False,num_workers=0,
                pin_memory=torch.cuda.is_available())
            test_loader=DataLoader(ms.MissionDataset(Xt,yt),
                batch_size=ms.BATCH_SIZE,shuffle=False,num_workers=0,
                pin_memory=torch.cuda.is_available())

            # Match E1 horizon seeding.
            ms.seed_everything(SEED+H)
            model=ms.MissionShieldForecaster(directed_mask).to(device)

            pos=float(ytr.sum()); neg=float(len(ytr)-pos)
            pos_weight=torch.tensor(neg/max(pos,1.0),dtype=torch.float32,device=device)
            criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer=torch.optim.AdamW(model.parameters(),lr=ms.LR,
                                        weight_decay=ms.WEIGHT_DECAY)

            ckpt=CHECKPOINTS/f"e4b_loao_missionshield_{held}_H{H}.pt"
            best_ap=-np.inf; best_epoch=0; patience=0

            for epoch in range(1,ms.EPOCHS+1):
                model.train(); running=0.0; seen=0
                for X,y in train_loader:
                    X=X.to(device,non_blocking=True)
                    y=y.to(device,non_blocking=True).float()
                    optimizer.zero_grad(set_to_none=True)
                    logits,_,_=model(X)
                    loss=criterion(logits,y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                    optimizer.step()
                    running+=float(loss.item())*len(X); seen+=len(X)

                _,vp=ms.predict(model,val_loader,device)
                vap=safe_ap(yv,vp)
                print(f"Epoch {epoch:02d} | train_loss={running/max(seen,1):.6f} | "
                      f"val_AUPRC={vap:.6f}")

                if np.isfinite(vap) and vap>best_ap+1e-8:
                    best_ap=vap; best_epoch=epoch; patience=0
                    torch.save(model.state_dict(),ckpt)
                else:
                    patience+=1
                    if patience>=ms.PATIENCE:
                        print(f"Early stop at epoch {epoch}")
                        break

            model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True))
            ty,tp,node_weights,gate_values=ms.predict(
                model,test_loader,device,return_explanations=True)

            auc=safe_auc(ty,tp); ap=safe_ap(ty,tp)

            pred=pd.DataFrame(meta_t)
            pred["label"]=ty.astype(int)
            pred["probability"]=tp
            pred["fusion_temporal_gate"]=gate_values
            for j,node in enumerate(ms.NODES):
                pred[f"importance_{node}"]=node_weights[:,j]

            pred_file=RESULTS/f"e4b_loao_missionshield_{held}_H{H}_predictions.csv"
            pred.to_csv(pred_file,index=False)

            rows.append({
                "model":"MissionShield","horizon":H,"held_out_family":held,
                "train_samples":len(tr),"train_positive":int(tr.label.sum()),
                "validation_samples":len(va),"validation_positive":int(va.label.sum()),
                "test_samples":len(te),"test_positive":int(te.label.sum()),
                "best_epoch":best_epoch,"best_validation_auprc":best_ap,
                "test_auroc":auc,"test_auprc":ap})
            print(f"SELECTED epoch={best_epoch} val_AUPRC={best_ap:.6f} | "
                  f"LOAO test AUROC={auc:.6f} AUPRC={ap:.6f}")
            print("Saved:",pred_file)

    summary=pd.DataFrame(rows)
    outfile=RESULTS/"e4b_loao_missionshield_summary.csv"
    summary.to_csv(outfile,index=False)
    print("\n"+"="*96)
    print("E4-B LOAO SUMMARY")
    print("="*96)
    print(summary.to_string(index=False))
    print("\nSaved:",outfile)
    print("\nE4-B COMPLETE")

if __name__=="__main__":
    main()
