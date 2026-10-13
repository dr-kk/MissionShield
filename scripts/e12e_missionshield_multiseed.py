"""E12-E: frozen E4-B MissionShield LOAO, ten seeds, isolated outputs.

From D:\\MissionShield:
  python scripts/e12e_missionshield_multiseed.py --smoke
  python scripts/e12e_missionshield_multiseed.py

Default runs 3 horizons x 3 held-out families x 10 seeds = 90.
--smoke runs DDoS/H30/seed101 for 2 epochs in separate smoke files.
Uses the exact E4-B model, preprocessing, objective, optimizer, and early-stop
criterion. Only random seed and output destinations change. E4-B and E1-E12-C
artifacts are not modified.
"""
import argparse
import gc
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
RESULTS = ROOT / 'results'
OUT = RESULTS / 'e12'
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
HORIZONS = [10, 30, 60]
FAMILIES = ['ddos', 'multi_stage', 'service_degradation']


def load_ms():
    path = SCRIPTS / 'e1_missionshield_forecaster.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('e12e_e1_ms', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    result = 0.0
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        mask = (p >= low) & ((p < high) if i < bins-1 else (p <= high))
        if mask.any():
            result += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return result


def atomic_csv(frame, path):
    temp = path.with_suffix('.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def splits(base, held):
    tr = base[(base.split == 'train') & (base.attack_family != held)].copy()
    va = base[(base.split == 'validation') & (base.attack_family != held)].copy()
    te = base[(base.split == 'test') & (base.attack_family == held)].copy()
    for name, subset in [('train', tr), ('validation', va), ('test', te)]:
        if subset.empty or subset.sample_id.duplicated().any():
            raise RuntimeError(f'Empty or duplicate {name} for {held}')
    for a, b in [(tr, va), (tr, te), (va, te)]:
        if set(a.run_id) & set(b.run_id):
            raise RuntimeError('Run overlap')
    if held in set(tr.attack_family) | set(va.attack_family):
        raise RuntimeError('Held-out attack family in training/validation')
    if set(te.attack_family) != {held}:
        raise RuntimeError('Unexpected test family')
    if any(d.label.nunique() != 2 for d in (tr, va, te)):
        raise RuntimeError('Single-class partition; cannot compute AUROC')
    return tr, va, te


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--horizons', nargs='+', type=int, choices=HORIZONS, default=HORIZONS)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.horizons, args.families = [101], [30], ['ddos']
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seed IDs')
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '_smoke' if args.smoke else ''
    ckpt_dir = OUT / ('e12e_smoke_checkpoints' if args.smoke else 'e12e_checkpoints')
    pred_dir = OUT / ('e12e_smoke_predictions' if args.smoke else 'e12e_predictions')
    ckpt_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    summary = OUT / f'e12_missionshield_multiseed{suffix}.csv'
    existing = pd.read_csv(summary) if summary.exists() else pd.DataFrame()

    ms = load_ms()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('E12-E device:', device, flush=True)
    print('Frozen E4-B MissionShield architecture, training-only normalization.', flush=True)
    df = pd.read_csv(ms.DATA)
    lookup = ms.build_telemetry_lookup(df)
    directed_mask = ms.build_directed_attention_mask()
    print('Telemetry rows:', len(df), 'timestamps:', len(lookup), flush=True)

    for H in args.horizons:
        base = pd.read_csv(RESULTS / f'e1_final_samples_H{H}.csv')
        base['run_id'] = base.run_id.astype(str)
        base['attack_family'] = base.attack_family.astype(str)
        base['split'] = base.split.astype(str)
        base['label'] = base.label.astype(int)
        for held in args.families:
            tr, va, te = splits(base, held)
            manifest = pd.concat([tr, va, te], ignore_index=True)
            mean, std = ms.calculate_normalization(manifest, lookup)
            Xtr, ytr, _ = ms.build_sequences_from_manifest(manifest, 'train', lookup, mean, std)
            Xv, yv, _ = ms.build_sequences_from_manifest(manifest, 'validation', lookup, mean, std)
            Xt, yt, meta = ms.build_sequences_from_manifest(manifest, 'test', lookup, mean, std)
            if (len(ytr), len(yv), len(yt)) != (len(tr), len(va), len(te)):
                raise RuntimeError('Manifest-to-tensor size mismatch')
            if not np.array_equal(np.asarray(yt).astype(int),
                                  te.sort_values(['run_id', 'time']).label.to_numpy()):
                raise RuntimeError('Test sample order mismatch')
            print(f'\nH{H} {held}: train={len(tr)} ({int(sum(ytr))} pos), '
                  f'val={len(va)} ({int(sum(yv))} pos), '
                  f'test={len(te)} ({int(sum(yt))} pos); '
                  f'runs={tr.run_id.nunique()}/{va.run_id.nunique()}/{te.run_id.nunique()}', flush=True)
            for seed in args.seeds:
                key = {'model': 'MissionShield', 'horizon': H,
                       'held_out_family': held, 'seed': seed}
                if not existing.empty and all(k in existing for k in key):
                    matches = np.logical_and.reduce([
                        existing[k].astype(str).to_numpy() == str(v) for k, v in key.items()])
                    if matches.any() and not args.overwrite:
                        print('SKIP completed:', key, flush=True)
                        continue
                # Seed controls model initialization AND shuffled minibatch order.
                ms.seed_everything(seed)
                gen = torch.Generator().manual_seed(seed)
                loader_args = dict(batch_size=ms.BATCH_SIZE, num_workers=0,
                                   pin_memory=(device.type == 'cuda'))
                train_loader = DataLoader(ms.MissionDataset(Xtr, ytr), shuffle=True,
                                          generator=gen, **loader_args)
                val_loader = DataLoader(ms.MissionDataset(Xv, yv), shuffle=False,
                                        **loader_args)
                test_loader = DataLoader(ms.MissionDataset(Xt, yt), shuffle=False,
                                         **loader_args)
                model = ms.MissionShieldForecaster(directed_mask).to(device)
                pos, neg = float(np.sum(ytr)), float(len(ytr) - np.sum(ytr))
                criterion = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(neg / max(pos, 1.), dtype=torch.float32,
                                            device=device))
                optimizer = torch.optim.AdamW(model.parameters(), lr=ms.LR,
                                              weight_decay=ms.WEIGHT_DECAY)
                name = f'e12e_missionshield_{held}_H{H}_seed{seed}'
                ckpt = ckpt_dir / f'{name}.pt'
                best_ap, best_epoch, patience = -np.inf, 0, 0
                epochs = min(2, ms.EPOCHS) if args.smoke else ms.EPOCHS
                print('RUN', name, 'params=', sum(p.numel() for p in model.parameters()), flush=True)
                for epoch in range(1, epochs + 1):
                    model.train()
                    running, seen = 0., 0
                    for X, y in train_loader:
                        X = X.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True).float()
                        optimizer.zero_grad(set_to_none=True)
                        logits, _, _ = model(X)
                        loss = criterion(logits, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        running += float(loss.item()) * len(X)
                        seen += len(X)
                    _, vp = ms.predict(model, val_loader, device)
                    vap = float(average_precision_score(yv, vp))
                    print(f'  epoch {epoch:02d} | train_loss={running/max(seen,1):.6f} '
                          f'| val_AUPRC={vap:.6f}', flush=True)
                    if np.isfinite(vap) and vap > best_ap + 1e-8:
                        best_ap, best_epoch, patience = vap, epoch, 0
                        torch.save(model.state_dict(), ckpt)
                    else:
                        patience += 1
                        if patience >= ms.PATIENCE:
                            break
                if best_epoch == 0:
                    raise RuntimeError('No valid validation checkpoint')
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                ty, tp, node_weights, gates = ms.predict(
                    model, test_loader, device, return_explanations=True)
                ty = np.asarray(ty).astype(int)
                tp = np.asarray(tp)
                if not np.array_equal(ty, np.asarray(yt).astype(int)):
                    raise RuntimeError('Test labels mismatch')
                if not np.isfinite(tp).all() or np.any((tp < 0) | (tp > 1)):
                    raise RuntimeError('Invalid predicted probabilities')
                prediction = pd.DataFrame(meta)
                prediction['label'] = ty
                prediction['probability'] = tp
                prediction['fusion_temporal_gate'] = gates
                for j, node in enumerate(ms.NODES):
                    prediction[f'importance_{node}'] = node_weights[:, j]
                atomic_csv(prediction, pred_dir / f'{name}_predictions.csv')
                row = dict(**key, train_samples=len(ytr), train_positive=int(np.sum(ytr)),
                           validation_samples=len(yv), validation_positive=int(np.sum(yv)),
                           test_samples=len(ty), test_positive=int(np.sum(ty)),
                           test_runs=te.run_id.nunique(), best_epoch=best_epoch,
                           validation_auprc=best_ap,
                           auprc=float(average_precision_score(ty, tp)),
                           auroc=float(roc_auc_score(ty, tp)),
                           ece=ece(ty, tp), brier=float(brier_score_loss(ty, tp)),
                           parameters=sum(p.numel() for p in model.parameters()),
                           implementation='E4-B frozen MissionShield')
                if not existing.empty and all(k in existing for k in key):
                    mask = np.logical_and.reduce([
                        existing[k].astype(str).to_numpy() == str(v) for k, v in key.items()])
                    existing = existing.loc[~mask].copy()
                existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                atomic_csv(existing, summary)
                print('SAVED', row, flush=True)
                del model, optimizer, train_loader, val_loader, test_loader
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            del Xtr, Xv, Xt, ytr, yv, yt
            gc.collect()
    print('\nE12-E finished:', summary, 'rows:', len(existing), flush=True)


if __name__ == '__main__':
    main()
