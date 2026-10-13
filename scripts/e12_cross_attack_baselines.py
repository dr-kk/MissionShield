"""E12-C: E4-B-compatible leave-one-attack-family-out neural baselines.

Usage (PowerShell, from D:\\MissionShield):
  python scripts/e12_cross_attack_baselines.py --smoke
  python scripts/e12_cross_attack_baselines.py

Uses original E4-B partitions, training-only normalization, and E11 architectures.
The LSTM is a standalone E12 baseline, NOT asserted identical to E2 LSTM.
"""
import argparse
import gc
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))
RESULTS = ROOT / 'results'
OUT = RESULTS / 'e12'
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
FAMILIES = ['ddos', 'multi_stage', 'service_degradation']
HORIZONS = [10, 30, 60]
MODELS = ['lstm', 'patch', 'graphwave']

class E12LSTM(nn.Module):
    """Transparent standalone LSTM baseline: 36 input channels, hidden 64."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=36, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))

    def forward(self, x):
        z, _ = self.lstm(x)
        return self.head(z[:, -1]).squeeze(-1)


def seed_all(seed, e11):
    e11.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    result = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def audit_splits(base, held):
    tr = base.loc[(base.split == 'train') & (base.attack_family != held)].copy()
    va = base.loc[(base.split == 'validation') & (base.attack_family != held)].copy()
    te = base.loc[(base.split == 'test') & (base.attack_family == held)].copy()
    for label, d in [('train', tr), ('validation', va), ('test', te)]:
        if d.empty:
            raise RuntimeError(f'Empty {label} split for {held}')
        if d.sample_id.duplicated().any():
            raise RuntimeError(f'Duplicate sample_id in {label}')
    for a, b in [(tr, va), (tr, te), (va, te)]:
        if set(a.run_id) & set(b.run_id):
            raise RuntimeError('Run overlap between splits')
    if held in set(tr.attack_family) | set(va.attack_family):
        raise RuntimeError('Held-out family leaked into train/validation')
    if set(te.attack_family) != {held}:
        raise RuntimeError('Test includes non-held-out family')
    if tr.label.nunique() != 2 or va.label.nunique() != 2:
        raise RuntimeError('Training or validation has only one class')
    if te.label.nunique() != 2:
        raise RuntimeError('Test is single-class; AUROC undefined; condition not suitable')
    return tr, va, te


def write_csv_atomic(df, path):
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--horizons', nargs='+', type=int, choices=HORIZONS, default=HORIZONS)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--smoke', action='store_true', help='One family/horizon/seed, 2 epochs, all selected models')
    parser.add_argument('--overwrite', action='store_true', help='Re-run existing completed conditions')
    args = parser.parse_args()
    if args.smoke:
        args.horizons = [30]
        args.families = ['ddos']
        args.seeds = [101]
        args.epochs = 2
        args.patience = 2
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seed IDs')

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / ('smoke_checkpoints' if args.smoke else 'checkpoints')
    pred_dir = OUT / ('smoke_predictions' if args.smoke else 'predictions')
    ckpt_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    summary_file = OUT / ('e12_smoke_results.csv' if args.smoke else 'e12_multiseed_results.csv')
    existing = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()

    e11 = importlib.import_module('e11_recent_baselines')
    e2 = e11.load_e2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('E12-C device:', device, flush=True)
    print('E4-B compatible partitions; held-out family excluded from train/val.', flush=True)
    data_path = Path(e11.DATA)
    if not data_path.exists():
        raise FileNotFoundError(f'Telemetry dataset missing: {data_path}')
    telemetry = pd.read_csv(data_path)
    lookup = e2.build_node_lookup(telemetry)
    print('Telemetry rows:', len(telemetry), flush=True)

    for h in args.horizons:
        base = pd.read_csv(RESULTS / f'e1_final_samples_H{h}.csv')
        base['run_id'] = base['run_id'].astype(str)
        base['attack_family'] = base['attack_family'].astype(str)
        base['split'] = base['split'].astype(str)
        base['label'] = base['label'].astype(int)
        for held in args.families:
            tr, va, te = audit_splits(base, held)
            print(f'\nH{h} {held}: train={len(tr)} ({tr.label.sum()} pos), '
                  f'val={len(va)} ({va.label.sum()} pos), '
                  f'test={len(te)} ({te.label.sum()} pos); '
                  f'runs={tr.run_id.nunique()}/{va.run_id.nunique()}/{te.run_id.nunique()}', flush=True)
            manifest = pd.concat([tr, va, te], ignore_index=True)
            for model_name in args.models:
                if model_name == 'graphwave':
                    mean, std = e2.graph_normalization(manifest, lookup)
                    builder = e2.build_graph_sequences
                    constructor = e11.GraphWave
                else:
                    mean, std = e2.vector_normalization(manifest, lookup)
                    builder = e2.build_vector_sequences
                    constructor = e11.PatchTemporal if model_name == 'patch' else E12LSTM
                xtr, ytr, _ = builder(manifest, 'train', lookup, mean, std)
                xv, yv, _ = builder(manifest, 'validation', lookup, mean, std)
                xt, yt, meta = builder(manifest, 'test', lookup, mean, std)
                if (len(ytr), len(yv), len(yt)) != (len(tr), len(va), len(te)):
                    raise RuntimeError('Sequence builder count mismatch')
                if not np.array_equal(yt.astype(int), np.array([int(m) for m in te.sort_values(['run_id','time']).label])):
                    raise RuntimeError('Test label/order mismatch')
                arrays = ((xtr, ytr), (xv, yv), (xt, yt))
                for seed in args.seeds:
                    key = dict(model=model_name, horizon=h, held_out_family=held, seed=seed)
                    done = False
                    if not existing.empty and all(k in existing for k in key):
                        done = bool(np.logical_and.reduce([existing[k].astype(str).to_numpy() == str(v) for k, v in key.items()]).any())
                    if done and not args.overwrite:
                        print('SKIP completed:', key, flush=True)
                        continue
                    seed_all(seed, e11)
                    model = constructor()
                    filename = f'e12_{model_name}_{held}_H{h}_seed{seed}'
                    checkpoint = ckpt_dir / f'{filename}.pt'
                    print(f'RUN {filename} | params={sum(p.numel() for p in model.parameters()):,}', flush=True)
                    y, p, best, epoch = e11.train(model, arrays, seed, device, checkpoint,
                                                  epochs=args.epochs, patience=args.patience)
                    if not np.array_equal(y.astype(int), yt.astype(int)):
                        raise RuntimeError('Returned test labels changed')
                    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                        raise RuntimeError('Invalid probability output')
                    prediction = pd.DataFrame(meta)
                    prediction['label'] = y.astype(int)
                    prediction['probability'] = p
                    write_csv_atomic(prediction, pred_dir / f'{filename}_predictions.csv')
                    row = dict(**key, train_samples=len(ytr), train_positive=int(ytr.sum()),
                               validation_samples=len(yv), validation_positive=int(yv.sum()),
                               test_samples=len(y), test_positive=int(y.sum()),
                               test_runs=te.run_id.nunique(), best_epoch=epoch,
                               validation_auprc=float(best),
                               auprc=float(average_precision_score(y, p)),
                               auroc=float(roc_auc_score(y, p)),
                               ece=ece(y, p), brier=float(brier_score_loss(y, p)),
                               parameters=sum(t.numel() for t in model.parameters()),
                               implementation=('E12 standalone LSTM' if model_name == 'lstm' else 'E11 architecture'))
                    if not existing.empty and all(k in existing for k in key):
                        mask = np.logical_and.reduce([existing[k].astype(str).to_numpy() == str(v) for k, v in key.items()])
                        existing = existing.loc[~mask].copy()
                    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                    write_csv_atomic(existing, summary_file)
                    print('SAVED', row, flush=True)
                    del model
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                del xtr, ytr, xv, yv, xt, yt, arrays
                gc.collect()
    print('\nE12-C finished. Results:', summary_file, flush=True)
    print('Rows:', len(existing), flush=True)

if __name__ == '__main__':
    main()
