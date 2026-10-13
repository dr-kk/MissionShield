"""MissionShield E11: frozen-manifest PatchTST-style and Graph WaveNet-style adaptations.
Run from D:\\MissionShield: python scripts\\e11_recent_baselines.py
Requires scripts/e2_frozen_baselines.py, original data and frozen manifests.
"""
import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DATA = ROOT / 'data' / 'processed' / 'missionshield_e0_v2_dataset.csv'


def load_e2():
    p = ROOT / 'scripts' / 'e2_frozen_baselines.py'
    spec = importlib.util.spec_from_file_location('e2_frozen_baselines', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PatchTemporal(nn.Module):
    """PatchTST-inspired patch transformer; NOT an exact PatchTST reproduction."""
    def __init__(self, dim=36, patch_len=4, stride=2, hidden=64):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.embed = nn.Linear(dim * patch_len, hidden)
        self.pos = nn.Parameter(torch.zeros(1, 4, hidden))  # 10 steps -> 4 patches
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        # x: batch x 10 x 36
        patches = x.unfold(1, self.patch_len, self.stride)  # B x 4 x 36 x 4
        patches = patches.permute(0, 1, 3, 2).reshape(x.size(0), -1, self.patch_len * x.size(2))
        z = self.embed(patches)
        z = self.encoder(z + self.pos[:, :z.size(1)])
        return self.head(z.mean(dim=1)).squeeze(-1)


class GraphWave(nn.Module):
    """Graph WaveNet-inspired gated temporal convolution + fixed directed chain mixing.
    Adaptation, not exact published Graph WaveNet implementation.
    """
    def __init__(self, features=6, hidden=48, nodes=6):
        super().__init__()
        adjacency = torch.eye(nodes)
        for i in range(nodes - 1):
            adjacency[i + 1, i] = 1.0
        adjacency = adjacency / adjacency.sum(-1, keepdim=True)
        self.register_buffer('adjacency', adjacency)
        self.in_proj = nn.Conv2d(features, hidden, 1)
        self.filter = nn.Conv2d(hidden, hidden, (1, 3), padding=(0, 2), dilation=1)
        self.gate = nn.Conv2d(hidden, hidden, (1, 3), padding=(0, 2), dilation=1)
        self.out_proj = nn.Conv2d(hidden, hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        # B x T x N x F -> B x F x N x T
        x = x.permute(0, 3, 2, 1)
        z = self.in_proj(x)
        z = torch.tanh(self.filter(z)[..., :x.size(-1)]) * torch.sigmoid(self.gate(z)[..., :x.size(-1)])
        z = torch.einsum('ij,bcjt->bcit', self.adjacency, z)
        z = self.out_proj(z).mean(dim=(-1, -2))
        return self.head(z).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train(model, arrays, seed, device, ckpt, epochs=30, patience=5):
    (xtr, ytr), (xv, yv), (xt, yt) = arrays
    gen = torch.Generator().manual_seed(seed)
    loader_args = {'batch_size': 256, 'num_workers': 0, 'pin_memory': device.type == 'cuda'}
    train_dl = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                          shuffle=True, generator=gen, **loader_args)
    val_dl = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(xv), torch.from_numpy(yv)),
                        shuffle=False, **loader_args)
    test_dl = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt)),
                         shuffle=False, **loader_args)
    model.to(device)
    weight = (len(ytr) - float(ytr.sum())) / max(float(ytr.sum()), 1.)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bad, best_epoch = -1., 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        vy, vp = predict(model, val_dl, device)
        score = average_precision_score(vy, vp)
        print(f'  epoch {epoch:02d} | validation AUPRC {score:.5f}', flush=True)
        if score > best + 1e-4:
            best, bad, best_epoch = score, 0, epoch
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    ty, tp = predict(model, test_dl, device)
    return ty, tp, best, best_epoch


def validate_manifest(manifest, split_file, horizon):
    split = json.loads(split_file.read_text())
    if manifest.sample_id.duplicated().any():
        raise ValueError('Duplicate sample IDs')
    if not (manifest.horizon == horizon).all():
        raise ValueError('Horizon mismatch')
    for part in ('train', 'validation', 'test'):
        expected = set(map(str, split[part]))
        actual = set(manifest.loc[manifest.split == part, 'run_id'].astype(str))
        if actual != expected:
            raise ValueError(f'Split mismatch {part}: missing {len(expected - actual)} extra {len(actual - expected)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010])
    parser.add_argument('--horizons', nargs='+', type=int, choices=[10, 30, 60], default=[10, 30, 60])
    parser.add_argument('--models', nargs='+', choices=['patch', 'graphwave'], default=['patch', 'graphwave'])
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    e2 = load_e2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    df = pd.read_csv(DATA)
    lookup = e2.build_node_lookup(df)
    split_file = RESULTS / 'e1_v2_prefailure_run_split.json'
    outdir = RESULTS / 'e11'
    outdir.mkdir(exist_ok=True)
    all_rows = []
    for h in args.horizons:
        manifest = pd.read_csv(RESULTS / f'e1_final_samples_H{h}.csv')
        manifest['run_id'] = manifest['run_id'].astype(str)
        validate_manifest(manifest, split_file, h)
        print(f'\nH{h}: {len(manifest)} frozen samples', flush=True)
        for name in args.models:
            if name == 'patch':
                mean, std = e2.vector_normalization(manifest, lookup)
                build = e2.build_vector_sequences
                ctor = PatchTemporal
            else:
                mean, std = e2.graph_normalization(manifest, lookup)
                build = e2.build_graph_sequences
                ctor = GraphWave
            xtr, ytr, _ = build(manifest, 'train', lookup, mean, std)
            xv, yv, _ = build(manifest, 'validation', lookup, mean, std)
            xt, yt, metadata = build(manifest, 'test', lookup, mean, std)
            arrays = ((xtr, ytr), (xv, yv), (xt, yt))
            for seed in args.seeds:
                set_seed(seed)
                model = ctor()
                ckpt = outdir / f'e11_{name}_H{h}_seed{seed}.pt'
                print(f'\n{name} H{h} seed {seed}', flush=True)
                y, p, best, epoch = train(model, arrays, seed, device, ckpt, epochs=args.epochs)
                pred = pd.DataFrame(metadata)
                pred['label'] = y.astype(int)
                pred['probability'] = p
                pred.to_csv(outdir / f'e11_{name}_H{h}_seed{seed}_predictions.csv', index=False)
                row = dict(model=name, horizon=h, seed=seed, best_epoch=epoch,
                           validation_auprc=best, test_samples=len(y),
                           positive_rate=float(y.mean()),
                           auprc=float(average_precision_score(y, p)),
                           auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan'),
                           ece=float(e2.expected_calibration_error(y, p)),
                           brier=float(brier_score_loss(y, p)),
                           parameters=sum(v.numel() for v in model.parameters()))
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(outdir / 'e11_multiseed_results.csv', index=False)
                print('Result:', row, flush=True)
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    print('\nSaved:', outdir / 'e11_multiseed_results.csv')


if __name__ == '__main__':
    main()
