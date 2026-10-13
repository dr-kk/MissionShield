"""E23: controlled MissionShield ablations, reusing E1's exact training protocol.

Place alongside e1_missionshield_forecaster.py in D:\MissionShield\scripts.
Run: python scripts\e23_minimal_architecture_ablations.py --variant both

No original E1 manifests, checkpoints, predictions or results are overwritten.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['no_gat', 'mean_pool', 'both'], default='both')
args = parser.parse_args()

source_file = Path(__file__).with_name('e1_missionshield_forecaster.py')
if not source_file.is_file():
    raise SystemExit(f'Missing {source_file}; copy the original E1 script beside E23.')
original = source_file.read_text(encoding='utf-8')

# Keep RESULTS pointing to original frozen E1 manifest directory; redirect ONLY outputs.
original_prediction = 'f"e1_missionshield_H{H}_predictions.csv"'
original_summary = '"e1_missionshield_final_results.csv"'
original_checkpoint = 'f"e1_missionshield_H{H}.pt"'
for needle in (original_prediction, original_summary, original_checkpoint):
    if original.count(needle) != 1:
        raise SystemExit(f'E1 version mismatch; expected exactly one occurrence of {needle}')

variants = ['no_gat', 'mean_pool'] if args.variant == 'both' else [args.variant]
for variant in variants:
    source = original.replace(original_prediction, f'f"e23_{variant}_H{{H}}_predictions.csv"')
    source = source.replace(original_summary, f'"e23_{variant}_results.csv"')
    source = source.replace(original_checkpoint, f'f"e23_{variant}_H{{H}}.pt"')
    # The original training/evaluation loop, seeds, preprocessing, split manifests,
    # optimizer, early stopping, and evaluation metrics remain unchanged.
    # The forward method below changes only the graph-dependency computation.
    injection = '''
import torch as _torch

def _e23_dependency_forward(self, x):
    if _E23_VARIANT == "no_gat":
        # Identity graph propagation; retain learned node gate and all other blocks.
        h = x
        logits = self.node_gate(h).squeeze(-1)
        weights = _torch.softmax(logits, dim=-1)
        return (h * weights.unsqueeze(-1)).sum(dim=1), weights, None, None
    elif _E23_VARIANT == "mean_pool":
        # Retain BOTH graph-attention layers, replace learned pooling with uniform pooling.
        h, attn1 = self.layer1(x)
        h, attn2 = self.layer2(h)
        weights = _torch.full(h.shape[:2], 1.0 / h.shape[1], device=h.device, dtype=h.dtype)
        return h.mean(dim=1), weights, attn1, attn2
    raise RuntimeError("Unexpected variant")

DependencyEncoder.forward = _e23_dependency_forward
'''
    marker = 'if __name__ == "__main__":'
    if source.count(marker) != 1:
        raise SystemExit('Could not find E1 main guard. Check the source version.')
    source = source.replace(marker, injection + '\n' + marker)
    print(f'\n========== E23 ABLATION: {variant} ==========\n', flush=True)
    scope = {'__name__': '__main__', '__file__': str(source_file), '_E23_VARIANT': variant}
    exec(compile(source, str(source_file), 'exec'), scope)
print('\nE23 complete. Compare e23_*_results.csv against e1_missionshield_final_results.csv.')
