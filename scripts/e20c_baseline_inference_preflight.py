"""E20-C: read-only preflight for frozen baseline inference compatibility."""
from pathlib import Path
import ast, json, hashlib
import pandas as pd
ROOT=Path(r'D:\MissionShield')
S=ROOT/'scripts'; R=ROOT/'results'; C=ROOT/'checkpoints'
FILES=['e1_missionshield_forecaster.py','e1_graph_temporal_prefailure.py','e1_lstm_prefailure.py','e1_transformer_prefailure.py','e2_frozen_baselines.py','e3a_extract_validation_predictions.py','e8_independent_cohort_confirmation.py']
KEYS=('model','predict','sequence','dataset','normal','manifest','load','infer','feature','history')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
print('E20-C | READ-ONLY BASELINE INFERENCE PREFLIGHT')
for name in FILES:
 p=S/name
 print('\nSCRIPT:',name,'exists=',p.exists())
 if not p.exists():continue
 src=p.read_text(encoding='utf-8-sig',errors='replace')
 try:tree=ast.parse(src)
 except SyntaxError as ex:print('SYNTAX ERROR',ex);continue
 print('SHA256:',digest(p))
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and any(k in node.name.lower() for k in KEYS):
   print(f'  {type(node).__name__} {node.name} line={node.lineno}')
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name) and target.id in ('HISTORY','NODES','FEATURES','DATA','CHECKPOINTS'):
     print('  CONSTANT',target.id,ast.get_source_segment(src,node.value)[:160] if ast.get_source_segment(src,node.value) else '?')
print('\nFROZEN OPERATING POINTS:')
op=R/'e3b_operational_summary.csv'
if op.exists():
 df=pd.read_csv(op)
 cols=['model','horizon','temperature','threshold_calibrated','validation_far']
 print(df[[x for x in cols if x in df.columns]].to_string(index=False))
else:print('MISSING',op)
print('\nCHECKPOINTS:')
for model in ['missionshield','graphtemporal','lstm','transformer']:
 for h in (10,30,60):
  names=([f'e1_missionshield_H{h}.pt'] if model=='missionshield' else [f'e2_frozen_{model}_H{h}.pt'])
  for n in names:
   p=C/n;print(n,'FOUND' if p.exists() else 'MISSING',p.stat().st_size if p.exists() else '')
print('\n[PASS] Read-only preflight finished. Share this console output; do not retrain.')
