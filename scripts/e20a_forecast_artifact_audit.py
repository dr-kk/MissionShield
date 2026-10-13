"""E20-A read-only inventory of forecasting checkpoints, predictions, calibration.
Run from D:\MissionShield. Does not fit models or access held-out outcomes.
"""
from pathlib import Path
import json, csv, hashlib
ROOT=Path(r'D:\MissionShield')
OUT=ROOT/'results'/'e20_forecast_audit'
OUT.mkdir(parents=True,exist_ok=True)
KEYWORDS=('mission','graphwave','lstm','patch','forecast','calib','threshold','checkpoint','predict','alarm','e8','e20')
EXT={'.pt','.pth','.ckpt','.pkl','.pickle','.joblib','.onnx','.json','.csv','.npz','.npy','.yaml','.yml','.py'}
SKIP={'venv','.venv','node_modules','.git','__pycache__','site-packages','datasets','data','e20_forecast_audit'}
rows=[]
for base in (ROOT/'results',ROOT/'runs',ROOT/'models',ROOT/'checkpoints',ROOT/'scripts'):
    if not base.exists(): continue
    for p in base.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in EXT or any(x in SKIP for x in p.relative_to(ROOT).parts): continue
        rel=str(p.relative_to(ROOT))
        if not any(k in rel.lower() for k in KEYWORDS): continue
        stat=p.stat()
        rows.append({'path':rel,'extension':p.suffix.lower(),'bytes':stat.st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if stat.st_size<=50_000_000 else 'omitted_large_file'})
rows.sort(key=lambda r:r['path'].lower())
with (OUT/'e20a_artifact_inventory.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['path','extension','bytes','sha256']);w.writeheader();w.writerows(rows)
print('E20-A | READ-ONLY FORECAST ARTIFACT AUDIT')
print('Root:',ROOT,'| candidate files:',len(rows))
for r in rows[:120]: print(f"{r['bytes']:>12}  {r['path']}")
if len(rows)>120: print(f'... {len(rows)-120} more in CSV')
print('Saved:',OUT/'e20a_artifact_inventory.csv')
print('NEXT: identify model checkpoints, inference entry points, and frozen calibration artifacts before matched alarms.')
