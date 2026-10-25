from pathlib import Path
import json,numpy as np,hashlib
P=Path(__file__).resolve().parent;O=P/'evidence/piv';A=O/'audit_arrays';r=json.loads((O/'report.json').read_text(encoding='utf-8'))
with np.load(A/'normalization.npz') as z:mask=z['mask'];scale=z['scale'].astype(float);mean=z['mean'].astype(float)
maxdiff=0
for row in r['cases']:
 s=row['start_frame'];pfile=A/'test_predictions'/row['method']/(str(s)+'.npz');assert hashlib.sha256(pfile.read_bytes()).hexdigest()==row['file_sha256']
 with np.load(pfile) as z:p=z['prediction']
 with np.load(A/f'truth_{s}.npz') as z:truth=z['fields']
 assert hashlib.sha256(p.tobytes()).hexdigest()==row['prediction_sha256'];actual=[]
 for ch in [0,1]:
  delta=(p[:,ch].astype(float)-truth[:,ch].astype(float))*scale[ch];target=truth[:,ch].astype(float)*scale[ch]+mean[ch]
  actual.append(float(np.mean(np.linalg.norm(delta[:,mask],axis=1)/np.linalg.norm(target[:,mask],axis=1))))
 diff=float(np.max(np.abs(np.array(actual)-row['metrics']['mean_relative_L2'])));maxdiff=max(maxdiff,diff);assert diff<2e-6
assert len(r['cases'])==40
print('PIV_40_ARRAY_SETS_INDEPENDENTLY_RECOMPUTED',maxdiff)
