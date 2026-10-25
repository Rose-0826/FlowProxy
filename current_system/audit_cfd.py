from pathlib import Path
import numpy as np,json
P=Path(__file__).resolve().parent;r=json.loads((P/'evidence/current_system/fresh_comparison.json').read_text(encoding='utf-8'));cases=r['reports']['selected']['cases'];diff=0;vals=[]
taus=np.asarray([.05,.05,.08],dtype=np.float32)
for row in cases:
 with np.load(P/'reference_fields'/(row['id']+'.npz')) as z:truth=z['fields'][4:164];mask=z['mask']
 with np.load(P/'expected_predictions'/(row['id']+'.npz')) as z:pred=z['prediction']
 delta=(pred.astype(np.float64)-truth)*mask[None,None];den=np.linalg.norm(truth*mask[None,None],axis=(-2,-1));err=np.linalg.norm(delta,axis=(-2,-1))/np.maximum(den,1e-8)
 for horizon,n in [('8s',32),('40s',160)]:
  e=err[:n];R=float(max(np.max(e.mean(axis=0)/taus),np.max(e[-1]/taus)));diff=max(diff,abs(R-row['horizons'][horizon]['R']));assert abs(R-row['horizons'][horizon]['R'])<2e-6
  if n==160:vals.append(R)
assert len(vals)==12 and sum(v<=1 for v in vals)==12
print('CFD_12_ARRAY_SETS_INDEPENDENTLY_RECOMPUTED',dict(mean_R=float(np.mean(vals)),worst_R=max(vals),max_difference=diff,truth_slice='fields[4:164]'))
