"""Portable, model-free verification of all recorded prediction/CFD metrics."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
rows=list(csv.DictReader((root/'all_case_metrics.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows)==304
groups={}
for row in rows:groups.setdefault(tuple(row[k] for k in ['role','method','noise','seed','case']),[]).append(row)
maxdiff=0
for (role,method,mode,seed,cid),items in groups.items():
    folder='fresh8' if role=='fresh_independent8' else 'historical16'
    with np.load(root/'references'/folder/(cid+'.npz')) as z:
        truth=z['fields'][4:164];mask=z['mask']
    if role=='fresh_independent8':predpath=root/'predictions'/method/mode/seed/(cid+'.npz');regpath=root/'registry'/method/mode/seed/(cid+'.json')
    else:predpath=root/'historical16/predictions'/method/(cid+'.npz');regpath=root/'historical16/registry'/method/(cid+'.json')
    with np.load(predpath) as z:pred=z['prediction']
    reg=json.loads(regpath.read_text());assert hashlib.sha256(pred.tobytes()).hexdigest()==reg['prediction_sha256']
    for item in items:
        n=32 if item['seconds']=='8' else 160
        # Match the recorded metric: subtraction in float64, reference norm in its stored dtype.
        delta=(pred[:n].astype(np.float64)-truth[:n])*mask[None,None]
        reference=truth[:n]*mask[None,None]
        errors=np.linalg.norm(delta,axis=(-2,-1))/np.maximum(np.linalg.norm(reference,axis=(-2,-1)),1e-8)
        mean=errors.mean(0);end=errors[-1];reported=np.array([float(item[k]) for k in ['Ux_mean','Uy_mean','p_mean','Ux_end','Uy_end','p_end']])
        np.testing.assert_allclose(np.r_[mean,end],reported,rtol=1e-6,atol=1e-10)
        R=float(max(np.max(mean/np.array([.05,.05,.08])),np.max(end/np.array([.05,.05,.08]))))
        diff=abs(R-float(item['R']));maxdiff=max(maxdiff,diff)
        assert np.isclose(R,float(item['R']),rtol=1e-6,atol=1e-8)
        assert (float(item['R'])<=1)==(item['pass']=='True')
for p in (root/'inputs/fresh8').glob('*.npz'):
    if '_noise' in p.name:continue
    with np.load(p) as z:assert z['fields'].shape==(4,3,64,64) and np.isfinite(z['measurements']).sum()==76
receipt=dict(status='METRICS_VERIFIED',rows=len(rows),prediction_sets=len(groups),reference_cases=24,full_prediction_SHA_verified=True,relative_tolerance=1e-6,max_R_difference=maxdiff)
(root/'metrics_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
