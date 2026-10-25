"""Read-only audit:16clean arrays scored afresh,176stored error paths summarized."""
from pathlib import Path
import json,numpy as np,argparse,sys,hashlib
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=None);args=ap.parse_args();P=Path(__file__).resolve().parent;E=P/'evidence/fresh38';report=json.loads((E/'report.json').read_text(encoding='utf-8'));max_R=0;checks=[];clean_arrays=0;error_paths=0
def score(err,n):x=err[:n];tau=np.asarray([.05,.05,.08],np.float32);return float(max(np.max(x.mean(0)/tau),np.max(x[-1]/tau)))
for row in report['cases']:
 with np.load(E/'per_frame_errors'/row['method']/row['mode']/str(row['seed'])/(row['id']+'.npz')) as z:err=z['relative_L2'];error_paths+=1
 assert err.shape==(160,3) and np.isfinite(err).all()
 if row['mode']=='clean':
  with np.load(E/'references'/(row['id']+'.npz')) as z:truth=z['fields'][4:164];mask=z['mask']
  with np.load(E/'clean_predictions'/row['method']/(row['id']+'.npz')) as z:pred=z['prediction']
  assert hashlib.sha256(pred.tobytes()).hexdigest()==row['prediction_SHA']
  own_err=np.linalg.norm((pred.astype(float)-truth)*mask[None,None],axis=(-2,-1))/np.maximum(np.linalg.norm(truth*mask[None,None],axis=(-2,-1)),1e-8);assert np.allclose(own_err,err,rtol=0,atol=1e-12);clean_arrays+=1
 for h,n in [('8s',32),('40s',160)]:delta=abs(score(err,n)-row['horizons'][h]['R']);max_R=max(max_R,delta);assert delta<2e-6
 checks.append(dict(id=row['id'],method=row['method'],mode=row['mode'],seed=row['seed'],independent_clean_array_recompute=row['mode']=='clean'))
assert clean_arrays==16 and error_paths==176
receipt=dict(status='PASSED',clean_prediction_arrays_independently_recomputed=16,error_trajectories_summarized=176,max_R_difference=max_R,scope='Clean16 from original prediction/CFD arrays. Noise160 recomputes summaries from exported per-frame error trajectories; server receipt separately verifies all176 original prediction arrays.',checks=checks)
payload=json.dumps(receipt,indent=2)
if args.out is not None:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(payload+'\n',encoding='utf-8')
else:sys.stdout.write(payload+'\n')
