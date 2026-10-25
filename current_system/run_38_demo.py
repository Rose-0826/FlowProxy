"""Actual neural inference for fixed38-scalar confirmation cases.
Writes runtime/optional predictions only as an explicitly invoked inference run;
the independent NumPy auditing entries remain default read-only.
"""
from pathlib import Path
import sys,os,argparse,json,subprocess,hashlib,time
ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--all',action='store_true');ap.add_argument('--out',type=Path,default=None);args=ap.parse_args();P=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AIC_ROOT=str(P/('runtime_'+args.device)))
subprocess.run([sys.executable,str(P/'bootstrap_portable.py'),args.device],check=True)
import numpy as np
sys.path.insert(0,str(P/('runtime_'+args.device)/'work/evidence_upgrade_20261007'));from lean_entry import Predictor,e
import audit_clock_diagnostics as clock
def predict(model,row,values):
 assert row['fields'].shape==(4,3,64,64) and values.shape==(160,4) and np.isfinite(values).sum()==38
 expected=np.zeros((160,4),bool);expected[7:159:8,0]=True;expected[7:159:8,3]=True;assert np.array_equal(np.isfinite(values),expected)
 shape=row['q']['shape'];bundle=model.parent if shape in ['cylinder','square'] else model.added
 if shape in ['cylinder','square']:own,C,pa,_=e.old.p.bootstrap(bundle['model'],row,bundle['priors']);spatial=model.config['original_shapes']['spatial']
 else:
  if shape not in e.shapes.SPECS:raise ValueError('Unsupported known contour: '+shape)
  own,C,pa,_=e.bootstrap(bundle,row);spatial=dict(rank=min(24,len(bundle['libs'][shape]['A'])),width=.35,noise=.01,split=False,name='available_rank')
 modes,_=e.old.query_modes(bundle['libs'][shape],C,row,spatial);C,_=e.old.b.previous.fit(C,pa,row,modes,spatial['rank'],spatial['noise']);ob,di,_=clock.observe(C,pa,row,values,[0,3],8,'clock2');assert di['scalar_readings']==38
 return e.old.p.hybrid(own,ob)
model=Predictor(P/('runtime_'+args.device)/'system.json');E=P/'evidence/fresh38';results=[];paths=sorted((E/'allowed_inputs').glob('*.json'));paths=paths if args.all else paths[:1]
for path in paths:
 info=json.loads(path.read_text(encoding='utf-8'))
 with np.load(path.with_suffix('.npz')) as z:row={k:z[k] for k in ['fields','geom','mask','times']};values=z['values']
 row.update(id=info['id'],q=info['q']);t=time.perf_counter();prediction=predict(model,row,values);seconds=time.perf_counter()-t
 # Expected future arrays are loaded only after the inference result is complete.
 with np.load(E/'clean_predictions/clock38'/(info['id']+'.npz')) as z:expected=z['prediction']
 diff=float(np.max(np.abs(prediction-expected)));exact=np.array_equal(prediction,expected);assert exact if args.device=='cuda' else diff<2e-4
 if args.out is not None:args.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out/(info['id']+'.npz'),prediction=prediction)
 results.append(dict(id=info['id'],future_scalar_count=38,byte_exact=bool(exact),max_abs=diff,seconds=seconds));print(results[-1],file=sys.stderr,flush=True)
print(json.dumps(dict(status='PASSED',cases=len(results),device=args.device,results=results,scope='Actual two-network inference,38allowed scalars; functional replay timing, not formal benchmark.'),indent=2))
