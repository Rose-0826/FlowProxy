from pathlib import Path
import os,sys,json,hashlib,argparse,time,subprocess
ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--all',action='store_true');args=ap.parse_args();P=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AIC_ROOT=str(P/('runtime_'+args.device)))
subprocess.run([sys.executable,str(P/'bootstrap_portable.py'),args.device],check=True)
external_root_reads=[]
def audit(event,items):
 if event=='open' and isinstance(items[0],str):
  f=Path(items[0]).resolve()
  if f.is_relative_to('/root/autodl-tmp/flowproxy') and not f.is_relative_to(P):external_root_reads.append(str(f))
sys.addaudithook(audit)
import numpy as np
sys.path.insert(0,str(Path(os.environ['AIC_ROOT'])/'work/evidence_upgrade_20261007'));from lean_entry import Predictor
t=time.perf_counter();model=Predictor(Path(os.environ['AIC_ROOT'])/'system.json');load=time.perf_counter()-t;results=[]
inputs=sorted((P/'demo_inputs').glob('*_input.json'));inputs=inputs if args.all else inputs[:1]
for p in inputs:
 info=json.loads(p.read_text(encoding='utf-8'))
 with np.load(p.with_suffix('.npz')) as z:row={k:z[k] for k in ['fields','geom','mask','times']};values=z['values']
 row.update(id=info['id'],q=info['q']);t=time.perf_counter();pred,di=model.predict(row,values);seconds=time.perf_counter()-t
 with np.load(P/'expected_predictions'/(info['id']+'.npz')) as z:expected=z['prediction']
 delta=float(np.max(np.abs(pred-expected)));exact=hashlib.sha256(pred.tobytes()).hexdigest()==info['expected_prediction_SHA']
 if args.device=='cuda':assert exact,(info['id'],delta)
 else:assert delta<2e-4,(info['id'],delta)
 target=P/'replayed_predictions'/(info['id']+'.npz');target.parent.mkdir(exist_ok=True);np.savez_compressed(target,prediction=pred)
 results.append(dict(id=info['id'],device=args.device,bitwise_equal=exact,max_abs_difference=delta,seconds=seconds,future_scalars=76));print(results[-1],flush=True)
assert not external_root_reads,external_root_reads
receipt=dict(status='PASSED',device=args.device,load_seconds=load,results=results,original_workspace_file_reads=external_root_reads,scope='Inference/replay. CPU numerical tolerance2e-4; GPU exact replay. Timing is functional verification, not headline performance benchmark.')
(P/('standalone_verification_'+args.device+'.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
