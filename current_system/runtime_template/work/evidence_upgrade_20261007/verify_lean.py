from pathlib import Path
import os,sys,json,copy,time,hashlib
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path('/root/autodl-tmp/flowproxy');OUT=ROOT/'outputs/evidence_upgrade_20261007/lean';OUT.mkdir(exist_ok=True)
capturing=True;opened=set()
def audit(event,args):
 if event=='open' and capturing and isinstance(args[0],str):
  p=Path(args[0])
  if p.is_absolute() and p.is_relative_to(ROOT) and p.is_file():opened.add(str(p))
sys.addaudithook(audit)
sys.path.insert(0,str(ROOT/'work/evidence_upgrade_20261007'));from lean_entry import Predictor,digest
source=ROOT/'outputs/targeted_coverage_20261007/selected_system.json';cfg=json.loads(source.read_text());lean=copy.deepcopy(cfg);compat=lean['new_shapes'].pop('old_banks');lean['source_selected_system_sha256']=digest(source)
(OUT/'lean_system.json').write_text(json.dumps(lean,indent=2));import torch
torch.cuda.synchronize();t=time.perf_counter();model=Predictor(OUT/'lean_system.json');torch.cuda.synchronize();load=time.perf_counter()-t
capturing=False
sys.path.insert(0,str(ROOT/'work/targeted_coverage_20261007'));import optimize as c
original=c.check();replays=[];times=[];demo={}
for mode in ['clean','future1pct','startup_and_future1pct']:
 for q0 in original['fresh_confirmation']:
  q=dict(q0,cache=str(c.CACHE));row=c.startup(q);values=c.packets(q,row)
  if mode!='clean':
   rng=np.random.default_rng(int(hashlib.sha256(('7001:'+q['id']).encode()).hexdigest()[:8],16));values=values+rng.normal(size=values.shape)*.01*c.old.p.scales()[1]
   if mode=='startup_and_future1pct':row['fields']=row['fields']+rng.normal(size=row['fields'].shape).astype('float32')*.01*__import__('numpy').asarray(c.old.p.scales(),__import__('numpy').float32)[None,:,None,None]*row['mask'][None,None]
  capturing=True;torch.cuda.synchronize();t=time.perf_counter();p,di=model.predict(row,values);torch.cuda.synchronize();secs=time.perf_counter()-t;capturing=False
  folder=c.OUT/'fresh/selected'/mode if mode=='clean' else c.OUT/'fresh/selected'/mode/'7001'
  reg=json.loads((folder/'registry'/(q['id']+'.json')).read_text());assert hashlib.sha256(p.tobytes()).hexdigest()==reg['prediction_sha256'],(mode,q['id'])
  replays.append(dict(id=q['id'],mode=mode,bitwise_unchanged=True,seconds=secs,scalar_budget=76))
  print('REPLAY',len(replays),mode,q['id'],flush=True)
  if mode=='clean':times.append(dict(id=q['id'],seconds=secs));demo[q['id']]=dict(row=row,values=values,expected_prediction_SHA=reg['prediction_sha256'])
for name,module in list(sys.modules.items()):
 p=getattr(module,'__file__',None)
 if p:
  p=Path(p)
  if p.is_relative_to(ROOT) and p.suffix=='.py' and '/work/evidence_upgrade_20261007/' not in str(p) and '/work/targeted_coverage_20261007/' not in str(p):opened.add(str(p))
runtime=[dict(path=str(Path(p).relative_to(ROOT)),bytes=Path(p).stat().st_size,sha256=digest(p)) for p in sorted(opened) if Path(p).suffix not in ['.pyc'] and 'selected_system.json' not in p and 'lean_system.json' not in p]
compatbytes=sum(Path(x['path']).stat().st_size for x in compat.values());files=[Path(b['checkpoint']) for b in [lean['original_shapes'],lean['new_shapes']]]+[Path(x['path']) for b,key in [(lean['original_shapes'],'banks'),(lean['new_shapes'],'new_banks')] for x in b[key].values()]
receipt=dict(status='PASSED',source_selected_system_SHA=digest(source),lean_system_SHA=digest(OUT/'lean_system.json'),source_code_SHA=digest(ROOT/'work/evidence_upgrade_20261007/lean_entry.py'),replays=replays,compatibility_banks_omitted=2,compatibility_bytes_removed=compatbytes,legacy_guard_weight_not_deployed=True,actual_networks=2,actual_spatial_banks=8,weights_and_banks_bytes=sum(p.stat().st_size for p in files),initialized_CUDA_load_seconds=load,clean_case_seconds=times,runtime_files=runtime,scope='Bitwise 12 clean + 12 future1pct seed7001 +12 startup/future1pct seed7001. Timing one run each during diagnostic session; not a replacement for standalone nine-run timing protocol.')
c.save(OUT/'verification_receipt.json',receipt)
import numpy as np
for qid,d in demo.items():
 row=d['row'];np.savez_compressed(OUT/(qid+'_input.npz'),fields=row['fields'],geom=row['geom'],mask=row['mask'],times=row['times'],values=d['values']);c.save(OUT/(qid+'_input.json'),dict(id=qid,q=row['q'],expected_prediction_SHA=d['expected_prediction_SHA']))
print('LEAN_DEPLOYMENT_VERIFIED',len(replays),compatbytes,sum(p.stat().st_size for p in files),flush=True)
