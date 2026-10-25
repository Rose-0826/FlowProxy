from pathlib import Path
import os,sys,json,time,copy,shutil,traceback,subprocess,fcntl
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from concurrent.futures import ThreadPoolExecutor,as_completed
from expansion_core import *
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import corrected_benchmark_restart_v1 as rebuild
from research_v3_corner_cfd import original_initialize

def initialize(case,q,config):
 proxy=dict(q);proxy['shape']='square' if q['family']=='rectangle' else 'cylinder'
 original_initialize(case,proxy,config)
 seed=case/'training_donor_seed.vtu';dest=case/'training_donor_seed.xml';assert seed.exists() and not dest.exists();sha=old.p.digest(seed);seed.rename(dest);assert old.p.digest(dest)==sha
 f=case/'initialization.json';j=json.loads(f.read_text());j.update(target_shape=q['shape'],target_shape_spec=shapes.SPECS[q['shape']],seed_geometry_differs=True,meaning='Old TRAIN80 numerical field is seed only. New geometry has its own true CFD relaxation >=60s and force-cycle quality checks.');old.p.save(f,j)

def checks():
 cfg=json.loads((OUT/'run_config.json').read_text());parent=cfg['parent'];assert old.p.digest(parent['checkpoint'])==parent['checkpoint_sha256']
 for info in parent['banks'].values():assert old.p.digest(info['path'])==info['sha256']
 for name,sha in cfg['source_sha256'].items():assert old.p.digest(Path(__file__).parent/name)==sha
 return cfg

def generate(qs,stage):
 cfg=checks();protocol=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text());rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.write_case=shapes.write_case;rebuild.geometry_channels=shapes.geometry_channels;rebuild.initialize=initialize
 complete=[];errors=[];start=time.perf_counter()
 def one(q):
  if shutil.disk_usage(ROOT).free<40*2**30:raise RuntimeError('Disk free <40GiB. Preserve all data and halt new starts.')
  ready=CACHE/(q['id']+'.json')
  if ready.exists():
   j=json.loads(ready.read_text());assert j['CFD_stable'] and j['uniform_frames']==165 and j['cache_sha256']==old.p.digest(CACHE/(q['id']+'.npz'));return j
  if (RAW/'cases'/q['id']).exists():raise RuntimeError('Partial CFD preserved; recover explicitly: '+q['id'])
  return rebuild.one(q,protocol)
 old.p.save(OUT/'overall_status.json',dict(stage=stage,total=len(qs),completed=0,errors=[],utc=old.p.now()))
 with ThreadPoolExecutor(max_workers=6) as pool:
  jobs={pool.submit(one,q):q for q in qs}
  for f in as_completed(jobs):
   if f.cancelled():continue
   try:complete.append(f.result())
   except Exception as e:
    errors.append(dict(id=jobs[f]['id'],error=repr(e)))
    for other in jobs:other.cancel()
   checks();old.p.save(OUT/(stage+'_status.json'),dict(stage=stage,completed=complete,total=len(qs),errors=errors,wall_seconds=time.perf_counter()-start,utc=old.p.now()));old.p.save(OUT/'overall_status.json',dict(stage=stage if not errors else 'CFD_failure',completed=len(complete),total=len(qs),errors=errors,utc=old.p.now()));print(stage,len(complete),len(errors),flush=True)
 if errors:raise RuntimeError(errors)
 return complete

def preflight():
 cfg=checks();config=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())['config'];result=[]
 for shape in shapes.SPECS:
  poly=shapes.polygon(shape);distance=shapes.sdf(np.array([0.,4.]),np.zeros(2),shape);assert distance[0]<0 and distance[1]>0
  geo=shapes.geometry_channels(config,shape,0.,.3,1.1,120.)
  reflected=shapes.geometry_channels(config,shape,0.,-.3,1.1,120.)
  assert np.allclose(geo['geom'][0,::-1],reflected['geom'][0],atol=2e-6);assert np.array_equal(geo['mask'][::-1],reflected['mask'])
  area=np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))/2
  result.append(dict(shape=shape,vertices=len(poly),cross_flow_span=float(np.ptp(poly[:,1])),area=float(area),fluid_mask_pixels=int(geo['mask'].sum()),reflection_consistent=True))
 setup();bundle=transferred_bundle();q=cfg['design']['pilot'][0];base=old.p.startup(old.queries()[0]);row=dict(base);row['q']={k:q[k] for k in ['shape','Re','cx','cy','D']};geo=shapes.geometry_channels(config,**row['q']);row['geom']=geo['geom'];row['mask']=geo['mask'];row['fields']=base['fields']*geo['mask'][None,None];values=np.full((160,4),np.nan);values[7:159:8]=0.
 pred,di=forecast(bundle,row,values,library=False,wide=True);assert pred.shape==(160,3,64,64) and np.isfinite(pred).all()
 old.p.save(OUT/'preflight.json',dict(geometry=result,inference_interface_finite=True,smoke_input_is_synthetic_interface_only=True,not_a_newshape_accuracy_result=True,budget=76,source_sha256=cfg['source_sha256']));print('PREFLIGHT_OK',flush=True)

def run():
 lock=(OUT/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=checks();d=cfg['design'];preflight()
 manifest=OUT/'pilot_dataset.json'
 if not manifest.exists():old.p.save(manifest,dict(cases=d['pilot'],results=generate(d['pilot'],'pilot_CFD'),role='new shape TRAIN-only exploration, never final independent confirmation'))
 setup();bundle=transferred_bundle()
 if not (OUT/'pilot_transfer/report.json').exists():score_queries(bundle,d['pilot'],OUT/'pilot_transfer',library=False,wide=True)
 old.p.save(OUT/'overall_status.json',dict(stage='pilot_complete',utc=old.p.now(),pilot_report=str(OUT/'pilot_transfer/report.json'),next='remaining TRAIN CFD then finite adaptation; no held future fields used for training'))
 del bundle;torch.cuda.empty_cache()
 dataset=OUT/'training_dataset.json'
 if not dataset.exists():old.p.save(dataset,dict(cases=d['train'],results=generate(d['train'],'train_CFD'),role='24 newshape TRAIN cases; pilot included, no duplication'))
 # train_expansion.py provides paired finite adaptation and old72 checks.
 train=Path(__file__).parent/'train_expansion.py'
 if not train.exists():raise RuntimeError('Training implementation pending; CFD and pilot preserved. Continue from this checkpoint after implementing train_expansion.py.')
 subprocess.run([sys.executable,str(train)],cwd=Path(__file__).parent,env=os.environ,check=True)
 # Only score new evaluation cases after all candidate weights are fixed.
 frozen=OUT/'locked_candidates.json';assert frozen.exists()
 evaluation=OUT/'evaluation_dataset.json'
 if not evaluation.exists():old.p.save(evaluation,dict(cases=d['evaluation'],results=generate(d['evaluation'],'evaluation_CFD'),role='12 new held parameter cases; fixed six profiles, not unseen geometry-family proof'))
 subprocess.run([sys.executable,str(train),'--evaluate-held'],cwd=Path(__file__).parent,env=os.environ,check=True)
 checks();print('SHAPE_EXPANSION_COMPLETE',flush=True)

if __name__=='__main__':
 try:
  if '--preflight' in sys.argv:preflight()
  else:run()
 except Exception as e:
  old.p.save(OUT/'failure.json',dict(utc=old.p.now(),error=repr(e),traceback=traceback.format_exc()));old.p.save(OUT/'overall_status.json',dict(stage='failure',error=repr(e),utc=old.p.now()));raise
