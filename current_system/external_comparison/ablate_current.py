"""Fixed post-confirmation mechanism diagnostics; no selection, new CFD or FNO training."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,copy
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path('/root/autodl-tmp/flowproxy');WORK=ROOT/'work/evidence_upgrade_20261007';OUT=ROOT/'outputs/evidence_upgrade_20261007/ablation'
sys.path.insert(0,str(ROOT/'work/targeted_coverage_20261007'));import optimize as c
OUT.mkdir(exist_ok=True);save=c.save
def update(stage,**kw):save(OUT/'status.json',dict(stage=stage,utc=c.now(),**kw));print(stage,kw,flush=True)
def unadapted_banks(cfg,selected):
 folder=OUT/'unadapted_network_aligned_TRAIN_banks';cp=selected['original_shapes']['checkpoint']
 if (folder/'system.json').exists():return folder/'system.json'
 bundle=c.old.load_bundle(cp,{});qs=json.loads((c.e.OUT/'run_config.json').read_text())['design']['train']+cfg['additional_train'];assert len(qs)==56
 for shape,st in selected['new_shapes']['new_priors'].items():bundle['priors'][shape]=lambda x,st=st:np.full(len(x),st)
 records=[]
 for i,q in enumerate(qs):
  path=(c.CACHE if q['id'] in {q['id'] for q in cfg['additional_train']} else c.e.CACHE)/(q['id']+'.npz');meta=json.loads(path.with_suffix('.json').read_text());assert meta['CFD_stable'] and c.sha(path)==meta['cache_sha256']
  with np.load(path) as z:raw={k:z[k] for k in ['fields','times','mask','geom']}
  frequency=meta['St']/q['D'];B=c.old.p.basis(raw['times']-raw['times'][3],frequency);T=c.old.b.previous.w.constrain(np.linalg.lstsq(B,raw['fields'].reshape(165,-1),rcond=1e-10)[0].reshape(9,3,64,64),raw['mask'])
  row=dict(id=q['id'],q={k:q[k] for k in ['shape','Re','cx','cy','D']},fields=raw['fields'][:4],geom=raw['geom'],mask=raw['mask'],times=raw['times'])
  own,C,pa,di=c.e.bootstrap(bundle,row)
  records.append(dict(id=q['id'],q=row['q'],fold=0,group_key=q['id'],mask=raw['mask'],truthC=T,C=C.astype(np.float32),params=pa))
  update('aligned_unadapted_TRAIN_bank',completed=i+1,total=56)
 banks={}
 for shape in c.e.shapes.SPECS:
  rows=[z for z in records if z['q']['shape']==shape];c.old.b.build(rows,str(folder/shape));path=folder/shape/(shape+'_bank.npz');banks[shape]=dict(path=str(path),sha256=c.sha(path))
 info=copy.deepcopy(selected);info['new_shapes'].update(checkpoint=cp,checkpoint_sha256=c.sha(cp),old_banks=selected['original_shapes']['banks'],new_banks=banks,name='unadapted_FNO_with_same_TRAIN56_rebuilt_library')
 save(folder/'system.json',info);return folder/'system.json'
def components(bundle,row,values,spatial,library=True,observer=True,disable_all_spatial=False):
 own,C,pa,boot=c.e.bootstrap(bundle,row)
 if disable_all_spatial:C=c.old.p.neural_template(row,own,boot['prefix_steps'],pa[1])
 if library:
  modes,md=c.old.query_modes(bundle['libs'][row['q']['shape']],C,row,spatial);C,fit=c.old.b.previous.fit(C,pa,row,modes,spatial['rank'],spatial['noise'])
 if observer:ob,di=c.old.p.observe(C,pa[1],row,values,pa)
 else:
  t=row['times'][4:164]-row['times'][3];ob=(c.old.p.basis(t,pa[1],pa[0])@C.reshape(9,-1)).reshape(160,3,64,64).astype(np.float32)*row['mask'][None,None];di=dict(scalar_future_readings=0,fixed_startup_phase_and_frequency=True)
 return c.old.p.hybrid(own,ob),di
def run():
 cfg=c.check();c.e.setup();selected=json.loads((c.OUT/'selected_system.json').read_text());source=c.sha(c.OUT/'selected_system.json');save(OUT/'config.json',dict(selected_system_SHA=source,source_sha256=c.sha(__file__),role='Post-confirmation diagnostic on same12; no new independent score, no model tuning or selection.',controls=['full','without_local_residual_library','without_all_startup_spatial_calibration','without_phase_frequency_updates','unadapted_FNO_aligned_TRAIN56_library'],input_available='4 startup fields +76 Uy future scalars; no-observer deliberately uses0 of76',geometry_control='Original TRAIN100 FNO, same newTRAIN56 for priors and recomputed residual banks; original two shapes stay unchanged in all methods.'))
 controlpath=unadapted_banks(cfg,selected);models=dict(full=c.previous.Predictor(c.OUT/'selected_system.json'),unadapted=c.previous.Predictor(controlpath));rows=[];snaps={}
 for j,q0 in enumerate(cfg['fresh_confirmation']):
  q=dict(q0,cache=str(c.CACHE));row=c.startup(q);values=c.packets(q,row);shape=q['shape'];bundle=models['full'].parent if shape in ['cylinder','square'] else models['full'].added
  spatial=selected['original_shapes']['spatial'] if shape in ['cylinder','square'] else dict(rank=min(24,len(bundle['libs'][shape]['A'])),width=.35,noise=.01,split=False,name='available_rank')
  full,di=models['full'].predict(row,values);reg=json.loads((c.OUT/'fresh/selected/clean/registry'/(q['id']+'.json')).read_text());assert hashlib.sha256(full.tobytes()).hexdigest()==reg['prediction_sha256']
  methods=dict(full=full,without_local_residual_library=components(bundle,row,values,spatial,False)[0],without_all_startup_spatial_calibration=components(bundle,row,values,spatial,False,True,True)[0],without_phase_frequency_updates=components(bundle,row,values,spatial,True,False)[0],unadapted_FNO_aligned_TRAIN56_library=models['unadapted'].predict(row,values)[0])
  with np.load(c.CACHE/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
  for name,p in methods.items():
   assert np.isfinite(p).all();target=OUT/'predictions'/name/(q['id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,prediction=p)
   rows.append(dict(id=q['id'],q=q,horizons=c.old.p.metrics(p,truth,row['mask'],row['times']),prediction_SHA=hashlib.sha256(p.tobytes()).hexdigest(),file_sha256=c.sha(target),available_future_scalars=76,consumed_future_scalars=0 if name=='without_phase_frequency_updates' else 76))
   if j==0:snaps[name]=p[[31,79,159]]
  if j==0:snaps.update(truth=truth[[31,79,159]],mask=row['mask'])
  save(OUT/'partial.json',rows);update('post_confirmation_diagnostic',completed=j+1,total=12)
 summary={name:c.summarize([rows[i] for i in range(len(rows)) if i%len(methods)==k]) for k,name in enumerate(methods)}
 np.savez_compressed(OUT/'snapshots.npz',**snaps);save(OUT/'report.json',dict(status='COMPLETE',role='Current-method post-confirmation diagnostics. Original12 first confirmation remains unchanged; these controls are not a new holdout experiment.',summary=summary,cases=rows,all12_full_bitwise_preserved=True,selected_system_SHA=source,source_SHA=c.sha(__file__),geometry_control_bank_source_cases=56))
 c.check();update('COMPLETE',summary=summary)
if __name__=='__main__':
 try:run()
 except Exception:save(OUT/'failure.json',dict(traceback=traceback.format_exc()));update('FAILED');raise
