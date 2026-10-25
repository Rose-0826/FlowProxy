from pathlib import Path
import os,sys,json,time,hashlib,numpy as np,torch
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'work/practical_targeted_20261006')]
import target_core as old
import shapes
OUT=ROOT/'outputs/shape_expansion_20261006';RAW=ROOT/'data/raw/shape_expansion_20261006';CACHE=ROOT/'data/processed/shape_expansion_20261006'
original_body_mask=old.b.body.body_mask

def body_mask(shape):
 if shape not in shapes.SPECS:return original_body_mask(shape)
 return (shapes.sdf(old.b.body.BX,old.b.body.BY,shape)>0).astype(float)

def setup():
 old.setup();old.b.body.body_mask=body_mask

def startup(q):
 with np.load(CACHE/(q['id']+'_startup.npz')) as z:row={k:z[k] for k in ['fields','times','geom','mask']}
 row.update(id=q['id'],q={k:q[k] for k in ['shape','Re','cx','cy','D']});assert row['fields'].shape==(4,3,64,64);return row

def packets(q,row,noise=0.):
 ij,_=old.p.probes(row);a=np.full((160,4),np.nan)
 with np.load(CACHE/(q['id']+'.npz')) as z:
  for step in range(8,160,8):a[step-1]=[z['fields'][step+3,1,i,j] for i,j in ij]
 if noise:
  rng=np.random.default_rng(int(hashlib.sha256(('newshape:'+q['id']).encode()).hexdigest()[:8],16));a+=rng.normal(size=a.shape)*noise*old.p.scales()[1]
 assert np.isfinite(a).sum()==76;return a

def transferred_bundle():
 cfg=json.loads((OUT/'run_config.json').read_text());b=old.load_bundle(cfg['parent']['checkpoint'],cfg['parent']['banks'])
 # No target forces/future fields determine these proxy priors.
 for shape,spec in shapes.SPECS.items():b['priors'][shape]=b['priors']['square' if spec['family']=='rectangle' else 'cylinder']
 return b

def bootstrap(bundle,row,wide=False):
 if not wide:return old.p.bootstrap(bundle['model'],row,bundle['priors'])
 own,di=old.p.infer(bundle['model'],row,True);n=di['prefix_steps'];q=row['q'];center=float(bundle['priors'][q['shape']](old.p.features(q)[None])[0])/q['D']
 obs=np.concatenate([row['fields'],own[:n]]);times=row['times'][:n+4]-row['times'][0]
 f,fd=old.p.frequency(times,obs[:,1][:,row['mask']>.5].astype(float),4,center,.65)
 C=old.p.neural_template(row,own,n,f);params,initial=old.p.fit_startup(C,row,f,False)
 return own,old.p.correction(C,row,params,'hybrid8_pH1'),params,dict(frequency=f,frequency_prior=center,prefix_steps=n,self_fit=fd,startup=initial,network=di,proxy_prior=True,frequency_search_relative_radius=.65)

def forecast(bundle,row,values,library=True,wide=False):
 own,C,pa,boot=bootstrap(bundle,row,wide)
 if library:
  lib=bundle['libs'][row['q']['shape']];rank=min(24,len(lib['A']));spatial=dict(name='local_rank_available',rank=rank,width=.35,noise=.01,split=False)
  modes,md=old.query_modes(lib,C,row,spatial);C,fit=old.b.previous.fit(C,pa,row,modes,rank,.01)
 else:md=dict(no_newshape_spatial_library=True);fit={}
 observed,od=old.p.observe(C,pa[1],row,values,pa);pred=old.p.hybrid(own,observed)
 assert np.array_equal(pred[:32],own[:32]) and od['scalar_future_readings']==76
 return pred,dict(bootstrap=boot,source=md,fit=fit,observer=od,first8_own_exact=True)

def score_queries(bundle,qs,folder,library=True,wide=False,noise=0.,startup_noise=False):
 rows=[];folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 for index,q in enumerate(qs):
  row=startup(q);values=packets(q,row,noise)
  if startup_noise:
   rng=np.random.default_rng(int(hashlib.sha256(('newshape_start:'+q['id']).encode()).hexdigest()[:8],16));row['fields']=row['fields']+rng.normal(size=row['fields'].shape).astype(np.float32)*.01*np.asarray(old.p.scales(),dtype=np.float32)[None,:,None,None]*row['mask'][None,None]
  pred,di=forecast(bundle,row,values,library,wide)
  if index==0:
   poison=values.copy();poison[79:]+=100.;other,_=forecast(bundle,row,poison,library,wide);assert np.array_equal(pred[:80],other[:80])
  old.p.save(folder/'registry'/(q['id']+'.json'),dict(prediction_before_fullfield_score=True,prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),diagnostic=di,startup_full_fields=4,future_scalar_readings=76))
  with np.load(CACHE/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
  rows.append(dict(id=q['id'],q=q,group=q['family'],horizons=old.p.metrics(pred,truth,row['mask'],row['times'])))
  old.p.save(folder/'status.json',dict(completed=index+1,total=len(qs),utc=old.p.now()))
 groups=['all']+sorted({q['family'] for q in qs});summary={g:{h:old.p.summarize([z['horizons'][h] for z in rows if g=='all' or z['group']==g]) for h in ['8s','40s']} for g in groups}
 report=dict(summary=summary,cases=rows,noise_fraction=noise,startup_noise=startup_noise,role='TRAIN exploration' if qs[0]['role']=='newshape_train' else 'newshape held parameters, first evaluation for locked candidates');old.p.save(folder/'report.json',report);return report
