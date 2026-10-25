from pathlib import Path
import os,sys,json,time,hashlib,numpy as np,torch
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path.insert(0,str(ROOT/'work/data_fno_pair_20261006'))
from paired_core import *
from corrected_benchmark_train_v1 import field_loss_steps
OUT=ROOT/'outputs/practical_refine_20261006';PREV=ROOT/'outputs/practical_fno_next_20261006';FRESH_CACHE=ROOT/'data/processed/practical_fresh8_20261006'

def setup():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p.protect()

def queries():
 qs=p.dev_records();add=json.loads((PREV/'fresh8_dataset.json').read_text())['cases'];qs=qs+[{**q,'cache':str(FRESH_CACHE),'group':'recent8_development'} for q in add];assert len(qs)==72;return qs

def packets(q,row):
 if q['group']!='recent8_development':clean,noisy=p.get_measurements(q)
 else:
  ij,_=p.probes(row);clean=np.full((160,4),np.nan)
  with np.load(FRESH_CACHE/(q['id']+'.npz')) as z:
   for step in range(8,160,8):clean[step-1]=[z['fields'][4+step-1,1,i,j] for i,j in ij]
  rng=np.random.default_rng(int(hashlib.sha256(('refine_future:'+q['id']).encode()).hexdigest()[:8],16));noisy=clean+rng.normal(size=clean.shape)*.01*p.scales()[1]
 out=[]
 for a in [clean,noisy]:
  v=np.full((160,4),np.nan);v[7:159:8]=a[7:159:8];assert np.isfinite(v).sum()==76;out.append(v)
 return out

def tuned_observe(C,f,row,values,params,sensor_sigma=.05,freq_q=.0005):
 theta=float(params[0]);omega=2*np.pi*f;omega0=omega;ij,_=p.probes(row);H=np.stack([C[:,1,i,j] for i,j in ij],0);reader=p.CausalMeasurements(values,8);R=np.eye(4)*(sensor_sigma*p.scales()[1])**2;z=np.array([theta,omega]);P=np.diag([np.deg2rad(5.)**2,(.02*omega)**2]);A=np.array([[1.,.25],[0.,1.]]);Q=np.diag([np.deg2rad(.5)**2,(freq_q*omega)**2])*.25;predictions=[];phases=[];updates=[]
 for step in range(1,161):
  z=A@z;P=A@P@A.T+Q;bb,_=p.weights(z[0]);phases.append(float(z[0]));predictions.append((bb@C.reshape(9,-1)).reshape(3,64,64))
  if step%8==0 and step<160:
   y=reader.read(step);bb,d=p.weights(z[0]);h=np.c_[H@d,np.zeros(4)];gain=np.linalg.solve(h@P@h.T+R,h@P).T;residual=y-H@bb;delta=gain@residual;delta[0]=np.clip(delta[0],-np.pi/2,np.pi/2);z+=delta;z[1]=np.clip(z[1],.8*omega0,1.2*omega0);J=np.eye(2)-gain@h;P=J@P@J.T+gain@R@gain.T;updates.append(dict(step=step,frequency=float(z[1]/(2*np.pi)),sensor_residual_norm=float(np.linalg.norm(residual))))
  assert np.isfinite(P).all()
 pred=np.asarray(predictions,np.float32)*row['mask'][None,None];return pred,dict(scalar_future_readings=4*len(reader.access),observed_future_steps=reader.access,phases=phases,updates=updates,every_field_emitted_before_current_time_measurement=True,sensor_sigma=sensor_sigma,freq_q=freq_q)

def load_bundle(checkpoint,banks):
 base,saved=model_from(checkpoint,protocol()['config'],seed=1401);model=ProjectedState(ReflectionOperator(base),True);libs={}
 for shape,info in banks.items():
  path=Path(info['path']);assert p.digest(path)==info['sha256']
  with np.load(path) as z:bank={k:z[k] for k in z.files}
  bank.update(shape=shape,source=json.loads(path.with_suffix('.json').read_text())['source']);bank['gram']=bank['A']@bank['A'].T;libs[shape]=bank
 return dict(model=model,libs=libs,priors=p.geometry_priors())

def primary_bundle():
 cfg=json.loads((OUT/'run_config.json').read_text());assert p.digest(cfg['primary_checkpoint'])==cfg['primary_checkpoint_sha256'];return load_bundle(cfg['primary_checkpoint'],cfg['primary_banks'])

def summary(rows):
 select={'all72':lambda z:True,'all64':lambda z:z['group']!='recent8_development','DEV56':lambda z:z['group']=='DEV56','latest_seen8':lambda z:z['group']=='latest_seen8','recent8_development':lambda z:z['group']=='recent8_development'}
 # historical p.dev_records uses DEV56 label as a union of several subgroups.
 select['DEV56']=lambda z:z['group'] not in ['latest_seen8','recent8_development']
 return {g:{h:p.summarize([z['horizons'][h] for z in rows if fn(z)]) for h in ['8s','40s']} for g,fn in select.items()}

def corrected(bundle,row,values,clock=None):
 own,C,pa,boot=p.bootstrap(bundle['model'],row,bundle['priors']);modes,md=b.modes(bundle['libs'][row['q']['shape']],C,row,local=True,split=False);coef,fit=b.previous.fit(C,pa,row,modes,16,.01)
 obs,od=p.observe(coef,pa[1],row,values,pa) if clock is None else tuned_observe(coef,pa[1],row,values,pa,clock['sensor_sigma'],clock['freq_q']);pred=p.hybrid(own,obs);assert np.array_equal(pred[:32],own[:32]) and od['scalar_future_readings']==76;return pred,dict(bootstrap=boot,source=md,fit=fit,observer=od)

def build_evaluate(model,recs,data,folder):
 priors=p.geometry_priors();updated=[]
 for rec,raw in zip(recs,data):
  row=dict(id=rec['id'],q=rec['q'],fields=raw['fields'][:4].copy(),geom=raw['geom'],mask=raw['mask'],times=raw['times']);own,C,pa,di=p.bootstrap(model,row,priors);updated.append({**rec,'C':C.astype(np.float32),'params':pa})
 libs={s:b.build([z for z in updated if z['q']['shape']==s],str(folder/'banks'/s)) for s in ['cylinder','square']};bundle=dict(model=model,libs=libs,priors=priors);rows=[]
 for index,q in enumerate(queries()):
  row=p.startup(q);clean,_=packets(q,row);pred,di=corrected(bundle,row,clean)
  if index==0:
   poison=clean.copy();poison[79:]+=100.;other,_=corrected(bundle,row,poison);assert np.array_equal(pred[:80],other[:80])
  p.save(folder/'registry'/(q['id']+'.json'),dict(prediction_before_fullfield_score=True,prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),diagnostic=di))
  with np.load(Path(q['cache'])/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
  rows.append(dict(id=q['id'],q=row['q'],group=q['group'],horizons=p.metrics(pred,truth,row['mask'],row['times'])));p.save(folder/'status.json',dict(stage='development72',completed=index+1,total=72,utc=p.now()))
 s=summary(rows);banks={shape:dict(path=str(folder/'banks'/shape/(shape+'_bank.npz')),sha256=p.digest(folder/'banks'/shape/(shape+'_bank.npz'))) for shape in libs};p.save(folder/'report.json',dict(summary=s,cases=rows,banks=banks,role='all72 SEEN development; only TRAIN100 used to fit network/banks'));return s,banks
