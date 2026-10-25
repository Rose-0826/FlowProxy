from pathlib import Path
import argparse,hashlib,time,numpy as np,torch,json
from field_model import p,b,DEST,FieldNet,predict,fit_records,held_bootstrap,metadata
METHODS=['frozen_v5','matched_residual16','direct_NN_pH1','direct_NN_UyPcal']

def make_lib(records,folder):
 shape=records[0]['q']['shape'];body=b.body;ref,rs,_=body.to_body(records[0]['C'],records[0]['q'],records[0]['mask']);res=[];sup=[]
 for r in records:
  C,support,_=body.to_body(r['C'],r['q'],r['mask']);T,_=p.gauge_align(r['truthC'],r['C'],r['mask']);T,_,_=body.to_body(T,r['q'],r['mask']);_,angle=p.gauge_align(C,ref,(support&rs&(body.body_mask(shape)>.5)).astype(float));res.append(p.remove_tangent(p.rotate(T-C,angle),p.rotate(C,angle)));sup.append(support)
 res=np.asarray(res);sup=np.asarray(sup);count=sup.sum(0);mean=np.sum(res*sup[:,None,None],axis=0)/np.maximum(count,1)[None,None];filled=np.where(sup[:,None,None],res,mean[None]);filled*=body.body_mask(shape)[None,None,None];A=(filled/p.SCALE*p.HWEIGHT).reshape(len(records),-1)
 from scipy.linalg import eigh
 eig,U=eigh(A@A.T);ix=np.argsort(eig)[::-1][:16];modes=((U[:,ix].T@A)/np.sqrt(len(records))).reshape(16,9,3,128,128)*p.SCALE/p.HWEIGHT;lib=dict(shape=shape,reference=ref,reference_support=rs,modes=modes)
 p.save(folder/(shape+'_baseline_sources.json'),dict(ids=[r['id'] for r in records],groups=sorted({r['group_key'] for r in records}),rank=16,fit_only=True,library='exactprevious noncanonicalwidebody residuallibrary refit on currentfitgroups only',source_SHA=[dict(id=r['id'],C=hashlib.sha256(r['C'].tobytes()).hexdigest(),truthC=hashlib.sha256(r['truthC'].tobytes()).hexdigest()) for r in records]))
 return lib

def data_for(records):
 data={r['id']:r for r in p.dataset()}
 cache=p.ROOT/'data/processed/literature_training_support_20261006'
 for r in records:
  if r['id'] in data:continue
  with np.load(cache/(r['id']+'.npz')) as z:data[r['id']]=dict(id=r['id'],q=r['q'],**{k:z[k] for k in ['fields','times','geom','mask']})
 return data

def summary(result):
 return {m:{g:(p.summarize(group) if (group := [r['horizons']['40s'] for r in rows if g=='all' or r['q']['shape']==g or (g=='thin' and r['q']['D']<.9)]) else {'n': 0}) for g in ['all','cylinder','square','thin']} for m,rows in result.items()}

def run(fold):
 torch.set_num_threads(1);p.protect();folder=DEST/f'fold{fold}';reportdir=folder/'evaluation';reportdir.mkdir();records=metadata();fit=fit_records(fold);held=held_bootstrap(fold);manifest=json.loads((folder/'training_manifest.json').read_text());assert manifest['fit_ids']==[r['id'] for r in fit] and manifest['held_ids']==[r['id'] for r in held]
 saved=torch.load(folder/'field_15000.pt',map_location='cpu',weights_only=False);net=FieldNet(saved['n_in']).cuda();net.load_state_dict(saved['state']);net.eval();libs={s:make_lib([r for r in fit if r['q']['shape']==s],reportdir) for s in ['cylinder','square']};background=p.load(dict(path=str(p.WEIGHT)));priors=p.geometry_priors();data=data_for(records);result={m:[] for m in METHODS};begin=time.perf_counter();guards=[]
 for index,rec in enumerate(held):
  raw=data[rec['id']];row=dict(id=rec['id'],q=rec['q'],fields=raw['fields'][:4].copy(),geom=raw['geom'],mask=raw['mask'],times=raw['times']);own,C,params,boot=p.bootstrap(background,row,priors);assert np.max(np.abs(C-rec['C']))<1e-5 and np.max(np.abs(params-rec['params']))<1e-5
  modes=b.previous.modes_for(libs[row['q']['shape']],C,row);corrected,rd=b.previous.fit(C,params,row,modes,16,.01);nnC=predict(net,row);nnparams,nd=p.fit_startup(nnC,row,params[1],False);ij,_=p.probes(row);values=np.full((160,4),np.nan)
  for step in range(8,160,8):values[step-1]=[raw['fields'][4+step-1,1,i,j] for i,j in ij]
  frozen=[]
  for method in METHODS:
   pa=params
   if method=='frozen_v5':coef=C;di=dict(baseline=True)
   elif method=='matched_residual16':coef=corrected;di=rd
   else:pa=nnparams;coef=p.correction(nnC,row,pa,'hybrid8_pH1' if method.endswith('pH1') else 'hybrid8_UyPcal');di=dict(direct_geometry_structure=True,checkpoint_sha256=p.digest(folder/'field_15000.pt'),startup=nd,calibration=method)
   observer,obs=p.observe(coef,pa[1],row,values,pa);pred=p.hybrid(own,observer);assert np.array_equal(pred[:32],own[:32]) and obs['scalar_future_readings']==76
   if index==0:
    poison=values.copy();poison[79:]+=100.;other,_=p.observe(coef,pa[1],row,poison,pa);assert np.array_equal(p.hybrid(own,other)[:80],pred[:80])
   frozen.append((method,pred,di));guards.append(dict(id=rec['id'],method=method,first8_exact=True,scalar_future=76,initial_case_causal_poison=index==0))
  p.save(reportdir/'registry'/f'{rec["id"]}.json',dict(before_any_held_case_future_field_score=True,fit_group_excluded=True,checkpoint_sha256=p.digest(folder/'field_15000.pt'),candidates=[dict(method=m,sha256=hashlib.sha256(x.tobytes()).hexdigest(),fit=d) for m,x,d in frozen]))
  for method,pred,di in frozen:result[method].append(dict(id=rec['id'],q=rec['q'],fold=fold,horizons=p.metrics(pred,raw['fields'][4:164],row['mask'],row['times'])))
  p.save(reportdir/'partial.json',dict(summary=summary(result),cases=result));p.save(reportdir/'status.json',dict(completed=index+1,total=len(held),utc=p.now()));print('FIELD_EVAL',fold,index+1,rec['id'],flush=True)
 p.save(reportdir/'report.json',dict(role='internalgeometryholdout ofnewstructuremodel andresidualmodule, notentiremodelindependent; backgroundFNO/frequencyprior sawold96',fold=fold,fit_ids=[r['id'] for r in fit],held_ids=[r['id'] for r in held],summary=summary(result),cases=result,guards=guards,wall_seconds=time.perf_counter()-begin,seenDEV64_not_evaluated=True));p.protect();p.save(reportdir/'status.json',dict(status='complete',utc=p.now()));print('FIELD_EVAL_COMPLETE',fold,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);a=ap.parse_args();run(a.fold)
