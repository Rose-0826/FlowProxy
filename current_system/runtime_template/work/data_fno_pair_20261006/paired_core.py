from pathlib import Path
import json,sys,numpy as np,torch,hashlib
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path[:0]=[str(ROOT/x) for x in ['work/periodic_field_pilot_20261006','work/literature_attack_20261006','scripts','src']]+[str(ROOT)]
from field_model import p,b,metadata,template_path
from evaluate_field import data_for
from corrected_benchmark_train_v1 import model_from,protocol,train_loss
from research_v2_reflection_train import ReflectionOperator
from research_v2_state_probe import ProjectedState
from research_v2_outer_support_train import cycle
DEST=ROOT/'outputs/data_fno_pair_20261006';DATA16=ROOT/'outputs/data_expansion16_20261006';CACHE16=ROOT/'data/processed/data_expansion16_20261006'

def records():
 out=[]
 for rec in metadata():
  with np.load(template_path(rec)) as z:out.append({**rec,**{k:z[k] for k in z.files}})
 for rec in json.loads((DATA16/'support116_manifest.json').read_text())['records'][100:]:
  with np.load(DATA16/'train_templates'/(rec['id']+'.npz')) as z:out.append({**rec,**{k:z[k] for k in z.files}})
 assert len(out)==116;return out

def training_data(recs):
 data=data_for(recs[:100]);out=[data[z['id']] for z in recs[:100]]
 for rec in recs[100:]:
  with np.load(CACHE16/(rec['id']+'.npz')) as z:out.append(dict(id=rec['id'],q=rec['q'],**{k:z[k] for k in ['fields','times','geom','mask']}))
 assert [z['id'] for z in out]==[z['id'] for z in recs];return out

def evaluate(model,recs,data,folder):
 priors=p.geometry_priors();updated=[]
 for rec in recs:
  raw=data[rec['id']] if isinstance(data,dict) else next(z for z in data if z['id']==rec['id']);row=dict(id=rec['id'],q=rec['q'],fields=raw['fields'][:4].copy(),geom=raw['geom'],mask=raw['mask'],times=raw['times']);own,C,params,di=p.bootstrap(model,row,priors);updated.append({**rec,'C':C.astype(np.float32),'params':params})
 libs={s:b.build([z for z in updated if z['q']['shape']==s],str(folder/'banks'/s)) for s in ['cylinder','square']};result=[];guards=[]
 qs=p.dev_records()[:56];assert len(qs)==56 and all(q['group']!='latest_seen8' for q in qs)
 for q in qs:
  row=p.startup(q);own,C,pa,boot=p.bootstrap(model,row,priors);modes,md=b.modes(libs[row['q']['shape']],C,row,local=True,split=False);coef,fit=b.previous.fit(C,pa,row,modes,16,.01);values,_=p.get_measurements(q);obs,od=p.observe(coef,pa[1],row,values,pa);pred=p.hybrid(own,obs);assert np.array_equal(pred[:32],own[:32]) and od['scalar_future_readings']==76
  if not result:
   poisoned=values.copy();poisoned[79:]+=100;other,_=p.observe(coef,pa[1],row,poisoned,pa);assert np.array_equal(p.hybrid(own,other)[:80],pred[:80])
  p.save(folder/'registry'/f'{q["id"]}.json',dict(before_future_field_score=True,prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),fit=fit,bootstrap=boot,source=md));
  with np.load(Path(q['cache'])/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
  result.append(dict(id=q['id'],q=row['q'],group=q['group'],horizons=p.metrics(pred,truth,row['mask'],row['times'])));guards.append(dict(id=q['id'],first8_exact=True,scalar_future=76))
 p.save(folder/'report.json',dict(role='SEEN DEV56 selection only; updated FNO and residual templates fitted exclusively on registered TRAIN arm; latestseen8 not rescored during selection; reserved8 not read',summary=p.eval_summary(result),cases=result,guards=guards,train_ids=[z['id'] for z in recs]));
 # Banks are regenerable; keep construction receipts. No true CFD removed.
 for file in (folder/'banks').rglob('*_bank.npz'):assert file.resolve().is_relative_to(DEST.resolve());file.unlink()
 return p.eval_summary(result)
