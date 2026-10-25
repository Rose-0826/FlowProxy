"""Explicit shape-family routing, with four-field startup and 76 Uy readings."""
from expansion_core import *

class Predictor:
 def __init__(self,path=None):
  setup();self.config=json.loads(Path(path or OUT/'selected_system.json').read_text());parent=self.config['original_shapes']
  assert old.p.digest(parent['checkpoint'])==parent['checkpoint_sha256']
  self.parent=old.load_bundle(parent['checkpoint'],parent['banks'])
  from train_expansion import load_candidate
  self.added=load_candidate(self.config['new_shapes'])

 def predict(self,row,values):
  assert row['fields'].shape==(4,3,64,64) and np.isfinite(row['fields']).all()
  assert row['geom'].shape==(4,64,64) and row['mask'].shape==(64,64)
  assert values.shape==(160,4) and np.isfinite(values).sum()==76
  # Only the known geometry profile determines routing. Case ID is unused.
  shape=row['q']['shape']
  if shape in ['cylinder','square']:
   return old.forecast(self.parent,row,values,self.config['original_shapes']['spatial'])
  if shape in shapes.SPECS:return forecast(self.added,row,values)
  raise ValueError('Unsupported geometry profile: '+shape)

def verify():
 model=Predictor();replays=[]
 for q in [next(q for q in old.queries() if q['shape']==shape) for shape in ['cylinder','square']]:
  row=old.p.startup(q);values=old.packets(q,row)[0]
  expected,_=old.forecast(model.parent,row,values,model.config['original_shapes']['spatial']);got,di=model.predict(row,values);assert np.array_equal(expected,got)
  replays.append(dict(id=q['id'],original_branch_bitwise_preserved=True,prediction_sha256=hashlib.sha256(got.tobytes()).hexdigest(),future_scalar_readings=76))
 cfg=json.loads((OUT/'run_config.json').read_text());name=model.config['new_shapes']['name']
 for family in ['ellipse','rectangle','irregular']:
  q=next(q for q in cfg['design']['evaluation'] if q['family']==family);row=startup(q);values=packets(q,row);pred,di=model.predict(row,values);registry=json.loads((OUT/'held'/name/'registry'/(q['id']+'.json')).read_text());digest=hashlib.sha256(pred.tobytes()).hexdigest();assert digest==registry['prediction_sha256']
  replays.append(dict(id=q['id'],held_prediction_replay_bitwise=True,prediction_sha256=digest,future_scalar_readings=di['observer']['scalar_future_readings']))
 reports=json.loads((OUT/'held_comparison.json').read_text())['reports'];assert len(reports)==5
 for report in reports.values():
  assert len(report['cases'])==12
  for g in report['summary']:
   cases=[z for z in report['cases'] if g=='all' or z['group']==g]
   for h in ['8s','40s']:
    actual=old.p.summarize([z['horizons'][h] for z in cases]);assert actual==report['summary'][g][h]
 noise=json.loads((OUT/'noise_report.json').read_text());assert len(noise)==2
 for conditions in noise.values():
  assert len(conditions)==2
  for report in conditions.values():assert len(report['cases'])==12
 old.p.save(OUT/'verification_receipt.json',dict(utc=old.p.now(),entry_replays=replays,held_candidate_count=5,held_cases_each=12,noise_methods=2,noise_conditions_each=2,noise_cases_each=12,held_summaries_recomputed=True,newshape_budget=76,old_network_family_routing_explicit=True,unseen_profiles_evaluated=False))
 print('SYSTEM_ENTRY_VERIFIED',flush=True)

if __name__=='__main__':verify()
