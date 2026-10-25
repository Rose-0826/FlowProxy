"""Current two-network predictor, omitting unused compatibility banks.

Runtime module copies may be relocated by bootstrap_portable.py. Numerical
functions/weights remain identical to the original selected system.
"""
from pathlib import Path
import os,sys,json,hashlib
import numpy as np
ROOT=Path(os.environ.get('AIC_ROOT','/root/autodl-tmp/flowproxy')).resolve()
sys.path.insert(0,str(ROOT/'work/shape_expansion_20261006'))
import expansion_core as e
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
class Predictor:
 def __init__(self,config):
  self.path=Path(config).resolve();self.config=json.loads(self.path.read_text());base=self.path.parent
  def resolved(p):p=Path(p);return p if p.is_absolute() else base/p
  for b in [self.config['original_shapes'],self.config['new_shapes']]:
   b['checkpoint']=str(resolved(b['checkpoint']))
   for key in ['banks','new_banks']:
    for v in b.get(key,{}).values():v['path']=str(resolved(v['path']))
  # The old frozen-baseline guard only checked an unused historical weight.
  # This runtime guards the actual two weights and all eight deployed banks.
  def protect():
   for b in [self.config['original_shapes'],self.config['new_shapes']]:
    assert digest(b['checkpoint'])==b['checkpoint_sha256']
    for key in ['banks','new_banks']:
     for v in b.get(key,{}).values():assert digest(v['path'])==v['sha256']
  e.old.p.protect=protect;e.setup()
  a=self.config['original_shapes'];self.parent=e.old.load_bundle(a['checkpoint'],a['banks'])
  b=self.config['new_shapes'];self.added=e.old.load_bundle(b['checkpoint'],{})
  for shape,record in b['new_banks'].items():
   p=Path(record['path'])
   with np.load(p) as z:lib={k:z[k] for k in z.files}
   lib.update(shape=shape,source=json.loads(p.with_suffix('.json').read_text())['source']);lib['gram']=lib['A']@lib['A'].T;self.added['libs'][shape]=lib
   st=b['new_priors'][shape];self.added['priors'][shape]=lambda x,st=st:np.full(len(x),st)
 def predict(self,row,values):
  assert row['fields'].shape==(4,3,64,64) and np.isfinite(row['fields']).all()
  assert row['geom'].shape==(4,64,64) and row['mask'].shape==(64,64)
  assert values.shape==(160,4) and np.isfinite(values).sum()==76
  shape=row['q']['shape']
  if shape in ['cylinder','square']:return e.old.forecast(self.parent,row,values,self.config['original_shapes']['spatial'])
  if shape in e.shapes.SPECS:return e.forecast(self.added,row,values)
  raise ValueError('Unsupported known geometry profile: '+shape)
