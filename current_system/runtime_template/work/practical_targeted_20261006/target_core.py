from pathlib import Path
import sys
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path.insert(0,str(ROOT/'work/practical_refine_20261006'))
import refine_core as previous
previous.OUT=ROOT/'outputs/practical_targeted_20261006'
from refine_core import *
OUT=previous.OUT

def query_modes(lib,C,row,config):
 if config['split']:return b.modes(lib,C,row,local=True,split=True)
 can,support,_=b.positive(C,row['q'],row['mask']);valid=support&lib['reference_support']&(b.body.body_mask(lib['shape'])>.5);_,angle=p.gauge_align(can,lib['reference'],valid.astype(float));n=len(lib['A']);dist=np.sum((lib['features']-b.features(row['q']))**2,axis=1);raw=np.exp(-dist/(2*config['width']**2));ww=.95*raw/max(raw.sum(),1e-20)+.05/n;sw=np.sqrt(ww);g=lib['gram']*sw[:,None]*sw[None];eig,U=b.eigh(g);idx=np.argsort(eig)[::-1][:config['rank']];z=((U[:,idx].T*sw[None])@lib['A']).reshape(config['rank'],9,3,128,128);z=z*p.SCALE[None]/p.HWEIGHT[None];result=[]
 for mode in z:
  raw=p.rotate(mode,-angle)
  if row['q']['cy']<0:raw=b.mirrored(raw)
  physical=b.body.to_physical(raw,row['q'],row['mask'],delta=True);result.append(p.delta_constraints(p.remove_tangent(physical,C),row['mask']))
 return np.asarray(result),dict(configuration=config,weights=ww.tolist(),sources=[s['id'] for s in lib['source']])

def forecast(bundle,row,values,spatial=None):
 if spatial is None:return previous.corrected(bundle,row,values,None)
 own,C,pa,boot=p.bootstrap(bundle['model'],row,bundle['priors']);modes,md=query_modes(bundle['libs'][row['q']['shape']],C,row,spatial);coef,fit=b.previous.fit(C,pa,row,modes,spatial['rank'],spatial['noise']);obs,od=p.observe(coef,pa[1],row,values,pa);pred=p.hybrid(own,obs);assert np.array_equal(pred[:32],own[:32]) and od['scalar_future_readings']==76;return pred,dict(bootstrap=boot,source=md,fit=fit,observer=od)
