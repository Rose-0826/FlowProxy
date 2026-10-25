from pathlib import Path
import os,sys,json,numpy as np,torch
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path.insert(0,str(ROOT/'work/literature_attack_20261006'))
import local_bank as b
p=b.p;DEST=ROOT/'outputs/periodic_field_pilot_20261006'
from flowproxy.geometry import geometry_channels
from corrected_benchmark_train_v1 import protocol
from research_v4_sparse_observer import constrain

class FieldNet(torch.nn.Module):
 def __init__(self,n_in):
  super().__init__();parts=[torch.nn.Linear(n_in,128),torch.nn.SiLU()]
  for _ in range(4):parts += [torch.nn.Linear(128,128),torch.nn.SiLU()]
  parts.append(torch.nn.Linear(128,27));self.net=torch.nn.Sequential(*parts)
 def forward(self,x):return self.net(x)

def canonical(row):
 q=dict(row['q']);mask=row['mask']
 if q['cy']<0:q['cy']=-q['cy'];mask=mask[::-1].copy()
 return q,mask

def feature(q,mask):
 geo=geometry_channels(protocol()['config'],**q);x=(p.X-q['cx'])/q['D'];y=(p.Y-q['cy'])/q['D'];dist=(p.X-(q['cx']+4*q['D']))**2+(p.Y-q['cy'])**2;dist[mask<=.5]=np.inf;i,j=np.unravel_index(np.argmin(dist),dist.shape)
 columns=[x/12,y/4,p.X/20,p.Y/8,np.full_like(x,(q['Re']-140)/60),np.full_like(x,(q['D']-1)/.25),np.full_like(x,q['cy']),np.clip(geo['sdf_obs']/q['D'],-1,20)/5,geo['sdf_wall']/8,np.full_like(x,float(q['shape']=='cylinder')),np.full_like(x,float(q['shape']=='square')),np.full_like(x,x[i,j]/4),np.full_like(x,y[i,j])]
 for v in [(x-x[i,j])/12,y/4]:
  for f in [1,2,4,8,16]:columns.extend([np.sin(np.pi*f*v),np.cos(np.pi*f*v)])
 return np.stack(columns,-1).reshape(4096,-1).astype(np.float32),(i,j)

def label(rec):
 q,mask=canonical(rec);T=rec['truthC'].astype(float).copy()
 if rec['q']['cy']<0:T=b.mirrored(T)
 X,ij=feature(q,mask);complex_anchor=T[1,1,ij[0],ij[1]]-1j*T[2,1,ij[0],ij[1]];assert abs(complex_anchor)>1e-6
 angle=-np.angle(complex_anchor);T=p.rotate(T,angle);assert abs(T[2,1,ij[0],ij[1]])<1e-8 and T[1,1,ij[0],ij[1]]>0
 rms=np.sqrt(np.mean(np.sum((T*p.HWEIGHT)**2,axis=0)[:,mask>.5],axis=1));rms=np.maximum(rms,1e-6)
 shifted=T.copy();shifted[0,0]-=1.;Y=(shifted/p.SCALE*p.HWEIGHT).reshape(27,-1).T.astype(np.float32)
 weights=np.tile(p.SCALE[0,:,0,0]/rms,9).astype(np.float32);active=mask.ravel()>.5
 return X[active],Y[active],np.broadcast_to(weights,(active.sum(),27)).copy(),dict(id=rec['id'],q=rec['q'],canonical_q=q,anchor_grid=list(map(int,ij)),phase_angle=float(angle),channel_rms=rms.tolist(),full_trainingtruthC_used=True,body_regrid_label_used=False)

def predict(net,row):
 q,mask=canonical(row);X,ij=feature(q,mask)
 with torch.inference_mode():a=net(torch.as_tensor(X,device='cuda')).cpu().numpy().T.reshape(9,3,64,64).astype(float)
 C=a*p.SCALE/p.HWEIGHT;C[0,0]+=1.;C=constrain(C,mask)
 if row['q']['cy']<0:C=b.mirrored(C)
 return constrain(C,row['mask'])

def metadata():
 source=ROOT/'outputs/literature_attack_20261006/support100/training_manifest.json';records=json.loads(source.read_text())['records'];foldmap=p.folds(records)
 for r in records:r['fold']=foldmap[r['group_key']]
 assert len(records)==100;return records

def template_path(rec):
 old=p.DEST/'train_templates'/(rec['id']+'.npz')
 return old if old.exists() else ROOT/'outputs/literature_attack_20261006/support100/train_templates'/(rec['id']+'.npz')

def fit_records(fold):
 result=[]
 for r in metadata():
  if r['fold']==fold:continue
  with np.load(template_path(r)) as z:result.append({**r,**{k:z[k] for k in z.files}})
 return result

def held_bootstrap(fold):
 result=[]
 for r in metadata():
  if r['fold']!=fold:continue
  with np.load(template_path(r)) as z:result.append(dict(**r,C=z['C'],params=z['params']))
 return result
