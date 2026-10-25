"""Finite public experimental velocity pilot; original CFD system stays frozen.

Adaptation of startup/calibrated spatial residual/causal phase-clock mechanisms,
not a zero-shot validation of the three-channel CFD predictor. External FLRONet
is executed from the authors' unmodified source, with a causal input adapter.
"""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,copy,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import scipy.io as sio
from scipy.ndimage import map_coordinates
from scipy.signal import welch
import torch
import torch.nn.functional as F
ROOT=Path('/root/autodl-tmp/flowproxy');WORK=ROOT/'work/evidence_upgrade_20261007';OUT=ROOT/'outputs/evidence_upgrade_20261007/piv'
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'src'),str(WORK/'FLRONet')]
from flowproxy.models.fno import FNO2d
from model.flronet import FLRONetFNO
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DT=.05;H=64;N=160;SEED=20261007;READOUT=list(range(7,159,8))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()));tmp.replace(p)
def status(stage,**kw):save(OUT/'status.json',dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw));print(stage,kw,flush=True)

def prepare():
 status('preparing_experimental_velocity')
 raw=ROOT/'data/experimental/zenodo_20765567/cylinder_vel.mat';z=sio.loadmat(raw)
 x,y=z['x'],z['y'];u,v=z['u'],z['v'];assert u.shape==v.shape==(135,80,8000)
 # Locate physical x/y directions from coordinate grids, then resample only
 # unmasked PIV vectors. No pressure labels or CFD geometry are fabricated.
 xaxis=int(np.argmax([np.abs(np.diff(x,axis=a)).mean() for a in [0,1]]));yaxis=1-xaxis
 if xaxis==0:x=x.T;y=y.T;u=u.transpose(1,0,2);v=v.transpose(1,0,2)
 if x[0,-1]<x[0,0]:x=x[:,::-1];y=y[:,::-1];u=u[:,::-1];v=v[:,::-1]
 if y[-1,0]<y[0,0]:x=x[::-1];y=y[::-1];u=u[::-1];v=v[::-1]
 yy,xx=np.meshgrid(np.linspace(0,u.shape[0]-1,H),np.linspace(0,u.shape[1]-1,H),indexing='ij')
 coords=np.array([yy,xx]);native_valid=np.any((u[:,:,:4800]!=0)|(v[:,:,:4800]!=0),axis=2)
 mask=map_coordinates(native_valid.astype(float),coords,order=1,mode='nearest')>.999
 a=np.empty((8000,2,H,H),np.float32)
 for t in range(8000):
  a[t,0]=map_coordinates(u[:,:,t],coords,order=1,mode='nearest')*mask
  a[t,1]=map_coordinates(v[:,:,t],coords,order=1,mode='nearest')*mask
 del z,u,v
 mean=a[:4800].mean(axis=0);scale=np.sqrt(np.mean((a[:4800]-mean[None])[:,:,mask]**2,axis=(0,2)));assert np.all(scale>0)
 b=((a-mean[None])/scale[None,:,None,None])*mask[None,None]
 variance=b[:4800,1].var(axis=0);probes=[]
 # One high-variance valid point in each quadrant, using TRAIN only.
 for ry,rx in [(slice(0,32),slice(0,32)),(slice(0,32),slice(32,64)),(slice(32,64),slice(0,32)),(slice(32,64),slice(32,64))]:
  score=variance[ry,rx].copy();score[~mask[ry,rx]]=-1
  i,j=np.unravel_index(score.argmax(),score.shape);probes.append((i+(ry.start or 0),j+(rx.start or 0)))
 assert len(set(probes))==4 and all(mask[i,j] for i,j in probes)
 trace=np.stack([b[:4800,1,i,j] for i,j in probes],axis=1);fs,ps=welch(trace,fs=20,nperseg=2048,axis=0)
 use=(fs>.02)&(fs<3.0);f=float(fs[use][ps[use].sum(axis=1).argmax()]);assert f>0
 trainstarts=list(range(0,4800-164+1,164));valstarts=list(range(5000,6000-164+1,164));teststarts=list(range(6200,8000-164+1,164));assert len(teststarts)==10
 xi=map_coordinates(x,coords,order=1);yi=map_coordinates(y,coords,order=1)
 np.savez_compressed(OUT/'prepared_velocity.npz',fields=b,mean=mean,scale=scale,mask=mask,x=xi,y=yi,probes=np.array(probes),frequency=f)
 config=dict(source='https://zenodo.org/records/20765567',DOI='10.5281/zenodo.20765567',source_md5='4cc876439c48f7970afa477ea17d217a',raw_sha256=sha(raw),Re=413,dt_seconds=DT,
  startup_full_velocity_frames=4,future_scalar_readings=76,future_measured_component='Uy',future_readout_steps=READOUT,forecast_seconds=8.0,
  train_frames=[0,4800],validation_frames=[5000,6000],test_frames=[6200,8000],gap_each_frames=200,
  trainstarts=trainstarts,valstarts=valstarts,teststarts=teststarts,test_overlap=False,train_test_recording='Same single experimental recording; time-block holdout, not a new experimental run or unknown Reynolds number.',
  native_grid=[135,80],grid=[64,64],resampling='Bilinear velocity interpolation; retain cells whose TRAIN-derived valid-vector support is >.999; coordinates physical meters. Common grid for all methods.',
  normalization='TRAIN temporal mean field and per-channel fluctuation RMS only',channel_scales_m_per_s=scale.tolist(),probes_ij=probes,probe_selection='TRAIN Uy variance, one fixed sensor per quadrant',TRAIN_frequency_Hz=f,
  original_CFD_system='Unmodified; pilot uses separately trained velocity-only FNO, TRAIN spatial residual modes and causal phase/frequency observer.',
  pressure_available=False,main_CFD_R_thresholds_applicable=False,steps_each=2000,seed=SEED,supervised_target_frames_each=32000,FNO_training='batch4, 4 self-fed steps per sample',FLRONet_training='batch4, 4 queried outputs per sample; each shares only measurements earlier than its earliest query',
  official_baseline='FLRONetFNO, unmodified official source, causal wrapper retains 4 startup full velocity fields plus latest 4 available Uy-only Voronoi frames. No future measurements relative to queried output.',
  baseline_configuration=dict(channels=2,layers=4,modes=[12,12],width=24,stacked_networks=2,grid=[64,64],capacity='Compact common-width configuration, not the paper default width64/modes24,48 nor a reproduction of its headline score.'),
  validation_selection='Both networks select 1000/2000 checkpoint on normalized MSE at future indices15,63,127,159 in the same six validation blocks.',
  upstream_commit=subprocess.check_output(['git','-C',str(WORK/'FLRONet'),'rev-parse','HEAD'],text=True).strip(),upstream_license='GPL-3.0',source_sha256=sha(__file__))
 save(OUT/'config.json',config);return b,mean,scale,mask,np.array(probes),f,config

def load():
 with np.load(OUT/'prepared_velocity.npz') as z:return z['fields'],z['mean'],z['scale'],z['mask'],z['probes'],float(z['frequency']),json.loads((OUT/'config.json').read_text())
def basis(t,f):
 t=np.asarray(t);return np.stack([np.ones_like(t),np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t),np.cos(4*np.pi*f*t),np.sin(4*np.pi*f*t)],axis=-1)
def make_fno():return FNO2d(8,2,n_modes=(12,12),hidden_channels=24,n_layers=4,lifting_channels=48,projection_channels=48).cuda()
def roll(model,start,steps=160):
 hist=start.copy();out=[]
 with torch.no_grad():
  for t in range(steps):
   p=model(torch.as_tensor(hist.reshape(1,8,H,H),device='cuda')).cpu().numpy()[0]*MASK[None]
   out.append(p);hist=np.concatenate([hist[1:],p[None]])
 return np.asarray(out)
def val_fno(m,starts):
 errs=[]
 for s in starts:
  p=roll(m,B[s:s+4],160);ix=[15,63,127,159];errs.append(float(np.mean((p[ix]-B[s+4:s+164][ix])[:,:,MASK]**2)))
 return float(np.mean(errs))

def train_fno(cfg):
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);m=make_fno();opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-5)
 data=torch.as_tensor(B[:4800],device='cuda');mask=torch.as_tensor(MASK,device='cuda');history=[];best=float('inf')
 for step in range(1,2001):
  idx=rng.integers(0,4792,size=4);hist=torch.stack([data[i:i+4] for i in idx]);opt.zero_grad();loss=0
  for k in range(4):
   p=m(hist.reshape(4,8,H,H))*mask[None,None];truth=torch.stack([data[i+4+k] for i in idx]);loss=loss+(p-truth)[:,:,mask].square().mean()/4
   hist=torch.cat([hist[:,1:],p[:,None]],dim=1)
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step%100==0:history.append(dict(step=step,train_loss=float(loss.detach())));save(OUT/'fno_history.json',history);status('training_PIV_FNO',step=step,total=2000)
  if step in [1000,2000]:
   m.eval();err=val_fno(m,cfg['valstarts']);history[-1]['validation_common_horizons_MSE']=err
   if err<best:best=err;torch.save(m.state_dict(),OUT/'velocity_fno.pt')
   m.train()
 save(OUT/'fno_history.json',history);m.load_state_dict(torch.load(OUT/'velocity_fno.pt',weights_only=True));m.eval();return m

def own_coeff(model,s,f):
 own=roll(model,s,80);obs=np.concatenate([s,own]);t=np.arange(84)*DT
 C=np.linalg.lstsq(basis(t,f),obs.reshape(84,-1),rcond=1e-6)[0].reshape(5,2,H,H)
 return own,C
def build_library(m,cfg,f):
 residual=[];truthC=[]
 for j,s in enumerate(cfg['trainstarts']):
  own,C=own_coeff(m,B[s:s+4],f);target=B[s:s+164]
  Ct=np.linalg.lstsq(basis(np.arange(164)*DT,f),target.reshape(164,-1),rcond=1e-6)[0].reshape(5,2,H,H)
  residual.append((Ct-C).reshape(-1));truthC.append(Ct)
  status('building_PIV_TRAIN_residuals',completed=j+1,total=len(cfg['trainstarts']))
 A=np.asarray(residual);_,sv,V=np.linalg.svd(A,full_matrices=False);rank=min(24,len(A));modes=V[:rank].reshape(rank,5,2,H,H)
 np.savez_compressed(OUT/'velocity_residual_modes.npz',modes=modes,singular_values=sv,truth_coefficients=np.array(truthC),source_starts=cfg['trainstarts'])
 return modes,np.array(truthC)

def calibrate(C,start,f,modes):
 design=basis(np.arange(4)*DT,f);prediction=np.einsum('tk,kchw->tchw',design,C)
 X=np.einsum('tk,rkchw->tchwr',design,modes).reshape(-1,len(modes))
 w=np.tile(MASK.reshape(-1),8);X=X[w];r=(start-prediction).reshape(-1)[w]
 reg=.01**2*len(r);coef=np.linalg.solve(X.T@X+reg*np.eye(len(modes)),X.T@r)
 return C+np.einsum('r,rkchw->kchw',coef,modes)
def packets(start,noise=0,seed=0):
 values=np.full((N,4),np.nan);rng=np.random.default_rng(seed)
 for j in READOUT:values[j]=[B[start+4+j,1,i,k] for i,k in PROBES]
 if noise:values+=rng.normal(size=values.shape)*noise
 assert np.isfinite(values).sum()==76;return values

def observed(C,f,values):
 out=[];phase=0.;freq=f;P=np.diag([.02,.001**2]);read=0
 cp=C[:,1,PROBES[:,0],PROBES[:,1]]
 for j in range(N):
  t=(j+4)*DT;theta=2*np.pi*freq*t+phase
  bs=np.array([1,np.cos(theta),np.sin(theta),np.cos(2*theta),np.sin(2*theta)])
  out.append(np.einsum('k,kchw->chw',bs,C)*MASK[None])
  P=P+np.diag([1e-5,1e-8])
  if np.isfinite(values[j]).all():
   pred=bs@cp;der=np.array([0,-np.sin(theta),np.cos(theta),-2*np.sin(2*theta),2*np.cos(2*theta)])@cp
   J=np.stack([der,der*2*np.pi*t],axis=1);R=np.eye(4)*.05**2;K=np.linalg.solve(J@P@J.T+R,J@P).T
   update=K@(values[j]-pred);phase+=float(np.clip(update[0],-.15,.15));freq=float(np.clip(freq+update[1],f*.85,f*1.15));P=(np.eye(2)-K@J)@P;P=(P+P.T)/2;read+=4
 return np.asarray(out,dtype=np.float32),dict(scalar_readings=read,final_frequency_Hz=freq,final_phase_rad=phase)
def hybrid(m,start,values,f,modes,use_library=True):
 own,C=own_coeff(m,start,f)
 if use_library:C=calibrate(C,start,f,modes)
 ob,di=observed(C,f,values);p=ob.copy();p[:32]=own[:32]
 for j in range(32,80):w=(j-31)/49;p[j]=(1-w)*own[j]+w*ob[j]
 return p,di

def sparse_input(s,j,probe_frames,device=True):
 # Query output j, emitted before measurement at j. Only earlier packets.
 times=[-3,-2,-1,0];frames=[B[s+k] for k in range(4)]
 available=[k for k in READOUT if k<j];recent=available[-4:]
 while len(recent)<4:recent.insert(0,None)
 for k in recent:
  if k is None:times.append(0);frames.append(np.zeros((2,H,H),np.float32))
  else:times.append(k+1);frames.append(probe_frames[k])
 a=np.asarray(frames,dtype=np.float32);tt=np.array(times,np.float32)*DT
 if device:return torch.as_tensor(tt[None],device='cuda'),torch.as_tensor(a[None],device='cuda'),torch.tensor([[(j+1)*DT]],device='cuda')
 return tt,a
def embeddings(values):
 yy,xx=np.meshgrid(np.arange(H),np.arange(H),indexing='ij');distance=(yy[None]-PROBES[:,0,None,None])**2+(xx[None]-PROBES[:,1,None,None])**2;nearest=distance.argmin(axis=0)
 output={}
 for k in READOUT:
  frame=np.zeros((2,H,H),np.float32);frame[1]=values[k][nearest]*MASK;output[k]=frame
 return output
def make_flro():return FLRONetFNO(n_channels=2,n_fno_layers=4,n_hmodes=12,n_wmodes=12,embedding_dim=24,n_stacked_networks=2).cuda()
def flro_val(m,cfg):
 errs=[]
 with torch.no_grad():
  for s in cfg['valstarts']:
   emb=embeddings(packets(s))
   for j in [15,63,127,159]:
    st,sv,ft=sparse_input(s,j,emb);p=m(st,sv,ft)[0,0]*TMASK[None];truth=torch.as_tensor(B[s+4+j],device='cuda');errs.append(float((p-truth)[:,TMASK].square().mean()))
 return float(np.mean(errs))
def train_flro(cfg):
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);m=make_flro();opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-5);history=[];best=float('inf')
 for step in range(1,2001):
  sts=[];svs=[];fts=[];targets=[]
  for _ in range(4):
   s=int(rng.integers(0,4800-164));j=int(rng.integers(0,157));emb=embeddings(packets(s));st,sv=sparse_input(s,j,emb,device=False)
   sts.append(st);svs.append(sv);fts.append(np.arange(j+1,j+5,dtype=np.float32)*DT);targets.append(B[s+4+j:s+8+j])
  st=torch.as_tensor(np.array(sts),device='cuda');sv=torch.as_tensor(np.array(svs),device='cuda');ft=torch.as_tensor(np.array(fts),device='cuda');truth=torch.as_tensor(np.array(targets),device='cuda')
  opt.zero_grad();p=m(st,sv,ft)*TMASK[None,None,None];loss=(p-truth)[:,:,:,TMASK].square().mean()
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step%100==0:history.append(dict(step=step,train_loss=float(loss.detach())));save(OUT/'FLRONet_history.json',history);status('training_official_FLRONet_causal_adapter',step=step,total=2000)
  if step in [1000,2000]:
   m.eval();err=flro_val(m,cfg);history[-1]['validation_MSE']=err
   if err<best:best=err;torch.save(m.state_dict(),OUT/'FLRONet.pt')
   m.train()
 save(OUT/'FLRONet_history.json',history);m.load_state_dict(torch.load(OUT/'FLRONet.pt',weights_only=True));m.eval();return m
def predict_flro(m,s,values):
 emb=embeddings(values);output=[]
 with torch.no_grad():
  for j in range(N):st,sv,ft=sparse_input(s,j,emb);output.append((m(st,sv,ft)[0,0]*TMASK[None]).cpu().numpy())
 return np.array(output)
def metric(pred,truth):
 # Per-channel relative L2 on physical velocities, plus fluctuation-normalized
 # error to avoid letting Ux mean flow conceal wake reconstruction error.
 phys_err=(pred-truth)*SCALE[None,:,None,None];physical_truth=truth*SCALE[None,:,None,None]+MEAN[None]
 e=np.sqrt(np.sum(phys_err[:,:,MASK]**2,axis=2)/np.maximum(np.sum(physical_truth[:,:,MASK]**2,axis=2),1e-16))
 ef=np.sqrt(np.mean((pred-truth)[:,:,MASK]**2,axis=2))
 return dict(mean_relative_L2=e.mean(axis=0).tolist(),end_relative_L2=e[-1].tolist(),max_relative_L2=e.max(axis=0).tolist(),mean_TRAIN_fluctuation_RMS_error=ef.mean(axis=0).tolist())
def evaluate(m,flro,modes,cfg,f):
 rows=[];saved={}
 for n,s in enumerate(cfg['teststarts']):
  values=packets(s);start=B[s:s+4].copy();truth=B[s+4:s+164]
  methods={'adapted_FNO_free_rollout':roll(m,start),'adapted_mechanisms':hybrid(m,start,values,f,modes)[0],'without_TRAIN_residual_modes':hybrid(m,start,values,f,modes,False)[0],'official_FLRONet_causal_adapter':predict_flro(flro,s,values)}
  changed=values.copy();changed[79:]+=100;poison=hybrid(m,start,changed,f,modes)[0];assert np.array_equal(methods['adapted_mechanisms'][:80],poison[:80])
  other=predict_flro(flro,s,changed);assert np.array_equal(methods['official_FLRONet_causal_adapter'][:80],other[:80])
  for name,p in methods.items():
   assert p.shape==truth.shape and np.isfinite(p).all();target=OUT/'test_predictions'/name/(str(s)+'.npz');target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,prediction=p)
   rows.append(dict(start_frame=s,method=name,future_scalar_readings=76,metrics=metric(p,truth),prediction_sha256=hashlib.sha256(p.tobytes()).hexdigest(),file_sha256=sha(target)))
  if n==0:
   saved.update(truth=truth[[31,79,159]],mask=MASK,mean=MEAN,scale=SCALE,probes=PROBES)
   for name,p in methods.items():saved[name]=p[[31,79,159]]
  save(OUT/'test_partial.json',rows);status('PIV_time_block_holdout',completed=n+1,total=len(cfg['teststarts']))
 summary={}
 for name in methods:
  r=[z['metrics'] for z in rows if z['method']==name];summary[name]={k:np.mean([z[k] for z in r],axis=0).tolist() for k in r[0]};summary[name]['worst_block_channel_mean_relative_L2']=np.max([z['mean_relative_L2'] for z in r],axis=0).tolist()
 np.savez_compressed(OUT/'field_snapshots.npz',**saved)
 report=dict(status='COMPLETE',summary=summary,cases=rows,config=cfg,models=dict(FNO_sha256=sha(OUT/'velocity_fno.pt'),FLRONet_sha256=sha(OUT/'FLRONet.pt')),causal_future_poison_checks=20,
  claim='Separate velocity-only experimental adaptation on a chronological held-out section of one Re413 recording. Does not validate pressure, unknown shapes, original frozen CFD network, or a new experimental realization.',steps=160,seconds=8.0)
 save(OUT/'report.json',report);return report

def main():
 global B,MEAN,SCALE,MASK,PROBES,TMASK
 B,MEAN,SCALE,MASK,PROBES,f,cfg=load() if (OUT/'prepared_velocity.npz').exists() and (OUT/'config.json').exists() else prepare();TMASK=torch.as_tensor(MASK,device='cuda')
 fno_done=(OUT/'fno_history.json').exists() and json.loads((OUT/'fno_history.json').read_text())[-1]['step']==2000
 if (OUT/'velocity_fno.pt').exists() and fno_done:m=make_fno();m.load_state_dict(torch.load(OUT/'velocity_fno.pt',weights_only=True));m.eval()
 else:m=train_fno(cfg)
 if (OUT/'velocity_residual_modes.npz').exists():
  with np.load(OUT/'velocity_residual_modes.npz') as z:modes=z['modes']
 else:modes,_=build_library(m,cfg,f)
 flro_done=(OUT/'FLRONet_history.json').exists() and json.loads((OUT/'FLRONet_history.json').read_text())[-1]['step']==2000
 if (OUT/'FLRONet.pt').exists() and flro_done:flro=make_flro();flro.load_state_dict(torch.load(OUT/'FLRONet.pt',weights_only=True));flro.eval()
 else:flro=train_flro(cfg)
 save(OUT/'selection_lock.json',dict(model_SHA=[sha(OUT/'velocity_fno.pt'),sha(OUT/'FLRONet.pt')],test_fullfields_not_used_in_training_or_validation=True,source_sha256=sha(__file__),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 r=evaluate(m,flro,modes,cfg,f);status('COMPLETE',summary=r['summary'])
if __name__=='__main__':
 try:main()
 except Exception:
  save(OUT/'failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
