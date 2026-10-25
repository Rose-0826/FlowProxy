"""Portable replay of post-confirmation clock diagnostics. NumPy only.
Contexts contain frozen startup-derived spatial C and FNO48-frame prefix;
this is an observer/control replay, not a substitute for run_demo.py's network.
No future truth is read until predictions have been constructed.
"""
from pathlib import Path
import numpy as np,json,hashlib,argparse,os,sys
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def block_diag(*parts):
 out=np.zeros((sum(x.shape[0] for x in parts),sum(x.shape[1] for x in parts)));i=j=0
 for x in parts:out[i:i+x.shape[0],j:j+x.shape[1]]=x;i+=x.shape[0];j+=x.shape[1]
 return out


def weights(theta):
    k=np.arange(1,5);b=np.empty(9);b[0]=1.;b[1::2]=np.cos(k*theta);b[2::2]=np.sin(k*theta)
    d=np.empty(9);d[0]=0.;d[1::2]=-k*np.sin(k*theta);d[2::2]=k*np.cos(k*theta)
    return b,d

def probes(row):
    q=row['q'];xx,yy=np.meshgrid(np.linspace(-8,20,64),np.linspace(-8,8,64))
    chosen=[];record=[]
    for offset in [2.,4.,6.,8.]:
        x=q['cx']+offset*q['D'];y=q['cy'];dist=(xx-x)**2+(yy-y)**2
        dist[row['mask']<=.5]=np.inf
        i,j=np.unravel_index(np.argmin(dist),dist.shape);assert np.isfinite(dist[i,j])
        chosen.append((int(i),int(j)));record.append({'requested':[x,y],'grid':[int(i),int(j)],'actual':[float(xx[i,j]),float(yy[i,j])]})
    assert len(set(chosen))==4
    return chosen,record

def hybrid(own,pred):
    """Exact original autonomous fields for first8s, fixed4s transition afterwards."""
    out=pred.copy();out[:32]=own[:32]
    for j in range(32,48):
        alpha=(j+1-32)/16
        out[j]=(1-alpha)*own[j]+alpha*pred[j]
    assert np.array_equal(out[:32],own[:32])
    return out

UY_SCALE=0.12362884730100632

def observe(C,params,row,values,indices,interval,kind):
 theta=float(params[0]);f=params[1];omega=2*np.pi*f;omega0=omega
 ij,_=probes(row);ij=[ij[k] for k in indices];H=np.stack([C[:,1,i,j] for i,j in ij],0);m=len(indices)
 measR=np.eye(m)*(.05*UY_SCALE)**2
 z=np.array([theta,omega]);P=np.diag([np.deg2rad(5.)**2,(.02*omega)**2]);A=np.array([[1.,.25],[0.,1.]])
 Q=np.diag([np.deg2rad(.5)**2,(.0005*omega)**2])*.25
 state=weights(theta)[0][1:];PK=np.eye(8)*.05**2
 rotation=block_diag(*[np.array([[np.cos(k*omega*.25),-np.sin(k*omega*.25)],[np.sin(k*omega*.25),np.cos(k*omega*.25)]]) for k in range(1,5)])
 out=[];updates=[];prefix_states=[];coefSHA=ah(C)
 for step in range(1,161):
  if kind=='coefficient_KF8':
   state=rotation@state;PK=rotation@PK@rotation.T+np.eye(8)*.0001*.25;b=np.r_[1.,state]
  else:
   z=A@z;P=A@P@A.T+Q;b,_=weights(z[0])
  out.append((b@C.reshape(9,-1)).reshape(3,64,64));prefix_states.append(b.copy())
  if interval and step%interval==0 and step<160:
   y=values[step-1,indices].copy();assert np.isfinite(y).all()
   if kind=='coefficient_KF8':
    h=H[:,1:];gain=np.linalg.solve(h@PK@h.T+measR,h@PK).T;residual=y-H[:,0]-h@state;state=state+gain@residual
    J=np.eye(8)-gain@h;PK=J@PK@J.T+gain@measR@gain.T;assert np.isfinite(PK).all()
    detail=dict(harmonic_state=state.tolist())
   else:
    b,d=weights(z[0]);h=np.c_[H@d,np.zeros(m)];gain=np.linalg.solve(h@P@h.T+measR,h@P).T;residual=y-H@b
    delta=gain@residual;delta[0]=np.clip(delta[0],-np.pi/2,np.pi/2);z+=delta;z[1]=np.clip(z[1],.8*omega0,1.2*omega0)
    J=np.eye(2)-gain@h;P=J@P@J.T+gain@measR@gain.T;detail=dict(phase=float(z[0]),frequency=float(z[1]/(2*np.pi)))
   updates.append(dict(step=step,observed_scalar_count=m,sensor_residual_norm=float(np.linalg.norm(residual)),**detail))
  assert np.isfinite(P).all()
 assert ah(C)==coefSHA
 return np.asarray(out,np.float32)*row['mask'][None,None],dict(updates=updates,scalar_readings=m*len(updates),spatial_coefficients_unchanged=True,initial_phase=theta,initial_frequency=float(f),kind=kind,fixed_frequency=(kind=='coefficient_KF8'),every_output_before_current_observation=True),np.array(prefix_states)

def phase1(C,params,row,values):
 ij,_=probes(row);H=np.stack([C[:,1,i,j] for i,j in ij],0);theta=float(params[0]);omega=2*np.pi*params[1];P=np.deg2rad(5.)**2;Q=np.deg2rad(.5)**2*.25
 measR=np.eye(4)*(.05*UY_SCALE)**2;out=[];updates=[];C_SHA=ah(C)
 for step in range(1,161):
  theta+=omega*.25;P+=Q;b,_=weights(theta);out.append((b@C.reshape(9,-1)).reshape(3,64,64))
  if step%8==0 and step<160:
   y=values[step-1].copy();assert np.isfinite(y).all();b,deriv=weights(theta);h=(H@deriv)[:,None]
   gain=np.linalg.solve(h*P@h.T+measR,h*P).T;delta=float((gain@(y-H@b))[0]);theta+=float(np.clip(delta,-np.pi/2,np.pi/2));J=1-float((gain@h)[0,0]);P=J*P*J+float((gain@measR@gain.T)[0,0]);updates.append(dict(step=step,phase=theta,fixed_frequency=float(params[1])))
  assert np.isfinite(P)
 assert ah(C)==C_SHA
 return np.asarray(out,np.float32)*row['mask'][None,None],dict(updates=updates,scalar_readings=76,spatial_coefficients_unchanged=True,fixed_frequency=True,initial_phase=float(params[0]),initial_frequency=float(params[1]))


def four(C,pa,row,values):
 # Odd harmonics share fundamental phase, even harmonics share second-harmonic
 # phase. Four spatial harmonics and all channels retained; C never updated.
 theta,frequency=pa;omega=2*np.pi*frequency;z=np.array([theta,omega,2*theta,2*omega]);base=z.copy()
 A=block_diag(np.array([[1.,.25],[0.,1.]]),np.array([[1.,.25],[0.,1.]]))
 P=np.diag([np.deg2rad(5.)**2,(.02*omega)**2,np.deg2rad(10.)**2,(.04*omega)**2]);Q=np.diag([np.deg2rad(.5)**2,(.0005*omega)**2,np.deg2rad(1.)**2,(.001*omega)**2])*.25
 ij,_=probes(row);H=np.stack([C[:,1,i,j] for i,j in ij]);measR=np.eye(4)*(.05*UY_SCALE)**2;out=[];updates=[];h0=ah(C)
 def basis(z):
  angles=[z[0],z[2],3*z[0],2*z[2]];b=np.ones(9);deriv=np.zeros((9,4))
  for k,(angle,state,factor) in enumerate(zip(angles,[0,2,0,2],[1,1,3,2])):i=1+2*k;b[i]=np.cos(angle);b[i+1]=np.sin(angle);deriv[i,state]=-factor*np.sin(angle);deriv[i+1,state]=factor*np.cos(angle)
  return b,deriv
 for step in range(1,161):
  z=A@z;P=A@P@A.T+Q;b,deriv=basis(z);out.append((b@C.reshape(9,-1)).reshape(3,64,64))
  if step%8==0 and step<160:
   y=values[step-1];assert np.isfinite(y).all();h=H@deriv;gain=np.linalg.solve(h@P@h.T+measR,h@P).T;delta=gain@(y-H@b);delta[0]=np.clip(delta[0],-np.pi/2,np.pi/2);delta[2]=np.clip(delta[2],-np.pi,np.pi);z+=delta
   for index in [1,3]:z[index]=np.clip(z[index],.8*base[index],1.2*base[index])
   J=np.eye(4)-gain@h;P=J@P@J.T+gain@measR@gain.T;updates.append(dict(step=step,phase1=float(z[0]),frequency1=float(z[1]/(2*np.pi)),phase2=float(z[2]),frequency2=float(z[3]/(2*np.pi))))
  assert np.isfinite(P).all()
 assert ah(C)==h0
 return np.asarray(out,np.float32)*row['mask'][None,None],dict(updates=updates,scalar_readings=76,fixed_spatial_C=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parent/'evidence/clock_principle');ap.add_argument('--references',type=Path,default=Path(__file__).resolve().parent/'reference_fields');ap.add_argument('--out',type=Path,default=None,help='Optional output receipt; default writes JSON to stdout only.');a=ap.parse_args()
 report=json.loads((a.evidence/'combined_report.json').read_text(encoding='utf-8'));checks=[];max_abs=0;max_R_diff=0;byte_exact=0
 for ctx in sorted((a.evidence/'contexts').glob('*.npz')):
  info=json.loads(ctx.with_suffix('.json').read_text(encoding='utf-8'));cid=info['id']
  with np.load(ctx) as z:C=z['C'];pa=z['params'];own=z['own_prefix'];values=z['values'];row={k:z[k] for k in ['mask','fields','geom','times']}
  row.update(id=cid,q=info['q']);assert ah(C)==info['C_SHA'];pending=[]
  for record in [r for r in report['cases'] if r['id']==cid]:
   key=record['method'];packet=values.copy()
   if 'seed' in key:
    seed=int(key.split('seed')[-1]);rng=np.random.default_rng(int(hashlib.sha256((str(seed)+':'+cid).encode()).hexdigest()[:8],16));packet+=rng.normal(size=values.shape)*.01*UY_SCALE
   if key.startswith('two_clock4'):ob,di=four(C,pa,row,packet)
   elif key.startswith('phase1'):ob,di=phase1(C,pa,row,packet)
   else:
    m=1 if '_m1_' in key else (2 if '_m2_' in key else 4);idx={1:[0],2:[0,3],4:[0,1,2,3]}[m];interval=0 if key=='fixed_startup_clock_no_future' else (16 if '_dt4_' in key else 8);kind='coefficient_KF8' if key.startswith('coefficient') else 'clock2';ob,di,_=observe(C,pa,row,packet,idx,interval,kind)
   pred=hybrid(own,ob);assert np.isfinite(pred).all() and np.array_equal(pred[:32],own[:32]);assert di['scalar_readings']==record['future_scalars']<=76
   equal=ah(pred)==record['prediction_SHA'];byte_exact+=int(equal);pending.append((record,pred,equal))
  # Truth accessed only after all34 methods for this case have been produced.
  with np.load(a.references/(cid+'.npz')) as z:truth=z['fields'][4:164]
  for record,pred,equal in pending:
   err=np.linalg.norm((pred.astype(float)-truth)*row['mask'][None,None],axis=(-2,-1))/np.maximum(np.linalg.norm(truth*row['mask'][None,None],axis=(-2,-1)),1e-8);taus=np.array([.05,.05,.08],np.float32)
   for h,n in [('8s',32),('40s',160)]:
    x=err[:n];score=float(max(np.max(x.mean(0)/taus),np.max(x[-1]/taus)));delta=abs(score-record['horizons'][h]['R']);max_R_diff=max(max_R_diff,delta);assert delta<2e-6,(cid,record['method'],delta)
   checks.append(dict(id=cid,method=record['method'],prediction_bitwise_same=equal))
  print('REPLAY',cid,len(pending),flush=True,file=sys.stderr)
 assert len(checks)==len(report['cases'])
 receipt=dict(status='PASSED',prediction_sets=len(checks),bitwise_same=byte_exact,max_R_difference=max_R_diff,checks=checks,scope='Frozen contexts and observer controls, not new model evaluation; CPU BLAS rounding may differ from server hashes.')
 payload=json.dumps(receipt,indent=2)
 if a.out is not None:
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(payload+'\n',encoding='utf-8')
 else:sys.stdout.write(payload+'\n')
if __name__=='__main__':main()
