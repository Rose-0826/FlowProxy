"""Read-only independent native-cycle, mesh and dense Courant verification."""
from pathlib import Path
import json,numpy as np,gzip,re,hashlib,sys,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=None);args=ap.parse_args();E=Path(__file__).resolve().parent/'evidence/fresh38';cfg=json.loads((E/'config.json').read_text(encoding='utf-8'));records=[]
integrate=getattr(np,'trapezoid',None) or np.trapz
def moments(t,q,start,end):
 mid=(t>start)&(t<end)
 def at(stamp):
  i=np.searchsorted(t,stamp,side='right')-1
  if t[i]==stamp or i==len(t)-1:return q[i]
  a=(stamp-t[i])/(t[i+1]-t[i]);return (1-a)*q[i]+a*q[i+1]
 ts=np.r_[start,t[mid],end];qs=np.concatenate([at(start)[None],q[mid],at(end)[None]]);mu=integrate(qs,ts,axis=0)/(end-start);second=integrate(qs*qs,ts,axis=0)/(end-start);return mu,np.sqrt(np.maximum(second-mu*mu,0))
for q in cfg['cases']:
 cid=q['id'];D=E/'CFD_quality'/cid;mesh=json.loads((D/'true2d_build.json').read_text(encoding='utf-8'));src=json.loads((D/'dataset_source.json').read_text(encoding='utf-8'));assert mesh['exact_single_layer'] and mesh['mesh_OK'] and len(mesh['z_planes'])==2
 p=next(D.glob('*checkMesh*'));b=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes();assert hashlib.sha256(b).hexdigest()==mesh['checkMesh_sha256'] and b'Mesh OK.' in b
 p=next(D.glob('*icoFoam_dense*'));b=gzip.decompress(p.read_bytes()).decode() if p.suffix=='.gz' else p.read_text();co=[float(v) for v in re.findall(r'Courant Number mean:\s+\S+\s+max:\s+(\S+)',b)];assert co and max(co)<1
 with np.load(E/'references'/(cid+'.npz')) as z:a=z['forces'];assert z['fields'].shape==(165,3,64,64) and np.allclose(np.diff(z['times']),.25)
 meta=json.loads((E/'references'/(cid+'.json')).read_text(encoding='utf-8'));assert max(co)==meta['max_Courant_dense'];end=src['window_start']+41;a=a[(a[:,0]>=end-40)&(a[:,0]<=end)];lift=a[:,2]-integrate(a[:,2],a[:,0])/(a[-1,0]-a[0,0]);cross=[]
 for k in np.flatnonzero((lift[:-1]<=0)&(lift[1:]>0)):cross.append(float(a[k,0]-lift[k]*(a[k+1,0]-a[k,0])/(lift[k+1]-lift[k])))
 assert len(cross)>=4;cycles=[]
 for start,stop in zip(cross[-4:-1],cross[-3:]):mu,rms=moments(a[:,0],a[:,1:],start,stop);cycles.append(dict(Cd_mean=float(mu[0]),Cl_rms=float(rms[1]),St=q['D']/(stop-start)))
 changes=[{k:abs(b[k]-a[k])/max(abs(a[k]),1e-8) for k in ['Cd_mean','Cl_rms','St']} for a,b in zip(cycles[:-1],cycles[1:])];assert all(x[k]<v for x in changes for k,v in [('Cd_mean',.005),('Cl_rms',.01),('St',.005)]);assert np.isclose(np.mean([x['St'] for x in cycles]),src['quality']['St'],rtol=1e-9);records.append(dict(id=cid,max_Courant=max(co),cycles_independently_recomputed=True,mesh_SHA_verified=True,changes=changes))
assert len(records)==8;receipt=dict(status='PASSED',cases=records);payload=json.dumps(receipt,indent=2)
if args.out is not None:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(payload+'\n',encoding='utf-8')
else:sys.stdout.write(payload+'\n')
