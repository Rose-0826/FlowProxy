"""Recompute periodic quality from native force records; check mesh/log provenance."""
from pathlib import Path
import sys,json,gzip,re,hashlib
import numpy as np
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
integrate=getattr(np,'trapezoid',None) or np.trapz

def moment(t,q,start,end):
    middle=(t>start)&(t<end)
    def at(stamp):
        i=np.searchsorted(t,stamp,side='right')-1
        if t[i]==stamp or i==len(t)-1:return q[i]
        a=(stamp-t[i])/(t[i+1]-t[i]);return (1-a)*q[i]+a*q[i+1]
    ts=np.r_[start,t[middle],end];qs=np.concatenate([at(start)[None],q[middle],at(end)[None]])
    mu=integrate(qs,ts,axis=0)/(end-start);second=integrate(qs*qs,ts,axis=0)/(end-start)
    return mu,np.sqrt(np.maximum(second-mu*mu,0))

records=[]
for q in json.loads((root/'records/config.json').read_text())['cases']:
    cid=q['id'];d=root/'CFD_quality'/cid;mesh=json.loads((d/'true2d_build.json').read_text());src=json.loads((d/'dataset_source.json').read_text())
    assert mesh['exact_single_layer'] and mesh['mesh_OK'] and len(mesh['z_planes'])==2
    meshlogs=[p for p in d.glob('*checkMesh*')]
    assert len(meshlogs)==1
    payload=gzip.decompress(meshlogs[0].read_bytes()) if meshlogs[0].suffix=='.gz' else meshlogs[0].read_bytes()
    assert hashlib.sha256(payload).hexdigest()==mesh['checkMesh_sha256'] and b'Mesh OK.' in payload
    denselogs=list(d.glob('*icoFoam_dense*'));assert len(denselogs)==1
    log=gzip.decompress(denselogs[0].read_bytes()).decode() if denselogs[0].suffix=='.gz' else denselogs[0].read_text()
    Co=[float(x) for x in re.findall(r'Courant Number mean:\s+\S+\s+max:\s+(\S+)',log)];assert Co and max(Co)<1
    with np.load(root/'references/fresh8'/(cid+'.npz')) as z:
        a=z['forces'];times=z['times'];assert len(times)==165 and np.allclose(np.diff(times),.25)
    end=src['window_start']+41;a=a[(a[:,0]>=end-40)&(a[:,0]<=end)]
    lift=a[:,2]-integrate(a[:,2],a[:,0])/(a[-1,0]-a[0,0]);cross=[]
    for k in np.flatnonzero((lift[:-1]<=0)&(lift[1:]>0)):cross.append(float(a[k,0]-lift[k]*(a[k+1,0]-a[k,0])/(lift[k+1]-lift[k])))
    assert len(cross)>=4
    cycles=[]
    for start,stop in zip(cross[-4:-1],cross[-3:]):
        mu,rms=moment(a[:,0],a[:,1:],start,stop);cycles.append(dict(Cd_mean=float(mu[0]),Cl_rms=float(rms[1]),St=q['D']/(stop-start)))
    changes=[{k:abs(b[k]-a[k])/max(abs(a[k]),1e-8) for k in ['Cd_mean','Cl_rms','St']} for a,b in zip(cycles[:-1],cycles[1:])]
    assert all(z[k]<v for z in changes for k,v in [('Cd_mean',.005),('Cl_rms',.01),('St',.005)])
    assert np.isclose(np.mean([z['St'] for z in cycles]),src['quality']['St'],rtol=1e-9)
    meta=json.loads((root/'references/fresh8'/(cid+'.json')).read_text());assert max(Co)==meta['max_Courant_dense']
    records.append(dict(id=cid,single_layer=True,mesh_SHA_verified=True,max_Courant=max(Co),last_cycles_recomputed=True,changes=changes))
assert len(records)==8
receipt=dict(status='CFD_QUALITY_VERIFIED',case_count=8,all_mesh_and_cycle_checks_pass=True,cases=records)
(root/'CFD_quality_verification.json').write_text(json.dumps(receipt,indent=2));print('CFD_QUALITY_VERIFIED_8',flush=True)
