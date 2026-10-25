from pathlib import Path
import os,sys,json,time,hashlib
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,'/root/autodl-tmp/flowproxy/work/phase_transport_20261005')
import full_startup_space as previous
p=previous.p;body=previous.body;ROOT=p.ROOT;DEST=ROOT/'outputs/literature_attack_20261006'

def mirrored(C):
    z=C[...,::-1,:].copy();z[:,1]*=-1.;return z

def positive(C,q,mask):
    z,support,diag=body.to_body(C,q,mask)
    if q['cy']<0:z=mirrored(z);support=support[::-1,:].copy()
    return z,support,diag

def features(q):return np.array([q['Re']/120.,abs(q['cy']),np.log(q['D'])/.4])

def build(records,name):
    shape=records[0]['q']['shape'];ref,rs,_=positive(records[0]['C'],records[0]['q'],records[0]['mask']);res=[];supports=[];metas=[]
    for row in records:
        C,support,diag=positive(row['C'],row['q'],row['mask']);T,_=p.gauge_align(row['truthC'],row['C'],row['mask']);T,_,_=positive(T,row['q'],row['mask'])
        _,angle=p.gauge_align(C,ref,(support&rs&(body.body_mask(shape)>.5)).astype(float));r=p.remove_tangent(p.rotate(T-C,angle),p.rotate(C,angle))
        res.append(r);supports.append(support);metas.append(dict(id=row['id'],q=row['q'],fold=row['fold'],group_key=row['group_key'],angle=angle,**diag))
    res=np.asarray(res);supports=np.asarray(supports);count=supports.sum(0);mean=np.sum(res*supports[:,None,None],axis=0)/np.maximum(count,1)[None,None]
    filled=np.where(supports[:,None,None],res,mean[None]);filled*=body.body_mask(shape)[None,None,None]
    A=(filled/p.SCALE*p.HWEIGHT).reshape(len(records),-1);x=np.stack([features(s['q']) for s in records]);lib=dict(shape=shape,reference=ref,reference_support=rs,A=A,gram=A@A.T,features=x,source=metas,count=count)
    folder=DEST/name;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(shape+'_bank.npz'),reference=ref,reference_support=rs,A=A,features=x,count=count)
    p.save(folder/(shape+'_bank.json'),dict(shape=shape,source=metas,canonical='reflect negativecy bodyUy andy thenalignH1phase; reflectionaugmentation not extraindependentCFD',support='fit-source supported-mean imputation only',bank_sha256=p.digest(folder/(shape+'_bank.npz'))))
    return lib

def modes(lib,C,row,local=False,split=False):
    can,support,_=positive(C,row['q'],row['mask']);valid=support&lib['reference_support']&(body.body_mask(lib['shape'])>.5);_,angle=p.gauge_align(can,lib['reference'],valid.astype(float))
    n=len(lib['A']);w=np.ones(n)/n
    if local:
        dist=np.sum((lib['features']-features(row['q']))**2,axis=1)
        # Fixed geometry-kernel plus5% globalmass; no held-field or future-score choices.
        raw=np.exp(-dist/(2*.35**2));w=.95*raw/max(raw.sum(),1e-20)+.05/n
    def pod(A,rank,gram=None):
        sw=np.sqrt(w);gram=A@A.T if gram is None else gram;g=gram*sw[:,None]*sw[None];eig,U=eigh(g);idx=np.argsort(eig)[::-1][:rank];e=np.maximum(eig[idx],0.)
        # std1 coefficient modes: eigenvectors scaled by RMS sigma.
        return (U[:,idx].T*sw[None])@A
    if split:
        a=lib['A'].reshape(n,9,3,128,128);a0=a[:,0].reshape(n,-1);a1=a[:,1:].reshape(n,-1)
        b0=pod(a0,4).reshape(4,1,3,128,128);b1=pod(a1,12).reshape(12,8,3,128,128);z=np.zeros((16,9,3,128,128));z[:4,:1]=b0;z[4:,1:]=b1
    else:z=pod(lib['A'],16,lib['gram']).reshape(16,9,3,128,128)
    z=z*p.SCALE[None]/p.HWEIGHT[None];result=[]
    for b in z:
        raw=p.rotate(b,-angle)
        if row['q']['cy']<0:raw=mirrored(raw)
        physical=body.to_physical(raw,row['q'],row['mask'],delta=True);result.append(p.delta_constraints(p.remove_tangent(physical,C),row['mask']))
    return np.asarray(result),dict(local_geometry_kernel=local,split_mean4_oscillation12=split,weights=w.tolist(),sources=[s['id'] for s in lib['source']])

def original_lib(shape):
    folder=ROOT/'outputs/phase_transport_20261005/full_startup_space'
    with np.load(folder/(shape+'_library.npz')) as z:a={k:z[k] for k in z.files}
    a['shape']=shape;return a
