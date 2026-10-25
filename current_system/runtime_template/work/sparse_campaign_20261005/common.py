#!/usr/bin/env python3
"""Shared sealed-input protocol. All experiment files stay on the server."""
from pathlib import Path
import os,sys,json,hashlib,datetime,subprocess
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
ROOT=Path('/root/autodl-tmp/flowproxy'); WORK=ROOT/'work/sparse_campaign_20261005'; OUT=ROOT/'outputs/sparse_campaign_20261005'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'work/spatial_observer_20261005')]
import pilot as p
from research_v4_sparse_observer import constrain
from round2_complete40_v1 import frame_summary,summarize
SCALES=np.asarray(p.scales());X,Y=p.X,p.Y
HISTORY_STEPS=np.arange(0,57,8);HISTORY_TIMES=np.arange(-14.,1.,2.);ORIGIN=56;FUTURE_STEPS=np.arange(8,160,8)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,obj):return p.save(path,obj)
def digest(path):return p.digest(path)
def grouped_fold(q):return p.folds([{'q':s['q']} for s in p.read_train()])[p.group_key(q)]
def group_key(q):return p.group_key(q)
def harmonic_basis(t,f,phase=0.):return p.basis(np.asarray(t),f,phase)
def center_probes(q,mask):return np.asarray(p.probes({'q':q,'mask':mask})[0],int)
def mapped_probes(q,mask,coordinates):
    ij=[];yy,xx=np.where(mask>.5)
    for dx,dy in coordinates:
        dist=(X[yy,xx]-(q['cx']+dx*q['D']))**2+(Y[yy,xx]-(q['cy']+dy*q['D']))**2
        for i in np.argsort(dist):
            loc=(int(yy[i]),int(xx[i]))
            if loc not in ij:ij.append(loc);break
    assert len(set(ij))==len(coordinates)
    return np.asarray(ij,int)
def load_train():
    data=p.dataset();templates={s['id']:s for s in p.read_train()}
    for row in data:
        s=templates[row['id']];row.update(truthC=s['truthC'],truth_frequency=float(s['truth_frequency']),fold=s['fold'],group_key=s['group_key'],origin_index=56,future_n=104)
        assert len(row['fields'])>=165 and np.allclose(np.diff(row['times']),.25)
    return data

def load_dev(extended=False):
    rows=[]
    for q0 in p.dev_records():
        q=dict(q0);path=Path(q['cache'])/(q['id']+'.npz')
        extension=OUT/'data_extension/cache'/(q['id']+'.npz')
        if extended:
            if not extension.exists() or not extension.with_suffix('.json').exists():continue
            path=extension
        with np.load(path) as a:meta={k:a[k] for k in ['mask','geom','times']}
        # Parent predictor sees no target full fields, including startup.
        rows.append({'id':q['id'],'q':{k:q[k] for k in ['shape','Re','cx','cy','D']},'group':q['group'],
                     'source_path':str(path),'origin_index':56,'future_n':160 if len(meta['times'])>=217 else 104,**meta})
    return rows

def _key(row,ij):
    return hashlib.sha256(json.dumps([row['id'],row.get('source_path','TRAIN'),np.asarray(ij).tolist()],sort_keys=True).encode()).hexdigest()[:20]

def packet(row,ij):
    ij=np.asarray(ij,int);n=int(row.get('future_n',160));assert ij.ndim==2 and ij.shape[1]==2
    assert len(set(map(tuple,ij)))==len(ij) and len(ij) in (4,8)
    if 'fields' in row:
        # TRAIN only, full labels are expressly allowed during fitting and fold scoring.
        fields=row['fields'];historical=np.stack([fields[HISTORY_STEPS,1,i,j] for i,j in ij],axis=1)
        n=min(n,len(fields)-57);future=np.full((n,len(ij)),np.nan,dtype=np.float64)
        steps=np.arange(8,n,8)
        for step in steps:future[step-1]=[fields[ORIGIN+step,1,i,j] for i,j in ij]
        return {'history':historical.astype(float),'history_times':HISTORY_TIMES.copy(),'future':future,'future_steps':steps,'ij':ij,'history_scalars':8*len(ij),'future_scalars':len(steps)*len(ij)}
    path=OUT/'sealed_inputs'/(_key(row,ij)+'.npz')
    request=OUT/'sealed_inputs'/(_key(row,ij)+'.request.json');path.parent.mkdir(exist_ok=True)
    if not path.exists():
        save(request,{'source_path':row['source_path'],'ij':ij.tolist(),'dest':str(path),'n':n,'origin_index':56,'id':row['id']})
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--collect',str(request)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Separate measurement collector failed: '+result.stderr[-1800:])
    with np.load(path) as a:obj={k:a[k].copy() for k in a.files}
    obj['future_steps']=np.arange(8,n,8);obj['ij']=ij;obj['history_times']=HISTORY_TIMES.copy();obj['history_scalars']=8*len(ij);obj['future_scalars']=len(obj['future_steps'])*len(ij)
    assert np.all(np.isnan(obj['future'][np.setdiff1d(np.arange(n),obj['future_steps']-1)]))
    return obj

def history(row,ij):return packet(row,ij)['history']
def future_measurements(row,ij):return packet(row,ij)['future']

def collect(request):
    req=json.loads(Path(request).read_text());ij=np.asarray(req['ij'],int)
    with np.load(req['source_path']) as a:fields=a['fields']
    n=req['n'];assert len(fields)>=57+n
    historical=np.stack([fields[HISTORY_STEPS,1,i,j] for i,j in ij],axis=1)
    future=np.full((n,len(ij)),np.nan)
    steps=np.arange(8,n,8)
    for step in steps:future[step-1]=[fields[ORIGIN+step,1,i,j] for i,j in ij]
    target=Path(req['dest']);tmp=target.with_suffix('.tmp.npz');np.savez_compressed(tmp,history=historical,future=future);tmp.replace(target)
    save(target.with_suffix('.audit.json'),{'id':req['id'],'source_sha256':digest(req['source_path']),'history_times':HISTORY_TIMES.tolist(),'future_seconds':(steps*.25).tolist(),
          'history_scalars':historical.size,'future_scalars':len(steps)*len(ij),'separate_process':True,'no_full_fields_in_packet':True,'utc':now()})

def score_forecast(pred,truth,mask,horizon=None):
    pred=np.asarray(pred);truth=np.asarray(truth);assert pred.shape==truth.shape and len(pred)>0
    requested=len(pred)*.25 if horizon is None else float(horizon);assert abs(len(pred)*.25-requested)<1e-8
    delta=(pred.astype(np.float64)-truth)*mask[None,None];den=np.linalg.norm(truth*mask[None,None],axis=(-2,-1))
    err=np.linalg.norm(delta,axis=(-2,-1))/np.maximum(den,1e-8)
    return frame_summary(err,0.,requested,requested,True)

def score_after_freeze(row,pred,record_path):
    pred=np.asarray(pred,np.float32);record_path=Path(record_path)
    save(record_path,{'id':row['id'],'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'utc':now(),
             'this_candidate_prediction_frozen_before_its_scoring':True,'protocol':'32 history Uy scalars + causal future probes; no startup fields'})
    if 'fields' in row:truth=row['fields'][57:57+len(pred)]
    else:
        with np.load(row['source_path']) as a:truth=a['fields'][57:57+len(pred)]
    horizons={}
    for sec in [2,8,20,26,40]:
        n=int(sec*4)
        if len(pred)>=n:horizons[f'{sec}s']=score_forecast(pred[:n],truth[:n],row['mask'],sec)
    return horizons

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--collect':collect(sys.argv[2])
    else:raise SystemExit('Use --collect REQUEST only')
