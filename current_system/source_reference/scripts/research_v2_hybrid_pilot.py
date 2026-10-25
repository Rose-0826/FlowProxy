#!/usr/bin/env python3
"""Train-case feasibility pilot: neural delta jumps plus real CFD recovery."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gzip,json,re,shutil,sys,time
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,truth,protected,now,scales
from corrected_benchmark_train_v1 import model_from,protocol
from research_v2_reflection_train import ReflectionOperator
from research_v2_state_probe import ProjectedState
from cycle_landmarks_train_v1 import FORMAL
from warmstart_iface import replace_internal
from geometry_attack_v2 import save,sha,NativeSampler
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import frame_summary
import true2d_recheck_v1 as runner

DEST=OUT/'hybrid_training_pilot';RAW=ROOT/'data/raw/research_v2_hybrid_training_pilot'
CID='cylinder_Re100_cy0_D1';HORIZON=8.;runner.ENV['HWLOC_COMPONENTS']='linux,stop'

def read_text(p):
    if p.exists():return p.read_text()
    return gzip.open(Path(str(p)+'.gz'),'rt').read()

def internal(text,components):
    m=re.search(r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(',text)
    assert m,'Expected nonuniform ASCII field'
    body=text[m.end():].split('\n)',1)[0].replace('(',' ').replace(')',' ')
    data=np.fromstring(body,sep=' ');assert data.size==int(m[1])*components
    return data.reshape(int(m[1]),components)

def prepare_case(mode,source,start):
    case=RAW/mode/CID;case.mkdir(parents=True,exist_ok=False)
    for name in ['constant','system']:shutil.copytree(source/name,case/name)
    target=case/f'{start:g}';target.mkdir()
    for name in ['U','p','phi']:
        src=source/f'{start:g}'/name
        if not src.exists():src=Path(str(src)+'.gz')
        assert src.exists();shutil.copyfile(src,target/src.name)
    return case

def export(case,start,end,index):
    t=time.perf_counter()
    runner.command(case,f'foamToVTK -useTimeName -no-boundary -fields "(U p)" -time "{start:g}:{end:g}"',
        f'log.VTK_{index}',600)
    return time.perf_counter()-t

def pure(mode,source,row,dt):
    begin=time.perf_counter();start=float(row['times'][3]);end=start+HORIZON
    case=prepare_case(mode,source,start);prep=time.perf_counter()-begin
    runner.configure(case,start,end,dt,.25)
    t=time.perf_counter();log=runner.command(case,'icoFoam','log.icoFoam',1800);runner.assert_solver(log,end)
    solve=time.perf_counter()-t;export_seconds=export(case,start,end,0)
    q=row['q'];geo=geometry_channels(protocol()['config'],**q);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
    items=[(t,f) for t,f in frames(case) if start+1e-7<t<=end+1e-7]
    assert len(items)==32
    prediction=np.stack([sampler.sample(f) for t,f in items])
    file=DEST/(mode+'_forecast.npz');np.savez_compressed(file,prediction=prediction)
    return {'method':mode,'file':str(file),'sha256':sha(file),'prediction':prediction,
        'cost':{'prepare':prep,'solve':solve,'VTK':export_seconds,'total':time.perf_counter()-begin}}

def neural_steps(model,row,history,n):
    g=torch.as_tensor(row['geom'][None],device='cuda')
    window=[torch.as_tensor(f[None],device='cuda') for f in history[-4:]];outputs=[]
    torch.cuda.synchronize();begin=time.perf_counter()
    with torch.inference_mode():
        for _ in range(n):
            x=torch.cat([g,torch.stack(window[-4:],1).reshape(1,12,64,64)],1)
            y=model(x)[:,:3];window.append(y);outputs.append(y[0].cpu().numpy())
    torch.cuda.synchronize();return np.stack(outputs),time.perf_counter()-begin

def hybrid(source,row):
    begin=time.perf_counter();start=float(row['times'][3]);case=prepare_case('AI_delta_jump',source,start)
    centers=internal(read_text(source/'0/C'),3)
    xx,yy=np.meshgrid(np.linspace(-8,20,64),np.linspace(-8,8,64));active=row['mask']>.5
    points=np.column_stack([xx[active],yy[active]]);nearest=cKDTree(points).query(centers[:,:2])[1]
    # Construct barycentric lift once; every update uses only predicted coarse deltas.
    from scipy.spatial import Delaunay
    tri=Delaunay(points);simplex=tri.find_simplex(centers[:,:2]);valid=simplex>=0;safe=np.maximum(simplex,0)
    tr=tri.transform[safe];b=np.einsum('nij,nj->ni',tr[:,:2],centers[:,:2]-tr[:,2])
    weight=np.column_stack([b,1-b.sum(1)]);vertices=tri.simplices[safe]
    def lift(delta):
        values=delta[:,active].T
        result=np.einsum('nk,nkc->nc',weight,values[vertices])
        result[~valid]=values[nearest[~valid]];return result
    base,_=model_from(FORMAL,protocol()['config']);model=ProjectedState(ReflectionOperator(base),True);model.eval()
    prep=time.perf_counter()-begin
    history=list(row['fields']);predictions=[];current=start;index=0;records=[]
    q=row['q'];geo=geometry_channels(protocol()['config'],**q);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
    while len(predictions)<32:
        remaining=32-len(predictions)
        # Reserve four solver-generated frames for every neural coast.
        recovery=min(4,remaining);jump=min(8,max(remaining-recovery,0))
        assert jump>0
        neural,net_seconds=neural_steps(model,row,history,jump)
        predictions.extend(neural);jump_time=current+.25*jump
        t=time.perf_counter();old=case/f'{current:g}'
        utext=read_text(old/'U');ptext=read_text(old/'p')
        u=internal(utext,3);p=internal(ptext,1)[:,0]
        delta=lift(neural[-1]-history[-1])
        u[:,:2]+=delta[:,:2];p+=delta[:,2]
        assert np.isfinite(u).all() and np.isfinite(p).all()
        target=case/f'{jump_time:g}';target.mkdir(exist_ok=False)
        ub='internalField nonuniform List<vector>\n'+str(len(u))+'\n(\n'+\
            '\n'.join(f'({a:.12g} {b:.12g} {c:.12g})' for a,b,c in u)+'\n);\n'
        pb='internalField nonuniform List<scalar>\n'+str(len(p))+'\n(\n'+\
            '\n'.join(f'{a:.12g}' for a in p)+'\n);\n'
        for name,template,body in [('U',utext,ub),('p',ptext,pb)]:
            text=replace_internal(template,body)
            text=re.sub(r'location\s+"[^"]+";',f'location "{jump_time:g}";',text,count=1)
            (target/name).write_text(text)
        # phi deliberately regenerated from jumped U by icoFoam, not stale old flux.
        mapping=time.perf_counter()-t;end=jump_time+.25*recovery
        runner.configure(case,jump_time,end,.005,.25)
        t=time.perf_counter();log=runner.command(case,'icoFoam',f'log.icoFoam_{index}',1800)
        runner.assert_solver(log,end);solve=time.perf_counter()-t
        vtk=export(case,jump_time,end,index)
        items=[(t,f) for t,f in frames(case) if jump_time+1e-7<t<=end+1e-7]
        assert len(items)==recovery,(len(items),jump_time,end)
        recovered=[sampler.sample(f) for t,f in items]
        predictions.extend(recovered);history.extend(neural);history.extend(recovered)
        current=end;records.append({'jump_steps':jump,'CFD_steps':recovery,'end':end,
            'network_seconds':net_seconds,'mapping_seconds':mapping,'solver_seconds':solve,'VTK_seconds':vtk,
            'minimum_pressure':float(p.min()),'maximum_velocity':float(np.max(np.linalg.norm(u,axis=1)))})
        index+=1;save(DEST/'hybrid_progress.json',{'frames':len(predictions),'target':32,'cycles':records,'utc':now()})
    prediction=np.stack(predictions);assert abs(current-start-HORIZON)<1e-7
    file=DEST/'AI_delta_jump_forecast.npz';np.savez_compressed(file,prediction=prediction)
    return {'method':'AI_delta_jump','file':str(file),'sha256':sha(file),'prediction':prediction,
        'cost':{'prepare':prep,'total':time.perf_counter()-begin},'cycles':records}

def run():
    protected();torch.set_num_threads(1);DEST.mkdir(exist_ok=False);RAW.mkdir(exist_ok=False)
    q=next(q for q in rows(True) if q['id']==CID);assert q['group']=='train80'
    row=startup(q);source=ROOT/'data/raw/corrected_benchmark_restart_v1/cases'/CID
    save(DEST/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),'id':CID,'role':'TRAINING interface/feasibility pilot ONLY',
        'input':'full native state at fourth startup time plus4 true coarse frames; explicitly MORE information than pure4-frame proxy',
        'horizon_seconds':8,'AI_jump_seconds':2,'CFD_recovery_seconds':1,
        'lifting':'old native state + fluid-only linear coarse delta; nearest outside convex hull; same target mesh',
        'feedback':'only OWN CFD-generated last4 states; no target future reset',
        'methods':['pure_CFD_dt005','pure_CFD_dt010','AI_delta_jump'],
        'cost':'preparation + inference + mapping + solver + nativeVTK export + sampling; no cold-start or1300x claim',
        'solver':'serial icoFoam; corrected original mesh/BC unchanged; do not rebuild mesh',
        'next_gate':'pure dt005 restart numerical reproduction first; hybrid field accuracy and actual cost must beat dt010 baseline before development40s'})
    save(DEST/'status.json',{'status':'running','utc':now()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(pure,n,source,row,dt) for n,dt in [('pure_CFD_dt005',.005),('pure_CFD_dt010',.01)]]
        result=[hybrid(source,row)]+[f.result() for f in futures]
    save(DEST/'forecasts_frozen.json',{'utc':now(),'forecasts':[{k:v for k,v in a.items() if k!='prediction'} for a in result],
        'all_saved_and_hashed_before_future_scoring':True})
    target=truth(q)[4:36];mask=row['mask'][None,None];reports=[]
    for a in result:
        pred=a.pop('prediction');error=np.linalg.norm((pred-target)*mask,axis=(-2,-1))/np.maximum(np.linalg.norm(target*mask,axis=(-2,-1)),1e-8)
        assert np.isfinite(error).all()
        reports.append({**a,'metric':frame_summary(error,row['times'][3],row['times'][35],8,True),
            'max_error_each_channel':error.max(0).tolist()})
    control=next(a for a in reports if a['method']=='pure_CFD_dt005')
    reproduction=max(control['max_error_each_channel'])<.002
    ai=next(a for a in reports if a['method']=='AI_delta_jump');coarse=next(a for a in reports if a['method']=='pure_CFD_dt010')
    signal=reproduction and ai['metric']['R']<1 and ai['cost']['total']<.8*coarse['cost']['total']
    protected();save(DEST/'report.json',{'utc':now(),'cases':reports,'pure_restart_reproduces_reference':reproduction,
        'go_to_development40s':signal,'no_competition_fallback_replacement':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'go_to_development40s':signal})

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(DEST/'failure.json',{'utc':now(),'error':repr(e)})
        save(DEST/'status.json',{'status':'failed','utc':now(),'error':repr(e)});raise
