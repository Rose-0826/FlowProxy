#!/usr/bin/env python3
"""Finite TRAIN96-only spatial observer pilot. Existing mainline is immutable."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import argparse, datetime, hashlib, json, sys, time
ROOT = Path(os.environ.get('AIC_ROOT', '/root/autodl-tmp/flowproxy'))
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from scipy.linalg import eigh
from scipy.optimize import least_squares
from research_v2_common import OUT, rows, startup, scales, now
from research_v2_final_confirmation import load, infer
from research_v2_outer_support_train import dataset
from research_v2_template_probe import basis, fit_startup, gauge_align
from research_v3_clock_probe import features
from startup_observability_v1 import frequency
from research_v4_sparse_observer import probes, weights, CausalMeasurements, forecast
from research_v5_split_observer import neural_template, correction, observe, hybrid, rotate
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

DEST = ROOT/'outputs/spatial_observer_20261005'
WEIGHT = OUT/'final_research_confirmation/selected_research_base_weights.pt'
EXPECTED_SHA = '9b860db72359a0e874a7e5942ea2ccd14d2f1ea29339af49dd435a89128ac1aa'
SCALE = np.asarray(scales(), np.float64)[None, :, None, None]
HWEIGHT = np.r_[1., np.full(8, 1/np.sqrt(2))][:, None, None, None]
X, Y = np.meshgrid(np.linspace(-8,20,64), np.linspace(-8,8,64))
REFERENCE_Q = dict(cx=0., cy=0., D=1.)

def save(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def protect():
    assert digest(WEIGHT)==EXPECTED_SHA, 'Frozen baseline changed'

def group_key(q):
    return json.dumps([q['shape'], float(q['Re']), abs(float(q['cy'])), float(q['D'])])

def folds(records):
    result={}
    for shape in ['cylinder','square']:
        keys=sorted({group_key(s['q']) for s in records if s['q']['shape']==shape},
                    key=lambda k:hashlib.sha256(('spatial101:'+k).encode()).hexdigest())
        for i,k in enumerate(keys): result[k]=i%4
    return result

def geometry_priors():
    obj=json.loads((OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json').read_text())
    return {a['shape']:RBFInterpolator(np.asarray([g['features'] for g in a['geometry_groups']]),
            np.asarray([g['mean_field_St'] for g in a['geometry_groups']]),
            kernel='thin_plate_spline',degree=1,smoothing=a['selected']['smoothing'])
            for a in obj['models']}

def bootstrap(model, row, priors):
    own,di=infer(model,row,True); n=di['prefix_steps']
    center=float(priors[row['q']['shape']](features(row['q'])[None])[0])/row['q']['D']
    obs=np.concatenate([row['fields'],own[:n]])
    t=row['times'][:n+4]-row['times'][0]
    f,fd=frequency(t,obs[:,1][:,row['mask']>.5].astype(np.float64),4,center,.2)
    C=neural_template(row,own,n,f); params,initial=fit_startup(C,row,f,False)
    return own,correction(C,row,params,'hybrid8_pH1'),params,{
        'frequency':float(f),'frequency_prior':center,'prefix_steps':n,
        'network':di,'self_fit':fd,'startup':initial}

def dev_records():
    result=[dict(q) for q in rows()]
    manifest=json.loads((OUT/'next_stage_split_observer/new_confirmation8_cfd/reference_manifest.json').read_text())
    for q in manifest['cases']:
        result.append({**q,'group':'latest_seen8','cache':str(ROOT/'data/processed/research_v5_split_confirmation8_cases')})
    assert len(result)==64 and len({q['id'] for q in result})==64
    return result

def prepare():
    import torch
    torch.set_num_threads(1); protect(); begin=time.perf_counter()
    DEST.mkdir(parents=True,exist_ok=True); (DEST/'train_templates').mkdir(exist_ok=True)
    (DEST/'dev_bootstrap').mkdir(exist_ok=True)
    model=load({'path':str(WEIGHT)}); priors=geometry_priors()
    data=dataset(); periods=json.loads((OUT/'outer_support_training/periods96.json').read_text())['rows']
    assert [r['id'] for r in data]==[p['id'] for p in periods]
    records=[]
    for index,(r,p) in enumerate(zip(data,periods)):
        path=DEST/'train_templates'/(r['id']+'.npz')
        records.append({'id':r['id'],'q':r['q'],'group_key':group_key(r['q'])})
        if not path.exists():
            row={**r,'fields':r['fields'][:4]}
            own,C,params,diag=bootstrap(model,row,priors)
            t=r['times']-r['times'][0]
            truthC=np.linalg.lstsq(basis(t,p['frequency']),r['fields'].reshape(len(t),-1),rcond=1e-10)[0].reshape(9,3,64,64)
            np.savez_compressed(path,C=C.astype(np.float32),truthC=truthC.astype(np.float32),
                mask=r['mask'],params=params,times=r['times'],truth_frequency=p['frequency'])
            save(path.with_suffix('.json'),{'id':r['id'],'q':r['q'],'diagnostic':diag,'role':'TRAIN96 only'})
        print('TRAIN_TEMPLATE',index+1,len(data),r['id'],flush=True)
        save(DEST/'status.json',{'stage':'prepare_train','completed':index+1,'total':96,'utc':now(),'wall_seconds':time.perf_counter()-begin})
    foldmap=folds(records)
    for s in records:s['fold']=foldmap[s['group_key']]
    save(DEST/'train_manifest.json',{'records':records,'groups':foldmap,
        'limitation':'FNO already trained on all96; folds hold out only spatial-correction training'})
    del data
    for index,q in enumerate(dev_records()):
        path=DEST/'dev_bootstrap'/(q['id']+'.npz')
        if not path.exists():
            row=startup(q); own,C,params,diag=bootstrap(model,row,priors)
            np.savez_compressed(path,own=own,C=C.astype(np.float64),params=params,mask=row['mask'],times=row['times'])
            save(path.with_suffix('.json'),{'q':q,'diagnostic':diag,'future_full_fields_read':False})
        print('DEV_BOOTSTRAP',index+1,64,q['id'],flush=True)
        save(DEST/'status.json',{'stage':'prepare_dev','completed':index+1,'total':64,'utc':now(),'wall_seconds':time.perf_counter()-begin})
    protect();save(DEST/'prepare_complete.json',{'utc':now(),'wall_seconds':time.perf_counter()-begin,'train':96,'seen_dev':64})

def delta_constraints(C,mask):
    C=np.asarray(C,np.float64).copy()*mask[None,None]
    C[:,0,:,0]=0.; C[:,1,:,0]=0.; C[:,1,0,:]=0.; C[:,1,-1,:]=0.
    return C

def transport(C, source, target):
    xx=source['cx']+(X-target['cx'])*source['D']/target['D']
    yy=source['cy']+(Y-target['cy'])*source['D']/target['D']
    index=np.array([(yy+8)*63/16,(xx+8)*63/28])
    return np.stack([map_coordinates(v,index,order=1,mode='constant',cval=0.,prefilter=False)
                    for v in C.reshape(-1,64,64)]).reshape(C.shape)

def canonical(C,q,coordinate):
    return C.copy() if coordinate=='physical' else transport(C,q,REFERENCE_Q)

def tangent(C):
    T=np.zeros_like(C)
    for k in range(1,5):T[2*k-1]=k*C[2*k];T[2*k]=-k*C[2*k-1]
    return T

def remove_tangent(R,C):
    T=tangent(C);tw=T/SCALE*HWEIGHT
    alpha=float(np.sum((R/SCALE*HWEIGHT)*tw)/max(np.sum(tw*tw),1e-20))
    return R-alpha*T

def read_train():
    manifest=json.loads((DEST/'train_manifest.json').read_text())['records']; result=[]
    for s in manifest:
        with np.load(DEST/'train_templates'/(s['id']+'.npz')) as a:
            result.append({**s,**{k:a[k] for k in a.files}})
    return result

def make_library(records,coordinate):
    ref=canonical(records[0]['C'],records[0]['q'],coordinate)
    residuals=[]
    for s in records:
        C=canonical(s['C'],s['q'],coordinate)
        Tc,_=gauge_align(s['truthC'],s['C'],s['mask'])
        Tc=canonical(Tc,s['q'],coordinate)
        _,angle=gauge_align(C,ref,np.ones((64,64)))
        R=remove_tangent(rotate(Tc-C,angle),rotate(C,angle))
        residuals.append((R/SCALE*HWEIGHT).ravel())
    A=np.asarray(residuals); G=A@A.T
    eig,U=eigh(G);order=np.argsort(eig)[::-1];eig=np.maximum(eig[order],0);U=U[:,order]
    sigma=np.sqrt(eig[:4]); V=(U[:,:4].T@A)/np.maximum(sigma[:,None],1e-20)
    modes=(V*(sigma/np.sqrt(len(A)))[:,None]).reshape(4,9,3,64,64)*SCALE/HWEIGHT
    return {'reference':ref,'modes':modes,'energy':(eig/np.sum(eig)).tolist(),
            'coordinate':coordinate,'donors':[s['id'] for s in records]}

def modes_for(lib,C,q,mask,rank):
    canonicalC=canonical(C,q,lib['coordinate'])
    _,angle=gauge_align(canonicalC,lib['reference'],np.ones((64,64)))
    modes=[]
    for B in lib['modes'][:rank]:
        raw=rotate(B,-angle)
        if lib['coordinate']!='physical':raw=transport(raw,REFERENCE_Q,q)
        modes.append(delta_constraints(remove_tangent(raw,C),mask))
    return np.asarray(modes)

def probe_information(C,B,params,row):
    ij,_=probes(row);H=np.stack([C[:,1,i,j] for i,j in ij])
    HB=np.stack([np.stack([v[:,1,i,j] for i,j in ij]) for v in B])
    M=[];N=[]
    for t in np.arange(2.,18.,2.):
        b,d=weights(params[0]+2*np.pi*params[1]*t)
        M.append(np.einsum('rij,j->ir',HB,b)); v=H@d;N.append(np.c_[v,t*v])
    M=np.vstack(M);N=np.vstack(N);M-=N@np.linalg.lstsq(N,M,rcond=1e-10)[0]
    s=np.linalg.svd(M,compute_uv=False);noise=.05*SCALE[0,1,0,0]
    rank=int(np.sum(s>max(s[0]*1e-3,noise))) if len(s) else 0
    return {'rank_after_clock_nuisance':rank,'state_dimensions':len(B),'singular_values':s.tolist(),
            'condition_number':float(s[0]/max(s[-1],1e-20)),'min_singular_over_noise':float(s[-1]/noise)}

def representation():
    begin=time.perf_counter();records=read_train(); results=[]
    (DEST/'libraries').mkdir(exist_ok=True)
    for coordinate in ['physical','body_relative_D_normalized']:
        for shape in ['cylinder','square']:
            part=[s for s in records if s['q']['shape']==shape]
            full=make_library(part,coordinate)
            np.savez_compressed(DEST/'libraries'/f'{shape}__{coordinate}.npz',reference=full['reference'],modes=full['modes'])
            save(DEST/'libraries'/f'{shape}__{coordinate}.json',{k:v for k,v in full.items() if k not in ['reference','modes']})
            for fold in range(4):
                fit=[s for s in part if s['fold']!=fold];held=[s for s in part if s['fold']==fold]
                assert not {s['group_key'] for s in fit}&{s['group_key'] for s in held}
                lib=make_library(fit,coordinate)
                for s in held:
                    Tc,_=gauge_align(s['truthC'],s['C'],s['mask'])
                    mask=s['mask']>.5;R=(Tc-s['C'])[:,1,mask]*HWEIGHT[:,0,0,0,None]
                    den=np.linalg.norm(Tc[:,1,mask]*HWEIGHT[:,0,0,0,None])
                    base=float(np.linalg.norm(R)/max(den,1e-20))
                    for rank in [2,4]:
                        B=modes_for(lib,s['C'],s['q'],s['mask'],rank)
                        M=(B[:,:,1,mask]*HWEIGHT[None,:,0,0,0,None]).reshape(rank,-1).T
                        a=np.linalg.lstsq(M,R.ravel(),rcond=1e-8)[0]
                        oracle=float(np.linalg.norm(R.ravel()-M@a)/max(den,1e-20))
                        row={'mask':s['mask'],'q':s['q']}
                        info=probe_information(s['C'],B,s['params'],row)
                        results.append({'id':s['id'],'shape':shape,'D':s['q']['D'],'fold':fold,'coordinate':coordinate,
                            'rank':rank,'baseline_aligned_template_Uy':base,'oracle_template_Uy':oracle,
                            'oracle_ratio':oracle/max(base,1e-20),'oracle_a':a.tolist(),'probe_information':info})
                print('REPRESENTATION',coordinate,shape,fold,'held',len(held),flush=True)
    summary=[]
    for coordinate in ['physical','body_relative_D_normalized']:
        for rank in [2,4]:
            for shape in ['cylinder','square','all']:
                p=[x for x in results if x['coordinate']==coordinate and x['rank']==rank and (shape=='all' or x['shape']==shape)]
                summary.append({'coordinate':coordinate,'rank':rank,'shape':shape,'n':len(p),
                    'baseline_template_Uy':float(np.mean([x['baseline_aligned_template_Uy'] for x in p])),
                    'oracle_template_Uy':float(np.mean([x['oracle_template_Uy'] for x in p])),
                    'median_ratio':float(np.median([x['oracle_ratio'] for x in p])),
                    'fully_observable_cases':sum(x['probe_information']['rank_after_clock_nuisance']==rank for x in p)})
    save(DEST/'representation_report.json',{'role':'NONDEPLOYABLE future-truth representation upper bound; correction-module group CV only',
         'cases':results,'summary':summary,'wall_seconds':time.perf_counter()-begin})
    save(DEST/'status.json',{'stage':'representation_complete','utc':now()});protect()

def load_library(shape,coordinate):
    with np.load(DEST/'libraries'/f'{shape}__{coordinate}.npz') as a:
        return {'reference':a['reference'],'modes':a['modes'],'coordinate':coordinate}

def observable_directions(J,N):
    clean=J-N@np.linalg.lstsq(N,J,rcond=1e-10)[0]
    _,s,V=np.linalg.svd(clean,full_matrices=False)
    threshold=max((s[0] if len(s) else 0.)*1e-3,.05*SCALE[0,1,0,0])
    keep=s>threshold
    return V[keep].T,s

def spatial_window(C,B,params,row,values,kind='joint'):
    """Sliding<=16s ridge fit; emit before measurement; never revise emitted fields."""
    r=len(B);ij,_=probes(row)
    H=np.stack([C[:,1,i,j] for i,j in ij])
    HB=np.stack([np.stack([v[:,1,i,j] for i,j in ij]) for v in B])
    flat=C.reshape(9,-1);BF=B.reshape(r,9,-1)
    a=np.zeros(r);theta=float(params[0]);omega=2*np.pi*params[1];omega0=omega
    reader=CausalMeasurements(values,8);history=[];pred=[];updates=[];last=np.zeros(2+r)
    noise=.05*SCALE[0,1,0,0]
    if kind=='fixedclock':
        _,clock=observe(C,params[1],row,values,params)
        phases=np.asarray(clock['phases']);clock_updates={q['step']:q for q in clock['updates']}
    for step in range(1,161):
        t=step*.25
        theta=phases[step-1] if kind=='fixedclock' else theta+omega*.25
        b,d=weights(theta)
        pred.append((b@(flat+np.einsum('r,rjf->jf',a,BF))).reshape(3,64,64))
        if step%8==0 and step<160:
            y=reader.read(step);history.append((t,y))
            history=[v for v in history if t-v[0]<16.]
            dt=np.asarray([v[0]-t for v in history]);obs=np.asarray([v[1] for v in history])
            theta_prior=theta;omega_prior=omega;a_prior=a.copy()
            if kind=='fixedclock':omega_prior=2*np.pi*clock_updates[step]['frequency']
            times_basis=np.asarray([weights(theta_prior+omega_prior*v)[0] for v in dt])
            times_derivative=np.asarray([weights(theta_prior+omega_prior*v)[1] for v in dt])
            J=np.einsum('rij,tj->tir',HB,times_basis).reshape(-1,r)
            Hcurrent=H+np.einsum('r,rij->ij',a,HB)
            phasej=(times_derivative@Hcurrent.T).ravel()
            nuisance=np.c_[phasej,np.repeat(dt,4)*phasej] if kind=='joint' else phasej[:,None]
            directions,s=observable_directions(J,nuisance)
            nr=directions.shape[1]
            if nr:
                def unpack(v):
                    if kind=='joint':return theta_prior+v[0],omega_prior+v[1],a_prior+directions@v[2:]
                    return theta_prior,omega_prior,a_prior+directions@v
                def residual(v):
                    ph,om,aa=unpack(v)
                    BB=np.asarray([weights(ph+om*tt)[0] for tt in dt])
                    hc=H+np.einsum('r,rij->ij',aa,HB)
                    e=((BB@hc.T-obs)/noise).ravel()
                    regular=[aa, (aa-a_prior)/.5]
                    if kind=='joint':regular.append(np.asarray([v[0]/np.deg2rad(5.),v[1]/(.02*omega0)]))
                    return np.concatenate([e]+regular)
                nvar=nr+(2 if kind=='joint' else 0)
                lo=np.full(nvar,-2.);hi=np.full(nvar,2.)
                if kind=='joint':lo[:2]=[-np.pi/2,.8*omega0-omega_prior];hi[:2]=[np.pi/2,1.2*omega0-omega_prior]
                opt=least_squares(residual,np.zeros(nvar),bounds=(lo,hi),max_nfev=35,
                    ftol=1e-7,xtol=1e-7,gtol=1e-7)
                ph,om,aa=unpack(opt.x)
                a=np.clip(aa,-3.,3.)
                if kind=='joint':theta=ph;omega=float(om)
                fit_success=bool(opt.success);nfev=int(opt.nfev)
            else:fit_success=True;nfev=0
            updates.append({'step':step,'time_seconds':t,'a':a.tolist(),'phase':float(theta),'frequency':float(omega/(2*np.pi)),
                'observable_rank':nr,'singular_values':s.tolist(),'window_times':[v[0] for v in history],
                'fit_success':fit_success,'nfev':nfev})
        assert np.isfinite(a).all() and np.isfinite(theta) and np.isfinite(omega)
    result=np.asarray(pred,np.float32)*row['mask'][None,None]
    return result,{'updates':updates,'scalar_future_readings':4*len(reader.access),
        'observed_future_steps':reader.access,'emit_before_measurement':True,'window_seconds_max':16.,'end_gap_seconds':2.}

def eval_summary(cases):
    result={}
    group_selectors={
        'DEV56':lambda c:c['group']!='latest_seen8',
        'latest_seen8':lambda c:c['group']=='latest_seen8',
        'all_seen64':lambda c:True,
        'DEV56_thin_cylinder':lambda c:c['group']!='latest_seen8' and c['q']['shape']=='cylinder' and c['q']['D']<.9,
        'DEV56_thin_square':lambda c:c['group']!='latest_seen8' and c['q']['shape']=='square' and c['q']['D']<.9,
        'latest_seen8_thin_cylinder':lambda c:c['group']=='latest_seen8' and c['q']['shape']=='cylinder' and c['q']['D']<.9,
        'latest_seen8_thin_square':lambda c:c['group']=='latest_seen8' and c['q']['shape']=='square' and c['q']['D']<.9}
    for name,select in group_selectors.items():
        selected=[c for c in cases if select(c)]
        if selected:result[name]={h:summarize([c['horizons'][h] for c in selected]) for h in ['8s','40s']}
    return result

def get_measurements(q):
    prefix=OUT/('next_stage_split_observer/new_confirmation8_eval' if q['group']=='latest_seen8' else 'sparse_observer_pilot')
    with np.load(prefix/'measurements'/(q['id']+'.npz')) as a:return a['clean'],a['noisy']

def run_method(method,C,params,own,row,values,libs):
    if method=='frozen_v5':p,di=observe(C,params[1],row,values,params)
    elif method=='same_template_linearKF':p,di=forecast(C,params[1],row,values,8,'linear')
    else:
        coordinate,rank,kind=method.split('__')
        B=modes_for(libs[(row['q']['shape'],coordinate)],C,row['q'],row['mask'],int(rank[1:]))
        p,di=spatial_window(C,B,params,row,values,kind)
    p=hybrid(own,p);assert np.array_equal(p[:32],own[:32]);assert di['scalar_future_readings']==76
    return p,di

def causality_check(C,params,own,row,clean,libs,methods):
    changed=clean.copy();changed[79:]+=100.
    result=[]
    for m in methods:
        a,_=run_method(m,C,params,own,row,clean,libs);b,_=run_method(m,C,params,own,row,changed,libs)
        gap=float(np.max(np.abs(a[:80]-b[:80])))
        assert gap==0.,(m,'future observation affected emitted past',gap)
        result.append({'method':m,'later_observation_poison_max_difference_first80':gap,'startup_short8_exact':True})
    return result

def evaluate():
    begin=time.perf_counter();protect()
    libs={(s,c):load_library(s,c) for s in ['cylinder','square'] for c in ['physical','body_relative_D_normalized']}
    methods=['frozen_v5','same_template_linearKF']+[
        f'{c}__r{r}__{k}' for c in ['physical','body_relative_D_normalized'] for r in [2,4] for k in ['fixedclock','joint']]
    results={m:[] for m in methods};records=dev_records();checks=[]
    (DEST/'observer_states').mkdir(exist_ok=True)
    for index,q in enumerate(records):
        row=startup(q)
        with np.load(DEST/'dev_bootstrap'/(q['id']+'.npz')) as a:C=a['C'];params=a['params'];own=a['own']
        clean,noisy=get_measurements(q);predictions={};registry=[]
        for m in methods:
            p,di=run_method(m,C,params,own,row,clean,libs);predictions[m]=p
            # State trajectories and prediction digest are written before reading this case's future field.
            save(DEST/'observer_states'/f'{m}__{q["id"]}.json',{'id':q['id'],'method':m,
                'prediction_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'observer':di})
            registry.append({'method':m,'sha256':hashlib.sha256(p.tobytes()).hexdigest()})
        save(DEST/'observer_states'/f'frozen__{q["id"]}.json',{'all_candidates_frozen_before_scoring':True,'registry':registry})
        if q['group']=='latest_seen8':
            path=OUT/'next_stage_split_observer/new_confirmation8_eval/predictions'/('v5_inference_split__'+q['id']+'.npz')
            with np.load(path) as a:gap=float(np.max(np.abs(a['prediction']-predictions['frozen_v5'])))
            assert gap<2e-6,(q['id'],'baseline mismatch',gap);checks.append({'id':q['id'],'frozen_max_abs_gap':gap})
        if index==0:
            checks.extend(causality_check(C,params,own,row,clean,libs,methods))
        with np.load(Path(q['cache'])/(q['id']+'.npz')) as a:truth=a['fields'][4:164]
        for m,p in predictions.items():
            results[m].append({'id':q['id'],'q':row['q'],'group':q['group'],'horizons':metrics(p,truth,row['mask'],row['times'])})
        save(DEST/'eval_partial.json',{m:{'summary':eval_summary(part),'cases':part} for m,part in results.items()})
        save(DEST/'status.json',{'stage':'causal_evaluation','completed':index+1,'total':64,'utc':now(),'wall_seconds':time.perf_counter()-begin})
        print('CAUSAL_EVAL',index+1,64,q['id'],flush=True)
    report={m:{'summary':eval_summary(part),'cases':part} for m,part in results.items()}
    base=report['frozen_v5']['summary'];entries=[]
    for m,p in report.items():
        if m in ['frozen_v5','same_template_linearKF']:continue
        s=p['summary'];a=s['DEV56']['40s'];b=base['DEV56']['40s']
        guards={
          'mean_R_at_least15pct_lower':a['mean_R']<=.85*b['mean_R'],
          'DEV56_max_R_not_higher':a['max_R']<=b['max_R'],
          'DEV56_short8_passes_preserved':s['DEV56']['8s']['n_pass']>=base['DEV56']['8s']['n_pass'],
          'latest8_max_R_not_higher':s['latest_seen8']['40s']['max_R']<=base['latest_seen8']['40s']['max_R']}
        for g in ['DEV56_thin_cylinder','DEV56_thin_square','latest_seen8_thin_cylinder','latest_seen8_thin_square']:
            if g in s:guards[g+'_improves']=s[g]['40s']['mean_R']<base[g]['40s']['mean_R']
        entries.append({'method':m,'DEV56_mean_R_ratio':a['mean_R']/b['mean_R'],'guards':guards,'signal':all(guards.values())})
    save(DEST/'causal_report.json',{'utc':now(),'role':'SEEN DEVELOPMENT ONLY; no new independent confirmation',
        'methods':report,'candidates':entries,'checks':checks,'wall_seconds':time.perf_counter()-begin,
        'prediction_protocol':'four complete startup fields+76Uy readings; emit before read; first8 exactly autonomous'})
    protect();save(DEST/'status.json',{'stage':'causal_evaluation_complete','utc':now()})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','representation','evaluate']);args=ap.parse_args()
    try:globals()[args.stage]()
    except Exception as exc:
        save(DEST/'failure.json',{'utc':now(),'stage':args.stage,'error':repr(exc)});raise
