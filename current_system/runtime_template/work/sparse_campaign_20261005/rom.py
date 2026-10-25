#!/usr/bin/env python3
"""TRAIN-only parametric harmonic POD and gappy history/causal observers.
Every target startup input is exactly 8 x 4 Uy samples; pressure has fixed CFD gauge.
"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,time,hashlib,sys,argparse
import numpy as np
from scipy.linalg import eigh,qr
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
import common as c
from research_v2_template_probe import gauge_align
from research_v4_sparse_observer import weights
DEST=c.OUT/'rom'
SCALE=c.SCALES[None,:,None,None]
HWEIGHT=np.r_[1.,np.full(8,1/np.sqrt(2))][:,None,None,None]

def geometry_features(q):return np.asarray([q['Re']/120.,q['cy'],np.log(q['D'])/.4],float)
def frequency_features(q):return np.asarray([q['Re']/120.,q['cy']**2,np.log(q['D'])/.4],float)

def frequency_model(records):
    groups={}
    for s in records:
        x=tuple(frequency_features(s['q']))
        groups.setdefault(x,[]).append(float(s['truth_frequency'])*s['q']['D'])
    X=np.asarray(list(groups));Y=np.asarray([np.mean(v) for v in groups.values()]);options=[]
    for smoothing in [0.,1e-6,1e-3]:
        errors=[]
        for i in range(len(X)):
            keep=np.arange(len(X))!=i
            fn=RBFInterpolator(X[keep],Y[keep],kernel='thin_plate_spline',degree=1,smoothing=smoothing)
            errors.append((float(fn(X[i:i+1])[0])/Y[i]-1)**2)
        options.append(dict(smoothing=smoothing,training_group_LOO_relative_MSE=float(np.mean(errors))))
    best=min(options,key=lambda s:(s['training_group_LOO_relative_MSE'],s['smoothing']))
    return RBFInterpolator(X,Y,kernel='thin_plate_spline',degree=1,smoothing=best['smoothing']),dict(options=options,selected=best,unique_reflection_groups=len(X))

def design_qr(records):
    """Phase-aligned TRAIN Uy template snapshots, body candidates and pivoted QR.
    Greedy spacing prevents four candidate coordinates collapsing at thin D.
    """
    xb=np.arange(1.,10.001,.5);yb=np.arange(-3.,3.001,.5)
    candidates=np.asarray([(x,y) for y in yb for x in xb],float)
    columns=[]
    ref=np.asarray(records[0]['truthC'],float)
    for s in records:
        C,_=gauge_align(np.asarray(s['truthC'],float),ref,np.ones((64,64)))
        q=s['q'];xx=q['cx']+candidates[:,0]*q['D'];yy=q['cy']+candidates[:,1]*q['D']
        coords=np.array([(yy+8.)*63/16.,(xx+8.)*63/28.])
        val=np.asarray([map_coordinates(v,coords,order=1,mode='constant',cval=0.,prefilter=False) for v in C[:,1]])
        val*=HWEIGHT[:,0,0,0,None];columns.append(val/c.SCALES[1])
    A=np.vstack(columns);residual=A.copy();chosen=[];pivots=[]
    for k in range(4):
        norms=np.sum(residual**2,axis=0)
        for i in chosen:norms[np.linalg.norm(candidates-candidates[i],axis=1)<.9]=-np.inf
        index=int(np.argmax(norms));assert np.isfinite(norms[index])
        vector=residual[:,index]/max(np.linalg.norm(residual[:,index]),1e-15)
        pivots.append(float(norms[index]));residual-=vector[:,None]*(vector@residual)[None,:];chosen.append(index)
    selected=candidates[chosen]
    return selected,dict(source='TRAIN only phase-aligned Uy harmonic templates; finite body-coordinate candidate grid',
        candidate_region=[1.,10.,-3.,3.],candidate_step_D=.5,min_candidate_spacing_D=.9,
        coordinates_D=selected.tolist(),residual_pivot_norm_squared=pivots,training_ids=[s['id'] for s in records])

def build_lib(records,rank=32):
    assert len({s['q']['shape'] for s in records})==1
    shape=records[0]['q']['shape'];rank=min(int(rank),len(records)-1)
    reference=np.asarray(records[0]['truthC'],np.float64);templates=[];angles=[]
    for s in records:
        val,angle=gauge_align(np.asarray(s['truthC'],np.float64),reference,np.ones((64,64)))
        templates.append(val);angles.append(float(angle))
    templates=np.asarray(templates);mean=templates.mean(axis=0)
    A=((templates-mean)/SCALE*HWEIGHT).reshape(len(records),-1)
    eig,U=eigh(A@A.T);order=np.argsort(eig)[::-1];eig=np.maximum(eig[order],0.);U=U[:,order]
    sigma=np.sqrt(eig[:rank]);V=(U[:,:rank].T@A)/np.maximum(sigma[:,None],1e-15)
    amps=sigma/np.sqrt(len(records));modes=(V*amps[:,None]).reshape(rank,9,3,64,64)*SCALE/HWEIGHT
    scores=(A@V.T)/np.maximum(amps[None],1e-15)
    X=np.asarray([geometry_features(s['q']) for s in records]);candidates=[]
    # Inner interpolation CV may use this outer-fit POD; all labels remain outer-fit only.
    folds=sorted({int(s['fold']) for s in records})
    for smoothing in [0.,1e-3]:
        errors=[]
        for fold in folds:
            keep=np.asarray([s['fold']!=fold for s in records]);held=~keep
            if keep.sum()<5 or not held.any():continue
            fn=RBFInterpolator(X[keep],scores[keep],kernel='thin_plate_spline',degree=1,smoothing=smoothing)
            errors.extend(np.sum(((fn(X[held])-scores[held])*amps[None])**2,axis=1))
        candidates.append(dict(smoothing=smoothing,fit_group_CV_weighted_coefficient_MSE=float(np.mean(errors))))
    chosen=min(candidates,key=lambda s:(s['fit_group_CV_weighted_coefficient_MSE'],s['smoothing']))
    rbf=RBFInterpolator(X,scores,kernel='thin_plate_spline',degree=1,smoothing=chosen['smoothing'])
    fmodel,fd=frequency_model(records);qr_body,qd=design_qr(records)
    return dict(shape=shape,rank=rank,mean=mean,modes=modes,scores=scores,features=X,reference=reference,
        rbf=rbf,frequency_prior=fmodel,qr_body=qr_body,ids=[s['id'] for s in records],
        gauge_angles=angles,gauge_definition='Aligned C = rotate(raw TRAIN truthC, gauge_angle); basis phase raw minus gauge_angle.',
        amplitude=amps,energy=float(eig[:rank].sum()/max(eig.sum(),1e-15)),geometry_CV=candidates,selected_geometry_smoothing=chosen,
        frequency_diagnostic=fd,qr_diagnostic=qd,source_group_keys=[s['group_key'] for s in records])

def save_library(lib,name):
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/(name+'.npz')
    np.savez_compressed(path,**{k:lib[k] for k in ['mean','modes','scores','features','reference','qr_body','amplitude']})
    c.save(path.with_suffix('.json'),{k:lib[k] for k in ['shape','rank','ids','gauge_angles','gauge_definition','energy',
        'geometry_CV','selected_geometry_smoothing','frequency_diagnostic','qr_diagnostic','source_group_keys']})

def query_prior(lib,q,mask,rank=None):
    nr=lib['rank'] if rank is None else min(int(rank),lib['rank'])
    scores=lib['rbf'](geometry_features(q)[None])[0]
    C=lib['mean']+np.einsum('r,rjchw->jchw',scores[:nr],lib['modes'][:nr]);C=c.constrain(C,mask)
    f=float(lib['frequency_prior'](frequency_features(q)[None])[0])/q['D']
    assert f>0 and np.isfinite(C).all()
    return C,f,dict(geometry_RBF_scores=scores[:nr].tolist(),prior_frequency=f,rank=nr,target_full_fields_used=False)

def probe_indices(q,mask,lib=None,layout='center'):
    if layout=='center':return np.asarray(c.center_probes(q,mask),int)
    assert layout=='qr' and lib is not None
    return c.mapped_probes(q,mask,lib['qr_body'])

def _hist(row,ij):
    if 'history_values' in row:return np.asarray(row['history_values'],float)
    if 'history' in row:return np.asarray(row['history'],float)
    return c.history(row,ij)

def _weights(t,theta,f):return c.harmonic_basis(t,f,theta)
def _constraints_delta(B,mask):
    out=np.asarray(B,float).copy()*mask[None,None,None]
    out[:,:,0,:,0]=0.;out[:,:,1,:,0]=0.;out[:,:,1,0,:]=0.;out[:,:,1,-1,:]=0.
    return out

def fit_history(C,f,row,ij,B=None):
    """Fit origin phase/frequency and optional <=4 gappy shape corrections.
    Fields and pressure of target are never accessed. f is TRAIN geometry prior.
    """
    hist=_hist(row,ij);assert hist.shape==(8,len(ij)) and np.isfinite(hist).all()
    t=np.asarray(row.get('history_times',c.HISTORY_TIMES),float)
    H=np.stack([C[:,1,i,j] for i,j in ij]);r=0 if B is None else len(B)
    HB=np.empty((0,9,len(ij))) if r==0 else np.asarray([[v[:,1,i,j] for i,j in ij] for v in B]).transpose(0,2,1)
    noise=.05*c.SCALES[1]
    angles=np.linspace(-np.pi,np.pi,40,endpoint=False);freqs=np.linspace(.8*f,1.2*f,41)
    AA=np.tile(angles,len(freqs));FF=np.repeat(freqs,len(angles));theta=2*np.pi*FF[:,None]*t[None]+AA[:,None]
    basis=np.ones((len(AA),len(t),9))
    for k in range(1,5):basis[:,:,2*k-1]=np.cos(k*theta);basis[:,:,2*k]=np.sin(k*theta)
    errors=np.sum((np.einsum('gtj,ij->gti',basis,H)-hist[None])**2,axis=(1,2))/noise**2+((FF/f-1)/.05)**2
    candidates=np.argsort(errors)[:4];solutions=[]
    def residual(v):
        ph,ff=v[:2];a=v[2:]
        b=_weights(t,ph,ff);hc=H if r==0 else H+np.einsum('r,rji->ij',a,HB)
        res=((b@hc.T-hist)/noise).ravel()
        return np.r_[res,(ff/f-1)/.05,a]
    for k in candidates:
        start=np.r_[AA[k],FF[k],np.zeros(r)];lo=np.r_[AA[k]-np.pi,.8*f,np.full(r,-3.)];hi=np.r_[AA[k]+np.pi,1.2*f,np.full(r,3.)]
        opt=least_squares(residual,start,bounds=(lo,hi),max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        solutions.append(opt)
    best=min(solutions,key=lambda opt:np.sum(opt.fun**2));params=best.x[:2];a=best.x[2:]
    diag=dict(phase=float(params[0]),frequency=float(params[1]),prior_frequency=f,shape_coefficients=a.tolist(),
        history_Uy_RMSE=float(np.sqrt(np.mean(residual(best.x)[:hist.size]**2))*noise),optimizer_success=bool(best.success),
        nfev=int(best.nfev),frequency_at_bound=bool(abs(params[1]/f-1)>.1999),history_scalars=hist.size,target_pressure_used=False,
        sparse_history_grid_search=dict(phases=40,frequencies=41,refined_starts=4))
    return params,a,diag

def correction_basis(lib,C,params,row,ij,count=4,rank=32):
    full=_constraints_delta(lib['modes'][:min(rank,lib['rank'])],row['mask'])
    H=np.stack([C[:,1,i,j] for i,j in ij]);HB=np.asarray([[v[:,1,i,j] for i,j in ij] for v in full]).transpose(0,2,1)
    times=c.HISTORY_TIMES;b=c.harmonic_basis(times,params[1],params[0]);ds=[]
    for tt in times:ds.append(weights(params[0]+2*np.pi*params[1]*tt)[1])
    J=np.einsum('tj,rji->tir',b,HB).reshape(-1,len(full));derivative=(np.asarray(ds)@H.T).ravel()
    N=np.c_[derivative,np.repeat(times,len(ij))*derivative]
    J-=N@np.linalg.lstsq(N,J,rcond=1e-10)[0]
    _,sv,V=np.linalg.svd(J,full_matrices=False)
    keep=min(count,int(np.sum(sv>max(sv[0]*1e-3,.05*c.SCALES[1]))))
    B=np.einsum('vr,rjchw->vjchw',V[:keep],full)
    return B,dict(singular_values=sv.tolist(),active_shape_dimensions=keep,requested=count,
        selection='TRAIN POD projected onto most observable sparse-history directions after phase/f nuisance')

def forecast(C,B,params,row,ij,values,nsteps=None):
    """<=16s causal window ridge, emitted before each current measurement.
    nsteps=104 yields 26s+48 future scalars; nsteps=160 yields 40s+76.
    """
    nsteps=len(values) if nsteps is None else int(nsteps);assert nsteps in (104,160)
    r=0 if B is None else len(B);B=np.zeros((0,9,3,64,64)) if B is None else B
    H=np.stack([C[:,1,i,j] for i,j in ij]);HB=np.asarray([[v[:,1,i,j] for i,j in ij] for v in B]).transpose(0,2,1) if r else np.empty((0,9,len(ij)))
    hist=_hist(row,ij);history=[(float(tt),yy.copy()) for tt,yy in zip(c.HISTORY_TIMES,hist)]
    theta=float(params[0]);f=float(params[1]);f0=f;a=np.zeros(r);flat=C.reshape(9,-1);BF=B.reshape(r,9,-1) if r else None
    pred=[];updates=[];access=[];noise=.05*c.SCALES[1]
    for step in range(1,nsteps+1):
        t=step*.25;theta+=2*np.pi*f*.25;b,d=weights(theta)
        template=flat if not r else flat+np.einsum('r,rjf->jf',a,BF)
        pred.append((b@template).reshape(3,64,64))
        if step%8==0 and step<nsteps:
            y=np.asarray(values[step-1],float);assert np.isfinite(y).all();access.append(step)
            history.append((t,y.copy()));history=[v for v in history if t-v[0]<16.]
            dt=np.asarray([v[0]-t for v in history]);obs=np.asarray([v[1] for v in history]);ph0=theta;ff0=f;aa0=a.copy()
            def residual(v):
                ph=ph0+v[0];ff=ff0+v[1];aa=aa0+v[2:]
                bb=_weights(dt,ph,ff);hc=H if not r else H+np.einsum('r,rji->ij',aa,HB)
                return np.r_[((bb@hc.T-obs)/noise).ravel(),v[0]/np.deg2rad(5.),v[1]/(.02*f0),aa,(aa-aa0)/.5]
            lo=np.r_[-np.pi/2,.8*f0-ff0,-3.-aa0];hi=np.r_[np.pi/2,1.2*f0-ff0,3.-aa0]
            opt=least_squares(residual,np.zeros(2+r),bounds=(lo,hi),max_nfev=35,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            theta=ph0+opt.x[0];f=ff0+opt.x[1];a=aa0+opt.x[2:]
            updates.append(dict(step=step,time_seconds=t,phase=float(theta),frequency=float(f),a=a.tolist(),
                nfev=int(opt.nfev),optimizer_success=bool(opt.success),window_times=[v[0] for v in history]))
    pred=np.asarray(pred,np.float32)*row['mask'][None,None];assert np.isfinite(pred).all()
    return pred,dict(updates=updates,observed_future_steps=access,scalar_future_readings=len(access)*len(ij),
        history_scalars=hist.size,every_field_emitted_before_current_time_measurement=True,window_seconds_max=16.,
        end_forecast_gap_seconds=nsteps*.25-(access[-1]*.25 if access else 0),target_full_fields_used=False)

def summaries(cases):
    out={};selectors={'all':lambda s:True,'cylinder':lambda s:s['q']['shape']=='cylinder','square':lambda s:s['q']['shape']=='square',
        'thin_cylinder':lambda s:s['q']['shape']=='cylinder' and s['q']['D']<.9,'thin_square':lambda s:s['q']['shape']=='square' and s['q']['D']<.9,
        'DEV56':lambda s:'group' in s and s['group']!='latest_seen8','latest_seen8':lambda s:s.get('group')=='latest_seen8'}
    for name,select in selectors.items():
        part=[s for s in cases if select(s)]
        if not part:continue
        horizons=set.intersection(*[set(s['horizons']) for s in part])
        out[name]={h:c.summarize([s['horizons'][h] for s in part]) for h in sorted(horizons)}
    return out

def one(lib,row,rank,layout,mode_count):
    ij=probe_indices(row['q'],row['mask'],lib,layout);pack=c.packet(row,ij)
    sealed={k:row[k] for k in ['id','q','mask']};sealed.update(history_values=pack['history'],history_times=pack['history_times'])
    C,f,query=query_prior(lib,row['q'],row['mask'],rank);params,_,di=fit_history(C,f,sealed,ij)
    B,bi=correction_basis(lib,C,params,sealed,ij,mode_count,rank)
    if len(B):
        params,a,di=fit_history(C,f,sealed,ij,B);C=c.constrain(C+np.einsum('r,rjchw->jchw',a,B),row['mask'])
    pred,observer=forecast(C,B,params,sealed,ij,pack['future'],nsteps=len(pack['future']))
    return pred,dict(query=query,history_fit=di,shape_basis=bi,observer=observer,probe_indices=ij.tolist()),sealed,C,B,params,pack

def run(stage='all',extended=False):
    begin=time.perf_counter();c.p.protect();DEST.mkdir(parents=True,exist_ok=True)
    prereg=DEST/'preregister.json'
    if not prereg.exists():c.save(prereg,dict(utc=c.now(),ranks=[16,32],layouts=['center','qr'],shape_state_dimensions={16:2,32:4},
        library='TRAIN96 CFD truth H4; physical grid; phases aligned on fit groups only; POD geometry RBF and frequency and QR retrained for every outer fold',
        startup='8x4 Uy at -14,-12,...,0 =32 scalars, geometry/Re/mask; no target fullfield or target pressure reference adjustment',
        future='emit BEFORE readings at 2,...,38 s,76 scalars at40s; development26s readings2,...,24s,48scalars; final2s no readings',
        parameter_selection='Geometry smoothing0/1e-3 inner fit-group CV; frequency smoothing0/1e-6/1e-3 fit-group LOO; no DEV selection',
        observer='Sparse history prior frequency+-20%, <=4 identifiable shape combinations; <=16s causal window',
        no_neural_fields=True,pressure='fixed raw CFD reference; no target pressure read',evaluation='TRAIN four-fold geometry held-out and already-seen DEV; not independent'))
    train=c.load_train()
    for s in train:s['future_n']=104
    methods=[f'POD_r{rank}__{layout}' for rank in [16,32] for layout in ['center','qr']]
    checks=[]
    if stage in ('all','cv'):
        results={m:[] for m in methods}
        for fold in range(4):
            libraries={}
            for shape in ['cylinder','square']:
                fit=[s for s in train if s['q']['shape']==shape and s['fold']!=fold]
                held=[s for s in train if s['q']['shape']==shape and s['fold']==fold]
                assert not {s['group_key'] for s in fit}&{s['group_key'] for s in held}
                lib=build_lib(fit,32);libraries[shape]=lib;save_library(lib,f'{shape}_fold{fold}')
                print('LIBRARY',fold,shape,len(fit),lib['rank'],lib['qr_body'].tolist(),flush=True)
            for row in [s for s in train if s['fold']==fold]:
                for rank in [16,32]:
                    for layout in ['center','qr']:
                        method=f'POD_r{rank}__{layout}'
                        pred,di,*_=one(libraries[row['q']['shape']],row,rank,layout,2 if rank==16 else 4)
                        horizons=c.score_after_freeze(row,pred,DEST/'registry_cv'/f'{method}__{row["id"]}.json')
                        case=dict(id=row['id'],q=row['q'],fold=fold,horizons=horizons,diagnostic=di);results[method].append(case)
                c.save(DEST/'cv_partial.json',{m:dict(summary=summaries(v),cases=v) for m,v in results.items()})
                c.save(DEST/'status.json',dict(stage='geometry_CV',fold=fold,completed=len(results[methods[0]]),total=96,utc=c.now(),wall_seconds=time.perf_counter()-begin))
                print('ROM_CV',fold,len(results[methods[0]]),row['id'],flush=True)
        c.save(DEST/'cv_report.json',dict(role='Four-fold TRUE geometry CV: every model/library/prior/layout retrained only fit groups',methods={m:dict(summary=summaries(v),cases=v) for m,v in results.items()},wall_seconds=time.perf_counter()-begin))
    if stage in ('all','eval'):
        libraries={}
        for shape in ['cylinder','square']:
            lib=build_lib([s for s in train if s['q']['shape']==shape],32);libraries[shape]=lib;save_library(lib,shape+'_full')
        rows=c.load_dev(extended=extended);results={m:[] for m in methods}
        for index,row in enumerate(rows):
            frozen=[]
            for rank in [16,32]:
                for layout in ['center','qr']:
                    method=f'POD_r{rank}__{layout}'
                    pred,di,sealed,C,B,params,pack=one(libraries[row['q']['shape']],row,rank,layout,2 if rank==16 else 4)
                    if index==0:
                        poison=pack['future'].copy();poison[79:]=np.where(np.isfinite(poison[79:]),poison[79:]+100.,poison[79:])
                        other,_=forecast(C,B,params,sealed,pack['ij'],poison,len(poison))
                        delta=float(np.max(np.abs(other[:80]-pred[:80])));assert delta==0.
                        assert np.max(np.abs(pred[:,:,row['mask']<=.5]))==0.
                        checks.append(dict(method=method,late_measurement_poison_max_gap_first80=delta,solid_zero=True,target_full_fields_unread=True))
                    frozen.append((method,pred,di))
            c.save(DEST/('registry_eval40' if extended else 'registry_eval26')/f'all_candidates__{row["id"]}.json',dict(id=row['id'],utc=c.now(),
                all_candidates_frozen_before_any_target_future_field_score=True,
                registry=[dict(method=m,prediction_bytes_sha256=hashlib.sha256(pred.tobytes()).hexdigest()) for m,pred,_ in frozen]))
            for method,pred,di in frozen:
                horizons=c.score_after_freeze(row,pred,DEST/('registry_eval40' if extended else 'registry_eval26')/f'{method}__{row["id"]}.json')
                case=dict(id=row['id'],q=row['q'],group=row['group'],horizons=horizons,diagnostic=di);results[method].append(case)
            c.save(DEST/('eval40_partial.json' if extended else 'eval26_partial.json'),{m:dict(summary=summaries(v),cases=v) for m,v in results.items()})
            c.save(DEST/'status.json',dict(stage='seen_evaluation40' if extended else 'seen_evaluation26',completed=index+1,total=len(rows),utc=c.now(),wall_seconds=time.perf_counter()-begin))
            print('ROM_EVAL',index+1,len(rows),row['id'],flush=True)
        c.save(DEST/('eval40_report.json' if extended else 'eval26_report.json'),dict(role='Already-seen DEVELOPMENT; four-Uy-history task has no four-fullfield inherited scores',methods={m:dict(summary=summaries(v),cases=v) for m,v in results.items()},checks=checks,wall_seconds=time.perf_counter()-begin))
    c.p.protect();c.save(DEST/'status.json',dict(stage=stage,status='complete',utc=c.now(),wall_seconds=time.perf_counter()-begin))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','cv','eval'],default='all');ap.add_argument('--extended',action='store_true');args=ap.parse_args()
    try:run(args.stage,args.extended)
    except Exception as exc:DEST.mkdir(parents=True,exist_ok=True);c.save(DEST/'failure.json',dict(utc=c.now(),error=repr(exc)));raise
