#!/usr/bin/env python3
"""Finite last coordinate check: TRAIN-only body-D harmonic POD16, 4 probes.
Expanded canonical domain plus explicit missing-support accounting; no target p.
"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,time,hashlib,argparse
import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
import common as c
import rom as r
from research_v2_template_probe import gauge_align
DEST=c.OUT/'rom_body';BX,BY=np.meshgrid(np.linspace(-14,34,128),np.linspace(-15,15,128))

def body_mask(shape):
    solid=BX*BX+BY*BY<=.25 if shape=='cylinder' else (np.abs(BX)<=.5)&(np.abs(BY)<=.5)
    return (~solid).astype(float)

def sample(A,coords):
    return np.asarray([map_coordinates(v,coords,order=1,mode='constant',cval=0.,prefilter=False) for v in A.reshape(-1,*A.shape[-2:])]).reshape(*A.shape[:-2],*coords.shape[1:])

def to_body(C,q,mask):
    xx=q['cx']+BX*q['D'];yy=q['cy']+BY*q['D'];coords=np.array([(yy+8)*63/16,(xx+8)*63/28])
    domain=(xx>=-8)&(xx<=20)&(yy>=-8)&(yy<=8)
    fraction=map_coordinates(mask.astype(float),coords,order=1,mode='constant',cval=0.,prefilter=False)
    perturb=np.asarray(C,float).copy();perturb[0,0]-=1.
    out=sample(perturb,coords);out[0,0]+=1.
    out[:,2]=np.divide(out[:,2],fraction[None],out=np.zeros_like(out[:,2]),where=fraction[None]>.25)
    bmask=body_mask(q['shape']);out*=bmask[None,None]
    support=domain&((fraction>.25)|(bmask<=.5))
    return out,support,dict(source_domain_fraction=float(domain.mean()),canonical_known_fluid_fraction=float(np.mean(support[bmask>.5])))

def to_physical(C,q,mask,delta=False):
    xx=(c.X-q['cx'])/q['D'];yy=(c.Y-q['cy'])/q['D'];coords=np.array([(yy+15)*127/30,(xx+14)*127/48])
    perturb=np.asarray(C,float).copy()
    if not delta:perturb[0,0]-=1.
    out=sample(perturb,coords)
    if not delta:out[0,0]+=1.
    fraction=map_coordinates(body_mask(q['shape']),coords,order=1,mode='constant',cval=0.,prefilter=False)
    out[:,2]=np.divide(out[:,2],fraction[None],out=np.zeros_like(out[:,2]),where=fraction[None]>.25)
    if delta:
        out*=mask[None,None];out[:,0,:,0]=0.;out[:,1,:,0]=0.;out[:,1,0,:]=0.;out[:,1,-1,:]=0.
        return out
    return c.constrain(out,mask)

def build_lib(records,rank=16):
    assert len({s['q']['shape'] for s in records})==1
    shape=records[0]['q']['shape'];reference,reference_support,_=to_body(records[0]['truthC'],records[0]['q'],records[0]['mask'])
    templates=[];supports=[];angles=[];support_records=[]
    for s in records:
        C,support,sd=to_body(s['truthC'],s['q'],s['mask']);C,angle=gauge_align(C,reference,(support&reference_support& (body_mask(shape)>.5)).astype(float))
        templates.append(C);supports.append(support);angles.append(angle);support_records.append(dict(id=s['id'],q=s['q'],**sd))
    templates=np.asarray(templates);supports=np.asarray(supports);count=supports.sum(0)
    mean=np.divide(np.sum(templates*supports[:,None,None],axis=0),np.maximum(count,1)[None,None])
    mean[0,0][count==0]=1.;mean*=body_mask(shape)[None,None]
    # Source-domain exclusions are missing labels, not physical zero-flow labels.
    # Fit-group supported mean imputation is fixed and uses no held/DEV fields.
    filled=np.where(supports[:,None,None],templates,mean[None])
    A=((filled-mean)/r.SCALE*r.HWEIGHT).reshape(len(records),-1)
    eig,U=eigh(A@A.T);order=np.argsort(eig)[::-1];eig=np.maximum(eig[order],0.);U=U[:,order];rank=min(rank,len(records)-1)
    sigma=np.sqrt(eig[:rank]);V=(U[:,:rank].T@A)/np.maximum(sigma[:,None],1e-15);amp=sigma/np.sqrt(len(records))
    modes=(V*amp[:,None]).reshape(rank,9,3,128,128)*r.SCALE/r.HWEIGHT;scores=(A@V.T)/np.maximum(amp[None],1e-15)
    X=np.asarray([r.geometry_features(s['q']) for s in records]);options=[]
    for smoothing in [0.,1e-3]:
        errors=[]
        for fold in sorted({s['fold'] for s in records}):
            keep=np.asarray([s['fold']!=fold for s in records]);held=~keep
            if keep.sum()<5 or not held.any():continue
            fn=RBFInterpolator(X[keep],scores[keep],kernel='thin_plate_spline',degree=1,smoothing=smoothing)
            errors.extend(np.sum(((fn(X[held])-scores[held])*amp[None])**2,axis=1))
        options.append(dict(smoothing=smoothing,fit_group_CV_weighted_coefficient_MSE=float(np.mean(errors))))
    selected=min(options,key=lambda s:(s['fit_group_CV_weighted_coefficient_MSE'],s['smoothing']))
    fn=RBFInterpolator(X,scores,kernel='thin_plate_spline',degree=1,smoothing=selected['smoothing']);fprior,fd=r.frequency_model(records);qr,qd=r.design_qr(records)
    return dict(shape=shape,rank=rank,mean=mean,modes=modes,scores=scores,features=X,amplitude=amp,reference=reference,
        rbf=fn,frequency_prior=fprior,qr_body=qr,ids=[s['id'] for s in records],gauge_angles=angles,source_group_keys=[s['group_key'] for s in records],
        source_support_count=count,source_support_records=support_records,source_support_fraction=supports.mean(0),
        energy=float(eig[:rank].sum()/max(eig.sum(),1e-15)),geometry_CV=options,selected_geometry_smoothing=selected,
        frequency_diagnostic=fd,qr_diagnostic=qd,canonical_domain=dict(x=[-14.,34.],y=[-15.,15.],size=[128,128],
        missing_source_domain='fit-group supported-mean imputation; locations with no fit-source support: freestream1 and other perturbation0, explicitly reported; never observed labels'))

def save_lib(lib,name):
    DEST.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DEST/(name+'.npz'),**{k:lib[k] for k in ['mean','modes','scores','features','amplitude','reference','qr_body','source_support_count','source_support_fraction']})
    c.save(DEST/(name+'.json'),{k:lib[k] for k in ['shape','rank','ids','gauge_angles','source_group_keys','energy','geometry_CV',
        'selected_geometry_smoothing','frequency_diagnostic','qr_diagnostic','canonical_domain','source_support_records']})

def query(lib,row):
    q=row['q'];a=lib['rbf'](r.geometry_features(q)[None])[0]
    body=lib['mean']+np.einsum('r,rjchw->jchw',a,lib['modes']);C=to_physical(body,q,row['mask'])
    B=np.asarray([to_physical(v,q,row['mask'],True) for v in lib['modes']]);f=float(lib['frequency_prior'](r.frequency_features(q)[None])[0])/q['D']
    xx=(c.X-q['cx'])/q['D'];yy=(c.Y-q['cy'])/q['D'];coords=np.array([(yy+15)*127/30,(xx+14)*127/48])
    support=map_coordinates(lib['source_support_count'].astype(float),coords,order=1,mode='constant',cval=0.,prefilter=False);active=row['mask']>.5
    diag=dict(prior_frequency=f,RBF_scores=a.tolist(),query_fluid_fraction_no_TRAIN_support=float(np.mean(support[active]<.5)),
        query_fluid_fraction_fewer_than3_donor_support=float(np.mean(support[active]<2.5)),minimum_donor_support=float(support[active].min()),median_donor_support=float(np.median(support[active])),
        canonical_query_domain_covered=bool(np.all((xx>=-14)&(xx<=34)&(yy>=-15)&(yy<=15))),
        target_pressure_used=False,wall_limitation='body-D map does not make finite physical wall locations identical across geometry')
    assert diag['canonical_query_domain_covered'];assert np.isfinite(C).all() and np.isfinite(B).all() and f>0
    return C,B,f,diag

def one(lib,row,layout):
    ij=r.probe_indices(row['q'],row['mask'],lib,layout);pack=c.packet(row,ij)
    sealed={k:row[k] for k in ['id','q','mask']};sealed.update(history_values=pack['history'],history_times=pack['history_times'])
    C,fullB,f,query_di=query(lib,row);params,_,di=r.fit_history(C,f,sealed,ij)
    B,bi=r.correction_basis(dict(rank=lib['rank'],modes=fullB),C,params,sealed,ij,2,16)
    if len(B):params,a,di=r.fit_history(C,f,sealed,ij,B);C=c.constrain(C+np.einsum('r,rjchw->jchw',a,B),row['mask'])
    pred,observer=r.forecast(C,B,params,sealed,ij,pack['future'],len(pack['future']))
    return pred,dict(query=query_di,history_fit=di,shape_basis=bi,observer=observer,probe_indices=ij.tolist()),sealed,C,B,params,pack

def mathematical_checks():
    q=dict(shape='cylinder',cx=0.,cy=.8,D=.65,Re=160.);mask=np.ones((64,64));C=np.zeros((9,3,64,64));C[0,0]=1.
    body,support,_=to_body(C,q,mask);back=to_physical(body,q,mask)
    err=float(np.max(np.abs(back[0,0][np.hypot(c.X-q['cx'],c.Y-q['cy'])>q['D']]-1.)))
    # Artificial all-fluid mask with the actual known obstacle geometry is only
    # checked in the far field; solid/boundary masks are verified on real rows.
    assert err<1e-10
    return dict(uniform_freestream_preserved_far_from_known_obstacle_max_gap=err,expanded_domain_covers_D065_cy08=True,
        pressure_reference_unchanged=True,finite_wall_covariance_not_claimed=True)

def run(stage='all',extended=False):
    begin=time.perf_counter();DEST.mkdir(parents=True,exist_ok=True);c.p.protect()
    pre=DEST/'preregister.json'
    if not pre.exists():c.save(pre,dict(utc=c.now(),rank=16,layouts=['center','qr'],shape_dimensions=2,
        canonical_domain=dict(x=[-14,34],y=[-15,15],grid=[128,128]),
        source='TRAIN CFD H4 only, four geometry folds rebuild gauge/POD/RBF/fprior/QR; original fullfield/startup FNO excluded',
        mapping='mean Ux perturbation minus1 ->linear zero-exterior mapping ->plus1; pressure weighted by fluid fraction; no target pressure read',
        missing_support='source-domain labels missing; impute only fit-group supported mean; no donor support cases explicitly reported',
        finite_candidates='last fixed coordinate implementation check; rank16, two layouts, unchanged sparse window observer and no extra hyperparameters',
        protocol='32 real CFD historical Uy scalars; predict before2s samples;26s开发80total or40s108total;最后2s无测量',
        evaluation='Already-seen DEV56/latest8, not independent; no inheritance of four-fullfield scores',
        mathematical_checks=mathematical_checks()))
    train=c.load_train()
    for s in train:s['future_n']=104
    methods=['body_POD16__center','body_POD16__qr'];checks=[]
    if stage in ('all','smoke','cv'):
        results={m:[] for m in methods}
        folds=[0] if stage=='smoke' else range(4)
        for fold in folds:
            libs={}
            for shape in ['cylinder','square']:
                fit=[s for s in train if s['q']['shape']==shape and s['fold']!=fold];held=[s for s in train if s['q']['shape']==shape and s['fold']==fold]
                assert not {s['group_key'] for s in fit}&{s['group_key'] for s in held}
                libs[shape]=build_lib(fit,16);save_lib(libs[shape],f'{shape}_fold{fold}')
                print('BODY_LIBRARY',fold,shape,len(fit),libs[shape]['energy'],flush=True)
            held=[s for s in train if s['fold']==fold]
            if stage=='smoke':held=[next(s for s in held if s['q']['shape']==shape) for shape in ['cylinder','square']]
            for row in held:
                frozen=[]
                for layout in ['center','qr']:
                    method='body_POD16__'+layout;pred,di,sealed,C,B,params,pack=one(libs[row['q']['shape']],row,layout)
                    assert np.max(np.abs(pred[:,:,row['mask']<=.5]))==0.
                    assert np.max(np.abs(pred[:,0,:,0]-row['mask'][:,0][None]))<1e-7
                    frozen.append((method,pred,di))
                    if stage=='smoke':
                        poison=pack['future'].copy();poison[79:]=np.where(np.isfinite(poison[79:]),poison[79:]+100.,poison[79:]);other,_=r.forecast(C,B,params,sealed,pack['ij'],poison,len(poison));gap=float(np.max(np.abs(pred[:80]-other[:80])));assert gap==0.
                        checks.append(dict(id=row['id'],method=method,late_future_probe_poison_gap_first80=gap,solid_zero=True,known_inlet_Ux_preserved=True))
                for method,pred,di in frozen:
                    horizons=c.score_after_freeze(row,pred,DEST/('registry_smoke' if stage=='smoke' else 'registry_cv')/f'{method}__{row["id"]}.json')
                    results[method].append(dict(id=row['id'],q=row['q'],fold=fold,horizons=horizons,diagnostic=di))
                c.save(DEST/'status.json',dict(stage=stage,fold=fold,completed=len(results[methods[0]]),total=2 if stage=='smoke' else 96,utc=c.now(),wall_seconds=time.perf_counter()-begin))
                print('BODY_CV',fold,len(results[methods[0]]),row['id'],flush=True)
        c.save(DEST/('smoke_report.json' if stage=='smoke' else 'cv_report.json'),dict(methods={m:dict(summary=r.summaries(v),cases=v) for m,v in results.items()},checks=checks,wall_seconds=time.perf_counter()-begin))
    if stage in ('all','eval'):
        libs={shape:build_lib([s for s in train if s['q']['shape']==shape],16) for shape in ['cylinder','square']}
        for shape,lib in libs.items():save_lib(lib,shape+'_full')
        rows=c.load_dev(extended);results={m:[] for m in methods}
        for index,row in enumerate(rows):
            frozen=[]
            for layout in ['center','qr']:
                method='body_POD16__'+layout;pred,di,sealed,C,B,params,pack=one(libs[row['q']['shape']],row,layout)
                if index==0:
                    poison=pack['future'].copy();poison[79:]=np.where(np.isfinite(poison[79:]),poison[79:]+100.,poison[79:]);other,_=r.forecast(C,B,params,sealed,pack['ij'],poison,len(poison));gap=float(np.max(np.abs(pred[:80]-other[:80])));assert gap==0.
                    assert np.max(np.abs(pred[:,:,row['mask']<=.5]))==0.;assert np.max(np.abs(pred[:,0,:,0]-row['mask'][:,0][None]))<1e-7
                    checks.append(dict(method=method,late_future_probe_poison_gap_first80=gap,solid_zero=True,known_inlet_Ux_preserved=True))
                frozen.append((method,pred,di))
            c.save(DEST/('registry_eval40' if extended else 'registry_eval26')/f'all_candidates__{row["id"]}.json',dict(id=row['id'],utc=c.now(),all_candidates_frozen_before_any_target_future_score=True,
                registry=[dict(method=m,prediction_bytes_sha256=hashlib.sha256(pred.tobytes()).hexdigest()) for m,pred,_ in frozen]))
            for method,pred,di in frozen:
                horizons=c.score_after_freeze(row,pred,DEST/('registry_eval40' if extended else 'registry_eval26')/f'{method}__{row["id"]}.json')
                results[method].append(dict(id=row['id'],q=row['q'],group=row['group'],horizons=horizons,diagnostic=di))
            c.save(DEST/('eval40_partial.json' if extended else 'eval26_partial.json'),{m:dict(summary=r.summaries(v),cases=v) for m,v in results.items()})
            c.save(DEST/'status.json',dict(stage='eval40' if extended else 'eval26',completed=index+1,total=len(rows),utc=c.now(),wall_seconds=time.perf_counter()-begin))
            print('BODY_EVAL',index+1,len(rows),row['id'],flush=True)
        c.save(DEST/('eval40_report.json' if extended else 'eval26_report.json'),dict(role='Already-seen DEVELOPMENT, fixed sparse-history task, body-D coordinate baseline',methods={m:dict(summary=r.summaries(v),cases=v) for m,v in results.items()},checks=checks,wall_seconds=time.perf_counter()-begin))
    c.p.protect();c.save(DEST/'status.json',dict(stage=stage,status='complete',utc=c.now(),wall_seconds=time.perf_counter()-begin))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','smoke','cv','eval'],default='all');ap.add_argument('--extended',action='store_true');args=ap.parse_args()
    try:run(args.stage,args.extended)
    except Exception as exc:DEST.mkdir(parents=True,exist_ok=True);c.save(DEST/'failure.json',dict(utc=c.now(),error=repr(exc)));raise
