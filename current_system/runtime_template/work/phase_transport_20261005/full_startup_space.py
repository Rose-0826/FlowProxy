#!/usr/bin/env python3
"""Finite richer-startup residual coefficient test, frozen v5 and same 76 future readings."""
from pathlib import Path
import os,sys,json,time,hashlib
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase_warp as w
p=w.p
sys.path.insert(0,str(p.ROOT/'work/sparse_campaign_20261005'))
import rom_body as body
DEST=w.DEST/'full_startup_space'
METHODS=['frozen_v5']+[f'body_r{rank}_noise{noise}' for rank in [4,16,32] for noise in [1,3]]

def make_lib(records):
    shape=records[0]['q']['shape'];ref,refsupport,_=body.to_body(records[0]['C'],records[0]['q'],records[0]['mask'])
    residuals=[];supports=[];meta=[]
    for row in records:
        C,support,sd=body.to_body(row['C'],row['q'],row['mask'])
        T,_=p.gauge_align(row['truthC'],row['C'],row['mask']);T,_,_=body.to_body(T,row['q'],row['mask'])
        _,angle=p.gauge_align(C,ref,(support&refsupport&(body.body_mask(shape)>.5)).astype(float))
        residual=p.remove_tangent(p.rotate(T-C,angle),p.rotate(C,angle))
        residuals.append(residual);supports.append(support);meta.append(dict(id=row['id'],q=row['q'],angle=angle,**sd))
    residuals=np.asarray(residuals);supports=np.asarray(supports);count=supports.sum(0)
    mean=np.sum(residuals*supports[:,None,None],axis=0)/np.maximum(count,1)[None,None]
    # Missing coordinates are supported-mean imputation, never zero-flow labels.
    filled=np.where(supports[:,None,None],residuals,mean[None]);filled*=body.body_mask(shape)[None,None,None]
    A=(filled/p.SCALE*p.HWEIGHT).reshape(len(records),-1)
    eig,U=eigh(A@A.T);order=np.argsort(eig)[::-1];eig=np.maximum(eig[order],0.);U=U[:,order]
    rank=min(32,len(records)-1);sigma=np.sqrt(eig[:rank]);V=(U[:,:rank].T@A)/np.maximum(sigma[:,None],1e-15)
    modes=(V*(sigma/np.sqrt(len(records)))[:,None]).reshape(rank,9,3,128,128)*p.SCALE/p.HWEIGHT
    lib=dict(shape=shape,reference=ref,reference_support=refsupport,modes=modes,support_count=count,energy=(np.cumsum(eig)/max(eig.sum(),1e-20)).tolist(),source=meta)
    np.savez_compressed(DEST/(shape+'_library.npz'),reference=ref,reference_support=refsupport,modes=modes,support_count=count)
    p.save(DEST/(shape+'_library.json'),{k:lib[k] for k in ['shape','source','energy']})
    return lib

def modes_for(lib,C,row):
    can,support,_=body.to_body(C,row['q'],row['mask']);valid=support&lib['reference_support']&(body.body_mask(lib['shape'])>.5)
    _,angle=p.gauge_align(can,lib['reference'],valid.astype(float))
    modes=[]
    for m in lib['modes']:
        z=p.rotate(m,-angle);z=body.to_physical(z,row['q'],row['mask'],delta=True)
        modes.append(p.delta_constraints(p.remove_tangent(z,C),row['mask']))
    return np.asarray(modes)

def fit(C,params,row,modes,rank,noise):
    modes=modes[:rank];B=p.basis(row['times'][:4]-row['times'][3],params[1],params[0])
    active=row['mask']>.5
    target=row['fields']-np.einsum('tj,jchw->tchw',B,C)
    designs=np.einsum('tj,kjchw->ktchw',B,modes)
    scaled=designs/p.SCALE[None];residual=target/p.SCALE
    J=scaled[:,:,:,active].reshape(rank,-1).T;v=residual[:,:,active].ravel();N=len(v)
    G=J.T@J/N;lam=(noise/100.)**2;a=np.linalg.solve(G+lam*np.eye(rank),J.T@v/N)
    out=w.constrain(C+np.einsum('k,kjchw->jchw',a,modes),row['mask'])
    eig=np.linalg.eigvalsh(G);df=float(np.sum(eig/(eig+lam)))
    return out,dict(rank=rank,prior_coefficient_std=1.,assumed_field_noise_fraction=noise/100.,ridge=lam,coefficients=a.tolist(),
        effective_degrees_of_freedom=df,startup_normal_matrix_eigenvalues=eig.tolist(),startup_fit_MSE_before=float(np.mean(v*v)),
        startup_fit_MSE_after=float(np.mean((J@a-v)**2)),observed_fields=4,future_field_used=False,spatial_states_fixed_after_startup=True)

def oracle(C,T,modes,row,rank):
    active=row['mask']>.5;mm=modes[:rank]/p.SCALE[None]*p.HWEIGHT[None]
    residual=(T-C)/p.SCALE*p.HWEIGHT;A=mm[:,:,:,active].reshape(rank,-1).T;v=residual[:,:,active].ravel()
    a=np.linalg.lstsq(A,v,rcond=1e-8)[0];out=C+np.einsum('k,kjchw->jchw',a,modes[:rank])
    vals=[]
    for j in range(3):
        den=np.linalg.norm(T[:,j,active]*p.HWEIGHT[:,0,0,0,None]);vals.append(float(np.linalg.norm((out-T)[:,j,active]*p.HWEIGHT[:,0,0,0,None])/max(den,1e-12)))
    return vals

def run():
    begin=time.perf_counter();p.protect()
    p.save(DEST/'preregister.json',dict(utc=p.now(),source_sha256=p.digest(__file__),methods=METHODS,
        task='ORIGINAL 4 full UxUyP startup fields +76 future Uy readings; 40s; exact first8s',
        new_mechanism='estimate 4/16/32 residual spatial coefficients from complete startup fields, then freeze space and observe only theta/f; no four-sensor highdimensional update',
        library='TRAIN96 true H4 minus frozen FNO H4, uniform-phase aligned and phase tangent removed; per shape expanded body128 grid x[-14,34],y[-15,15]; supported mean missing imputation, noncentered SVD',
        fit='startup normalized all three channels; coefficient prior variance1 and fixed 1%/3% field error ridge; no target future for fit',
        evaluation='all64 have already been seen, DEVELOPMENT ONLY; all method predictions hashed before each case futurefield scoring; subsequent correction-only grouped CV required if gates pass',
        guards='DEV meanR>=15%lower andmaxnotworse; latest8 at least6passes/meanR<1 andmaxnotworse; first8exact; then fourfold correction-module CV andnoise/perturbation beforeindependentfreeze',
        storage='server-only; compact failures then remove regenerable dictionaries; no promotion fromdevelopment'))
    records=p.read_train();libs={shape:make_lib([x for x in records if x['q']['shape']==shape]) for shape in ['cylinder','square']}
    # TRAIN-only oracle is clearly segregated from deployable startup coefficients.
    diag=[]
    for row in records:
        row={**row,'times':row['times'],'fields':None};C=row['C'];T,_=p.gauge_align(row['truthC'],C,row['mask']);modes=modes_for(libs[row['q']['shape']],C,row)
        diag.append(dict(id=row['id'],q=row['q'],oracle_only=True,errors_UxUyP={str(k):oracle(C,T,modes,row,k) for k in [4,16,32]}))
    p.save(DEST/'TRAIN_oracle_diagnostic.json',dict(role='TRAIN96 oracle representation only, neither startup predictor nor wholeFNO-heldout',cases=diag))
    result={m:[] for m in METHODS};checks=[]
    for index,q in enumerate(p.dev_records()):
        row=p.startup(q)
        with np.load(p.DEST/'dev_bootstrap'/(q['id']+'.npz')) as z:C=z['C'];params=z['params'];own=z['own']
        modes=modes_for(libs[row['q']['shape']],C,row);values,_=p.get_measurements(q);frozen=[]
        for method in METHODS:
            if method=='frozen_v5':coef=C.copy();di={'baseline':True}
            else:
                parts=method.split('_');rank=int(parts[1][1:]);noise=int(parts[2][5:]);coef,di=fit(C,params,row,modes,rank,noise)
            raw,obs=p.observe(coef,params[1],row,values,params);pred=p.hybrid(own,raw)
            assert np.array_equal(pred[:32],own[:32]) and obs['scalar_future_readings']==76
            assert np.max(abs(pred[:,:,row['mask']<=.5]))==0.
            if index==0:
                poison=values.copy();poison[79:]+=100.;other,_=p.observe(coef,params[1],row,poison,params)
                gap=float(np.max(abs(pred[:80]-p.hybrid(own,other)[:80])));assert gap==0.;checks.append(dict(method=method,later_measurement_poison_gap=gap,first8exact=True))
            frozen.append((method,pred,di,obs))
        p.save(DEST/'registry'/f'{q["id"]}.json',dict(utc=p.now(),all_candidates_before_case_target_future_scoring=True,
            candidates=[dict(method=m,sha256=hashlib.sha256(pred.tobytes()).hexdigest(),fit=di,observer=obs) for m,pred,di,obs in frozen]))
        with np.load(Path(q['cache'])/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
        for method,pred,di,obs in frozen:result[method].append(dict(id=q['id'],q=row['q'],group=q['group'],horizons=p.metrics(pred,truth,row['mask'],row['times'])))
        p.save(DEST/'status.json',dict(stage='fixed_seen64',completed=index+1,total=64,utc=p.now()))
        p.save(DEST/'partial.json',{m:dict(summary=p.eval_summary(a),cases=a) for m,a in result.items()});print('FULL_STARTUP',index+1,q['id'],flush=True)
    report={m:dict(summary=p.eval_summary(a),cases=a) for m,a in result.items()};base=report['frozen_v5']['summary'];guards=[]
    for method,a in report.items():
        if method=='frozen_v5':continue
        ds=a['summary']['DEV56']['40s'];ls=a['summary']['latest_seen8']['40s'];bd=base['DEV56']['40s'];bl=base['latest_seen8']['40s']
        g=dict(DEV_mean15pct_lower=ds['mean_R']<=.85*bd['mean_R'],DEVmaxnotworse=ds['max_R']<=bd['max_R'],latest8passge6=ls['n_pass']>=6,latest8meanlt1=ls['mean_R']<1,latest8maxnotworse=ls['max_R']<=bl['max_R'])
        guards.append(dict(method=method,guards=g,passed=all(g.values())))
    p.save(DEST/'report.json',dict(role='SEEN development ONLY',methods=report,guards=guards,checks=checks,wall_seconds=time.perf_counter()-begin))
    p.protect();p.save(DEST/'status.json',dict(status='complete',utc=p.now()));print('VERDICT',guards,flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:p.save(DEST/'failure.json',dict(error=repr(e),utc=p.now()));raise
