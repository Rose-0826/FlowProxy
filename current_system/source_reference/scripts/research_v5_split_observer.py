#!/usr/bin/env python3
"""Separate short forecast, causal clock, and startup-only spatial calibration."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,score,summaries,scales,protected,now
from research_v4_sparse_observer import probes,weights,CausalMeasurements,constrain
from research_v2_template_probe import fit_startup,calibrate,basis
from geometry_attack_v2 import save,sha

DEST=OUT/'next_stage_split_observer'
PILOT=DEST/'inference_pilot'
METHODS=['observer_baseline','hybrid8','hybrid8_pmean','hybrid8_pH1',
         'hybrid8_pH1_half','hybrid8_H1_fixedmean','hybrid8_H1_fixedmean_half',
         'hybrid8_UyPcal','hybrid8_allcal']

def rotate(C,phase):
    out=C.copy()
    for k in range(1,5):
        c,s=np.cos(k*phase),np.sin(k*phase)
        out[2*k-1]=C[2*k-1]*c+C[2*k]*s
        out[2*k]=-C[2*k-1]*s+C[2*k]*c
    return out

def neural_template(row,own,n,f):
    obs=np.concatenate([row['fields'],own[:n]])
    t=row['times'][:n+4]-row['times'][0]
    return constrain(np.linalg.lstsq(basis(t,f),obs.reshape(len(t),-1),
        rcond=1e-10)[0].reshape(9,3,64,64),row['mask'])

def correction(C,row,params,method):
    result=C.copy()
    if method in ['observer_baseline','hybrid8']:return result
    t=row['times'][:4]-row['times'][3]
    B=basis(t,params[1],params[0])
    residual=row['fields'].reshape(4,-1)-B@C.reshape(9,-1)
    if method=='hybrid8_pmean':
        result[0,2]+=residual.reshape(4,3,64,64)[:,2].mean(0)
    elif method in ['hybrid8_H1_fixedmean','hybrid8_H1_fixedmean_half']:
        A=B[:,1:3];G=A.T@A;lam=.03*np.trace(G)/2
        delta=np.linalg.solve(G+lam*np.eye(2),A.T@residual).reshape(2,3,64,64)
        result[1:3]+=(.5 if method.endswith('half') else 1.)*delta
    else:
        delta=calibrate(C,row,params)-C
        if method in ['hybrid8_pH1','hybrid8_pH1_half']:
            result[:,2]+=(.5 if method.endswith('half') else 1.)*delta[:,2]
        elif method=='hybrid8_UyPcal':result[:,1:]+=delta[:,1:]
        elif method=='hybrid8_allcal':result+=delta
        else:raise ValueError(method)
    return constrain(result,row['mask'])

def observe(C,f,row,values,params,interval=8):
    """Fixed startup phase: spatial changes never trigger a second full-field fit."""
    theta=float(params[0]);omega=2*np.pi*f;omega0=omega
    ij,_=probes(row);H=np.stack([C[:,1,i,j] for i,j in ij],0)
    reader=CausalMeasurements(values,interval)
    R=np.eye(4)*(.05*scales()[1])**2
    z=np.array([theta,omega]);P=np.diag([np.deg2rad(5.)**2,(.02*omega)**2])
    A=np.array([[1.,.25],[0.,1.]])
    Q=np.diag([np.deg2rad(.5)**2,(.0005*omega)**2])*.25
    predictions=[];phases=[];updates=[]
    for step in range(1,161):
        z=A@z;P=A@P@A.T+Q;b,_=weights(z[0]);phases.append(float(z[0]))
        predictions.append((b@C.reshape(9,-1)).reshape(3,64,64))
        if step%interval==0 and step<160:
            y=reader.read(step);b,d=weights(z[0]);h=np.c_[H@d,np.zeros(4)]
            gain=np.linalg.solve(h@P@h.T+R,h@P).T;residual=y-H@b
            delta=gain@residual;delta[0]=np.clip(delta[0],-np.pi/2,np.pi/2)
            z+=delta;z[1]=np.clip(z[1],.8*omega0,1.2*omega0)
            J=np.eye(2)-gain@h;P=J@P@J.T+gain@R@gain.T
            updates.append({'step':step,'frequency':float(z[1]/(2*np.pi)),
                'sensor_residual_norm':float(np.linalg.norm(residual))})
        assert np.isfinite(P).all()
    pred=np.asarray(predictions,np.float32)*row['mask'][None,None]
    return pred,{'scalar_future_readings':4*len(reader.access),
        'observed_future_steps':reader.access,'last_reading_seconds':reader.access[-1]*.25,
        'end_gap_seconds':40-reader.access[-1]*.25,
        'every_field_emitted_before_current_time_measurement':True,
        'initial_phase':theta,'initial_frequency':f,'phases':phases,'updates':updates}

def hybrid(own,pred):
    """Exact original autonomous fields for first8s, fixed4s transition afterwards."""
    out=pred.copy();out[:32]=own[:32]
    for j in range(32,48):
        alpha=(j+1-32)/16
        out[j]=(1-alpha)*own[j]+alpha*pred[j]
    assert np.array_equal(out[:32],own[:32])
    return out

def context(q):
    row=startup(q);source=OUT/'next_round/clock_probe96'
    registry=json.loads((source/'forecasts_frozen.json').read_text())['registry']
    d=next(a['diagnostic'] for a in registry if a['id']==q['id'] and a['method']=='self_H4')
    with np.load(source/'predictions'/('frozen_research__'+q['id']+'.npz')) as a:own=a['prediction']
    C=neural_template(row,own,d['prefix_steps'],d['self_frequency'])
    params,initial=fit_startup(C,row,d['self_frequency'],False)
    with np.load(OUT/'sparse_observer_pilot/measurements'/(q['id']+'.npz')) as a:
        clean=a['clean'];noisy=a['noisy']
    return row,own,C,params,initial,clean,noisy

def balanced(summary,h='40s'):
    return .5*(summary['development_inside24'][h]['mean_R']+summary['seen_extreme32'][h]['mean_R'])

def candidate_guard(summary,baseline,short):
    a=summary['development_all56'];b=baseline['development_all56']
    return (a['8s']['n_pass']>=short['development_all56']['8s']['n_pass'] and
        a['40s']['n_pass']>=b['40s']['n_pass'] and
        a['40s']['max_R']<=b['40s']['max_R']*1.01 and
        all(summary[g]['40s']['mean_R']<=baseline[g]['40s']['mean_R']*1.01
            for g in ['development_inside24','seen_extreme32']))

def main():
    protected();PILOT.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    config={'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'selection_data':'old56 DEVELOPMENT only; all previous independent8 batches excluded',
        'hybrid':'exact frozen autonomous branch first32steps;linear mix steps33..48;observer afterwards',
        'pressure':'startup-only mean or fixed-ridge mean/H1 correction;baseline startup phase frozen',
        'H1':'optional fixed-mean H1 ridge on4 startup fields,lambda=.03 trace/2',
        'observations':'same4 Uy probes every2s,76 scalars;no t40 reading;emit before current-time reading',
        'selection':'preserve original short8s passes,at least main-observer40s passes,no worse either-group meanR or maxR (1% tolerance);min balancedR40',
        'signal':'selected balancedR40 at least10% lower and >=48/56 pass,noise pass loss<=2',
        'fresh_confirmation':'only after final two-seed learned experiment or inference winner frozen;no automatic fallback replacement'}
    save(PILOT/'preregister.json',config)
    forecasts={m:[] for m in METHODS};registry=[];noise_forecasts={};checks=[]
    for q in rows():
        row,own,C,params,initial,clean,noisy=context(q)
        base_pred,diag=observe(C,params[1],row,clean,params)
        with np.load(OUT/'sparse_observer_pilot/predictions'/('neural_EKF_2s__'+q['id']+'.npz')) as a:old=a['prediction']
        gap=float(np.max(np.abs(old-base_pred)));assert gap<2e-6
        checks.append({'id':q['id'],'old_observer_max_abs_gap':gap})
        for method in METHODS:
            coeff=correction(C,row,params,method)
            pred,detail=observe(coeff,params[1],row,clean,params)
            if method!='observer_baseline':pred=hybrid(own,pred)
            forecasts[method].append((q,row,pred))
            registry.append({'id':q['id'],'method':method,
                'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),
                'scalar_future_readings':detail['scalar_future_readings'],
                'startup_phase_fixed':True,'first8s_equal_original':method!='observer_baseline'})
        save(PILOT/'status.json',{'status':'predicting','utc':now(),
            'completed_cases':len(checks),'total':56,'wall_seconds':time.perf_counter()-begin})
    save(PILOT/'forecasts_frozen.json',{'utc':now(),'all_before_future_scoring':True,'registry':registry})
    results={}
    for m,part in forecasts.items():
        cases=[score(q,pred,row) for q,row,pred in part]
        results[m]={'summary':summaries(cases),'cases':cases}
    baseline=results['observer_baseline']['summary']
    short=json.loads((OUT/'final_research_confirmation/outer96_seed622_development.json').read_text())['summary']
    entries=[{'method':m,'balanced_R40':balanced(a['summary']),
        'eligible':candidate_guard(a['summary'],baseline,short)} for m,a in results.items()]
    eligible=[a for a in entries if a['eligible'] and a['method']!='observer_baseline']
    guarded=bool(eligible)
    chosen=min(eligible if eligible else [a for a in entries if a['method']=='hybrid8'],
        key=lambda a:(a['balanced_R40'],a['method']))
    selected=chosen['method'];(PILOT/'selected_predictions').mkdir();noisy_cases=[]
    for q in rows():
        row,own,C,params,initial,clean,noisy=context(q)
        coeff=correction(C,row,params,selected)
        pred,detail=observe(coeff,params[1],row,noisy,params);pred=hybrid(own,pred)
        noisy_cases.append((q,row,pred))
        main_pred=next(p for qq,rr,p in forecasts[selected] if qq['id']==q['id'])
        np.savez_compressed(PILOT/'selected_predictions'/(q['id']+'.npz'),prediction=main_pred)
    save(PILOT/'noise_forecasts_frozen.json',{'utc':now(),'registry':[
        {'id':q['id'],'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest()} for q,row,pred in noisy_cases]})
    noise_summary=summaries([score(q,pred,row) for q,row,pred in noisy_cases])
    chosen_summary=results[selected]['summary'];s=chosen_summary['development_all56']['40s']
    signal=(guarded and chosen['balanced_R40']<=.9*balanced(baseline) and s['n_pass']>=48 and
        noise_summary['development_all56']['40s']['n_pass']>=s['n_pass']-2)
    save(PILOT/'report.json',{'utc':now(),'results':results,'candidates':entries,
        'selected':chosen,'noise_summary':noise_summary,'research_signal_passed':signal,'preservation_guard_passed':guarded,
        'implementation_checks':checks,'fallback_unchanged':True})
    save(PILOT/'selected_lock.json',{'utc':now(),'selected_method':selected,
        'source_sha256':sha(__file__),'preregister_sha256':sha(PILOT/'preregister.json'),
        'selection_data':'old56 development; NOT prior independent8','signal':signal})
    protected();save(PILOT/'status.json',{'status':'complete','utc':now(),
        'selected':selected,'signal':signal,'wall_seconds':time.perf_counter()-begin})
    for m,a in results.items():
        s=a['summary']['development_all56'];print('SPLIT',m,s['8s']['n_pass'],
            s['40s']['n_pass'],round(s['40s']['mean_R'],5),round(s['40s']['max_R'],5),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        PILOT.mkdir(parents=True,exist_ok=True);save(PILOT/'failure.json',{'utc':now(),'error':repr(e)});raise
