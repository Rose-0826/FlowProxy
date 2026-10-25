#!/usr/bin/env python3
"""Four true startup frames only. Fit periodic closure to an autonomous neural trajectory."""
from pathlib import Path
import argparse,sys,json,datetime,time
import numpy as np
import h5py
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import sha,save
from startup_observability_v1 import basis,frequency
from round2_complete40_v1 import frame_summary,summarize
OUT=ROOT/'outputs/self_trajectory_closure_v1'
DATA=ROOT/'data/processed/dataset_long40_v1.h5'


def feature(q):return np.array([[q['Re']/200,q['cy']/.8,q['D']]],np.float64)


def build_priors():
    rows=json.loads((ROOT/'outputs/startup_observability_v1/training_priors.json').read_text());models={}
    for shape in ['cylinder','square']:
        selected=[r for r in rows if r['shape']==shape];X=np.array([r['X'] for r in selected]);y=np.array([r['St'] for r in selected])
        models[shape]=(RBFInterpolator(X,y,kernel='thin_plate_spline',smoothing=1e-6),float(y.mean()))
    return models


def closure(q,support4,pred,times,mask,prior,candidate):
    P=json.loads((ROOT/'outputs/geometry_attack_v2/preregister.json').read_text())
    N,H,method,blend=(candidate[k] for k in ['bootstrap_steps','H','frequency_method','blend_steps'])
    observed=np.concatenate([support4,pred[:N]],0).astype(np.float64)*mask[None,None]
    t=np.array(times[:4+N],np.float64);t-=float(times[3]);rbf,mean=prior[q['shape']]
    center=(float(rbf(feature(q))[0]) if method!='identified_shape_mean' and method!='shape_mean_prior' else mean)*P['config']['physics']['U_inf']/q['D']
    diagnostic={}
    if method.startswith('identified'):
        f,diagnostic=frequency(t,observed[:,1][:,mask>0.5],H,center,.10)
    else:f=center
    B=basis(t,f,H);coef=np.linalg.lstsq(B,observed.reshape(len(observed),-1),rcond=1e-10)[0]
    future_t=np.array(times[4:164],np.float64)-float(times[3])
    projected=(basis(future_t,f,H)@coef).reshape(160,3,64,64).astype(np.float32)
    result=pred.copy();result[N:]=projected[N:]
    for k in range(min(blend,160-N)):
        alpha=(k+1)/blend;result[N+k]=(1-alpha)*pred[N+k]+alpha*projected[N+k]
    if not np.isfinite(result).all():raise ValueError('Nonfinite periodic closure')
    assert np.array_equal(result[:N],pred[:N])
    return result,{'frequency':f,'frequency_fit':diagnostic,'condition_number':float(np.linalg.cond(B))}


def metrics(pred,true,mask,times):
    delta=(pred.astype(np.float64)-true)*mask[None,None];truth=true*mask[None,None]
    error=np.linalg.norm(delta,axis=(-2,-1))/np.maximum(np.linalg.norm(truth,axis=(-2,-1)),1e-8)
    return {f'{sec}s':frame_summary(error[:n],float(times[3]),float(times[n+3]),sec,True) for sec,n in [(2,8),(8,32),(20,80),(40,160)]}


def validation():
    P=json.loads((ROOT/'outputs/geometry_attack_v2/preregister.json').read_text())
    OUT.mkdir(exist_ok=False)
    candidates=[]
    for model in ['F0','K8']:
        for N in [16,24,32,48,64]:
            for H in [2,4]:
                for method in ['RBF_prior','shape_mean_prior','identified_RBF','identified_shape_mean']:
                    for blend in [0,4]:
                        candidates.append({'base_model':model,'bootstrap_steps':N,'H':H,'frequency_method':method,'blend_steps':blend})
    for i,c in enumerate(candidates):c['id']=f'c{i:03d}'
    save(OUT/'preregister.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidates':candidates,'true_support':'only indices0..3, four real fields','fit_support':'four real fields plus N SELF-predicted fields; no extra target measurements',
        'frequency_prior':'train80 only, same fixed Fourier-fitted priors as observability diagnostic',
        'selection':'original val8 minimum40s meanR; ties maxR then candidate index; boot>=32 preserves original8s prediction exactly',
        'fresh8_not_read':True,'no_future_pressure_reference_change_or_time_alignment':True,'script_sha256':sha(__file__)})
    prior=build_priors();reports={c['id']:{'candidate':c,'cases':[]} for c in candidates};started=time.perf_counter()
    with h5py.File(DATA,'r') as h:
        for cid in P['val_ids']:
            g=h['cases'][cid];q={k:str(g.attrs[k]) if k=='shape' else float(g.attrs.get(k,0)) for k in ['shape','Re','cy','cx','D']}
            fields=g['fields'][...].astype(np.float64);times=g['times'][...];mask=g['mask'][...]
            predictions={name:np.load(ROOT/'outputs/geometry_attack_evaluation_v1/v1_bank/val/predictions'/f'{name}__{cid}.npz')['prediction'] for name in ['F0','K8']}
            for c in candidates:
                pred,diag=closure(q,fields[:4],predictions[c['base_model']],times,mask,prior,c)
                hs=metrics(pred,fields[4:164],mask,times)
                reports[c['id']]['cases'].append({'id':cid,'horizons':hs,'closure':diag})
            print('CLOSURE_VAL',cid,round(time.perf_counter()-started,1),flush=True)
    for report in reports.values():report['summary']={key:summarize([r['horizons'][key] for r in report['cases']]) for key in ['2s','8s','20s','40s']}
    selected=min(reports.values(),key=lambda r:(r['summary']['40s']['mean_R'],r['summary']['40s']['max_R'],r['candidate']['id']))
    save(OUT/'report_val.json',{'reports':reports,'selected':selected['candidate']})
    save(OUT/'selected_val.json',selected)
    save(OUT/'summary_val.json',{k:{'candidate':r['candidate'],'summary':r['summary']} for k,r in reports.items()})
    save(OUT/'status.json',{'status':'complete','selected':selected['candidate'],'seconds':time.perf_counter()-started})
    print('CLOSURE_SELECTED',selected['candidate'],selected['summary']['40s'],flush=True)


if __name__=='__main__':validation()
