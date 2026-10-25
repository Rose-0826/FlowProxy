#!/usr/bin/env python3
"""Four-startup-frame clock anchoring and reflection projection, fixed network."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import time

import h5py
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from geometry_attack_v2 import save, sha
from startup_observability_v1 import basis, frequency
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

OLD = ROOT/'outputs/corrected_benchmark_restart_v1'
OUT = ROOT/'outputs/startup_anchored_periodic_v1'
CACHE = ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
TRAINVAL = ROOT/'data/processed/dataset_corrected_trainval_v1.h5'
MODES = [
    {'id':'normal_unanchored','mirror':False,'anchor':False},
    {'id':'normal_anchored','mirror':False,'anchor':True},
    {'id':'mirror_unanchored','mirror':True,'anchor':False},
    {'id':'mirror_anchored','mirror':True,'anchor':True},
]


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def summarize_rows(rows):
    return {h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']}


def q_from(g):
    return {k:str(g.attrs[k]) if k=='shape' else float(g.attrs[k]) for k in ['shape','Re','cx','cy','D']}


def reflection(a):
    out=np.flip(a,axis=-2).copy()
    out[...,1,:,:] *= -1
    return out


def projected_forward(model, input_tensor):
    import torch
    reflected=torch.flip(input_tensor,(-2,)).clone()
    reflected[:,5::3] *= -1
    raw=model(torch.cat([input_tensor,reflected],0))[:,:3]
    reflected_back=torch.flip(raw[1:2],(-2,)).clone()
    reflected_back[:,1] *= -1
    return .5*(raw[:1]+reflected_back)


def mirror_predict(model,row):
    import torch
    geom=torch.as_tensor(row['geom'][None],device='cuda')
    window=[torch.as_tensor(f[None],device='cuda') for f in row['fields'][:4]]
    out=[]
    model.eval()
    with torch.inference_mode():
        for k in range(160):
            hist=torch.stack(window[-4:],1).reshape(1,12,64,64)
            pred=projected_forward(model,torch.cat([geom,hist],1))
            out.append(pred[0].cpu().numpy()); window.append(pred)
    return np.stack(out)


def phase_basis(t,f,H,phi):
    return np.column_stack([np.ones(len(t))]+[a for k in range(1,H+1)
            for a in [np.cos(k*(2*np.pi*f*t+phi)),np.sin(k*(2*np.pi*f*t+phi))]])


def fit_parameters(row,base,prior,anchor):
    N,H=48,4
    mask=row['mask']>.5
    observed=np.concatenate([row['fields'][:4],base[:N]]).astype(np.float64)*mask[None,None]
    t=row['times'][:4+N].astype(np.float64)-float(row['times'][3])
    center=prior[row['q']['shape']]/row['q']['D']
    f,diag=frequency(t,observed[:,1][:,mask],H,center,.25)
    B=basis(t,f,H)
    coefficients=np.linalg.lstsq(B,observed.reshape(len(observed),-1),rcond=1e-10)[0]
    fitted_frequency=f; phi=0.
    anchor_diag={'enabled':False}
    if anchor:
        # Freeze spatial coefficients. Only two clock numbers see the four true startup fields.
        C=coefficients.reshape(2*H+1,3,64,64)[:,1][:,mask]
        Q,R=np.linalg.qr(C.T,mode='reduced')
        target=row['fields'][:4,1][:,mask].astype(np.float64)@Q
        startup=row['times'][:4].astype(np.float64)-float(row['times'][3])
        def residual(x):
            return (phase_basis(startup,f*(1+x[0]),H,x[1])@R.T-target).ravel()
        initial=residual(np.zeros(2))
        fit=least_squares(residual,np.zeros(2),bounds=([-.10,-.25],[.10,.25]),
                          ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=100)
        fitted_frequency=f*(1+float(fit.x[0]));phi=float(fit.x[1])
        anchor_diag={'enabled':True,'relative_frequency_change':float(fit.x[0]),
            'phase_radians':phi,'initial_projected_squared_error':float(initial@initial),
            'final_projected_squared_error':float(fit.fun@fit.fun),'optimizer_success':bool(fit.success),
            'function_evaluations':int(fit.nfev),'at_frequency_bound':bool(abs(fit.x[0])>.0999),
            'at_phase_bound':bool(abs(fit.x[1])>.2499),
            'true_fields_used':4,'spatial_coefficients_frozen':True}
    return coefficients, {'original_frequency':f,'frequency':fitted_frequency,'phase_radians':phi,
                         'frequency_fit':diag,'anchor':anchor_diag,'condition_number':float(np.linalg.cond(B))}


def extend(base,times,coefficients,diag):
    t=times[4:164].astype(np.float64)-float(times[3])
    out=(phase_basis(t,diag['frequency'],4,diag['phase_radians'])@coefficients).reshape(160,3,64,64).astype(np.float32)
    out[:48]=base[:48]
    assert np.array_equal(out[:48],base[:48]) and np.isfinite(out).all()
    return out


def save_params(folder,cid,mode,row,base,prior):
    coefficients,diag=fit_parameters(row,base,prior,mode['anchor'])
    pred=extend(base,row['times'],coefficients,diag)
    path=folder/(mode['id']+'__'+cid+'_parameters.npz')
    np.savez_compressed(path,coefficients=coefficients,prefix=base[:48],times=row['times'],
                        frequency=diag['frequency'],phase_radians=diag['phase_radians'])
    return {'id':cid,'mode':mode['id'],'path':str(path),'sha256':sha(path),'diagnostic':diag,
            'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest()}


def restore(entry):
    with np.load(entry['path']) as a:
        diag={'frequency':float(a['frequency']),'phase_radians':float(a['phase_radians'])}
        base=np.zeros((160,3,64,64),np.float32);base[:48]=a['prefix']
        pred=extend(base,a['times'],a['coefficients'],diag)
    assert hashlib.sha256(pred.tobytes()).hexdigest()==entry['prediction_bytes_sha256']
    return pred


def run():
    OUT.mkdir(exist_ok=False)
    started=time.perf_counter()
    p=json.loads((OLD/'preregister.json').read_text())
    lock=json.loads((OLD/'selected_model_lock.json').read_text())
    assert sha(lock['chosen']['path'])==lock['chosen']['sha256']
    protected={str(fp):sha(fp) for fp in [Path(lock['chosen']['path']),OLD/'selected_model_lock.json',
                      OLD/'already_seen_test32/report.json',OLD/'fresh_confirmation12/report.json']}
    save(OUT/'preregister.json',{'registered_utc':now(),'modes':MODES,'network':lock['chosen'],
        'scope':'small 2x2 experiment, no training and no new CFD','startup_true_frames':4,
        'bootstrap_self_steps':48,'harmonics':4,'frequency_search':'train80 shape mean +-25%; previous pilot showed no val8 difference to original +-10%',
        'reflection':'flip y; negate Uy; flip ALL geometry/history inputs; average prediction and transformed-back mirror prediction each step',
        'boundary_evidence':'symmetric y domain, symmetric top/bottom velocity/pressure BC, horizontal uniform inlet, fixed outlet p=0',
        'clock_anchor':'freeze self-trajectory spatial coefficients; fit frequency +-10% and phase +-0.25rad to startup4 Uy only',
        'selection':'val8 minimum40s meanRi, tie maximumRi and mode order; baseline included',
        'acceptance':'meanRi <=0.85*original; maxRi <=original; short8s 8/8 passed; selection on val8 only',
        'score_reuse':'32+12 are already-seen exploratory evaluation, not fresh confirmation',
        'future_prediction_information':'no future fields, no target force/frequency, no future phase, raw pressure unchanged',
        'storage':'small prediction parameters+48step prefix; bitwise reconstruction checked; no raw CFD copies',
        'protected':protected,'script_sha256':sha(__file__)})
    save(OUT/'status.json',{'status':'running','stage':'validation','started_utc':now()})
    from corrected_benchmark_train_v1 import model_from
    from flowproxy.geometry import geometry_channels
    import torch
    torch.set_num_threads(1)
    model,_=model_from(lock['chosen']['path'],p['config']);model.eval()
    prior=lock['shape_mean_train80_St']
    valfolder=OUT/'validation';valfolder.mkdir()
    results={m['id']:[] for m in MODES}
    direct_mirror=[]
    smoke=None
    with h5py.File(TRAINVAL,'r') as h:
        for cid in p['splits']['val']:
            g=h['cases'][cid]
            row={'id':cid,'q':q_from(g),'fields':g['fields'][:4],'geom':g['geom'][...],
                 'mask':g['mask'][...],'times':g['times'][...]}
            if smoke is None:
                assert np.array_equal(reflection(reflection(row['fields'])),row['fields'])
                q=row['q'];mirgeom=geometry_channels(p['config'],shape=q['shape'],cx=q['cx'],cy=-q['cy'],D=q['D'],Re=q['Re'])['geom']
                geom_diff=float(np.max(abs(np.flip(row['geom'],-2)-mirgeom)))
                assert geom_diff<3e-6
                inp=torch.cat([torch.as_tensor(row['geom'][None],device='cuda'),
                    torch.as_tensor(row['fields'][None],device='cuda').reshape(1,12,64,64)],1)
                mir=torch.flip(inp,(-2,)).clone();mir[:,5::3]*=-1
                with torch.inference_mode():
                    a=projected_forward(model,inp);b=projected_forward(model,mir)
                    expected=torch.flip(a,(-2,)).clone();expected[:,1]*=-1
                    equivariance=float(torch.max(abs(b-expected)))
                assert equivariance<1e-5
                smoke={'reflection_involution_exact':True,'geometry_mirror_max_difference':geom_diff,
                       'projected_operator_equivariance_max_difference':equivariance,'checked_before_scoring':True}
                save(OUT/'interface_smoke.json',smoke)
            normal=np.load(ROOT/'outputs/periodic_clock_refinement_v1/validation'/(cid+'_base.npz'))['prediction']
            mirrored=mirror_predict(model,row)
            np.savez_compressed(valfolder/(cid+'_mirror_base.npz'),prediction=mirrored)
            registry=[save_params(valfolder,cid,mode,row,mirrored if mode['mirror'] else normal,prior) for mode in MODES]
            save(valfolder/(cid+'_predictions_frozen.json'),{'registry':registry,'before_future_read':True})
            truth=g['fields'][4:164]
            direct_mirror.append({'id':cid,'horizons':metrics(mirrored,truth,row['mask'],row['times'])})
            for entry in registry:
                results[entry['mode']].append({'id':cid,'q':row['q'],'horizons':metrics(restore(entry),truth,row['mask'],row['times']),
                                             'clock':entry['diagnostic']})
            print('ANCHORED_VAL',cid,round(time.perf_counter()-started,1),flush=True)
    summaries={key:summarize_rows(rows) for key,rows in results.items()}
    original=json.loads((Path(lock['chosen']['path']).parent/'selected_validation.json').read_text())['summary']['closure']
    assert np.allclose(summaries['normal_unanchored']['40s']['mean'],original['40s']['mean'],atol=1e-6)
    winner=min(MODES,key=lambda m:(summaries[m['id']]['40s']['mean_R'],summaries[m['id']]['40s']['max_R'],MODES.index(m)))
    chosen=summaries[winner['id']]
    accepted=chosen['40s']['mean_R']<=.85*original['40s']['mean_R'] and chosen['40s']['max_R']<=original['40s']['max_R'] and chosen['8s']['n_pass']==8
    save(OUT/'validation_report.json',{'cases':results,'summary':summaries,'mirror_direct':summarize_rows(direct_mirror),
                                      'original_frozen_validation':original})
    save(OUT/'selection.json',{'locked_utc':now(),'mode':winner,'accepted':accepted,'validation':chosen,'selected_on_val8_only':True})
    print('ANCHORED_SELECTED',winner['id'],'ACCEPTED',accepted,'R',chosen['40s']['mean_R'],'MAX',chosen['40s']['max_R'],flush=True)
    if accepted:
        for role in ['already_seen_test32','fresh_confirmation12']:
            folder=OUT/role;folder.mkdir()
            frozen=json.loads((OLD/role/'report.json').read_text())
            registry=[];startup=[]
            for case in frozen['cases']['corrected_selected']:
                cid=case['id']
                with np.load(CACHE/(cid+'_startup.npz')) as a:
                    row={'id':cid,'q':case['q'],'fields':a['fields'],'geom':a['geom'],'mask':a['mask'],'times':a['times']}
                normal=np.load(OLD/role/('corrected_selected__'+cid+'.npz'))['prediction']
                mirrored=mirror_predict(model,row)
                for mode in MODES:
                    registry.append(save_params(folder,cid,mode,row,mirrored if mode['mirror'] else normal,prior))
                startup.append(row)
                print('ANCHORED_PREDICT',role,cid,flush=True)
            save(folder/'predictions_frozen.json',{'registry':registry,'all_frozen_before_future_scoring':True,
                                                 'selected_config_sha256':sha(OUT/'selection.json')})
            scored={m['id']:[] for m in MODES}
            rowbyid={r['id']:r for r in startup}
            for entry in registry:
                row=rowbyid[entry['id']]
                with np.load(CACHE/(entry['id']+'.npz')) as a:truth=a['fields'][4:164]
                scored[entry['mode']].append({'id':entry['id'],'q':row['q'],
                    'horizons':metrics(restore(entry),truth,row['mask'],row['times']),'clock':entry['diagnostic']})
            sums={name:summarize_rows(rows) for name,rows in scored.items()}
            save(folder/'report.json',{'role':'exploratory evaluation of previously seen cases','selected_mode':winner,
                'summary':sums,'cases':scored,'original_baselines':frozen['summary'],'not_a_new_blind_test':True})
            print('ANCHORED_EXPLORATORY',role,[(k,round(v['40s']['mean_R'],3),v['40s']['n_pass']) for k,v in sums.items()],flush=True)
    unchanged={path:sha(path)==value for path,value in protected.items()};assert all(unchanged.values())
    save(OUT/'status.json',{'status':'complete','accepted':accepted,'selected':winner['id'],
                         'original_files_unchanged':unchanged,'seconds':time.perf_counter()-started})


if __name__=='__main__':
    try:run()
    except Exception as error:
        if OUT.exists():save(OUT/'status.json',{'status':'failed','error':repr(error),'utc':now()})
        raise
