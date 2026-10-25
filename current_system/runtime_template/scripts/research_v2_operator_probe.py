#!/usr/bin/env python3
"""Compare stepwise reflection, trajectory averaging and once-only body centering."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,baseline,scales,score,summaries,protected,now
from corrected_benchmark_train_v1 import model_from,protocol
from corrected_benchmark_controls_v1 import dmd_periodic
from cycle_landmarks_train_v1 import FORMAL
from geometry_attack_v2 import save,sha
from flowproxy.geometry import geometry_channels

METHODS=['step_reflection','trajectory_reflection_average','body_center_fft']

def reflect_input(x):
    y=x.flip(-2).clone();y[:,5::3]=-y[:,5::3];return y

def reflect_output(y):return y.flip(-2)*y.new_tensor([1.,-1.,1.])[None,:,None,None]

def shift_y(a,physical_shift):
    # Output at y samples input at y+physical_shift. Include native grid spacing.
    k=np.fft.fftfreq(64,d=16/63)
    phase=np.exp(2j*np.pi*k*physical_shift)
    shape=[1]*a.ndim;shape[-2]=64
    return np.fft.ifft(np.fft.fft(a,axis=-2)*phase.reshape(shape),axis=-2).real.astype(np.float32)

def rollout(model,row,method):
    geom=row['geom'].copy();initial=row['fields'].copy();cy=float(row['q']['cy'])
    if method=='body_center_fft':
        initial=shift_y(initial,cy)
        source={**row['q'],'cy':0.}
        geom=geometry_channels(protocol()['config'],**source)['geom']
    g=torch.as_tensor(geom[None],device='cuda');window=[torch.as_tensor(a[None],device='cuda') for a in initial]
    mirror=[]
    if method=='trajectory_reflection_average':
        mg=g.flip(-2);mirror=[reflect_output(a) for a in window]
    outputs=[]
    with torch.inference_mode():
        for step in range(160):
            x=torch.cat([g,torch.stack(window[-4:],1).reshape(1,12,64,64)],1)
            if method=='step_reflection':
                y=model(torch.cat([x,reflect_input(x)],0))[:,:3]
                pred=.5*(y[:1]+reflect_output(y[1:]))
            elif method=='trajectory_reflection_average':
                mx=torch.cat([mg,torch.stack(mirror[-4:],1).reshape(1,12,64,64)],1)
                y=model(torch.cat([x,mx],0))[:,:3];pred=y[:1];mirror.append(y[1:])
            else:pred=model(x)[:,:3]
            window.append(pred)
            displayed=.5*(pred+reflect_output(mirror[-1])) if method=='trajectory_reflection_average' else pred
            outputs.append(displayed[0].cpu().numpy())
    result=np.stack(outputs)
    if method=='body_center_fft':result=shift_y(result,-cy)
    return result

def check_symmetry(model,row):
    g=torch.as_tensor(row['geom'][None],device='cuda');h=torch.as_tensor(row['fields'][None],device='cuda').reshape(1,12,64,64)
    x=torch.cat([g,h],1)
    def projected(z):
        a=model(torch.cat([z,reflect_input(z)],0))[:,:3]
        return .5*(a[:1]+reflect_output(a[1:]))
    with torch.inference_mode():
        residual=(projected(x)-reflect_output(projected(reflect_input(x)))).abs().max()
    assert float(residual)<2e-5
    return float(residual)

def run():
    torch.set_num_threads(1);protected();out=OUT/'operator_probe';out.mkdir(exist_ok=False);(out/'predictions').mkdir()
    save(out/'configuration.json',{'utc':now(),'methods':METHODS,'network_unchanged':True,'source_sha256':sha(__file__),
        'inference_observations':4,'ensemble_baseline':'two independent trajectories averaged after rollout',
        'group_projection':'average one-step outputs before feeding the result back; exact reflection equivariance',
        'centering':'FFT shift4 initial physical fields once, use cy0 geometry, self-roll then shift results back once; boundaries approximate'})
    model,_=model_from(FORMAL,protocol()['config']);model.eval();cases=rows();registry=[];begin=time.perf_counter()
    eq=check_symmetry(model,startup(cases[0]));save(out/'symmetry_check.json',{'maximum_absolute_defect':eq,'passed':True})
    for q in cases:
        row=startup(q)
        for method in METHODS:
            started=time.perf_counter();direct=rollout(model,row,method)
            closed,diag=dmd_periodic(np.concatenate([row['fields'],direct[:48]]),row['times'][:52],row['times'][4:164],row['mask'],9,scales())
            closed[:48]=direct[:48]
            file=out/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,direct=direct,DMD9=closed)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'DMD':diag,'seconds_full160_plus_DMD_and_save':time.perf_counter()-started})
        cy=row['q']['cy'];roundtrip=shift_y(shift_y(row['fields'],cy),-cy)
        m=row['mask'][None,None];norm=np.linalg.norm(row['fields']*m,axis=(-2,-1))
        defect=np.linalg.norm((roundtrip-row['fields'])*m,axis=(-2,-1))/np.maximum(norm,1e-8)
        save(out/(q['id']+'_remapping_floor.json'),{'id':q['id'],'observed4_only_mean_error':defect.mean(0).tolist()})
        save(out/'status.json',{'status':'predicting','completed_cases':len(registry)//len(METHODS),'total_cases':56,'elapsed_seconds':time.perf_counter()-begin})
    save(out/'predictions_frozen.json',{'utc':now(),'all_predictions_before_target_future_scoring':True,'registry':registry})
    report={};scored={m:{k:[] for k in ['direct','DMD9']} for m in ['fallback']+METHODS}
    for q in cases:
        row=startup(q)
        for variant in ['direct','DMD9']:scored['fallback'][variant].append(score(q,baseline(q,variant),row))
        for method in METHODS:
            with np.load(out/'predictions'/(method+'__'+q['id']+'.npz')) as a:
                for variant in ['direct','DMD9']:scored[method][variant].append(score(q,a[variant],row))
    for method,variants in scored.items():
        report[method]={'summary':{k:summaries(v) for k,v in variants.items()},'cases':variants}
    protected();save(out/'report.json',report)
    save(out/'status.json',{'status':'complete','utc':now(),'wall_seconds':time.perf_counter()-begin,'fallback_unchanged':True})
    for method,r in report.items():
        print('OPERATOR_PROBE',method,flush=True)
        for group in ['development_inside24','seen_extreme32']:
            print(group,'R40',round(r['summary']['DMD9'][group]['40s']['mean_R'],6),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(OUT/'operator_probe/failure.json',{'utc':now(),'error':repr(e)})
        raise
