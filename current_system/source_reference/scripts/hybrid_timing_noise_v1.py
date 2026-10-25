#!/usr/bin/env python3
"""Resident-model wall timing and synthetic startup-noise diagnostics after frozen confirmation."""
from pathlib import Path
import sys,json,datetime,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_evaluate_v1 import Candidate,specs,CFG
from geometry_attack_v2 import save,sha,NativeSampler
from self_trajectory_closure_v1 import metrics,P
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/hybrid_timing_noise_v1'


def run():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'post_confirmation_diagnostic':True,
        'timing_includes':['geometry generation','H2D inputs','GPU rollout','CPU periodic fit if used','D2H outputs'],
        'timing_excludes':['CFD initial-field acquisition','reading input files','loading weights to GPU'],
        'timing':'3 warmups then20 repeats, same resident models and RAM startup4; one fixture, not independent20 cases',
        'noise_sigma_train_channel_RMS':[.001,.01],'noise_seeds':[101,202,303],'noise_only_four_true_startup_fields':True,
        'no_real_PIV_measurements':True,'no_primary_configuration_change':True,'script_sha256':sha(__file__)})
    sources=[]
    for folder in ['geometry_attack_confirmation_d1_v1','geometry_attack_confirmation_far_v1']:
        rows=json.loads((ROOT/'outputs'/folder/'preregister.json').read_text())['cases']
        for row in rows:
            q={k:row[k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(CFG,**q);mask=geo['mask'];raw=ROOT/'data/raw'/folder/row['id']
            items=frames(raw);sampler=NativeSampler(geo['xx'],geo['yy'],mask)
            fields=np.stack([sampler.sample(path) for _,path in items[:4]])
            truth=np.stack([sampler.sample(path) for _,path in items[4:164]])
            sources.append({'id':row['id'],'q':q,'initial':fields,'truth':truth,'mask':mask,'times':np.array([t for t,_ in items]),'raw':raw})
    available={s['name']:s for s in specs()};fixture=sources[0];timing={}
    for name in ['F0','K8','neural_periodic_closure']:
        candidate=Candidate(available[name]);elapsed=[]
        for k in range(23):
            torch.cuda.synchronize();begin=time.perf_counter()
            prediction,_=candidate.predict(fixture['q'],fixture['initial'],fixture['times'],fixture['raw'])
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            if k>=3:elapsed.append(seconds)
        timing[name]={'seconds':elapsed,'median_seconds':float(np.median(elapsed)),
                      'p10_seconds':float(np.quantile(elapsed,.1)),'p90_seconds':float(np.quantile(elapsed,.9))}
        del candidate;torch.cuda.empty_cache();print('TIMING',name,timing[name]['median_seconds'],flush=True)
    save(OUT/'timing.json',{'fixture':fixture['id'],'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,
                           'threads':torch.get_num_threads(),'models':timing,'initial_fields_acquisition_cost_not_removed':True})
    candidate=Candidate(available['neural_periodic_closure']);scales=np.array(P['normalization']['channel_rms'],np.float32);noise={}
    for sigma in [.001,.01]:
        for seed in [101,202,303]:
            rng=np.random.default_rng(seed);rows=[]
            for d in sources:
                perturb=(rng.standard_normal(d['initial'].shape)*sigma*scales[None,:,None,None]*d['mask'][None,None]).astype(np.float32)
                initial=d['initial']+perturb
                prediction,_=candidate.predict(d['q'],initial,d['times'],d['raw'])
                rows.append({'id':d['id'],'horizons':metrics(prediction,d['truth'],d['mask'],d['times'])})
            summary={k:summarize([r['horizons'][k] for r in rows]) for k in ['2s','8s','20s','40s']}
            noise[f'sigma{sigma}_seed{seed}']={'cases':rows,'summary':summary}
            print('NOISE',sigma,seed,summary['40s']['n_pass'],round(summary['40s']['mean_R'],4),flush=True)
    save(OUT/'noise.json',noise);save(OUT/'status.json',{'status':'complete'})


if __name__=='__main__':run()
