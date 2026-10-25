#!/usr/bin/env python3
"""Finite wait for new8, then fixed startup/probe predictions before future scoring."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,protected,now
from research_v2_final_confirmation import load,infer
from research_v4_sparse_observer import METHODS,library,probes,constrain,forecast
from research_v2_template_probe import mapped,gauge_align
from research_v3_clock_probe import features
from startup_observability_v1 import basis,frequency
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize
from geometry_attack_v2 import save,sha

CFD=OUT/'sparse_confirmation8_cfd';DEST=OUT/'sparse_confirmation8_eval'
MANIFEST=CFD/'reference_manifest.json'

def read_startup(q,cache):
    with np.load(cache/(q['id']+'_startup.npz')) as a:row={k:a[k] for k in ['fields','times','geom','mask']}
    assert row['fields'].shape==(4,3,64,64)
    row['q']={k:q[k] for k in ['shape','Re','cx','cy','D']};row['id']=q['id'];return row

def collect():
    manifest=json.loads(MANIFEST.read_text());cache=Path(manifest['cache']);(DEST/'measurements').mkdir()
    for q in manifest['cases']:
        meta=json.loads((cache/(q['id']+'.json')).read_text());assert meta['CFD_stable'] and sha(cache/(q['id']+'.npz'))==meta['cache_sha256']
        row=read_startup(q,cache);ij,_=probes(row)
        with np.load(cache/(q['id']+'.npz')) as a:target=a['fields'][4:164]
        clean=np.stack([target[:,1,i,j] for i,j in ij],1).astype(np.float64)
        import hashlib
        from research_v2_common import scales
        seed=int(hashlib.sha256(q['id'].encode()).hexdigest()[:8],16)
        noise=np.random.default_rng(seed).normal(0,.01*scales()[1],clean.shape)
        np.savez(DEST/'measurements'/(q['id']+'.npz'),clean=clean,noisy=clean+noise)

def confirm():
    torch.set_num_threads(1);protected();manifest=json.loads(MANIFEST.read_text());cache=Path(manifest['cache']);lock=json.loads((CFD/'observer_lock.json').read_text())
    assert sha(lock['weights'])==lock['weights_sha256'] and sha(ROOT/'scripts/research_v4_sparse_observer.py')==lock['observer_source_sha256']
    assert sha(OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json')==lock['TRAIN96_clock_prior_sha256']
    assert sha(OUT/'outer_support_training/periods96.json')==lock['TRAIN96_periods_sha256']
    save(DEST/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),'observer_lock_sha256':sha(CFD/'observer_lock.json'),
        'method_frozen_before_reference_generation':True,'all_predictions_before_full_field_future_scoring':True,
        'extra_observations_explicit':True,'no_posthoc_model_reselection':True})
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    with (DEST/'measurement_collection.log').open('x') as log:subprocess.run([sys.executable,__file__,'--collect'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    templates=library();model=load({'path':lock['weights']});prior={}
    data=json.loads((OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json').read_text())
    for a in data['models']:
        prior[a['shape']]=RBFInterpolator(np.array([q['features'] for q in a['geometry_groups']]),
            np.array([q['mean_field_St'] for q in a['geometry_groups']]),kernel='thin_plate_spline',degree=1,smoothing=a['selected']['smoothing'])
    registry=[]
    for q in manifest['cases']:
        row=read_startup(q,cache);own,diag=infer(model,row,True);n=diag['prefix_steps'];obs=np.concatenate([row['fields'],own[:n]])
        t=row['times'][:4+n]-row['times'][0];center=float(prior[q['shape']](features(q)[None])[0])/q['D']
        f,_=frequency(t,obs[:,1][:,row['mask']>.5].astype(np.float64),4,center,.2)
        neural=constrain(np.linalg.lstsq(basis(t,f,4),obs.reshape(len(t),-1),rcond=1e-10)[0].reshape(9,3,64,64),row['mask'])
        def distance(s):return sum(((q[k]-s['q'][k])/v)**2 for k,v in [('Re',120.),('cy',1.),('D',.5)])
        donors=sorted([s for s in templates if s['q']['shape']==q['shape']],key=lambda s:(distance(s),s['q']['id']))[:4];donor=donors[0]
        nearest=constrain(mapped(donor,q,row['mask']),row['mask']);fn=donor['f']*donor['q']['D']/q['D']
        coefficients=[mapped(s,q,row['mask']) for s in donors]
        aligned=[coefficients[0]]+[gauge_align(c,coefficients[0],row['mask'])[0] for c in coefficients[1:]]
        weights=1/(np.array([distance(s) for s in donors])+1e-6);weights/=weights.sum()
        mixed=constrain(np.einsum('s,skchw->kchw',weights,np.stack(aligned)),row['mask']);fm=sum(w*s['f']*s['q']['D'] for w,s in zip(weights,donors))/q['D']
        with np.load(DEST/'measurements'/(q['id']+'.npz')) as a:clean=a['clean'];noisy=a['noisy']
        for method in ['frozen_autonomous']+METHODS:
            if method=='frozen_autonomous':pred=own;detail=diag
            else:
                C,fr=(nearest,fn) if method.startswith('nearest') else ((mixed,fm) if method.startswith('aligned4') else (neural,f))
                interval=0 if method.endswith('open') else (16 if '4s' in method else 8)
                pred,detail=forecast(C,fr,row,noisy if 'noise' in method else clean,interval,'linear' if 'linearKF' in method else 'phase')
            file=DEST/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'diagnostic':detail})
    save(DEST/'forecasts_frozen.json',{'utc':now(),'all_before_full_future_field_scoring':True,'registry':registry})
    results={}
    for method in ['frozen_autonomous']+METHODS:
        cases=[]
        for q in manifest['cases']:
            row=read_startup(q,cache)
            with np.load(cache/(q['id']+'.npz')) as a:true=a['fields'][4:164]
            with np.load(DEST/'predictions'/(method+'__'+q['id']+'.npz')) as a:pred=a['prediction']
            cases.append({'id':q['id'],'horizons':metrics(pred,true,row['mask'],row['times'])})
        results[method]={'summary':{h:summarize([q['horizons'][h] for q in cases]) for h in ['2s','8s','20s','40s']},'cases':cases}
    primary=results['neural_EKF_2s']['summary']['40s'];noise=results['neural_EKF_2s_noise1pct']['summary']['40s']
    best_template=min(results[m]['summary']['40s']['mean_R'] for m in ['nearest96_EKF_2s','aligned4_96_EKF_2s'])
    signal=primary['n_pass']>=6 and primary['mean_R']<1 and primary['mean_R']<=.8*best_template and noise['n_pass']>=primary['n_pass']-1
    protected();save(DEST/'report.json',{'utc':now(),'results':results,'independent_signal_passed':signal,
        'role':'new eight-case confirmation of sparse virtual-probe assisted task,method frozen before CFD',
        'observations_required':True,'no_posthoc_model_reselection':True,'autonomous_results_unchanged':True,'fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'independent_signal_passed':signal})

def main():
    DEST.mkdir(exist_ok=False);(DEST/'predictions').mkdir();begin=time.perf_counter()
    while not MANIFEST.exists():
        if time.perf_counter()-begin>14400:raise RuntimeError('4h CFD dependency limit')
        if (CFD/'failure.json').exists():raise RuntimeError('Reference CFD failed; no selected-case scoring')
        save(DEST/'status.json',{'status':'waiting_for_new8_reference','utc':now(),'elapsed_seconds':time.perf_counter()-begin});time.sleep(30)
    confirm()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--collect',action='store_true');a=ap.parse_args()
    try:
        if a.collect:collect()
        else:main()
    except Exception as e:
        DEST.mkdir(exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
