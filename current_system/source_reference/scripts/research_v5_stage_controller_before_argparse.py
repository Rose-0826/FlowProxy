#!/usr/bin/env python3
"""Finite paired-training selection, then a new frozen eight-case confirmation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,sys,time,shutil,subprocess
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,protected,now,scales,score,summaries
from research_v5_split_observer import DEST,PILOT,context,neural_template,correction,observe,hybrid,balanced
from research_v5_harmonic_residual import ResidualNet,predicted_coefficients,SEEDS,PREP
from research_v5_wake_gain_observer import GAIN,gain_forecast
from research_v2_final_confirmation import load,infer
from research_v2_template_probe import fit_startup,mapped,gauge_align,calibrate
from research_v4_sparse_observer import METHODS,library,constrain,forecast,probes
from research_v4_sparse_confirmation_eval import read_startup
from research_v3_clock_probe import features
from startup_observability_v1 import frequency
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize
from geometry_attack_v2 import save,sha
import corrected_benchmark_restart_v1 as rebuild
from research_v2_outer_support_cfd import initialize as donor_initialize

CTRL=DEST/'controller';CFD=DEST/'new_confirmation8_cfd';EVAL=DEST/'new_confirmation8_eval'
RAW=ROOT/'data/raw/research_v5_split_confirmation8'
CACHE=ROOT/'data/processed/research_v5_split_confirmation8_cases'
BASE=OUT/'final_research_confirmation/selected_research_base_weights.pt'

def residual_load(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=ResidualNet().cuda();model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint['arm']

def select():
    pilot=json.loads((PILOT/'report.json').read_text());selected=pilot['selected']['method']
    ref=pilot['results'][selected]['summary'];base=pilot['results']['observer_baseline']['summary']
    reports=[json.loads((DEST/('residual_seed'+str(seed))/'report.json').read_text()) for seed in SEEDS]
    paired=all(r['selected'] is not None and r['selected']['balanced_R40']<=.95*balanced(ref)
        and r['selected']['summary']['development_all56']['40s']['n_pass']>=48 for r in reports)
    chosen=None;noise_summary=None
    gain=json.loads((GAIN/'report.json').read_text())
    gain_signal=gain['research_signal_passed']
    gain_method=gain['selected']['method'] if gain_signal else None
    if paired:
        chosen=min(reports,key=lambda r:(r['selected']['balanced_R40'],
            r['selected']['summary']['development_all56']['40s']['max_R'],r['seed']))
        path=DEST/('residual_seed'+str(chosen['seed']))/'selected.pt'
        assert sha(path)==chosen['selected_sha256']
        model,arm=residual_load(path);parts=[];registry=[]
        for q in rows():
            row,own,C,params,initial,clean,noisy=context(q)
            coeff=predicted_coefficients(model,row,C,params,arm)
            pred,_=observe(coeff,params[1],row,noisy,[0.,params[1]]);pred=hybrid(own,pred)
            parts.append((q,row,pred));registry.append({'id':q['id'],
                'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest()})
        save(CTRL/'learned_noise_forecasts_frozen.json',{'utc':now(),'all_before_future_scoring':True,'registry':registry})
        noise_summary=summaries([score(q,pred,row) for q,row,pred in parts])
        s=chosen['selected']['summary']['development_all56']['40s'];noisy_s=noise_summary['development_all56']['40s']
        paired=noisy_s['n_pass']>=s['n_pass']-2 and noisy_s['max_R']<=1.05*s['max_R']
        del model,parts;torch.cuda.empty_cache()
    usability=(pilot['preservation_guard_passed'] and
        ref['development_all56']['8s']['n_pass']>=base['development_all56']['8s']['n_pass']+4 and
        ref['development_all56']['40s']['end'][2]<=.8*base['development_all56']['40s']['end'][2])
    if paired:
        frozen=CTRL/'selected_residual.pt'
        shutil.copyfile(DEST/('residual_seed'+str(chosen['seed']))/'selected.pt',frozen)
        selected_model={'path':str(frozen),'sha256':sha(frozen),'arm':chosen['selected']['arm'],
            'seed':chosen['seed'],'step':chosen['selected']['step']}
    else:selected_model=None
    primary='v5_wake_gain_selected' if gain_signal else ('v5_learned_selected' if paired else 'v5_inference_split')
    result={'utc':now(),'inference_method':selected,'inference_usability_signal':usability,
        'paired_learned_signal':paired,'learned_model':selected_model,'noise_summary':noise_summary,
        'wake_gain_signal':gain_signal,'wake_gain_method':gain_method,'primary':primary,
        'confirmation_eligible':bool(paired or usability or gain_signal),'selection_data':'old56 DEVELOPMENT only',
        'previous_independent8_batches_excluded':True,'previous_primary_scores_unchanged':True,
        'training_reports':[{'seed':r['seed'],'selected':r['selected'],
            'selected_sha256':r['selected_sha256']} for r in reports]}
    save(CTRL/'selection.json',result);return result

def design():
    rng=np.random.default_rng(2026100423);result=[];old={q['id'] for q in rows(True)}
    for p in [OUT/'independent8_cfd/preregister.json',OUT/'sparse_confirmation8_cfd/preregister.json',
            OUT/'outer_support_training_cfd/preregister.json',OUT/'next_round/corner_training_cfd/preregister.json']:
        if p.exists():old.update(q['id'] for q in json.loads(p.read_text())['cases'])
    for shape in ['cylinder','square']:
        for sign in [-1,1]:
            for thin in [True,False]:
                Re=float(np.round(rng.uniform(90,190),1));cy=sign*float(np.round(rng.uniform(.55,.95),3))
                D=float(np.round(rng.uniform(.78,.86) if thin else rng.uniform(1.14,1.22),3))
                result.append({'id':f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}','shape':shape,'Re':Re,
                    'cx':0.,'cy':cy,'D':D,'role':'NEW split-observer independent confirmation ONLY'})
    assert len(result)==8 and len({q['id'] for q in result})==8
    assert not old.intersection(q['id'] for q in result)
    return result

def initialize(case,row,p):
    donor_initialize(case,row,p);a=case/'training_donor_seed.vtu';b=case/'training_donor_seed.xml'
    assert a.exists() and not b.exists();digest=sha(a);a.rename(b);assert sha(b)==digest

def cfd(selection):
    assert shutil.disk_usage(ROOT).free>28*2**30
    CFD.mkdir(exist_ok=False);RAW.mkdir(exist_ok=False);CACHE.mkdir(exist_ok=False)
    cases=design();p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    sources=['research_v5_split_observer.py','research_v5_harmonic_residual.py','research_v5_stage_controller.py','research_v4_sparse_observer.py','research_v5_wake_gain_observer.py']
    lock={'utc':now(),'primary':selection['primary'],'inference_method':selection['inference_method'],
        'learned_model':selection['learned_model'],'base_path':str(BASE),'base_sha256':sha(BASE),
        'wake_gain_method':selection['wake_gain_method'],
        'source_sha256':{name:sha(ROOT/'scripts'/name) for name in sources},
        'TRAIN96_prior_sha256':sha(OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json'),
        'TRAIN96_periods_sha256':sha(OUT/'outer_support_training/periods96.json'),
        'selection_sha256':sha(CTRL/'selection.json'),'primary_frozen_before_CFD':True}
    save(CFD/'method_lock.json',lock)
    save(CFD/'preregister.json',{'utc':now(),'seed':2026100423,'cases':cases,
        'method_lock_sha256':sha(CFD/'method_lock.json'),'workers':4,'MPI_ranks_each':2,
        'reference':'same corrected one-layer210x120,dt.005,.25output,165frames,same stability gate',
        'no_case_dropping':True,'input':'four initial full fields+76 future Uy scalars,final2s without measurement',
        'primary':selection['primary'],'previous_frozen_methods':'autonomous andv4 sparse primary scored alongside',
        'independent_target':'primary >=6/8 Ri40<1 andmeanRi40<1,preserve autonomous8s pass count;noise loss<=1',
        'comparison':'ordinary linearKF andsame-probe CFD nearest/aligned4 templates;startup-calibrated nearest secondary',
        'no_posthoc_reselection':True,'virtual_sensors_only':True})
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.initialize=initialize
    begin=time.perf_counter();complete=[];errors=[]
    save(CFD/'status.json',{'utc':now(),'status':'running','completed':[],'errors':[],'total':8})
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(rebuild.one,q,p):q['id'] for q in cases}
        for f in as_completed(jobs):
            if f.cancelled():continue
            try:complete.append(f.result())
            except Exception as e:
                errors.append({'id':jobs[f],'error':repr(e)})
                for j in jobs:j.cancel()
            protected();save(CFD/'status.json',{'utc':now(),'status':'failed' if errors else 'running',
                'total':8,'completed':complete,'errors':errors,'wall_seconds':time.perf_counter()-begin})
    if errors:raise RuntimeError(errors)
    save(CFD/'reference_manifest.json',{'utc':now(),'cases':cases,'results':complete,'cache':str(CACHE)})
    save(CFD/'status.json',{'utc':now(),'status':'complete','total':8,'completed':complete,
        'errors':[],'wall_seconds':time.perf_counter()-begin})

def collect():
    manifest=json.loads((CFD/'reference_manifest.json').read_text());(EVAL/'measurements').mkdir()
    for q in manifest['cases']:
        row=read_startup(q,CACHE);ij,_=probes(row)
        with np.load(CACHE/(q['id']+'.npz')) as a:target=a['fields'][4:164]
        values=np.stack([target[:,1,i,j] for i,j in ij],1).astype(np.float64)
        seed=int(hashlib.sha256(q['id'].encode()).hexdigest()[:8],16)
        noise=np.random.default_rng(seed).normal(0,.01*scales()[1],values.shape)
        np.savez(EVAL/'measurements'/(q['id']+'.npz'),clean=values,noisy=values+noise)

def evaluate_new(selection):
    protected();EVAL.mkdir(exist_ok=False);(EVAL/'predictions').mkdir()
    lock=json.loads((CFD/'method_lock.json').read_text());assert sha(BASE)==lock['base_sha256']
    for name,digest in lock['source_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    with (EVAL/'measurement_collection.log').open('x') as log:
        subprocess.run([sys.executable,__file__,'--collect'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=json.loads((CFD/'reference_manifest.json').read_text());templates=library();model=load({'path':str(BASE)})
    learned=None;arm=None
    if selection['learned_model']:
        assert sha(selection['learned_model']['path'])==selection['learned_model']['sha256']
        learned,arm=residual_load(selection['learned_model']['path'])
    methods=['frozen_autonomous']+METHODS+['v5_inference_split','v5_inference_split_noise1pct','nearest96_startupcal_EKF_2s']
    if learned is not None:methods+=['v5_learned_selected','v5_learned_selected_noise1pct']
    if selection['wake_gain_signal']:
        methods+=['v5_wake_gain_selected','v5_wake_gain_selected_noise1pct','nearest96_wake_gain4','aligned4_wake_gain4']
    prior={};data=json.loads((OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json').read_text())
    for a in data['models']:
        prior[a['shape']]=RBFInterpolator(np.array([q['features'] for q in a['geometry_groups']]),
            np.array([q['mean_field_St'] for q in a['geometry_groups']]),kernel='thin_plate_spline',
            degree=1,smoothing=a['selected']['smoothing'])
    registry=[]
    for q in manifest['cases']:
        row=read_startup(q,CACHE);own,diag=infer(model,row,True);n=diag['prefix_steps']
        obs=np.concatenate([row['fields'],own[:n]]);t=row['times'][:n+4]-row['times'][0]
        center=float(prior[q['shape']](features(q)[None])[0])/q['D']
        f,_=frequency(t,obs[:,1][:,row['mask']>.5].astype(np.float64),4,center,.2)
        C=neural_template(row,own,n,f);params,_=fit_startup(C,row,f,False)
        def distance(s):return sum(((q[k]-s['q'][k])/v)**2 for k,v in [('Re',120.),('cy',1.),('D',.5)])
        donors=sorted([s for s in templates if s['q']['shape']==q['shape']],key=lambda s:(distance(s),s['q']['id']))[:4]
        nearest=constrain(mapped(donors[0],q,row['mask']),row['mask']);fn=donors[0]['f']*donors[0]['q']['D']/q['D']
        transported=[mapped(s,q,row['mask']) for s in donors]
        aligned=[transported[0]]+[gauge_align(c,transported[0],row['mask'])[0] for c in transported[1:]]
        w=1/(np.array([distance(s) for s in donors])+1e-6);w/=w.sum()
        mixed=constrain(np.einsum('s,skchw->kchw',w,np.stack(aligned)),row['mask'])
        fm=sum(a*s['f']*s['q']['D'] for a,s in zip(w,donors))/q['D']
        with np.load(EVAL/'measurements'/(q['id']+'.npz')) as a:clean=a['clean'];noisy=a['noisy']
        for method in methods:
            values=noisy if 'noise' in method else clean
            if method=='frozen_autonomous':pred=own;detail=diag
            elif method.startswith('v5_wake_gain'):
                coef=correction(C,row,params,'hybrid8_pH1')
                count=1 if selection['wake_gain_method'].startswith('gain1') else 4
                pred,detail=gain_forecast(coef,f,row,values,params,count);pred=hybrid(own,pred)
            elif method in ['nearest96_wake_gain4','aligned4_wake_gain4']:
                raw,fr=(nearest,fn) if method.startswith('nearest') else (mixed,fm)
                pp,_=fit_startup(raw,row,fr,False)
                coef=correction(raw,row,pp,'hybrid8_pH1')
                pred,detail=gain_forecast(coef,fr,row,values,pp,4);pred=hybrid(own,pred)
            elif method.startswith('v5_inference'):
                coef=correction(C,row,params,selection['inference_method'])
                pred,detail=observe(coef,f,row,values,params);pred=hybrid(own,pred)
            elif method.startswith('v5_learned'):
                coef=predicted_coefficients(learned,row,C,params,arm)
                pred,detail=observe(coef,f,row,values,[0.,f]);pred=hybrid(own,pred)
            elif method=='nearest96_startupcal_EKF_2s':
                pp,_=fit_startup(nearest,row,fn,False)
                coef=constrain(calibrate(nearest,row,pp),row['mask'])
                pred,detail=forecast(coef,fn,row,values,8,'phase')
            else:
                coef,fr=(nearest,fn) if method.startswith('nearest') else ((mixed,fm) if method.startswith('aligned4') else (C,f))
                interval=0 if method.endswith('open') else (16 if '4s' in method else 8)
                pred,detail=forecast(coef,fr,row,values,interval,'linear' if 'linearKF' in method else 'phase')
            file=EVAL/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'diagnostic':detail})
        save(EVAL/'status.json',{'status':'predicting','utc':now(),'completed_cases':len(registry)//len(methods),'total':8})
    save(EVAL/'forecasts_frozen.json',{'utc':now(),'all_before_future_full_field_scoring':True,'registry':registry})
    results={}
    for method in methods:
        part=[]
        for q in manifest['cases']:
            row=read_startup(q,CACHE);meta=json.loads((CACHE/(q['id']+'.json')).read_text())
            assert meta['CFD_stable'] and sha(CACHE/(q['id']+'.npz'))==meta['cache_sha256']
            with np.load(CACHE/(q['id']+'.npz')) as a:target=a['fields'][4:164]
            with np.load(EVAL/'predictions'/(method+'__'+q['id']+'.npz')) as a:pred=a['prediction']
            part.append({'id':q['id'],'horizons':metrics(pred,target,row['mask'],row['times'])})
        results[method]={'cases':part,'summary':{h:summarize([q['horizons'][h] for q in part]) for h in ['2s','8s','20s','40s']}}
    primary=selection['primary'];s=results[primary]['summary'];noise=results[primary+'_noise1pct']['summary']
    signal=(s['40s']['n_pass']>=6 and s['40s']['mean_R']<1 and
        s['8s']['n_pass']>=results['frozen_autonomous']['summary']['8s']['n_pass'] and
        noise['40s']['n_pass']>=s['40s']['n_pass']-1)
    protected();save(EVAL/'report.json',{'utc':now(),'results':results,'primary':primary,
        'independent_signal_passed':signal,'no_posthoc_reselection':True,'virtual_probes_required':True,
        'fallback_unchanged':True,'previous_results_unchanged':True})
    save(EVAL/'status.json',{'utc':now(),'status':'complete','primary':primary,'signal':signal})

def main():
    protected();torch.set_num_threads(1);CTRL.mkdir(exist_ok=False);begin=time.perf_counter()
    save(CTRL/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'seeds':SEEDS,
        'finite_pipeline':'wait inference+paired learned training;select onold56;freeze;new8CFD;scorefixed methods once',
        'learned_gate':'BOTH selected seeds >=5% lower balancedR40 than inference winner and>=48/56pass,noise guard',
        'inference_usability_gate':'preserve40s guards,recover>=4 old56 short passes,and>=20% pressure-end reduction',
        'wake_gain_gate':'gain1/4 neural guard,>=48/56pass,>=10% lower balancedR40 than inference winner,20% better strengthened retrieval baselines;noise pass loss<=2',
        'no_40s_accuracy_claim_from_usability_gate':True,'confirmation_seed':2026100423,
        'training_timeout_seconds':7200,'no_automatic_fallback_replacement':True})
    dependencies=[PILOT,GAIN]+[DEST/('residual_seed'+str(s)) for s in SEEDS]
    while not all((d/'report.json').exists() for d in dependencies):
        if time.perf_counter()-begin>7200:raise RuntimeError('2h training dependency timeout')
        for d in dependencies+[PREP]:
            if (d/'failure.json').exists():raise RuntimeError('Dependency failure '+str(d))
        save(CTRL/'status.json',{'status':'waiting_for_paired_training','utc':now(),
            'wall_seconds':time.perf_counter()-begin});time.sleep(20)
    selection=select()
    if not selection['confirmation_eligible']:
        save(CTRL/'status.json',{'status':'complete_no_confirmation_signal','utc':now(),
            'selection':selection,'wall_seconds':time.perf_counter()-begin});return
    save(CTRL/'status.json',{'status':'selected_waiting_for_new_CFD','utc':now(),'primary':selection['primary']})
    cfd(selection);evaluate_new(selection)
    save(CTRL/'status.json',{'status':'complete','utc':now(),'wall_seconds':time.perf_counter()-begin,
        'primary':selection['primary'],'independent_report':str(EVAL/'report.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--collect',action='store_true');a=ap.parse_args()
    try:
        if a.collect:collect()
        else:main()
    except Exception as e:
        CTRL.mkdir(exist_ok=True);save(CTRL/'failure.json',{'utc':now(),'error':repr(e)});raise
