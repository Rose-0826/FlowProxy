#!/usr/bin/env python3
"""Fixed startup mean/H1 calibration for neural and CFD templates; no new training."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,score,summaries,protected,now
from research_v4_sparse_observer import library,constrain,forecast
from research_v2_template_probe import mapped,gauge_align,fit_startup,calibrate
from research_v4_sparse_confirmation_eval import read_startup
from startup_observability_v1 import basis
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize
from geometry_attack_v2 import save,sha

METHODS=['neural_EKF_2s_startupcal','neural_EKF_2s_startupcal_noise1pct',
         'nearest96_EKF_2s_startupcal','aligned4_96_EKF_2s_startupcal']

def predict_all(dest,fresh):
    if fresh:
        source=OUT/'sparse_confirmation8_eval';m=json.loads((OUT/'sparse_confirmation8_cfd/reference_manifest.json').read_text());case_list=m['cases'];cache=Path(m['cache'])
        frozen=json.loads((source/'forecasts_frozen.json').read_text())['registry']
        n_lookup={a['id']:a['diagnostic']['prefix_steps'] for a in frozen if a['method']=='frozen_autonomous'}
        f_lookup={a['id']:a['diagnostic']['initial']['frequency'] for a in frozen if a['method']=='neural_EKF_2s'}
    else:
        source=OUT/'sparse_observer_pilot';case_list=rows();cache=None
        frozen=json.loads((OUT/'next_round/clock_probe96/forecasts_frozen.json').read_text())['registry']
        n_lookup={a['id']:a['diagnostic']['prefix_steps'] for a in frozen if a['method']=='self_H4'}
        f_lookup={a['id']:a['diagnostic']['self_frequency'] for a in frozen if a['method']=='self_H4'}
    templates=library();registry=[]
    for q in case_list:
        row=read_startup(q,cache) if fresh else startup(q);n=n_lookup[q['id']];f=f_lookup[q['id']]
        prefix_file=(source/'predictions'/('frozen_autonomous__'+q['id']+'.npz')) if fresh else (OUT/'next_round/clock_probe96/predictions'/('frozen_research__'+q['id']+'.npz'))
        with np.load(prefix_file) as a:own=a['prediction'][:n]
        obs=np.concatenate([row['fields'],own]);t=row['times'][:n+4]-row['times'][0]
        neural=constrain(np.linalg.lstsq(basis(t,f,4),obs.reshape(len(t),-1),rcond=1e-10)[0].reshape(9,3,64,64),row['mask'])
        def distance(s):return sum(((q[k]-s['q'][k])/v)**2 for k,v in [('Re',120.),('cy',1.),('D',.5)])
        donors=sorted([s for s in templates if s['q']['shape']==q['shape']],key=lambda s:(distance(s),s['q']['id']))[:4]
        transported=[mapped(s,q,row['mask']) for s in donors];nearest=constrain(transported[0],row['mask'])
        fn=donors[0]['f']*donors[0]['q']['D']/q['D'];w=1/(np.array([distance(s) for s in donors])+1e-6);w/=w.sum()
        aligned=[transported[0]]+[gauge_align(c,transported[0],row['mask'])[0] for c in transported[1:]]
        mixed=constrain(np.einsum('s,skchw->kchw',w,np.stack(aligned)),row['mask']);fm=sum(a*s['f']*s['q']['D'] for a,s in zip(w,donors))/q['D']
        with np.load(source/'measurements'/(q['id']+'.npz')) as a:clean=a['clean'];noisy=a['noisy']
        for method in METHODS:
            C,fr=(nearest,fn) if method.startswith('nearest') else ((mixed,fm) if method.startswith('aligned4') else (neural,f))
            params,_=fit_startup(C,row,fr,False)
            C=constrain(calibrate(C,row,params),row['mask'])
            pred,diag=forecast(C,fr,row,noisy if 'noise' in method else clean,8,'phase')
            p=dest/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(p,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(p),'diagnostic':diag,'initial_calibration_uses_only4_real_startup_fields':True})
        save(dest/'status.json',{'status':'predicting','utc':now(),'completed_cases':len(registry)//len(METHODS),'total':len(case_list)})
    save(dest/'forecasts_frozen.json',{'utc':now(),'all_predictions_before_full_future_scoring':True,'registry':registry})
    results={}
    for method in METHODS:
        cases=[]
        for q in case_list:
            row=read_startup(q,cache) if fresh else startup(q)
            with np.load(dest/'predictions'/(method+'__'+q['id']+'.npz')) as a:pred=a['prediction']
            if fresh:
                with np.load(cache/(q['id']+'.npz')) as a:true=a['fields'][4:164]
                cases.append({'id':q['id'],'horizons':metrics(pred,true,row['mask'],row['times'])})
            else:cases.append(score(q,pred,row))
        results[method]={'summary':({h:summarize([q['horizons'][h] for q in cases]) for h in ['2s','8s','20s','40s']} if fresh else summaries(cases)),'cases':cases}
    protected();save(dest/'report.json',{'utc':now(),'results':results,'role':'predeclared secondary check, never replaces frozen primary method','new_confirmation':fresh,'fallback_unchanged':True})
    save(dest/'status.json',{'status':'complete','utc':now()})
    for method,a in results.items():
        z=a['summary']['40s'] if fresh else a['summary']['development_all56']['40s'];print('STARTUPCAL',method,round(z['mean_R'],5),round(z['max_R'],5),z['n_pass'],flush=True)

def main(fresh):
    protected();dest=OUT/('sparse_confirmation8_startupcal_secondary' if fresh else 'sparse_startupcal_development')
    dest.mkdir(exist_ok=False);(dest/'predictions').mkdir()
    save(dest/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'fixed_calibration':'existing template_probe calibrate():mean andH1 ridge residual on4 true fields;lambda=.03*trace(A.T@A)/3,diag(.5,1,1);no hyperparameter sweep',
        'fairness':'apply same startup spatial calibration to neural,nearest and4-donor templates; unchanged probe filter and76 scalar measurements',
        'role':'predeclared secondary comparison, primary sparse-observer method remains unchanged',
        'evaluation':'new8 references' if fresh else 'seen56 DEVELOPMENT only','no_primary_reselection':True,
        'registered_before_new_confirmation_results':True,'extra_future_full_fields':0})
    if fresh:
        begin=time.perf_counter();required=OUT/'sparse_confirmation8_eval/report.json'
        while not required.exists():
            if time.perf_counter()-begin>16200:raise RuntimeError('4.5h dependency limit')
            for p in [OUT/'sparse_confirmation8_cfd/failure.json',OUT/'sparse_confirmation8_eval/failure.json']:
                if p.exists():raise RuntimeError('Dependency failed '+str(p))
            save(dest/'status.json',{'status':'waiting_for_primary_confirmation','utc':now()});time.sleep(30)
    predict_all(dest,fresh)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--fresh',action='store_true');args=a.parse_args()
    try:main(args.fresh)
    except Exception as e:
        dest=OUT/('sparse_confirmation8_startupcal_secondary' if args.fresh else 'sparse_startupcal_development');dest.mkdir(exist_ok=True);save(dest/'failure.json',{'utc':now(),'error':repr(e)});raise
