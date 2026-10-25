#!/usr/bin/env python3
"""CPU-only fixed-method check: do joint-corner TRAIN labels repair geometry clock?"""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,score,summaries,protected,now
from research_v3_clock_probe import features,balanced
from startup_observability_v1 import basis
from geometry_attack_v2 import save,sha

DEST=OUT/'next_round/clock_prior104_probe'
SOURCE=OUT/'next_round/clock_probe96'
METHODS=['frozen_research','TRAIN96_geometry_H4','TRAIN104_geometry_H4','TRAIN104_half_blend_H4']

def train_prior():
    periods=json.loads((OUT/'next_round/paired_corner_training/periods104.json').read_text())['rows']
    q={r['id']:r for r in rows(True) if r['group']=='train80'}
    for p in [OUT/'outer_support_training_cfd/training_manifest.json',OUT/'next_round/corner_training_cfd/training_manifest.json']:
        for r in json.loads(p.read_text())['cases']:q[r['id']]=r
    assert len(q)==104 and {r['id'] for r in periods}==set(q)
    old=json.loads((SOURCE/'TRAIN96_frequency_prior.json').read_text());models={};records=[]
    for a in old['models']:
        shape=a['shape'];groups={}
        for r in periods:
            case=q[r['id']]
            if case['shape']==shape:groups.setdefault(tuple(features(case)),[]).append((r['id'],r['frequency']*case['D']))
        X=np.array(list(groups));Y=np.array([np.mean([v for _,v in g]) for g in groups.values()])
        alpha=a['selected']['smoothing']
        models[shape]=RBFInterpolator(X,Y,kernel='thin_plate_spline',degree=1,smoothing=alpha)
        errors=[]
        for i in range(len(Y)):
            keep=np.arange(len(Y))!=i
            fn=RBFInterpolator(X[keep],Y[keep],kernel='thin_plate_spline',degree=1,smoothing=alpha)
            errors.append(abs(float(fn(X[i:i+1])[0])/Y[i]-1))
        records.append({'shape':shape,'smoothing_frozen_from_TRAIN96':alpha,'unique_reflection_groups':len(Y),
            'mean_TRAIN104_LOO_relative_error':float(np.mean(errors)),
            'median_TRAIN104_LOO_relative_error':float(np.median(errors)),
            'groups':[{'features':list(k),'ids':[n for n,v in g],'mean_field_St':float(np.mean([v for n,v in g]))} for k,g in groups.items()]})
    save(DEST/'TRAIN104_prior.json',{'utc':now(),'models':records,'role':'TRAINING ONLY; kernel/features/smoothing same as TRAIN96'})
    return models

def main():
    protected();DEST.mkdir(exist_ok=False);(DEST/'predictions').mkdir();begin=time.perf_counter()
    save(DEST/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'single_factor':'TRAIN96 versus TRAIN104 scalar frequency labels; same frozen research network/bootstrap, feature map, RBF kernel and previously selected smoothing',
        'target_input':'four real startup fields +24/48 own generated states; no future clocks or forces',
        'blend':'fixed50/50 of TRAIN104 geometry clock and previous self-estimated clock, no new parameter search',
        'data_roles':'104 TRAINING frequency labels; seen56 DEVELOPMENT for scoring; independent8 excluded',
        'guards':'>=10% balancedRi40 improvement, neither group mean nor maxRi40 worse, balancedRi8<=1.05 and8s pass loss<=1',
        'source_forecasts_sha256':sha(SOURCE/'forecasts_frozen.json'),
        'no_new_CFD_or_network_training':True,'no_model_or_material_replacement':True})
    prior=train_prior();registry=[];lookup={a['id']:a['diagnostic'] for a in json.loads((SOURCE/'forecasts_frozen.json').read_text())['registry'] if a['method']=='self_H4'}
    for q in rows():
        row=startup(q);d=lookup[q['id']];n=d['prefix_steps'];fs=d['self_frequency']
        with np.load(SOURCE/'predictions'/('frozen_research__'+q['id']+'.npz')) as a:base=a['prediction']
        observed=np.concatenate([row['fields'],base[:n]]);t=row['times'][:4+n]-row['times'][0];future=row['times'][4:164]-row['times'][0]
        fg=float(prior[q['shape']](features(q)[None])[0])/q['D'];assert fg>0
        for method,f in [('TRAIN104_geometry_H4',fg),('TRAIN104_half_blend_H4',.5*(fg+fs))]:
            C=np.linalg.lstsq(basis(t,f,4),observed.reshape(len(t),-1).astype(np.float64),rcond=1e-10)[0]
            pred=(basis(future,f,4)@C).reshape(160,3,64,64).astype(np.float32)*row['mask'][None,None]
            pred[:n]=base[:n];assert np.isfinite(pred).all()
            p=DEST/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(p,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(p),'frequency':float(f),'prefix':n})
        save(DEST/'status.json',{'status':'predicting_CPU_only','utc':now(),'completed_cases':len(registry)//2,'total':56})
    save(DEST/'forecasts_frozen.json',{'utc':now(),'all_forecasts_before_target_future_scoring':True,'registry':registry})
    result={}
    for method in METHODS:
        cases=[];folder=SOURCE if method in ['frozen_research','TRAIN96_geometry_H4'] else DEST
        for q in rows():
            with np.load(folder/'predictions'/(method+'__'+q['id']+'.npz')) as a:cases.append(score(q,a['prediction'],startup(q)))
        result[method]={'summary':summaries(cases),'cases':cases}
    ref=result['frozen_research']['summary'];eligible=[]
    for method,a in result.items():
        if method in ['frozen_research','TRAIN96_geometry_H4']:continue
        s=a['summary'];ok=(balanced(s)<=.9*balanced(ref)
            and all(s[g]['40s']['mean_R']<=ref[g]['40s']['mean_R'] for g in ['development_inside24','seen_extreme32'])
            and s['development_all56']['40s']['max_R']<=ref['development_all56']['40s']['max_R']
            and balanced(s,'8s')<=1.05*balanced(ref,'8s')
            and s['development_all56']['8s']['n_pass']>=ref['development_all56']['8s']['n_pass']-1)
        if ok:eligible.append(method)
        print('PRIOR104',method,round(balanced(s),6),ok,flush=True)
    protected();save(DEST/'report.json',{'utc':now(),'methods':result,'eligible_signal_methods':eligible,
        'role':'seen56 DEVELOPMENT only','fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'eligible_signal_methods':eligible,'wall_seconds':time.perf_counter()-begin})

if __name__=='__main__':
    try:main()
    except Exception as e:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
