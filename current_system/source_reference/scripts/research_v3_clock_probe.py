#!/usr/bin/env python3
"""TRAIN96-only frequency prior and self-bootstrap H4; seen56 development only."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT, rows, startup, scales, score, summaries, protected, now
from research_v2_final_confirmation import load, infer
from research_v2_outer_support_train import dataset
from startup_observability_v1 import basis, frequency
from geometry_attack_v2 import save, sha

DEST=OUT/'next_round/clock_probe96'
INITIAL=OUT/'final_research_confirmation/selected_research_base_weights.pt'
PERIODS=OUT/'outer_support_training/periods96.json'
METHODS=['frozen_research','self_H4','TRAIN96_geometry_H4','half_blend_H4']

def features(q):
    return np.array([q['Re']/120.,q['cy']**2,np.log(q['D'])/.4],np.float64)

def prior():
    data=dataset();labels=json.loads(PERIODS.read_text())['rows'];assert [a['id'] for a in data]==[a['id'] for a in labels]
    byshape={};records=[]
    for shape in ['cylinder','square']:
        groups={}
        for r,a in zip(data,labels):
            if r['q']['shape']!=shape:continue
            key=tuple(features(r['q']).tolist());groups.setdefault(key,[]).append((r['id'],a['frequency']*r['q']['D']))
        X=np.array(list(groups));Y=np.array([np.mean([v for _,v in g]) for g in groups.values()]);assert len(X)>=10
        candidates=[]
        # Mirror twins stay together in leave-one-geometry-out CV; no validation/test labels.
        for smoothing in [0.,1e-6,1e-3]:
            errors=[]
            for i in range(len(Y)):
                keep=np.arange(len(Y))!=i
                fn=RBFInterpolator(X[keep],Y[keep],kernel='thin_plate_spline',degree=1,smoothing=smoothing)
                errors.append(abs(float(fn(X[i:i+1])[0])/Y[i]-1))
            candidates.append({'smoothing':smoothing,'mean_relative_LOO_error':float(np.mean(errors)),'median_relative_LOO_error':float(np.median(errors))})
        chosen=min(candidates,key=lambda a:(a['mean_relative_LOO_error'],a['smoothing']))
        byshape[shape]=RBFInterpolator(X,Y,kernel='thin_plate_spline',degree=1,smoothing=chosen['smoothing'])
        records.append({'shape':shape,'unique_reflection_groups':len(Y),'smoothing_candidates':candidates,'selected':chosen,
            'geometry_groups':[{'features':list(k),'ids':[n for n,v in g],'mean_field_St':float(np.mean([v for n,v in g]))} for k,g in groups.items()]})
    save(DEST/'TRAIN96_frequency_prior.json',{'utc':now(),'role':'TRAINING ONLY full-field frequencies; reflection twins grouped; no target future','periods_sha256':sha(PERIODS),'models':records})
    return byshape

def balanced(s,h='40s'):
    return .5*(s['development_inside24'][h]['mean_R']+s['seen_extreme32'][h]['mean_R'])

def main():
    torch.set_num_threads(1);protected();DEST.mkdir(parents=True,exist_ok=False);(DEST/'predictions').mkdir()
    save(DEST/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'initial_weights_sha256':sha(INITIAL),'input':'4 real initial fields +geometry/Re; same adaptive24/48 self-bootstrap',
        'clock_prior':'RBF fieldSt(TRAIN96 geometry)/D; smoothing picked by TRAIN96 unique-geometry LOO; reflected signs grouped',
        'self_H4':'fit only own bootstrap Uy; optimize frequency within+-20% training geometry prior; all3 spatial fields fit with4 harmonics',
        'blend':'one predeclared50/50 blend of self-estimated and training-prior frequency; no blend sweep',
        'phase':'least squares on4 real+generated bootstrap only; no future phase, no target future frequencies',
        'prefix':'all methods preserve exactly same24/48 neural prefix; only extrapolator changes',
        'evaluation':'seen56 DEVELOPMENT only; independent8 excluded; neither old fallback nor independently confirmed research candidate replaced',
        'signal':'>=10% balancedRi40 improvement over frozen research, neither group mean nor all56 maxRi worse; balanced8s<=1.05 and8s pass loss<=1/56'})
    models=prior();model=load({'path':str(INITIAL)});registry=[];begin=time.perf_counter()
    for q in rows():
        r=startup(q);base,diag=infer(model,r,True);n=diag['prefix_steps'];observed=np.concatenate([r['fields'],base[:n]])
        t=r['times'][:4+n]-r['times'][0];future=r['times'][4:164]-r['times'][0]
        center=float(models[q['shape']](features(q)[None])[0])/q['D'];assert center>0
        fs,fd=frequency(t,observed[:,1][:,r['mask']>.5].astype(np.float64),4,center,.2)
        estimates={'self_H4':fs,'TRAIN96_geometry_H4':center,'half_blend_H4':.5*(fs+center)}
        options={'frozen_research':(base,diag)}
        for name,f in estimates.items():
            C=np.linalg.lstsq(basis(t,f,4),observed.reshape(len(t),-1).astype(np.float64),rcond=1e-10)[0]
            p=(basis(future,f,4)@C).reshape(160,3,64,64).astype(np.float32)
            p*=r['mask'][None,None];p[:n]=base[:n];assert np.isfinite(p).all()
            options[name]=(p,{'frequency':float(f),'geometry_prior':center,'self_frequency':fs,'self_fit':fd,'prefix_steps':n})
        for method,(p,d) in options.items():
            file=DEST/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=p)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'diagnostic':d})
        save(DEST/'status.json',{'status':'predicting','utc':now(),'completed_cases':len(registry)//len(METHODS),'total':56,'wall_seconds':time.perf_counter()-begin})
    save(DEST/'forecasts_frozen.json',{'utc':now(),'all_forecasts_before_target_future_scoring':True,'registry':registry})
    result={}
    for method in METHODS:
        cases=[]
        for q in rows():
            with np.load(DEST/'predictions'/(method+'__'+q['id']+'.npz')) as a:cases.append(score(q,a['prediction'],startup(q)))
        result[method]={'summary':summaries(cases),'cases':cases}
    ref=result['frozen_research']['summary'];eligible=[]
    for method,a in result.items():
        if method=='frozen_research':continue
        s=a['summary'];ok=(balanced(s)<=.9*balanced(ref)
            and all(s[g]['40s']['mean_R']<=ref[g]['40s']['mean_R'] for g in ['development_inside24','seen_extreme32'])
            and s['development_all56']['40s']['max_R']<=ref['development_all56']['40s']['max_R']
            and balanced(s,'8s')<=1.05*balanced(ref,'8s')
            and s['development_all56']['8s']['n_pass']>=ref['development_all56']['8s']['n_pass']-1)
        if ok:eligible.append(method)
        print('CLOCK96',method,round(balanced(s),6),ok,flush=True)
    protected();save(DEST/'report.json',{'utc':now(),'methods':result,'eligible_signal_methods':eligible,
        'role':'seen56 DEVELOPMENT; not an independent gain','fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'eligible_signal_methods':eligible,'wall_seconds':time.perf_counter()-begin})

if __name__=='__main__':
    try:main()
    except Exception as e:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
