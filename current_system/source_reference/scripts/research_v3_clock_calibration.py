#!/usr/bin/env python3
"""Calibrate network clock bias on TRAIN96 own rollouts, not unseen future clocks."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,score,summaries,protected,now
from research_v2_outer_support_train import dataset
from research_v2_final_confirmation import load,infer
from research_v3_clock_probe import features,balanced
from startup_observability_v1 import basis,frequency
from geometry_attack_v2 import save,sha

DEST=OUT/'next_round/clock_bias_calibration'
INITIAL=OUT/'final_research_confirmation/selected_research_base_weights.pt'
METHODS=['frozen_research','TRAIN96_shape_clock_bias','TRAIN96_linear_clock_bias']

def vector(q,f):return np.r_[features(q), (f*q['D']-.17)/.05]
def regression(X,y,alpha):
    Z=np.c_[np.ones(len(X)),X];reg=np.diag([0.]+[alpha]*X.shape[1])
    return np.linalg.solve(Z.T@Z+reg,Z.T@y)

def main():
    torch.set_num_threads(1);protected();DEST.mkdir(parents=True,exist_ok=False);(DEST/'predictions').mkdir()
    save(DEST/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'purpose':'learn relative bias in this frozen network self-estimated frequency, rather than geometry-only clock',
        'labels':'TRAIN96 full-field true frequency divided by its OWN predicted bootstrap frequency minus1',
        'features':'Re/120,cy^2,logD/.4,self frequency*D; shape models separate; mirror groups held out together',
        'methods_fixed':'two predeclared calibrations: shape median bias; ridge linear bias with TRAIN96 grouped LOO alpha in1e-4,.01,1',
        'target_input':'same four observed fields, geometry,24/48 own bootstrap; no target future phase/frequency/forces',
        'selection_data':'seen56 DEVELOPMENT only; latest independent8 not used for model selection or training',
        'guards':'>=10% balancedRi40 reduction; neither group mean nor overall maxRi40 worse; balancedRi8<=1.05 and pass loss<=1',
        'frozen_research_weights_sha256':sha(INITIAL),'no_automatic_fallback_replacement':True})
    label=json.loads((OUT/'outer_support_training/periods96.json').read_text())['rows'];data=dataset()
    assert [r['id'] for r in data]==[q['id'] for q in label]
    p=json.loads((OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json').read_text())
    priors={}
    for a in p['models']:
        priors[a['shape']]=RBFInterpolator(np.array([g['features'] for g in a['geometry_groups']]),
            np.array([g['mean_field_St'] for g in a['geometry_groups']]),kernel='thin_plate_spline',degree=1,smoothing=a['selected']['smoothing'])
    model=load({'path':str(INITIAL)});training=[]
    for r,q in zip(data,label):
        row={**r,'fields':r['fields'][:4]};pred,diag=infer(model,row,True);n=diag['prefix_steps']
        obs=np.concatenate([row['fields'],pred[:n]]);t=row['times'][:4+n]-row['times'][0]
        center=float(priors[row['q']['shape']](features(row['q'])[None])[0])/row['q']['D']
        f,fd=frequency(t,obs[:,1][:,row['mask']>.5].astype(np.float64),4,center,.2)
        training.append({'id':r['id'],'q':row['q'],'self_frequency':f,'true_TRAIN_frequency':q['frequency'],
            'relative_clock_bias':q['frequency']/f-1,'self_fit':fd,'bootstrap':n})
        if len(training)%8==0:save(DEST/'status.json',{'status':'TRAIN96_clock_calibration','utc':now(),'completed_train':len(training),'total':96})
    save(DEST/'TRAIN96_self_clock_labels.json',{'rows':training,'role':'TRAINING ONLY frozen-network bias labels'})
    fits={};fitrecords=[]
    for shape in ['cylinder','square']:
        groups={}
        for a in training:
            if a['q']['shape']==shape:groups.setdefault(tuple(features(a['q'])),[]).append(a)
        X=np.array([np.mean([vector(a['q'],a['self_frequency']) for a in g],0) for g in groups.values()])
        y=np.array([np.mean([a['relative_clock_bias'] for a in g]) for g in groups.values()])
        options=[]
        for alpha in [1e-4,.01,1.]:
            errors=[]
            for i in range(len(y)):
                keep=np.arange(len(y))!=i;c=regression(X[keep],y[keep],alpha);errors.append(float((np.r_[1.,X[i]]@c-y[i])**2))
            options.append({'alpha':alpha,'LOO_bias_MSE':float(np.mean(errors))})
        best=min(options,key=lambda a:(a['LOO_bias_MSE'],a['alpha']))
        c=regression(X,y,best['alpha']);median=float(np.median(y));fits[shape]=(median,c)
        fitrecords.append({'shape':shape,'unique_geometry_groups':len(y),'median_bias':median,'ridge_coefficients':c.tolist(),'CV':options,'selected':best})
    save(DEST/'TRAIN96_calibration_model.json',{'utc':now(),'models':fitrecords,'training_roles_only':True})
    registry=[]
    for q in rows():
        row=startup(q);base,diag=infer(model,row,True);n=diag['prefix_steps'];obs=np.concatenate([row['fields'],base[:n]])
        t=row['times'][:n+4]-row['times'][0];future=row['times'][4:164]-row['times'][0]
        center=float(priors[q['shape']](features(q)[None])[0])/q['D'];fs,fd=frequency(t,obs[:,1][:,row['mask']>.5].astype(np.float64),4,center,.2)
        med,coef=fits[q['shape']];biases={'TRAIN96_shape_clock_bias':med,'TRAIN96_linear_clock_bias':float(np.r_[1.,vector(q,fs)]@coef)}
        predictions={'frozen_research':(base,diag)}
        for method,bias in biases.items():
            f=fs*(1+bias);assert f>0
            C=np.linalg.lstsq(basis(t,f,4),obs.reshape(len(t),-1).astype(np.float64),rcond=1e-10)[0]
            pred=(basis(future,f,4)@C).reshape(160,3,64,64).astype(np.float32)*row['mask'][None,None]
            pred[:n]=base[:n];assert np.isfinite(pred).all()
            predictions[method]=(pred,{'self_frequency':fs,'calibrated_frequency':f,'TRAIN_only_bias':bias,'prefix':n})
        for method,(pred,diag) in predictions.items():
            file=DEST/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'diagnostic':diag})
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
        print('CLOCK_BIAS',method,round(balanced(s),6),ok,flush=True)
    protected();save(DEST/'report.json',{'utc':now(),'methods':result,'eligible_signal_methods':eligible,'role':'seen56 DEVELOPMENT only','fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'eligible_signal_methods':eligible})

if __name__=='__main__':
    try:main()
    except Exception as e:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
