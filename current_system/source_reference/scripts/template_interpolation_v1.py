#!/usr/bin/env python3
"""Phase-aligned interpolation of training CFD cycles. Post-hoc stronger baseline; no target future fitting."""
from pathlib import Path
import sys,json,datetime,time
import numpy as np,h5py
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from nearest_cycle_template_v1 import predict,mapped_coefficients
from geometry_attack_v2 import save,sha,attrs,NativeSampler
from self_trajectory_closure_v1 import P,DATA,build_priors,metrics
from startup_observability_v1 import basis
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/template_interpolation_v1'


class ConstantPrior:
    def __init__(self,value):self.value=value
    def __call__(self,x):return np.full(len(x),self.value)


def prepare_library():
    rows=json.loads((ROOT/'outputs/startup_observability_v1/training_priors.json').read_text());freqs={r['id']:r['St'] for r in rows};templates=[]
    with h5py.File(DATA,'r') as h:
        for cid in P['train_ids']:
            g=h['cases'][cid];q=attrs(g);t=g['times'][:161].astype(np.float64);t-=160
            f=freqs[cid]*P['config']['physics']['U_inf']/q['D'];fields=g['fields'][:161].astype(np.float64)
            coef=np.linalg.lstsq(basis(t,f,4),fields.reshape(161,-1),rcond=1e-10)[0].reshape(9,3,64,64)
            force=g['forces'][...].astype(np.float64);force=force[(force[:,0]>=160)&(force[:,0]<=200)]
            fitted=np.linalg.lstsq(basis(force[:,0]-160,f,1),force[:,2],rcond=1e-10)[0]
            phase=np.arctan2(fitted[1],fitted[2])
            aligned=coef.copy()
            for k in range(1,5):
                c,s=np.cos(k*phase),np.sin(k*phase);a,b=coef[2*k-1],coef[2*k]
                aligned[2*k-1]=a*c-b*s;aligned[2*k]=a*s+b*c
            templates.append({'id':cid,'q':q,'coef':aligned,'mask':g['mask'][...],'St':freqs[cid],'phase_gauge':float(phase)})
    return templates


def select_sources(q,templates):
    eligible=[s for s in templates if s['q']['shape']==q['shape']]
    same=[s for s in eligible if abs(s['q']['cy']-q['cy'])<1e-6 and abs(s['q']['D']-q['D'])<1e-6]
    lower=[s for s in same if s['q']['Re']<=q['Re']];upper=[s for s in same if s['q']['Re']>=q['Re']]
    if lower and upper:
        lo=max(lower,key=lambda s:s['q']['Re']);hi=min(upper,key=lambda s:s['q']['Re'])
        if lo['id']==hi['id']:return [(lo,1.)],'exact_training_geometry_Re'
        w=(q['Re']-lo['q']['Re'])/(hi['q']['Re']-lo['q']['Re']);return [(lo,1-w),(hi,w)],'bracketed_Re_same_geometry'
    source=min(eligible,key=lambda s:(sum(((q[k]-s['q'][k])/scale)**2 for k,scale in [('Re',120),('cy',.8),('D',.2)]),s['id']))
    return [(source,1.)],'nearest_fallback_no_same_geometry_bracket'


def forecasts(q,support4,times,mask,library,priors):
    sources,selection=select_sources(q,library)
    coef=sum(w*mapped_coefficients(s,q,P['config']) for s,w in sources)
    st=sum(w*s['St'] for s,w in sources);synthetic={'id':'+'.join(s['id'] for s,w in sources),'q':q,'mask':mask,'coef':coef}
    result={}
    for name,frequencies in [('linear_frequency',{q['shape']:(ConstantPrior(st),st)}),('RBF_frequency',priors)]:
        predictions=predict(q,support4,times,mask,[synthetic],frequencies)
        for method,(prediction,diagnostic) in predictions.items():
            diagnostic.update({'sources':[{'id':s['id'],'weight':float(w)} for s,w in sources],'source_selection':selection,
                               'force_phase_alignment':'training force only, canonical Cl first harmonic = sin phase'})
            result[name+'_'+('startup_fit' if method.endswith('startup_frequency') else 'fixed')]=(prediction,diagnostic)
    return result


def run():
    OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'posthoc_after_initial12_primary_scores':True,
        'no_target_future_in_fit':True,'train80_only':True,'phase_gauge':'training lift first harmonic defines positive sine phase',
        'donors':'same shape/D/cy bracket Re if available, linear coefficients/St; otherwise exact original NN rule',
        'startup':'four real target fields fit phase and optional frequency; same +-10% bounds and optimizer as NN',
        'no_hyperparameter_selection':True,'no_primary_replacement':True,'script_sha256':sha(__file__)})
    library=prepare_library();priors=build_priors();results={};started=time.perf_counter()
    groups=[('val','',P['val_ids']),('old32_seen','',P['old_seen_test_ids']),('fresh_diameter8','geometry_attack_confirmation_v1',None),
            ('fresh_fixed_D1_4','geometry_attack_confirmation_d1_v1',None)]
    far=ROOT/'outputs/geometry_attack_confirmation_far_v1/evaluation/status.json'
    if far.exists():groups.append(('fresh_far_D1_4','geometry_attack_confirmation_far_v1',None))
    with h5py.File(DATA,'r') as h:
        for role,folder,ids in groups:
            isnew=ids is None
            if isnew:
                cases=json.loads((ROOT/'outputs'/folder/'preregister.json').read_text())['cases'];lookup={r['id']:r for r in cases};ids=list(lookup)
            report={}
            for cid in ids:
                if isnew:
                    q={k:lookup[cid][k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(P['config'],**q);mask=geo['mask']
                    items=frames(ROOT/'data/raw'/folder/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],mask)
                    support4=np.stack([sampler.sample(path) for _,path in items[:4]])
                else:g=h['cases'][cid];q=attrs(g);times=g['times'][...];mask=g['mask'][...];support4=g['fields'][:4]
                predictions=forecasts(q,support4,times,mask,library,priors)
                truth=np.stack([sampler.sample(path) for _,path in items[4:164]]) if isnew else h['cases'][cid]['fields'][4:164]
                for method,(prediction,diagnostic) in predictions.items():report.setdefault(method,[]).append({'id':cid,'horizons':metrics(prediction,truth,mask,times),'diagnostic':diagnostic})
            summary={name:{key:summarize([r['horizons'][key] for r in rows]) for key in ['2s','8s','20s','40s']} for name,rows in report.items()}
            results[role]={'cases':report,'summary':summary};print('INTERPOLATED_TEMPLATE',role,{name:(round(d['40s']['mean_R'],4),d['40s']['n_pass']) for name,d in summary.items()},flush=True)
    save(OUT/'report.json',results);save(OUT/'summary.json',{r:d['summary'] for r,d in results.items()});save(OUT/'status.json',{'status':'complete','seconds':time.perf_counter()-started})


if __name__=='__main__':run()
