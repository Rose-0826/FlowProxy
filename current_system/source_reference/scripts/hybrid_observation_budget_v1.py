#!/usr/bin/env python3
"""Fixed reduced-observation pilot; previously scored cases, no new training or formal score replacement."""
from pathlib import Path
import sys,json,datetime
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_evaluate_v1 import Candidate,specs,CFG
from geometry_attack_v2 import save,sha,NativeSampler
from cpu_data_prepare import frames
from flowproxy.geometry import geometry_channels
from self_trajectory_closure_v1 import build_priors
from startup_observability_v1 import basis,frequency
from round2_complete40_v1 import frame_summary,summarize
OUT=ROOT/'outputs/hybrid_observation_budget_v1'

@torch.inference_mode()
def predict(model,q,true4,times,mask,variant,priors):
    if variant=='true4':ids=[0,1,2,3];hist=true4.copy()
    elif variant=='true2':ids=[2,3];hist=true4[[2,2,2,3]].copy()
    elif variant=='true1':ids=[3];hist=np.repeat(true4[3:4],4,axis=0)
    elif variant=='velocity4_only':ids=[0,1,2,3];hist=true4.copy();hist[:,2]=0
    else:raise ValueError(variant)
    geom=torch.from_numpy(geometry_channels(CFG,**q)['geom']).to('cuda')[None]
    state=torch.from_numpy(hist.astype(np.float32)).to('cuda')[None];generated=[]
    for _ in range(48):
        x=torch.cat([geom,state.reshape(1,12,64,64)],dim=1)
        v=model.model(x)[:,:3];state=torch.cat([state[:,1:],v[:,None]],dim=1);generated.append(v[0])
    own=torch.stack(generated).cpu().numpy()
    supplied=np.concatenate([true4[ids],own],axis=0).astype(np.float64)*mask[None,None]
    tau=np.r_[times[ids],times[4:52]].astype(np.float64)-times[3]
    center=priors[q['shape']][1]*CFG['physics']['U_inf']/q['D']
    f,diagnostic=frequency(tau,supplied[:,1][:,mask>.5],4,center,.1)
    B=basis(tau,f,4);C=np.empty((9,3,64,64),dtype=np.float64)
    for c in range(3):
        if c==2 and variant=='velocity4_only':
            # Initial pressure is unknown; fit pressure only to self-generated fields.
            C[:,c]=np.linalg.lstsq(B[len(ids):],supplied[len(ids):,c].reshape(48,-1),rcond=1e-10)[0].reshape(9,64,64)
        else:C[:,c]=np.linalg.lstsq(B,supplied[:,c].reshape(len(supplied),-1),rcond=1e-10)[0].reshape(9,64,64)
    result=(basis(times[4:164]-times[3],f,4)@C.reshape(9,-1)).reshape(160,3,64,64).astype(np.float32)
    result[:48]=own
    return result,{'frequency':f,'frequency_fit':diagnostic,'actual_true_support_indices':ids,
      'actual_true_support_channels':['Ux','Uy'] if variant=='velocity4_only' else ['Ux','Uy','p'],
      'history_padding':'latest value or repeat latest two; no earlier hidden target fields',
      'pressure_initialization':'zero, excluded from pressure coefficient fit' if variant=='velocity4_only' else 'supplied pressure'}

def run():
    OUT.mkdir(exist_ok=False);(OUT/'predictions').mkdir()
    variants=['true4','true2','true1','velocity4_only'];folders=['geometry_attack_confirmation_v1','geometry_attack_confirmation_d1_v1','geometry_attack_confirmation_far_v1']
    config=json.loads((ROOT/'outputs/self_trajectory_closure_v1/selected_val.json').read_text())['candidate']
    spec=next(s for s in specs() if s['name']=='K8');torch.set_num_threads(1);model=Candidate(spec);priors=build_priors()
    save(OUT/'preregister.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variants':variants,
      'all16_previously_scored_cases_included':True,'posthoc_exploration_not_independent_confirmation':True,
      'training':False,'weights_sha256':sha(spec['path']),'base_configuration':config,
      'reduced_budget_fit_uses_actual_observed_times_only':True,'velocity_only_pressure_fit_uses_generated_fields_only':True,
      'candidate_gate':'on fixed D1 eight >=6/8 full40 and meanR<1; passing requires subsequent new confirmation before promotion',
      'no_new_hyperparameter_selection_on_cases':True,'no_formal_primary_replacement':True,'script_sha256':sha(__file__)})
    registry=[];lookup={};parity=[]
    for folder in folders:
        for row in json.loads((ROOT/'outputs'/folder/'preregister.json').read_text())['cases']:
            cid=row['id'];q={k:row[k] for k in ['shape','Re','D','cx','cy']};geo=geometry_channels(CFG,**q)
            items=frames(ROOT/'data/raw'/folder/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
            lookup[cid]={'folder':folder,'q':q,'mask':geo['mask'],'items':items,'times':times}
            true4=np.stack([sampler.sample(path) for _,path in items[:4]])
            for variant in variants:
                result,diag=predict(model,q,true4,times,geo['mask'],variant,priors)
                if variant=='true4':
                    old=np.load(ROOT/'outputs'/folder/'evaluation/predictions'/f'neural_periodic_closure__{cid}.npz')['prediction']
                    parity.append(float(np.max(np.abs(old-result))));assert parity[-1]<2e-6
                file=OUT/'predictions'/f'{variant}__{cid}.npz';np.savez_compressed(file,prediction=result)
                registry.append({'case':cid,'variant':variant,'file':str(file),'sha256':sha(file),'diagnostic':diag})
            print('OBSERVATION_PREDICTED',cid,flush=True)
    save(OUT/'predictions_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':registry,'true4_max_parity_delta':max(parity)})
    reports={v:[] for v in variants}
    for cid,d in lookup.items():
        geo=geometry_channels(CFG,**d['q']);sampler=NativeSampler(geo['xx'],geo['yy'],d['mask'])
        target=np.stack([sampler.sample(path) for _,path in d['items'][4:164]]).astype(np.float64)*d['mask'][None,None]
        norm=np.linalg.norm(target,axis=(-2,-1))
        for variant in variants:
            entry=next(r for r in registry if r['case']==cid and r['variant']==variant)
            assert sha(entry['file'])==entry['sha256'];pred=np.load(entry['file'])['prediction'].astype(np.float64)
            err=np.linalg.norm((pred-target)*d['mask'][None,None],axis=(-2,-1))/np.maximum(norm,1e-8)
            hs={f'{sec}s':frame_summary(err[:n],float(d['times'][3]),float(d['times'][n+3]),sec,True) for sec,n in [(2,8),(8,32),(20,80),(40,160)]}
            reports[variant].append({'id':cid,'D':d['q']['D'],'horizons':hs,'diagnostic':entry['diagnostic']})
    summary={}
    for group,predicate in [('fixed_D1_8',lambda r:abs(r['D']-1)<1e-6),('diameter8',lambda r:abs(r['D']-1)>1e-6),('all16',lambda r:True)]:
        summary[group]={v:{h:summarize([r['horizons'][h] for r in rows if predicate(r)]) for h in ['2s','8s','20s','40s']} for v,rows in reports.items()}
    save(OUT/'report.json',{'cases':reports,'summary':summary,'true4_reproduction_max_absolute_delta':max(parity),'no_primary_replacement':True})
    save(OUT/'status.json',{'status':'complete','n_cases':16,'n_variants':4,'exploratory_only':True})
    print('OBSERVATION_DONE',[(k,v['40s']['mean_R'],v['40s']['n_pass']) for k,v in summary['fixed_D1_8'].items()],flush=True)

if __name__=='__main__':run()
