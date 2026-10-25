#!/usr/bin/env python3
"""One fixed history-completion hypothesis, on already seen cases only."""
from pathlib import Path
import sys,json,datetime
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from hybrid_observation_budget_v1 import predict as old_predict
from geometry_attack_evaluate_v1 import Candidate,specs,CFG
from geometry_attack_v2 import save,sha,NativeSampler
from cpu_data_prepare import frames
from flowproxy.geometry import geometry_channels
from self_trajectory_closure_v1 import build_priors
from startup_observability_v1 import frequency,basis
from round2_complete40_v1 import frame_summary,summarize
OUT=ROOT/'outputs/hybrid_twoframe_history_v1'

@torch.inference_mode()
def predict_linear(model,q,support,times,mask,priors):
    # Only indices 2 and 3 are genuine observations. Earlier history slots are
    # deterministic backward linear extrapolations, never target field reads.
    rate=(support[3]-support[2])/(times[3]-times[2])
    hist=np.stack([support[2]+(times[i]-times[2])*rate for i in range(4)]).astype(np.float32)
    geom=torch.from_numpy(geometry_channels(CFG,**q)['geom']).to('cuda')[None]
    state=torch.from_numpy(hist).to('cuda')[None];generated=[]
    for _ in range(48):
        v=model.model(torch.cat([geom,state.reshape(1,12,64,64)],dim=1))[:,:3]
        state=torch.cat([state[:,1:],v[:,None]],dim=1);generated.append(v[0])
    own=torch.stack(generated).cpu().numpy()
    supplied=np.concatenate([support[2:4],own]).astype(np.float64)*mask[None,None]
    tau=np.r_[times[2:4],times[4:52]]-times[3]
    center=priors[q['shape']][1]*CFG['physics']['U_inf']/q['D']
    f,diag=frequency(tau,supplied[:,1][:,mask>.5],4,center,.1)
    C=np.linalg.lstsq(basis(tau,f,4),supplied.reshape(50,-1),rcond=1e-10)[0]
    result=(basis(times[4:164]-times[3],f,4)@C).reshape(160,3,64,64).astype(np.float32)
    result[:48]=own
    return result,{'frequency':f,'fit':diag,'actual_true_indices':[2,3],
      'history_slots0_1':'backward linear extrapolation using actual saved timestamps',
      'coefficient_fit_uses_only_two_actual_observations_and_48_generated_fields':True}

def run():
    OUT.mkdir(exist_ok=False);(OUT/'predictions').mkdir();torch.set_num_threads(1)
    folders=['geometry_attack_confirmation_v1','geometry_attack_confirmation_d1_v1',
      'geometry_attack_confirmation_far_v1','hybrid_twoframe_confirmation_v1']
    spec=next(s for s in specs() if s['name']=='K8');model=Candidate(spec);priors=build_priors()
    save(OUT/'preregister.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'all20_cases_already_scored':True,'posthoc_exploration_not_confirmation':True,
      'single_fixed_hypothesis':'backward linear history completion from two actual frames',
      'no_new_training_or_parameter_search':True,'no_primary_replacement':True,
      'actual_true_support':[2,3],'prediction_function_sha256':sha(__file__),
      'weights_sha256':sha(spec['path']),
      'pilot_gate':'D1 original8 >=6 pass and meanR<1, plus failed fresh4 >=3 pass and meanR<1; requires another independent confirmation if passed'})
    reports={'two_repeat':[],'two_linear':[]};lookup={};registry=[]
    for folder in folders:
        for row in json.loads((ROOT/'outputs'/folder/'preregister.json').read_text())['cases']:
            cid=row['id'];q={k:row[k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(CFG,**q)
            items=frames(ROOT/'data/raw'/folder/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
            support=np.zeros((4,3,64,64),np.float32)
            for i in [2,3]:support[i]=sampler.sample(items[i][1])
            lookup[cid]={'q':q,'geo':geo,'items':items,'times':times,'folder':folder}
            for variant in reports:
                pred,diag=predict_linear(model,q,support,times,geo['mask'],priors) if variant=='two_linear' else old_predict(model,q,support,times,geo['mask'],'true2',priors)
                f=OUT/'predictions'/f'{variant}__{cid}.npz';np.savez_compressed(f,prediction=pred)
                registry.append({'case':cid,'variant':variant,'file':str(f),'sha256':sha(f),'diagnostic':diag,'support_read_indices':[2,3]})
            print('HISTORY_PREDICTED',cid,flush=True)
    save(OUT/'predictions_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':registry,'before_this_runs_future_reads':True})
    for cid,d in lookup.items():
        g=d['geo'];sampler=NativeSampler(g['xx'],g['yy'],g['mask']);mask=g['mask'][None,None]
        target=np.stack([sampler.sample(p) for _,p in d['items'][4:164]]).astype(np.float64)*mask
        for variant in reports:
            r=next(r for r in registry if r['case']==cid and r['variant']==variant);assert sha(r['file'])==r['sha256']
            pred=np.load(r['file'])['prediction'].astype(np.float64)
            error=np.linalg.norm((pred-target)*mask,axis=(-2,-1))/np.maximum(np.linalg.norm(target,axis=(-2,-1)),1e-8)
            hs={f'{sec}s':frame_summary(error[:n],float(d['times'][3]),float(d['times'][n+3]),sec,True) for sec,n in [(2,8),(8,32),(20,80),(40,160)]}
            reports[variant].append({'id':cid,'D':d['q']['D'],'folder':d['folder'],'horizons':hs})
    predicates={'original_D1_8':lambda r:r['D']==1 and r['folder']!='hybrid_twoframe_confirmation_v1',
      'failed_twoframe_fresh4':lambda r:r['folder']=='hybrid_twoframe_confirmation_v1',
      'diameter8':lambda r:r['D']!=1,'all20':lambda r:True}
    summary={group:{v:{h:summarize([r['horizons'][h] for r in rows if keep(r)]) for h in ['2s','8s','20s','40s']} for v,rows in reports.items()} for group,keep in predicates.items()}
    a=summary['original_D1_8']['two_linear']['40s'];b=summary['failed_twoframe_fresh4']['two_linear']['40s']
    passed=a['n_pass']>=6 and a['mean_R']<1 and b['n_pass']>=3 and b['mean_R']<1
    save(OUT/'report.json',{'cases':reports,'summary':summary,'pilot_gate_passed':passed,'no_primary_replacement':True,'exploratory_only':True})
    save(OUT/'status.json',{'status':'complete','n':20,'pilot_gate_passed':passed})
    print('HISTORY_DONE',passed,[(g,v['two_linear']['40s']['n_pass'],round(v['two_linear']['40s']['mean_R'],4)) for g,v in summary.items()],flush=True)

if __name__=='__main__':run()
