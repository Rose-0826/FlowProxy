#!/usr/bin/env python3
"""Frozen predictor on two refined CFD anchors, as a reference-transfer diagnostic."""
from pathlib import Path
import sys,json,datetime,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_evaluate_v1 import Candidate,specs,CFG
from geometry_attack_v2 import save,sha,NativeSampler
from hybrid_observation_budget_v1 import predict
from cpu_data_prepare import frames
from flowproxy.geometry import geometry_channels
from self_trajectory_closure_v1 import build_priors
from round2_complete40_v1 import frame_summary,summarize
OUT=ROOT/'outputs/hybrid_refined_reference_v1';RAW=ROOT/'data/raw/cfd_numerics_round2_v1'

def prepare():
    OUT.mkdir(exist_ok=False)
    p=json.loads((ROOT/'outputs/cfd_numerics_round2_v1/preregister.json').read_text());spec=next(s for s in specs() if s['name']=='K8')
    save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':p['cases'],
      'variants':['fine_dt0005','third_dt0005'],'start_true_times':[281.,281.25,281.5,281.75],
      'future_scored':[282.,301.75],'future_duration':20.,'horizons':[2,8,20],
      'reference_transfer_diagnostic_not_new_independent_geometry_confirmation':True,
      'physical_times_exact_no_resampling':True,'frozen_weights_sha256':sha(spec['path']),
      'unchanged_N48_H4_frequency_prior_and_search':True,'no_model_selection_or_training':True,
      'coarse_training_to_fine_reference_not_expected_to_prove_full40':True,'script_sha256':sha(__file__)})

def run():
    p=json.loads((OUT/'protocol.json').read_text())
    while not (ROOT/'outputs/cfd_numerics_round2_v1/sensitivity_report.json').exists():time.sleep(10)
    torch.set_num_threads(1);spec=next(s for s in specs() if s['name']=='K8');assert sha(spec['path'])==p['frozen_weights_sha256'];model=Candidate(spec);prior=build_priors()
    (OUT/'predictions').mkdir(exist_ok=False);registry=[];lookup={}
    with torch.inference_mode():
        for variant in p['variants']:
            for row in p['cases']:
                cid=row['id'];q={k:row[k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(CFG,**q)
                items=frames(RAW/variant/cid);chosen=[(t,f) for t,f in items if 281-1e-7<=t<=301.75+1e-7]
                assert len(chosen)==84,(variant,cid,len(chosen));actual=np.array([t for t,_ in chosen]);assert np.max(np.abs(actual-(281+.25*np.arange(84))))<1e-7
                sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask']);support=np.stack([sampler.sample(f) for _,f in chosen[:4]])
                forecast_times=281+.25*np.arange(164)
                closure,diag=predict(model,q,support,forecast_times,geo['mask'],'true4',prior)
                geom=torch.from_numpy(geo['geom']).to('cuda')[None];state=torch.from_numpy(support.astype(np.float32)).to('cuda')[None];generated=[]
                for _ in range(80):
                    v=model.model(torch.cat([geom,state.reshape(1,12,64,64)],dim=1))[:,:3];state=torch.cat([state[:,1:],v[:,None]],dim=1);generated.append(v[0])
                direct=torch.stack(generated).cpu().numpy();key=variant+'__'+cid;lookup[key]=(geo,chosen,actual)
                for name,pred in [('K8',direct),('neural_periodic_closure',closure[:80])]:
                    f=OUT/'predictions'/f'{name}__{key}.npz';np.savez_compressed(f,prediction=pred)
                    registry.append({'case':cid,'variant':variant,'key':key,'model':name,'file':str(f),'sha256':sha(f),'diagnostic':diag if name!='K8' else {}})
                print('REFINED_PREDICTED',key,flush=True)
    save(OUT/'predictions_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':registry,'all_prediction_files_before_this_run_future_reads':True})
    reports=[]
    for key,(g,items,times) in lookup.items():
        sampler=NativeSampler(g['xx'],g['yy'],g['mask']);mask=g['mask'][None,None];target=np.stack([sampler.sample(f) for _,f in items[4:]]).astype(np.float64)*mask
        for r in [r for r in registry if r['key']==key]:
            assert sha(r['file'])==r['sha256'];pred=np.load(r['file'])['prediction'].astype(np.float64)
            error=np.linalg.norm((pred-target)*mask,axis=(-2,-1))/np.maximum(np.linalg.norm(target,axis=(-2,-1)),1e-8)
            reports.append({'case':r['case'],'variant':r['variant'],'model':r['model'],
              'horizons':{f'{sec}s':frame_summary(error[:n],times[3],times[n+3],sec,True) for sec,n in [(2,8),(8,32),(20,80)]}})
    summary={variant:{name:{h:summarize([r['horizons'][h] for r in reports if r['variant']==variant and r['model']==name]) for h in ['2s','8s','20s']} for name in ['K8','neural_periodic_closure']} for variant in p['variants']}
    save(OUT/'report.json',{'cases':reports,'summary':summary,'reference_transfer_diagnostic_only':True,'not_complete40':True,'does_not_replace_old_primary_scores':True})
    save(OUT/'status.json',{'status':'complete','n_references':4,'n_geometries':2})
    print('REFINED_REFERENCE_DONE',[(v,n,round(h['20s']['mean_R'],4),h['20s']['n_pass']) for v,a in summary.items() for n,h in a.items()],flush=True)

if __name__=='__main__':
    prepare() if '--prepare' in sys.argv else run()
