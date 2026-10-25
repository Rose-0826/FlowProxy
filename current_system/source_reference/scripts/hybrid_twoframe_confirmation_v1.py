#!/usr/bin/env python3
"""Four new geometry/Re cases, nominated two-real-frame pipeline before CFD."""
from pathlib import Path
import sys,json,copy,datetime,os,time,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'cfd/scripts')]
from geometry_attack_v2 import save,sha,NativeSampler
from aic_stop_control import stop_job,active_group_members
OUT=ROOT/'outputs/hybrid_twoframe_confirmation_v1';RAW=ROOT/'data/raw/hybrid_twoframe_confirmation_v1'

def prepare():
    assert not OUT.exists() and not RAW.exists();OUT.mkdir()
    cfg=json.loads((ROOT/'outputs/geometry_attack_confirmation_d1_v1/preregister.json').read_text())['CFD']['config']
    rows=[{'id':f'{s}_Re{r}_cy{y:g}_D1','shape':s,'Re':r,'cy':y,'cx':0.,'D':1.} for s in ['cylinder','square'] for r,y in [(125,-.65),(185,.65)]]
    nomination={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary':'two_true_frames_periodic',
      'checkpoint_sha256':sha(ROOT/'outputs/checkpoints/fno_k8_base/base.pt'),
      'frequency_priors_sha256':sha(ROOT/'outputs/startup_observability_v1/training_priors.json'),
      'prediction_function_sha256':sha(ROOT/'scripts/hybrid_observation_budget_v1.py'),
      'N':48,'H':4,'prior':'train80 shape mean; search +-10%, 61 points plus bounded refinement',
      'support_only':[2,3],'history_slots':[2,2,2,3],'fit_support':'two actual observed frames at their actual times plus 48 autonomous predictions',
      'pressure':'both observed frames include raw p; no future pressure reference changes','no_hyperparameter_retuning':True}
    save(OUT/'nomination.json',nomination)
    save(OUT/'preregister.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':rows,
      'CFD':{'config':cfg,'to':201,'record_from':160,'record_every':.25,'workers':2,'timeout':3600},
      'new_cases_not_in_original120_or_previous28_confirmations':True,'Re_interpolation_position_extrapolation_D1':True,
      'nomination_before_CFD':True,'primary':'two_true_frames_periodic','baseline':'two_true_frames_K8',
      'support_indices':[2,3],'not_reading_support_indices0_1':True,'future_indices':[4,163],
      'horizons':[2,8,20,40],'success':{'all4_8s_pass':True,'40s_at_least3_pass':True,'40s_meanR_below':1,'40s_maxR_below':1.5},
      'R':'max of channelwise time means and endpoints divided by .05/.05/.08; strict R<1',
      'reference_quality':'same original discrete CFD protocol; not claimed mesh independent',
      'no_replacement_of_original_four_frame_primary':True,'script_sha256':sha(__file__)})
    print('TWOFRAME_CONFIRMATION_REGISTERED',flush=True)

def run():
    from generate_case import write_case
    from run_long import two_stage
    p=json.loads((OUT/'preregister.json').read_text());assert not RAW.exists();RAW.mkdir()
    cfg=copy.deepcopy(p['CFD']['config']);cfg['cfd'].update({'end_time':160.,'write_interval':160.})
    done=[];errors=[]
    def one(row):
        begin=time.perf_counter();case=write_case(cfg,RAW,Re=row['Re'],cx=row['cx'],cy=row['cy'],D=row['D'],shape=row['shape'])
        two_stage(case,160.,.25,201.)
        with (case/'log.Allrun').open('x') as log:
            proc=subprocess.Popen(['bash',str(case/'Allrun')],cwd=case,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
            try:
                rc=proc.wait(timeout=p['CFD']['timeout'])
                if active_group_members(proc.pid):stop_job(proc);raise RuntimeError('Active child after solver ended')
            except BaseException:
                if proc.poll() is None:stop_job(proc)
                raise
        if rc:raise RuntimeError('CFD failed '+row['id'])
        assert 'End' in (case/'log.icoFoam2').read_text()[-1600:]
        result={'id':row['id'],'seconds':time.perf_counter()-begin,'normal_solver_end':True,'no_active_children':True,'solver_log_sha256':sha(case/'log.icoFoam2')}
        print('TWOFRAME_CFD_DONE',row['id'],round(result['seconds'],1),flush=True);return result
    save(OUT/'cfd_status.json',{'status':'running','completed':done,'errors':errors,'total':4})
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(one,row):row['id'] for row in p['cases']}
        for future in as_completed(fs):
            try:done.append(future.result())
            except Exception as e:errors.append({'case':fs[future],'error':repr(e)})
            save(OUT/'cfd_status.json',{'status':'running','completed':done,'errors':errors,'total':4})
    save(OUT/'cfd_status.json',{'status':'failed' if errors else 'complete','completed':done,'errors':errors,'total':4})
    if errors:raise RuntimeError('Preserve CFD failures')
    evaluate()

def evaluate():
    import numpy as np,torch
    from hybrid_observation_budget_v1 import predict
    from geometry_attack_evaluate_v1 import Candidate,specs,CFG
    from cpu_data_prepare import frames
    from flowproxy.geometry import geometry_channels
    from self_trajectory_closure_v1 import build_priors
    from round2_complete40_v1 import frame_summary,summarize
    p=json.loads((OUT/'preregister.json').read_text());nom=json.loads((OUT/'nomination.json').read_text())
    assert sha(ROOT/'outputs/checkpoints/fno_k8_base/base.pt')==nom['checkpoint_sha256']
    assert sha(ROOT/'scripts/hybrid_observation_budget_v1.py')==nom['prediction_function_sha256']
    assert sha(ROOT/'outputs/startup_observability_v1/training_priors.json')==nom['frequency_priors_sha256']
    out=OUT/'evaluation';out.mkdir(exist_ok=False);(out/'predictions').mkdir();torch.set_num_threads(1)
    model=Candidate(next(s for s in specs() if s['name']=='K8'));prior=build_priors();registry=[];lookup={}
    names=['two_true_frames_periodic','two_true_frames_K8']
    for row in p['cases']:
        cid=row['id'];q={k:row[k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(CFG,**q)
        items=frames(RAW/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
        support=np.zeros((4,3,64,64),np.float32)
        # Exactly two reference field files are opened by the prediction loader.
        for index in [2,3]:support[index]=sampler.sample(items[index][1])
        primary,diag=predict(model,q,support,times,geo['mask'],'true2',prior)
        baseline,_=model.predict(q,support[[2,2,2,3]],times,RAW/cid)
        for name,pred in zip(names,[primary,baseline]):
            file=out/'predictions'/f'{name}__{cid}.npz';np.savez_compressed(file,prediction=pred)
            registry.append({'case':cid,'model':name,'file':str(file),'sha256':sha(file),'opened_support_indices':[2,3]})
        lookup[cid]={'items':items,'times':times,'mask':geo['mask'],'q':q,'diagnostic':diag}
    save(out/'predictions_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_future_fields_not_read_before_commit':True,'rows':registry})
    reports={name:[] for name in names}
    for cid,d in lookup.items():
        geo=geometry_channels(CFG,**d['q']);sampler=NativeSampler(geo['xx'],geo['yy'],d['mask'])
        target=np.stack([sampler.sample(path) for _,path in d['items'][4:164]]).astype(np.float64)*d['mask'][None,None]
        for name in names:
            entry=next(r for r in registry if r['case']==cid and r['model']==name);assert sha(entry['file'])==entry['sha256']
            pred=np.load(entry['file'])['prediction'].astype(np.float64)
            error=np.linalg.norm((pred-target)*d['mask'][None,None],axis=(-2,-1))/np.maximum(np.linalg.norm(target,axis=(-2,-1)),1e-8)
            metrics={f'{sec}s':frame_summary(error[:n],float(d['times'][3]),float(d['times'][n+3]),sec,True) for sec,n in [(2,8),(8,32),(20,80),(40,160)]}
            reports[name].append({'id':cid,'horizons':metrics,'diagnostic':d['diagnostic'] if name==names[0] else {}})
    summary={name:{h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']} for name,rows in reports.items()}
    s=summary[names[0]];passed=s['8s']['n_pass']==4 and s['40s']['n_pass']>=3 and s['40s']['mean_R']<1 and s['40s']['max_R']<1.5
    save(out/'report.json',{'cases':reports,'summary':summary,'preregistered_gate_passed':passed,'support_read_count_per_case':2,'discrete_reference_only':True})
    save(out/'status.json',{'status':'complete','n':4,'gate_passed':passed})
    print('TWOFRAME_CONFIRMATION_EVALUATED',passed,s['40s'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    prepare() if a.prepare else evaluate() if a.evaluate else run()
