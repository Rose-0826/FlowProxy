#!/usr/bin/env python3
"""Fixed 32-second phase-start stress test on all already-scored sixteen cases."""
from pathlib import Path
import sys,json,datetime
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_evaluate_v1 import Candidate,specs,CFG
from geometry_attack_v2 import save,sha,NativeSampler
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import frame_summary,summarize
OUT=ROOT/'outputs/hybrid_start_phase_v1'

def run():
    OUT.mkdir(exist_ok=False);(OUT/'predictions').mkdir()
    folders=['geometry_attack_confirmation_v1','geometry_attack_confirmation_d1_v1','geometry_attack_confirmation_far_v1']
    offsets=[0,8,16,24];names=['K8','neural_periodic_closure']
    selected=[s for s in specs() if s['name'] in names]
    protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'posthoc_stress_test_after_all16_primary_scores':True,'not_new_independent_cases':True,
      'all16_included':True,'offset_frames':offsets,'nominal_start_offsets_seconds':[0,2,4,6],
      'horizon_steps':128,'horizon_seconds':32,'support_frames':4,
      'primary_and_baseline_frozen':[{**s,'file_sha256':sha(s['path'])} for s in selected],
      'selection_or_retuning':False,'script_sha256':sha(__file__),
      'all_predictions_committed_before_stress_future_read':True}
    save(OUT/'preregister.json',protocol);torch.set_num_threads(1)
    lookup={};supports={}
    for folder in folders:
        rows=json.loads((ROOT/'outputs'/folder/'preregister.json').read_text())['cases']
        for row in rows:
            cid=row['id'];q={k:row[k] for k in ['shape','Re','cx','cy','D']}
            raw=ROOT/'data/raw'/folder/cid;geo=geometry_channels(CFG,**q);items=frames(raw)
            sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask']);lookup[cid]={'row':row,'folder':folder,'items':items,'raw':raw,'mask':geo['mask'],'q':q}
            # Read only the declared support4 for each shifted task.
            for offset in offsets:
                support=np.stack([sampler.sample(path) for _,path in items[offset:offset+4]])
                times=np.array([t for t,_ in items[offset:offset+164]],dtype=np.float64)
                while len(times)<164:times=np.r_[times,times[-1]+.25]
                supports[(cid,offset)]=(support,times)
    registry=[]
    for spec in selected:
        model=Candidate(spec)
        for (cid,offset),(support,times) in supports.items():
            d=lookup[cid];prediction,_=model.predict(d['q'],support,times,d['raw'])
            file=OUT/'predictions'/f"{spec['name']}__{cid}__offset{offset}.npz"
            np.savez_compressed(file,prediction=prediction[:128])
            registry.append({'model':spec['name'],'case':cid,'offset':offset,'file':str(file),'sha256':sha(file)})
        del model;torch.cuda.empty_cache()
        print('PHASE_PREDICTIONS_DONE',spec['name'],flush=True)
    save(OUT/'predictions_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':registry,'support_only_read_for_this_test':True})
    reports={n:[] for n in names};prefix_delta=[]
    for cid,d in lookup.items():
        geo=geometry_channels(CFG,**d['q']);sampler=NativeSampler(geo['xx'],geo['yy'],d['mask'])
        truth=np.stack([sampler.sample(path) for _,path in d['items']])
        for offset in offsets:
            times=supports[(cid,offset)][1];bank={}
            for name in names:
                file=OUT/'predictions'/f'{name}__{cid}__offset{offset}.npz'
                saved=next(r for r in registry if r['file']==str(file));assert sha(file)==saved['sha256']
                pred=np.load(file)['prediction'];bank[name]=pred
                target=truth[offset+4:offset+132].astype(np.float64)
                delta=(pred.astype(np.float64)-target)*d['mask'][None,None];target*=d['mask'][None,None]
                error=np.linalg.norm(delta,axis=(-2,-1))/np.maximum(np.linalg.norm(target,axis=(-2,-1)),1e-8)
                metrics=frame_summary(error,float(times[3]),float(times[131]),32,True)
                reports[name].append({'id':cid,'offset_frames':offset,'start_time':float(times[3]),'D':d['q']['D'],
                  'folder':d['folder'],'metrics':metrics})
            prefix_delta.append(float(np.max(np.abs(bank['K8'][:48]-bank['neural_periodic_closure'][:48]))))
    summary={}
    for group,predicate in [('fixed_D1_8',lambda r:abs(r['D']-1)<1e-6),('diameter8',lambda r:abs(r['D']-1)>1e-6),('all16',lambda r:True)]:
        summary[group]={}
        for name,rows in reports.items():
            summary[group][name]={f'offset{offset}':summarize([r['metrics'] for r in rows if r['offset_frames']==offset and predicate(r)]) for offset in offsets}
            ids=set(r['id'] for r in rows if predicate(r))
            summary[group][name]['all_four_starts_passed_cases']=sum(all(r['metrics']['R']<1 for r in rows if r['id']==cid) for cid in ids)
            summary[group][name]['n_independent_cases']=len(ids)
    save(OUT/'report.json',{'protocol':protocol,'cases':reports,'summary':summary,'first48_K8_and_closure_max_delta':max(prefix_delta)})
    save(OUT/'status.json',{'status':'complete','independent_cases':16,'tasks':64,'models':2,'all_predictions_verified':True})
    print('PHASE_STRESS_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':run()
