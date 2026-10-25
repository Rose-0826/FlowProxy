#!/usr/bin/env python3
"""Validation-selected ordinary cycle continuation; already-seen development ONLY."""
from pathlib import Path
import json,sys,datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from flowproxy.models import build_model
from corrected_benchmark_train_v1 import protocol
from physical_timestep_train_v1 import predict
from corrected_benchmark_controls_v1 import dmd_periodic
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

TRAIN=ROOT/'outputs/physical_timestep_train_v1'
OUT=ROOT/'outputs/cycle_continuation_eval_v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def run():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=False);(OUT/'predictions').mkdir()
    training=json.loads((TRAIN/'completed_report.json').read_text());c=json.loads((TRAIN/'preregister.json').read_text())
    controls=[x for x in training['candidates'] if x['arm']=='control']
    chosen=min(controls,key=lambda x:(x['summary']['DMD9']['40s']['mean_R'],x['summary']['DMD9']['40s']['max_R']))
    assert chosen['seed']==55 and sha(chosen['path'])==chosen['sha256']
    previous=ROOT/'outputs/period_consistency_evaluate_v1'
    fresh4=ROOT/'outputs/period_consistency_confirmation4_v1'
    newcfg=json.loads((fresh4/'preregister.json').read_text())
    p=protocol();cache=ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
    groups=[('seen32',p['splits']['test'],cache,previous/'already_seen_test32/report.json','cycle_seed11_self_DMD_rank9'),
            ('seen12',[x['id'] for x in p['confirmation_cases']],cache,previous/'fresh_confirmation12/report.json','cycle_seed11_self_DMD_rank9'),
            ('seen4',[x['id'] for x in newcfg['cases']],ROOT/'data/processed/period_consistency_confirmation4_v1_cases',fresh4/'report.json','cycle_seed11_DMD9')]
    save(OUT/'preregister.json',{'registered_utc':now(),'role':'all48 cases ALREADY SEEN; exploratory development re-evaluation, not new blind confirmation',
       'selected_by':'original val8 only, best control continuation; physical timestep method gate FAILED',
       'network':chosen,'control':'unchanged cycle_seed11 frozen primary, same DMD9 rank/scales and48step prefix',
       'groups':[{'label':label,'ids':ids,'cache':str(folder)} for label,ids,folder,_,_ in groups],
       'no_further_training':True,'no_new_CFD':True,'script_sha256':sha(__file__),'showcase_policy':'current evening deck remains frozen old-primary evidence until development effect understood'})
    saved=torch.load(chosen['path'],map_location='cpu',weights_only=False)
    model=build_model(saved['model_name'],17,saved['out_channels'],p['config']).cuda();model.load_state_dict(saved['model'])
    registry=[];startup={};count=0
    for label,ids,folder,oldreport,oldkey in groups:
        for cid in ids:
            with np.load(folder/(cid+'_startup.npz')) as a:
                row={'id':cid,'fields':a['fields'],'times':a['times'],'geom':a['geom'],'mask':a['mask'],'q':{'D':1.}}
            assert row['fields'].shape[0]==4
            base=predict(model,row,'control',c)
            observed=np.concatenate([row['fields'],base[:48]])
            closed,diag=dmd_periodic(observed,row['times'][:52],row['times'][4:164],row['mask'],9,c['channel_scales_train80']);closed[:48]=base[:48]
            f=OUT/'predictions'/(cid+'.npz');np.savez_compressed(f,direct=base,DMD9=closed)
            registry.append({'id':cid,'group':label,'path':str(f),'sha256':sha(f),'DMD':diag});startup[cid]=row;count+=1
            save(OUT/'status.json',{'status':'predicting','completed_predictions':count,'total':48})
    save(OUT/'predictions_frozen.json',{'utc':now(),'all_predictions_before_future_scoring':True,'registry':registry})
    results={}
    for label,ids,folder,oldreport,oldkey in groups:
        scored={'direct':[],'DMD9':[]}
        for cid in ids:
            with np.load(folder/(cid+'.npz')) as a:truth=a['fields'][4:164]
            row=startup[cid]
            with np.load(OUT/'predictions'/(cid+'.npz')) as a:
                for key in scored:scored[key].append({'id':cid,'horizons':metrics(a[key],truth,row['mask'],row['times'])})
        summary={key:{h:summarize([x['horizons'][h] for x in vals]) for h in ['2s','8s','20s','40s']} for key,vals in scored.items()}
        before=json.loads(oldreport.read_text())['summary'][oldkey]
        results[label]={'role':'ALREADY SEEN exploratory re-evaluation','summary':summary,'previous_frozen_primary_DMD9':before,'cases':scored}
        save(OUT/(label+'_report.json'),results[label])
        print('CONTINUATION_EVAL',label,'endUy',summary['DMD9']['40s']['end'][1],'previous',before['40s']['end'][1],flush=True)
    save(OUT/'completed_report.json',{'role':'ALREADY SEEN development; no additional method novelty established','network':chosen,'groups':results})
    save(OUT/'status.json',{'status':'complete','total':48,'frozen_primary_changed':False})

if __name__=='__main__':
    try:run()
    except Exception as e:
        if OUT.exists():save(OUT/'status.json',{'status':'failed','error':repr(e),'utc':now()})
        raise
