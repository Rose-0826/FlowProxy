#!/usr/bin/env python3
"""Validation-selected candidates, evaluated only on already-seen development."""
from pathlib import Path
import argparse, datetime, json, sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from corrected_benchmark_train_v1 import protocol,model_from,predict
from corrected_benchmark_controls_v1 import dmd_periodic
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize
from cycle_landmarks_train_v1 import FORMAL,FORMAL_SHA

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def run():
    suite="joint_DMD40";allow_gate_failed=False;nonzero_only=True
    torch.set_num_threads(1)
    training=ROOT/'outputs/end_to_end_periodic_train_v1';out=ROOT/'outputs/end_to_end_periodic_development_v1'
    out.mkdir(exist_ok=False);(out/'predictions').mkdir()
    trained=json.loads((training/'training_summary.json').read_text());c=json.loads((training/'preregister.json').read_text())
    assert trained['method_gate_passed'] or allow_gate_failed
    candidates=trained['candidates']
    if nonzero_only:candidates=[item for item in candidates if item['selection']['step']>0]
    assert candidates
    for item in candidates:assert sha(item['path'])==item['sha256']
    p=protocol();previous=ROOT/'outputs/period_consistency_evaluate_v1';fresh4=ROOT/'outputs/period_consistency_confirmation4_v1'
    four=json.loads((fresh4/'preregister.json').read_text())['cases'];cache=ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
    groups=[('seen32',p['splits']['test'],cache,previous/'already_seen_test32/report.json','cycle_seed11_self_DMD_rank9'),
        ('seen12',[q['id'] for q in p['confirmation_cases']],cache,previous/'fresh_confirmation12/report.json','cycle_seed11_self_DMD_rank9'),
        ('seen4',[q['id'] for q in four],ROOT/'data/processed/period_consistency_confirmation4_v1_cases',fresh4/'report.json','cycle_seed11_DMD9')]
    save(out/'preregister.json',{'utc':now(),'suite':suite,'role':'already-seen48 DEVELOPMENT; not fresh confirmation',
        'candidates_all_validation_selected':candidates,'all_predictions_before_future_scoring':True,
        'method_gate_passed':trained['method_gate_passed'],'additional_development_reason':'validation-selected nonzero continuation candidates; novelty gate still FAILED' if allow_gate_failed else 'passed training gate',
        'decoder':'sameDMD9,four true startup and48 self-prefix,normalization train80',
        'no_further_training':True,'no_new_CFD':True,'formal_model_preserved':True,'script_sha256':sha(__file__)})
    startup={};registry=[]
    for item in candidates:
        label=f"{item['arm']}_seed{item['seed']}";model,_=model_from(item['path'],p['config'])
        for group,ids,folder,oldreport,key in groups:
            for cid in ids:
                with np.load(folder/(cid+'_startup.npz')) as a:
                    row={'id':cid,'fields':a['fields'],'times':a['times'],'geom':a['geom'],'mask':a['mask']}
                assert row['fields'].shape[0]==4
                base=predict(model,row)
                closed,diag=dmd_periodic(np.concatenate([row['fields'],base[:48]]),row['times'][:52],row['times'][4:164],row['mask'],9,c['channel_scales_train80'])
                closed[:48]=base[:48]
                file=out/'predictions'/(label+'__'+cid+'.npz');np.savez_compressed(file,direct=base,DMD9=closed)
                registry.append({'model':label,'group':group,'id':cid,'sha256':sha(file),'DMD':diag});startup[cid]=row
                save(out/'status.json',{'status':'predicting','completed_predictions':len(registry),'total':len(candidates)*48})
        del model;torch.cuda.empty_cache()
    save(out/'predictions_frozen.json',{'utc':now(),'registry':registry,'all_saved_before_future_scoring':True})
    report={}
    for group,ids,folder,oldreport,key in groups:
        scored={f"{item['arm']}_seed{item['seed']}":{'direct':[],'DMD9':[]} for item in candidates}
        for cid in ids:
            with np.load(folder/(cid+'.npz')) as a:truth=a['fields'][4:164]
            row=startup[cid]
            for label,variants in scored.items():
                with np.load(out/'predictions'/(label+'__'+cid+'.npz')) as a:
                    for name in variants:variants[name].append({'id':cid,'horizons':metrics(a[name],truth,row['mask'],row['times'])})
        summaries={label:{mode:{h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']} for mode,rows in variants.items()} for label,variants in scored.items()}
        baseline=json.loads(oldreport.read_text())['summary'][key]
        report[group]={'summary':summaries,'cases':scored,'unchanged_frozen_primary_DMD9':baseline}
        save(out/(group+'_report.json'),report[group])
        print('SPRINT_DEVELOPMENT',suite,group,[(label,round(s['DMD9']['40s']['end'][1],5)) for label,s in summaries.items()],flush=True)
    assert sha(FORMAL)==FORMAL_SHA
    save(out/'completed_report.json',{'suite':suite,'role':'already-seen development only','groups':report,'formal_model_unchanged':True})
    save(out/'status.json',{'status':'complete','utc':now(),'formal_model_unchanged':True})



def finish():
    import os, time, tarfile, hashlib, io
    training=ROOT/'outputs/end_to_end_periodic_train_v1'
    status_path=ROOT/'outputs/end_to_end_periodic_followup_v1.json'
    begin=time.monotonic()
    while True:
        p=training/'status.json'
        try: state=json.loads(p.read_text())
        except (FileNotFoundError,json.JSONDecodeError):state={}
        if state.get('status')=='failed':raise RuntimeError(state)
        if state.get('status')=='complete':break
        if time.monotonic()-begin>3600:raise TimeoutError('Finite 3600-second followup deadline')
        save(status_path,{'status':'waiting_for_training','utc':now(),'finite_timeout_seconds':3600,'pid':os.getpid()})
        time.sleep(15)
    result=json.loads((training/'training_summary.json').read_text())
    if result['method_gate_passed']:
        save(status_path,{'status':'development_evaluation','utc':now(),'method_gate_passed':True})
        run()
    else:
        save(status_path,{'status':'method_gate_failed_no_development','utc':now(),'method_gate_passed':False})
    assert sha(FORMAL)==FORMAL_SHA
    files=sorted(p for p in training.rglob('*') if p.is_file() and p.suffix in ['.json','.py','.log'])
    files+=sorted(training.glob('tasks_seed*.npz'))
    files+=[ROOT/'scripts/end_to_end_periodic_train_v1.py',Path(__file__)]
    development=ROOT/'outputs/end_to_end_periodic_development_v1'
    if development.exists():files+=sorted(development.glob('*.json'))
    blobs=[(str(p.relative_to(ROOT)),p.read_bytes()) for p in files]
    manifest={'utc':now(),'method_gate_passed':result['method_gate_passed'],
        'formal_weight_unchanged':True,'files':[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in blobs]}
    archive=ROOT/'work/end_to_end_periodic_final_20261003.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,b in blobs+[('snapshot_manifest.json',json.dumps(manifest,indent=2).encode())]:
            info=tarfile.TarInfo(name);info.size=len(b);tar.addfile(info,io.BytesIO(b))
    save(status_path,{'status':'complete','utc':now(),'pid':os.getpid(),'method_gate_passed':result['method_gate_passed'],
        'development_done':development.exists(),'formal_weight_unchanged':True,'final_archive':str(archive),'archive_sha256':sha(archive)})
    print('FINITE_JOINT_FOLLOWUP_COMPLETE','GATE',result['method_gate_passed'],flush=True)

if __name__=='__main__':
    try:finish()
    except Exception as error:
        save(ROOT/'outputs/end_to_end_periodic_followup_v1.json',{'status':'failed','utc':now(),'error':repr(error)})
        raise
