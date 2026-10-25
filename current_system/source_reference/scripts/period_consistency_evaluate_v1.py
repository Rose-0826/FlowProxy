#!/usr/bin/env python3
"""Freeze decoder on val8, then exploratory rescore previously seen 32+12."""
from pathlib import Path
import datetime
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from corrected_benchmark_train_v1 import model_from,predict,protocol
from self_trajectory_closure_v1 import closure,metrics
from corrected_benchmark_controls_v1 import controls
from round2_complete40_v1 import summarize

OLD=ROOT/'outputs/corrected_benchmark_restart_v1'
TRAIN=ROOT/'outputs/period_consistency_train_v1'
OUT=ROOT/'outputs/period_consistency_evaluate_v1'
TRAINVAL=ROOT/'data/processed/dataset_corrected_trainval_v1.h5'
CACHE=ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
MODES=['direct','closure','cycle_replay','self_DMD_rank9']


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def summary(rows):
    return {h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']}


def q_from(g):
    return {k:str(g.attrs[k]) if k=='shape' else float(g.attrs[k]) for k in ['shape','Re','cx','cy','D']}


def variants(model,row,p,prior,scales):
    base=predict(model,row)
    closed,diag=closure(row['q'],row['fields'][:4],base,row['times'],row['mask'],prior,p['closure'])
    alternatives=controls(row['fields'][:4],base,row['times'],row['mask'],p['closure'],diag['frequency'],scales)
    return {'direct':base,'closure':closed,**{k:v[0] for k,v in alternatives.items()}},diag


def run():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    report=json.loads((TRAIN/'completed_report.json').read_text());assert report['gate_passed']
    p=protocol();oldlock=json.loads((OLD/'selected_model_lock.json').read_text())
    candidates=[c for c in report['all_candidates'] if c['arm']=='cycle']
    selected=report['selected_cycle_for_followup_if_gate_passed']
    assert selected and all(sha(c['path'])==c['sha256'] for c in candidates)
    names={c['seed']:f'cycle_seed{c["seed"]}' for c in candidates}
    selected_name=names[selected['seed']]
    scales=json.loads((Path(oldlock['chosen']['path']).parent/'configuration.json').read_text())['normalization_train80']
    prior={k:(None,v) for k,v in oldlock['shape_mean_train80_St'].items()}
    save(OUT/'preregister.json',{'registered_utc':now(),'training_gate_passed':True,'selected_network':selected,
        'replicate_networks':candidates,'decoders_fixed':MODES,'bootstrap_steps':48,'H':4,'DMD_rank':9,
        'decoder_selection':'selected network only, val8 minimum40s meanRi, then maxRi, mode order; direct included',
        'decoder_selection_precedes_development_scoring':True,
        'comparisons':'both cycle seeds, original frozen network with SAME decoder and same protocol',
        'seen_case_roles':'32 old development geometries +12 previously scored confirmation; exploratory, not new blind confirmation',
        'startup_only':4,'no_target_force_or_future_fields_for_prediction':True,'pressure_reference_unchanged':True,
        'selected_cycle_weights_sha256':selected['sha256'],'training_gate_report_sha256':sha(TRAIN/'completed_report.json'),
        'script_sha256':sha(__file__)})
    models={names[c['seed']]:model_from(c['path'],p['config'])[0] for c in candidates}
    save(OUT/'status.json',{'status':'running','stage':'decoder_validation'})
    folder=OUT/'validation';folder.mkdir()
    scored={name+'_'+mode:[] for name in models for mode in MODES}
    with h5py.File(TRAINVAL,'r') as h:
        for cid in p['splits']['val']:
            g=h['cases'][cid]
            row={'id':cid,'q':q_from(g),'fields':g['fields'][:4],'geom':g['geom'][...],'mask':g['mask'][...],'times':g['times'][...]}
            frozen=[]
            for name,model in models.items():
                results,diag=variants(model,row,p,prior,scales)
                for mode,pred in results.items():
                    label=name+'_'+mode;fp=folder/(label+'__'+cid+'.npz');np.savez_compressed(fp,prediction=pred)
                    frozen.append({'id':cid,'label':label,'path':str(fp),'sha256':sha(fp),'closure_diagnostic':diag})
            save(folder/(cid+'_frozen.json'),{'all_saved_before_future_read':True,'registry':frozen})
            truth=g['fields'][4:164]
            for entry in frozen:
                pred=np.load(entry['path'])['prediction']
                scored[entry['label']].append({'id':cid,'q':row['q'],'horizons':metrics(pred,truth,row['mask'],row['times'])})
    sums={k:summary(rows) for k,rows in scored.items()}
    decoder=min(MODES,key=lambda mode:(sums[selected_name+'_'+mode]['40s']['mean_R'],sums[selected_name+'_'+mode]['40s']['max_R'],MODES.index(mode)))
    save(OUT/'validation_report.json',{'summary':sums,'cases':scored})
    save(OUT/'selected_pipeline_lock.json',{'locked_utc':now(),'network':selected,'network_name':selected_name,
                                          'decoder':decoder,'summary':sums[selected_name+'_'+decoder],
                                          'selected_only_on_val8':True,'no_development_or_confirmation_scoring_yet':True})
    print('PERIOD_DECODER_SELECTED',selected_name,decoder,'R',sums[selected_name+'_'+decoder]['40s']['mean_R'],flush=True)
    for role in ['already_seen_test32','fresh_confirmation12']:
        save(OUT/'status.json',{'status':'running','stage':'predict_'+role})
        folder=OUT/role;folder.mkdir()
        oldreport=json.loads((OLD/role/'report.json').read_text())
        registry=[];rows=[]
        for oldrow in oldreport['cases']['corrected_selected']:
            cid=oldrow['id']
            with np.load(CACHE/(cid+'_startup.npz')) as a:
                row={'id':cid,'q':oldrow['q'],'fields':a['fields'],'geom':a['geom'],'mask':a['mask'],'times':a['times']}
            for name,model in models.items():
                results,diag=variants(model,row,p,prior,scales)
                for mode,pred in results.items():
                    label=name+'_'+mode;fp=folder/(label+'__'+cid+'.npz');np.savez_compressed(fp,prediction=pred)
                    registry.append({'id':cid,'label':label,'path':str(fp),'sha256':sha(fp),'closure_diagnostic':diag})
            rows.append(row);print('PERIOD_EVAL_PREDICT',role,cid,flush=True)
        save(folder/'predictions_frozen.json',{'all_predictions_before_future_scoring':True,'registry':registry,
                                             'pipeline_lock_sha256':sha(OUT/'selected_pipeline_lock.json')})
        scored={name+'_'+mode:[] for name in models for mode in MODES};byid={r['id']:r for r in rows}
        for entry in registry:
            row=byid[entry['id']]
            with np.load(CACHE/(entry['id']+'.npz')) as a:truth=a['fields'][4:164]
            pred=np.load(entry['path'])['prediction']
            scored[entry['label']].append({'id':entry['id'],'q':row['q'],'horizons':metrics(pred,truth,row['mask'],row['times'])})
        sums={key:summary(rows) for key,rows in scored.items()}
        save(folder/'report.json',{'role':'already-seen exploratory evaluation','primary_label':selected_name+'_'+decoder,
                                   'summary':sums,'cases':scored,'old_frozen_baselines':oldreport['summary'],
                                   'no_model_or_decoder_selection_from_this_report':True})
        print('PERIOD_EVAL_RESULT',role,[(k,round(v['40s']['mean_R'],3),v['40s']['n_pass']) for k,v in sums.items()],flush=True)
    assert sha(selected['path'])==selected['sha256']
    save(OUT/'status.json',{'status':'complete','selected_label':selected_name+'_'+decoder,'selected_checkpoint_sha256':selected['sha256']})


if __name__=='__main__':
    try:run()
    except Exception as error:
        if OUT.exists():save(OUT/'status.json',{'status':'failed','error':repr(error),'utc':now()})
        raise
