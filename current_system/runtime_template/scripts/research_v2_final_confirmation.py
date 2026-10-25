#!/usr/bin/env python3
"""Finite research selection on seen56, then one sealed8 confirmation; fallback never replaced."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,score,summaries,scales,protected,now,baseline
from research_v2_reflection_train import ReflectionOperator
from research_v2_state_probe import ProjectedState
from research_v2_spectrum_guard_probe import coherence
from corrected_benchmark_train_v1 import model_from,protocol
from corrected_benchmark_controls_v1 import dmd_periodic
from cycle_landmarks_train_v1 import FORMAL
from geometry_attack_v2 import save,sha
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

DEST=OUT/'final_research_confirmation';TRAIN_REPORT=OUT/'outer_support_training/completed_report.json'
FRESH=OUT/'independent8_cfd/SEALED_reference_manifest.json'

def infer(model,row,adaptive):
    torch.cuda.synchronize();begin=time.perf_counter()
    g=torch.as_tensor(row['geom'][None],device='cuda')
    window=[torch.as_tensor(a[None],device='cuda') for a in row['fields'][:4]];predicted=[]
    def generate(n):
        with torch.inference_mode():
            while len(predicted)<n:
                x=torch.cat([g,torch.stack(window[-4:],1).reshape(1,12,64,64)],1)
                y=model(x)[:,:3];window.append(y);predicted.append(y[0].cpu().numpy())
    n=24 if adaptive else 48;generate(n)
    def decode(n):
        observed=np.concatenate([row['fields'],np.stack(predicted[:n])])
        pred,diag=dmd_periodic(observed,row['times'][:n+4],row['times'][4:164],row['mask'],9,scales())
        pred[:n]=np.stack(predicted[:n]);return pred,diag
    pred,diag=decode(n);guard=None
    if adaptive:
        accept,guard=coherence(diag)
        if not accept:
            n=48;generate(n);pred,diag=decode(n)
    torch.cuda.synchronize()
    return pred,{'prefix_steps':n,'guard':guard,'DMD':diag,'inference_seconds_including_transfer_and_CPU_decoder':time.perf_counter()-begin}

def load(candidate):
    base,_=model_from(candidate['path'],protocol()['config']);base.eval()
    return ProjectedState(ReflectionOperator(base),True).eval()

def select(report):
    candidates=[{'name':'fixed_fallback_weights_plus_physical_guard','path':str(FORMAL),'arm':'inference_only'}]
    # Honour the predeclared paired-data signal instead of chasing one lucky checkpoint.
    if report['research_signal_passed']:
        candidates += [{'name':f"outer96_seed{a['seed']}",'path':a['path'],'arm':'expanded_training'}
            for a in report['candidates'] if a['arm']=='outer96']
    original=json.loads((OUT/'operator_probe/report.json').read_text())['fallback']['summary']['DMD9']
    original_balanced8=.5*(original['development_inside24']['8s']['mean_R']+original['seen_extreme32']['8s']['mean_R'])
    entries=[]
    for candidate in candidates:
        model=load(candidate);forecasts=[];registry=[]
        for q in rows():
            row=startup(q);pred,diag=infer(model,row,True)
            forecasts.append((q,row,pred));registry.append({'id':q['id'],'diagnostic':diag,
                'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest()})
        save(DEST/(candidate['name']+'_development_forecasts_frozen.json'),{'utc':now(),'all_before_future_scoring':True,'registry':registry})
        scored=[score(q,pred,row) for q,row,pred in forecasts];summary=summaries(scored)
        score40=.5*(summary['development_inside24']['40s']['mean_R']+summary['seen_extreme32']['40s']['mean_R'])
        short=.5*(summary['development_inside24']['8s']['mean_R']+summary['seen_extreme32']['8s']['mean_R'])
        eligible=short<=1.05*original_balanced8 and summary['development_all56']['8s']['n_pass']>=original['development_all56']['8s']['n_pass']-1
        entry={**candidate,'summary':summary,'cases':scored,'balanced_R40':score40,'eligible_short':eligible,'weights_sha256':sha(candidate['path'])}
        entries.append(entry);save(DEST/(candidate['name']+'_development.json'),entry)
        del model,forecasts;torch.cuda.empty_cache()
    eligible=[a for a in entries if a['eligible_short']]
    if not eligible:raise RuntimeError('No research candidate preserves short-task guard; no independent scoring')
    chosen=min(eligible,key=lambda a:(a['balanced_R40'],a['summary']['development_all56']['40s']['max_R'],a['name']))
    target=DEST/'selected_research_base_weights.pt';shutil.copyfile(chosen['path'],target)
    assert sha(target)==chosen['weights_sha256']
    lock={'utc':now(),'selected':{k:v for k,v in chosen.items() if k not in ['cases','summary']},
        'locked_weight_copy':str(target),'weight_sha256':sha(target),'pipeline':'ProjectedState(ReflectionOperator(base),True);4 observed;prefix24 if5% harmonic coherence otherwise48;unit-circleDMD9',
        'selection_data':'seen56 DEVELOPMENT ONLY','independent8_not_scored_yet':True,
        'outer_training_signal':report['research_signal_passed'],'fallback_unchanged':True,
        'source_sha256':sha(__file__)}
    save(DEST/'selected_research_lock.json',lock);return lock

def confirm(lock):
    manifest=json.loads(FRESH.read_text());cache=Path(manifest['cache']);cases=manifest['cases'];assert len(cases)==8
    chosen=load({'path':lock['locked_weight_copy']});base,_=model_from(FORMAL,protocol()['config']);base.eval()
    forecast=[];registry=[]
    for q in cases:
        with np.load(cache/(q['id']+'_startup.npz')) as a:
            row={k:a[k] for k in ['fields','geom','mask','times']}
        assert row['fields'].shape==(4,3,64,64)
        row['q']={k:q[k] for k in ['shape','Re','cx','cy','D']};row['id']=q['id']
        for method,model,adaptive in [('fallback',base,False),('research_selected',chosen,True)]:
            pred,diag=infer(model,row,adaptive)
            file=DEST/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'diagnostic':diag,
                'startup_sha256':sha(cache/(q['id']+'_startup.npz'))})
            forecast.append((q,row,method,pred))
    save(DEST/'independent8_forecasts_frozen.json',{'utc':now(),'all_before_independent_future_field_reads':True,
        'selected_lock_sha256':sha(DEST/'selected_research_lock.json'),'registry':registry})
    scored={'fallback':[],'research_selected':[]}
    for q,row,method,pred in forecast:
        meta=json.loads((cache/(q['id']+'.json')).read_text());assert meta['CFD_stable'] and sha(cache/(q['id']+'.npz'))==meta['cache_sha256']
        with np.load(cache/(q['id']+'.npz')) as a:target=a['fields'][4:164]
        scored[method].append({'id':q['id'],'horizons':metrics(pred,target,row['mask'],row['times'])})
    summary={m:{h:summarize([a['horizons'][h] for a in part]) for h in ['2s','8s','20s','40s']} for m,part in scored.items()}
    protected();assert sha(lock['locked_weight_copy'])==lock['weight_sha256']
    save(DEST/'independent8_report.json',{'utc':now(),'role':'one independent confirmation after research model/pipeline freeze',
        'summary':summary,'cases':scored,'selected_lock':lock,'no_posthoc_model_reselection':True,
        'fallback_unchanged':True,'limits':'8 cases of combined far-position/diameter conditions; all-case accuracy claimed only if everyRi<1'})
    save(DEST/'status.json',{'utc':now(),'status':'complete','fallback_unchanged':True,'independent8_scored_once':True})

def run():
    torch.set_num_threads(1);protected();DEST.mkdir(exist_ok=False);(DEST/'predictions').mkdir();begin=time.perf_counter()
    save(DEST/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),
        'finite_pipeline':'wait paired training ->select only development56 ->lock weights/pipeline ->wait sealed8 ->predict startup4 ->score once',
        'candidate_policy':'fixed physical+spectral guard always considered; expanded-data candidates only if predeclared BOTH-seed signal passes',
        'no_automatic_competition_fallback_replacement':True,'dependency_timeout_seconds':14400})
    while not TRAIN_REPORT.exists():
        if time.perf_counter()-begin>14400:raise RuntimeError('4h training dependency limit')
        if (OUT/'outer_support_training_failure.json').exists():raise RuntimeError('Paired training dependency failed')
        save(DEST/'status.json',{'utc':now(),'status':'waiting_for_paired_training','elapsed_seconds':time.perf_counter()-begin})
        time.sleep(30)
    lock=select(json.loads(TRAIN_REPORT.read_text()))
    while not FRESH.exists():
        if time.perf_counter()-begin>14400:raise RuntimeError('4h independent CFD dependency limit')
        if (OUT/'independent8_cfd/failure.json').exists():raise RuntimeError('Independent CFD failed; no dropping cases')
        save(DEST/'status.json',{'utc':now(),'status':'research_locked_waiting_for_sealed8','elapsed_seconds':time.perf_counter()-begin})
        time.sleep(30)
    confirm(lock)

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(DEST/'failure.json',{'utc':now(),'error':repr(e)})
        save(DEST/'status.json',{'utc':now(),'status':'failed','error':repr(e)});raise
