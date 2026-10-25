#!/usr/bin/env python3
"""Four fresh, seeded interpolation cases; frozen models before new CFD."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import datetime
import gzip
import json
import re
import shutil
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
import corrected_benchmark_restart_v1 as rebuild
from corrected_benchmark_train_v1 import model_from,predict,protocol
from corrected_benchmark_controls_v1 import dmd_periodic
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

OLD=ROOT/'outputs/corrected_benchmark_restart_v1'
TRAIN=ROOT/'outputs/period_consistency_train_v1'
EVALUATED=ROOT/'outputs/period_consistency_evaluate_v1'
OUT=ROOT/'outputs/period_consistency_confirmation4_v1'
RAW=ROOT/'data/raw/period_consistency_confirmation4_v1'
CACHE=ROOT/'data/processed/period_consistency_confirmation4_v1_cases'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cases(p):
    rng=np.random.default_rng(20261003)
    rows=[]
    existing={r['id'] for r in p['cases']+p['confirmation_cases']}
    for shape in ['cylinder','square']:
        for lo,hi in [(80.,140.),(140.,200.)]:
            Re=float(np.round(rng.uniform(lo,hi),1))
            cy=float(np.round(rng.uniform(-.4,.4),3))
            D=float(np.round(rng.uniform(.9,1.1),3))
            cid=f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}'
            assert cid not in existing
            rows.append({'id':cid,'shape':shape,'Re':Re,'cy':cy,'cx':0.,'D':D,'role':'fresh_confirmation'})
    return rows


def summarize_rows(rows):
    return {h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']}


def dense_future_wall(case,t0,t1):
    path=case/'log.icoFoam_dense.gz'
    if not path.exists():return None
    with gzip.open(path,'rt') as f:text=f.read()
    stamps={}
    for block in re.split(r'\nTime = ',text)[1:]:
        stamp=float(block.splitlines()[0]);m=re.search(r'ClockTime = ([0-9.eE+-]+)',block)
        if m:stamps[stamp]=float(m.group(1))
    if not stamps:return None
    a=min(stamps,key=lambda t:abs(t-t0));b=min(stamps,key=lambda t:abs(t-t1))
    assert abs(a-t0)<1e-6 and abs(b-t1)<1e-6
    return {'future_start':a,'future_end':b,'wall_seconds':stamps[b]-stamps[a],
            'CFD_includes_dense_output':True,'concurrent_cases':4,'MPI_ranks_per_case':2,
            'source':'solver ClockTime at matching true startup-end and future-end timestamps'}


def predict_prefix(model,row,N):
    geom=torch.as_tensor(row['geom'][None],device='cuda')
    window=[torch.as_tensor(f[None],device='cuda') for f in row['fields'][:4]];out=[]
    model.eval()
    with torch.inference_mode():
        for k in range(N):
            hist=torch.stack(window[-4:],1).reshape(1,12,64,64)
            pred=model(torch.cat([geom,hist],1))[:,:3];out.append(pred[0].cpu().numpy());window.append(pred)
    return np.stack(out)


def dmd_from_base(row,base,scales):
    observed=np.concatenate([row['fields'][:4],base[:48]])
    pred,diag=dmd_periodic(observed,row['times'][:52],row['times'][4:164],row['mask'],9,scales)
    pred[:48]=base[:48]
    return pred,diag


def run():
    OUT.mkdir(exist_ok=False);RAW.mkdir(exist_ok=False);CACHE.mkdir(exist_ok=False)
    p=protocol();oldlock=json.loads((OLD/'selected_model_lock.json').read_text())
    trained=json.loads((TRAIN/'completed_report.json').read_text());assert trained['gate_passed']
    chosen=trained['selected_cycle_for_followup_if_gate_passed']
    models_meta={'baseline':oldlock['chosen'],**{f'cycle_seed{r["seed"]}':r for r in trained['all_candidates'] if r['arm']=='cycle'}}
    assert chosen['seed']==11
    for item in models_meta.values():assert sha(item['path'])==item['sha256']
    rows=cases(p)
    assert shutil.disk_usage(ROOT).free>35*2**30
    config=json.loads((Path(oldlock['chosen']['path']).parent/'configuration.json').read_text())
    scales=config['normalization_train80']
    prereg={'registered_utc':now(),'seed':20261003,'cases':rows,'count':4,
        'case_selection':'one case per shape x Re stratum; uniform within strata; cy full[-.4,.4],D full[.9,1.1]; no model scores used',
        'domain':'NEW INTERPOLATION ONLY, inside original training parameter bounds; does NOT replace already-seen32 extreme extrapolation or12 broader cases',
        'networks':models_meta,'primary':'cycle_seed11_DMD9',
        'network_selection':'seed11/update1000 chosen on original val8 before exploratory32+12',
        'decoder_choice':'DMD9 chosen for this NEW stage using already-seen32+12 development results, lower aggregate/worst error than replay/H4; older val-chosen replay report preserved',
        'primary_decoder':'fixed rank9 DMD projected to unit-circle; four true startup frames +48 self-generated frames, no frequency prior or future observation',
        'baselines':['same DMD9 with baseline network','both networks direct160','second trained seed+DMD9','copy last','linear last2'],
        'parameters_frozen_before_any_new_CFD':True,'future_fields_never_read_by_prediction_loader':True,
        'success_gate':'new primary short8s4/4 and40s>=3/4 under unchanged team5%/5%/8%; all failures still reported; no model changes from these scores',
        'model_effect':'compare period-consistent vs baseline network with SAME DMD9; not a claim DMD invented by team',
        'reference':'reuse locked single-layer210x120 dt.005 sample.25 and same stability gates, new isolated raw/cache dirs',
        'workers':4,'MPI_ranks_each':2,'quality_anchors':'unchanged2-case reference checks, not new grid-independence proof',
        'timing':'given stable four-frame initial state; efficient48step neural prefix+DMD; read-disk time excluded; CFD matching40s interval includes output,4concurrent2MPI cases',
        'old_selected_model_lock_sha256':sha(OLD/'selected_model_lock.json'),
        'training_report_sha256':sha(TRAIN/'completed_report.json'),
        'seen32_report_sha256':sha(EVALUATED/'already_seen_test32/report.json'),
        'seen12_report_sha256':sha(EVALUATED/'fresh_confirmation12/report.json'),
        'script_sha256':sha(__file__)}
    save(OUT/'preregister.json',prereg)
    completed=[];errors=[]
    save(OUT/'status.json',{'status':'CFD_running','completed':completed,'errors':errors,'total':4,'started_utc':now()})
    # Only directory globals used by one() are redirected; original dataset/status/model files stay untouched.
    rebuild.RAW=RAW;rebuild.CACHE=CACHE
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(rebuild.one,row,p):row['id'] for row in rows}
        for future in as_completed(futures):
            try:completed.append(future.result())
            except Exception as error:errors.append({'id':futures[future],'error':repr(error)})
            save(OUT/'status.json',{'status':'CFD_running','completed':completed,'errors':errors,'total':4})
    if errors:raise RuntimeError(errors)
    torch.set_num_threads(1)
    models={name:model_from(item['path'],p['config'])[0] for name,item in models_meta.items()}
    pred_dir=OUT/'predictions';pred_dir.mkdir();registry=[];startup=[]
    for q in rows:
        with np.load(CACHE/(q['id']+'_startup.npz')) as a:
            row={'id':q['id'],'q':q,'fields':a['fields'],'geom':a['geom'],'mask':a['mask'],'times':a['times']}
        results={}
        for name,model in models.items():
            base=predict(model,row);closed,diag=dmd_from_base(row,base,scales)
            results[name+'_direct']=(base,{})
            results[name+'_DMD9']=(closed,diag)
        support=row['fields'];elapsed=row['times'][4:164]-row['times'][3]
        slope=(support[3]-support[2])/(row['times'][3]-row['times'][2])
        results['copy_last']=(np.repeat(support[3:4],160,axis=0),{})
        results['linear_last2']=((support[3]+elapsed[:,None,None,None]*slope).astype(np.float32),{})
        for label,(pred,diag) in results.items():
            fp=pred_dir/(label+'__'+row['id']+'.npz');np.savez_compressed(fp,prediction=pred)
            registry.append({'id':row['id'],'label':label,'path':str(fp),'sha256':sha(fp),'diagnostic':diag})
        startup.append(row)
    save(OUT/'predictions_frozen.json',{'utc':now(),'all_predictions_saved_before_future_scoring':True,'registry':registry,
                                      'configuration_sha256':sha(OUT/'preregister.json')})
    byid={r['id']:r for r in startup};scored={e['label']:[] for e in registry}
    for entry in registry:
        row=byid[entry['id']]
        with np.load(CACHE/(entry['id']+'.npz')) as a:truth=a['fields'][4:164]
        pred=np.load(entry['path'])['prediction']
        scored[entry['label']].append({'id':row['id'],'q':row['q'],'horizons':metrics(pred,truth,row['mask'],row['times'])})
    summaries={key:summarize_rows(items) for key,items in scored.items()}
    timings=[]
    row=startup[0];model=models['cycle_seed11']
    for k in range(8):
        torch.cuda.synchronize();started=time.perf_counter()
        prefix=predict_prefix(model,row,48);base=np.zeros((160,3,64,64),np.float32);base[:48]=prefix
        prediction,_=dmd_from_base(row,base,scales);torch.cuda.synchronize()
        if k>=2:timings.append(time.perf_counter()-started)
    reference=np.load(pred_dir/('cycle_seed11_DMD9__'+row['id']+'.npz'))['prediction']
    assert np.allclose(prediction,reference,atol=1e-6,rtol=1e-5)
    cost={'neural48step_plus_DMD_cpu_wall_seconds':timings,'median_seconds':float(np.median(timings)),
          'given_startup_fields_in_memory':True,'includes_H2D_D2H_and_CPU_DMD':True,'disk_read_excluded':True,
          'CFD':[{'id':r['id'],**(dense_future_wall(RAW/'cases'/r['id'],float(r['times'][3]),float(r['times'][163])) or {})} for r in startup]}
    gate=summaries['cycle_seed11_DMD9']['8s']['n_pass']==4 and summaries['cycle_seed11_DMD9']['40s']['n_pass']>=3
    save(OUT/'report.json',{'role':'fresh interpolation confirmation4; fixed prior to new CFD','summary':summaries,'cases':scored,
                           'primary':'cycle_seed11_DMD9','success_gate_passed':gate,'timing':cost,
                           'does_not_establish_extreme_geometry_or_Re250_generalization':True})
    for item in models_meta.values():assert sha(item['path'])==item['sha256']
    save(OUT/'status.json',{'status':'complete','completed':completed,'errors':[],'total':4,'success_gate_passed':gate,
                          'primary':summaries['cycle_seed11_DMD9']})
    print('PERIOD_FRESH4_COMPLETE',gate,json.dumps(summaries['cycle_seed11_DMD9']['40s']),flush=True)


if __name__=='__main__':
    try:run()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',{'status':'failed','error':repr(error),'utc':now()})
        raise
