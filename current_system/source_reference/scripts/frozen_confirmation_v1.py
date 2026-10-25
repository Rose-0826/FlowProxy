#!/usr/bin/env python3
"""Registered 12-case confirmation of frozen F0/K8, with a prospectively fixed scope."""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/frozen_confirmation_v1'
RAW=ROOT/'data/raw/frozen_confirmation_v1'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'cfd/scripts'))
from flowproxy.paths import load_config
from aic_stop_control import stop_job, active_group_members
from round2_complete40_v1 import MODELS, frame_summary, summarize, horizon_selection, state_sha


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def prepare():
    if RAW.exists():raise FileExistsError('Confirmation raw directory already exists')
    OUT.mkdir(exist_ok=False)
    cfg=load_config(ROOT/'configs/default.yaml')
    registered=ROOT/'outputs/round2/preregister.json'
    cases=json.loads(registered.read_text())['cases']
    if len(cases)!=12 or len({r['id'] for r in cases})!=12:raise ValueError('Original list changed')
    primary=[r['id'] for r in cases if float(r['D'])==1. and 80<=float(r['Re'])<=200]
    if len(primary)!=4:raise ValueError('Expected four primary-scope confirmations')
    p={'utc_registered':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'task':'frozen F0/K8 short-horizon independent confirmation, all original 12 cases retained',
       'scope_decided_after_original_test32_group_analysis':True,
       'original_test32_not_an_independent_new_confirmation':True,
       'primary_scope':{'D':1.,'Re':[80,200],'abs_cy_max':.8,'main_horizon_seconds':8},
       'primary_ids':primary,'boundary_ids':[r['id'] for r in cases if r['id'] not in primary],
       'all_case_ids_and_parameters':cases,'original_registration_sha256':sha(registered),
       'models':{name:{'path':str(path.relative_to(ROOT)),'checkpoint_file_sha256':sha(path)} for name,path in MODELS.items()},
       'no_training_or_checkpoint_selection_on_confirmation':True,
       'success_definition':'primary four all R<1 at 8s; all 12 and all boundary results reported regardless',
       'horizons':[2,8,20,40],'support_indices':[0,1,2,3],'forecast_indices_inclusive':[4,163],
       'predictions_saved_before_future_raw_U_p_or_forces_read_for_scoring':True,
       'query_inputs':'geometry/Re and four 64x64 true support fields only',
       'pressure':'raw solver pressure; no future mean subtraction',
       'CFD':{'config':cfg,'end':201,'record_from':160,'record_every':.25,'workers':4,'timeout_per_case_seconds':3600},
       'resource_cost':'CFD generates reference truth; this run is not an end-to-end speedup benchmark',
       'script_sha256':sha(__file__),
       'support_sampling_code_sha256':sha(ROOT/'scripts/cpu_data_prepare.py'),
       'two_dimensional_mathematical_flow_task_including_Re250_boundary_cases':True}
    save(OUT/'preregister.json',p)
    print('prepared all 12; primary scope four',primary,flush=True)


def cfd_run():
    from generate_case import write_case
    from run_long import two_stage
    p=json.loads((OUT/'preregister.json').read_text())
    if RAW.exists():raise FileExistsError('Refusing to regenerate existing cases')
    RAW.mkdir(parents=True)
    cfg=copy.deepcopy(p['CFD']['config'])
    cfg['cfd']['end_time']=160.;cfg['cfd']['write_interval']=160.
    results=[]
    save(OUT/'cfd_status.json',{'status':'running','completed':results,'total':12})
    def one(row):
        dest=(RAW/row['id']).resolve()
        if not dest.is_relative_to(RAW.resolve()) or dest.exists():raise ValueError('Unsafe or existing target')
        case=write_case(cfg,RAW,Re=float(row['Re']),cx=0.,cy=float(row['cy']),D=float(row['D']),shape=row['shape'])
        if case.resolve()!=dest:raise ValueError('Generated ID differs')
        two_stage(case,160.,.25,201.)
        t0=time.perf_counter()
        logfile=case/'log.Allrun'
        print('CFD_START',row['id'],flush=True)
        with logfile.open('x') as log:
            proc=subprocess.Popen(['bash',str(case/'Allrun')],cwd=case,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
            try:
                rc=proc.wait(timeout=3600)
                remaining=active_group_members(proc.pid)
                if remaining:
                    stop=stop_job(proc)
                    raise RuntimeError('Job left active group members')
            except BaseException:
                if proc.poll() is None:stop_job(proc)
                raise
        if rc!=0:raise RuntimeError(f'{row["id"]}: Allrun failed rc={rc}')
        solverlog=case/'log.icoFoam2'
        tail=solverlog.read_text(errors='replace')[-1600:]
        if 'End' not in tail:raise RuntimeError('Solver did not finish normally')
        result={'id':row['id'],'status':'complete','reference_generation_wall_seconds':time.perf_counter()-t0,
                'solver_log_sha256':sha(solverlog),'no_active_group_members':True}
        print('CFD_DONE',row['id'],round(result['reference_generation_wall_seconds'],1),flush=True)
        return result
    errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(one,r):r['id'] for r in p['all_case_ids_and_parameters']}
        for f in as_completed(pending):
            try:results.append(f.result())
            except Exception as e:errors.append({'id':pending[f],'error':repr(e)})
            save(OUT/'cfd_status.json',{'status':'running','completed':results,'errors':errors,'total':12})
    save(OUT/'cfd_status.json',{'status':'failed' if errors else 'complete','completed':results,'errors':errors,'total':12})
    if errors:raise RuntimeError('CFD failures, retained for diagnosis')


def evaluate():
    import torch
    from flowproxy.geometry import geometry_channels
    from flowproxy.models import build_model
    from cpu_data_prepare import frames, SampleCache, read_forces
    from flowproxy.round2.metrics import per_frame_rel
    p=json.loads((OUT/'preregister.json').read_text())
    if (OUT/'report.json').exists():raise FileExistsError('Refusing to overwrite confirmation')
    status=json.loads((OUT/'cfd_status.json').read_text())
    if status['status']!='complete' or len(status['completed'])!=12:raise RuntimeError('CFD incomplete')
    cfg=p['CFD']['config'];device=torch.device('cuda')
    torch.set_num_threads(1)
    support_dir=OUT/'support';support_dir.mkdir(exist_ok=False)
    prediction_dir=OUT/'predictions';prediction_dir.mkdir(exist_ok=False)
    caches={};native={};geoms={};masks={};clocks={}
    for row in p['all_case_ids_and_parameters']:
        cid=row['id'];items=frames(RAW/cid)
        times=np.asarray([t for t,_ in items],dtype=np.float32)
        if len(times)!=165 or abs(float(times[163]-times[3])-40.)>1e-4:raise ValueError('Clock differs')
        geo=geometry_channels(cfg,shape=row['shape'],cx=0.,cy=float(row['cy']),D=float(row['D']),Re=float(row['Re']))
        masks[cid]=geo['mask'];geoms[cid]=geo['geom'];clocks[cid]=times;native[cid]=items
        sampler=SampleCache(cfg,geo['mask']);caches[cid]=sampler
        support=np.stack([sampler.sample(path) for _,path in items[:4]])
        np.savez_compressed(support_dir/(cid+'.npz'),fields=support,geom=geo['geom'],mask=geo['mask'],times=times[:4])
    predicted_hashes={}
    reference=json.loads((ROOT/'outputs/cpu_round1/full40_eval/complete40_val.json').read_text())
    # No future point U/p or target force history has been read at this point.
    for name,path in MODELS.items():
        if sha(path)!=p['models'][name]['checkpoint_file_sha256']:raise ValueError('Frozen weights changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        state_digest=state_sha(saved['model'])
        if state_digest!=reference['models'][name]['state_sha256']:raise ValueError('State checksum differs')
        model=build_model(saved['model_name'],saved['in_channels'],saved['out_channels'],cfg).to(device)
        model.load_state_dict(saved['model']);model.eval()
        predicted_hashes[name]={}
        with torch.inference_mode():
            for row in p['all_case_ids_and_parameters']:
                cid=row['id'];s=np.load(support_dir/(cid+'.npz'))
                geom=torch.from_numpy(s['geom'])[None].to(device)
                fields=torch.from_numpy(s['fields']).to(device)
                window=deque(fields.unbind(0),maxlen=4);pred=[]
                for _ in range(160):
                    hist=torch.stack(tuple(window)).reshape(1,12,64,64)
                    v=model(torch.cat([geom,hist],dim=1))[0,:3];pred.append(v);window.append(v)
                array=torch.stack(pred).cpu().numpy()
                if not np.isfinite(array).all():raise ValueError('Non-finite prediction')
                dest=prediction_dir/f'{name}_{cid}.npz'
                np.savez_compressed(dest,fields=array,times=clocks[cid][4:164])
                predicted_hashes[name][cid]=sha(dest)
        del model,saved
        torch.cuda.empty_cache()
    save(OUT/'predictions_committed_before_scoring.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'files_sha256':predicted_hashes,'future_raw_U_p_or_forces_read_for_scoring':False})
    report={'protocol':p,'models':{},'truth':{}}
    curves={name:[] for name in MODELS}
    for row in p['all_case_ids_and_parameters']:
        cid=row['id'];sampler=caches[cid]
        truth=np.stack([sampler.sample(path) for _,path in native[cid][4:164]])
        np.savez_compressed(OUT/(cid+'_future_truth.npz'),fields=truth,times=clocks[cid][4:164],mask=masks[cid])
        forces=read_forces(RAW/cid)
        report['truth'][cid]={'future_sha256':hashlib.sha256(truth.tobytes()).hexdigest(),
                            'force_rows':len(forces),'force_last_time':float(forces[-1,0]),
                            'native_last_time':float(native[cid][-1][0]),'future_finite':bool(np.isfinite(truth).all())}
        target=torch.from_numpy(truth).to(device);mask=torch.from_numpy(masks[cid]).to(device)
        for name in MODELS:
            prediction=np.load(prediction_dir/f'{name}_{cid}.npz')['fields']
            with torch.inference_mode():err=per_frame_rel(torch.from_numpy(prediction).to(device),target,mask).cpu().numpy()
            horizons={}
            for seconds in p['horizons']:
                chosen,complete=horizon_selection(clocks[cid],seconds)
                horizons[f'{seconds}s']=frame_summary(err[chosen],float(clocks[cid][3]),float(clocks[cid][4+chosen[-1]]),seconds,complete)
            curves[name].append({'id':cid,'horizons':horizons,'curve':err.tolist(),'times':clocks[cid][4:164].tolist()})
    for name,rows in curves.items():
        groups={}
        for group,ids in [('all12',[r['id'] for r in p['all_case_ids_and_parameters']]),
                          ('primary4',p['primary_ids']),('boundary8',p['boundary_ids'])]:
            groups[group]={}
            for seconds in p['horizons']:
                key=f'{seconds}s'
                selected=[{'id':r['id'],**r['horizons'][key]} for r in rows if r['id'] in ids]
                groups[group][key]={'summary':summarize(selected),'cases':selected}
        report['models'][name]={'groups':groups,'curves':rows}
    report['main_scope_confirmed']={name:report['models'][name]['groups']['primary4']['8s']['summary']['n_pass']==4 for name in MODELS}
    report['all_12_reported']=True
    save(OUT/'report.json',report)
    save(OUT/'summary.json',{'main_scope_confirmed':report['main_scope_confirmed'],'all_12_reported':True,
        'models':{name:{g:{h:b['summary'] for h,b in hs.items()} for g,hs in m['groups'].items()} for name,m in report['models'].items()}})
    print('CONFIRMATION_COMPLETE',json.dumps(report['main_scope_confirmed']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.evaluate:evaluate()
    else:
        cfd_run();evaluate()
