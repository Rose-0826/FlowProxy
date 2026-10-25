#!/usr/bin/env python3
"""Fresh eight-case reference generation, blinded until model nomination."""
import argparse
import copy
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/geometry_attack_confirmation_v1'
RAW=ROOT/'data/raw/geometry_attack_confirmation_v1'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'cfd/scripts'))
from flowproxy.paths import load_config
from aic_stop_control import stop_job,active_group_members

def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)

def prepare():
    if OUT.exists() or RAW.exists():raise FileExistsError('New reference already exists')
    OUT.mkdir()
    cfg=load_config(ROOT/'configs/default.yaml');rows=[]
    for shape in ['cylinder','square']:
        for D in [.8,1.2]:
            for Re,cy in [(110,-.2),(170,.2)]:
                rows.append({'id':f'{shape}_Re{Re}_cy{cy:g}_D{D:g}','shape':shape,'Re':Re,'cy':cy,'cx':0.,'D':D})
    save(OUT/'preregister.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'cases':rows,'primary_task':'geometry-general autonomous 8s forecasting, with four true startup fields',
         'full_cases_all_reported':True,'new_Re_in_original_range_and_diameter_extrapolation':True,
         'no_case_was_in_original120_or_previous_confirmation12':True,
         'primary_model_nomination':'must be written with checkpoint SHA256 before any new support or future field or forces are read',
         'bank':'formal F0/K8 plus val8-selected control/scale/warp64/warp_native, seeds11/22; no checkpoint selection on new8',
         'primary_success':'report every case; target at least6/8 R<1 at8s and meanR<1; '
             'claim own-method advantage only if paired comparisons with stronger frozen/continued FNO support it',
         'other_candidates_after_nomination':'secondary exploratory comparisons, cannot retrospectively replace failed nominated primary',
         'support_indices':[0,1,2,3],'future_indices_inclusive':[4,163],
         'horizons':[2,8,20,40],'pressure':'original solver reference, no future mean subtraction',
         'state_budget':'4096 sites/frame; warped native has different known query locations; original uniform baselines use uniform sites',
         'CFD':{'config':cfg,'to':201,'record_from':160,'record_every':.25,'workers':4,'timeout':3600},
         'resource_cost':'reference generation only, not a speedup measurement',
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('FRESH8_REGISTERED',[x['id'] for x in rows],flush=True)

def run():
    from generate_case import write_case
    from run_long import two_stage
    p=json.loads((OUT/'preregister.json').read_text())
    if RAW.exists():raise FileExistsError('Refusing existing raw directory')
    RAW.mkdir(parents=True);cfg=copy.deepcopy(p['CFD']['config'])
    cfg['cfd']['end_time']=160.;cfg['cfd']['write_interval']=160.
    done=[];errors=[]
    def one(row):
        expected=(RAW/row['id']).resolve()
        if not expected.is_relative_to(RAW.resolve()) or expected.exists():raise ValueError('Unsafe/existing target')
        case=write_case(cfg,RAW,Re=row['Re'],cx=0.,cy=row['cy'],D=row['D'],shape=row['shape'])
        if case.resolve()!=expected:raise ValueError('ID differs')
        two_stage(case,160.,.25,201.);t=time.perf_counter()
        print('CFD_START',row['id'],flush=True)
        with (case/'log.Allrun').open('x') as f:
            proc=subprocess.Popen(['bash',str(case/'Allrun')],cwd=case,stdout=f,stderr=subprocess.STDOUT,
                                  start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
            try:
                rc=proc.wait(timeout=3600)
                if active_group_members(proc.pid):
                    stop_job(proc);raise RuntimeError('Active children after completion')
            except BaseException:
                if proc.poll() is None:stop_job(proc)
                raise
        if rc!=0:raise RuntimeError('CFD failed '+row['id'])
        log=case/'log.icoFoam2'
        if 'End' not in log.read_text(errors='replace')[-1600:]:raise RuntimeError('CFD missing normal End')
        d={'id':row['id'],'wall_reference_seconds':time.perf_counter()-t,'no_active_group_members':True,
           'solver_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
        print('CFD_DONE',row['id'],round(d['wall_reference_seconds'],1),flush=True)
        return d
    save(OUT/'cfd_status.json',{'status':'running','completed':done,'errors':errors,'total':8})
    with ThreadPoolExecutor(max_workers=4) as ex:
        fs={ex.submit(one,r):r['id'] for r in p['cases']}
        for f in as_completed(fs):
            try:done.append(f.result())
            except Exception as e:errors.append({'id':fs[f],'error':repr(e)})
            save(OUT/'cfd_status.json',{'status':'running','completed':done,'errors':errors,'total':8})
    save(OUT/'cfd_status.json',{'status':'failed' if errors else 'complete','completed':done,'errors':errors,'total':8,
                             'new_target_fields_not_read_for_AI_or_scoring':True})
    if errors:raise RuntimeError('CFD errors; retained')
    print('CFD_COMPLETE_BLINDED',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');args=a.parse_args()
    prepare() if args.prepare else run()
