#!/usr/bin/env python3
"""Sixteen registered TRAINING additions; no validation or final-test reuse."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, copy, datetime, json, shutil, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
import corrected_benchmark_restart_v1 as rebuild

OUT=ROOT/'outputs/continuous_training_cfd_v1'
RAW=ROOT/'data/raw/continuous_training_cfd_v1'
CACHE=ROOT/'data/processed/continuous_training_cfd_v1_cases'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def new_cases(p):
    rng=np.random.default_rng(2026100401);rows=[]
    # Deliberate low/high diameter coverage: the earlier four confirmations were all near D=1.
    for size_index,(dlo,dhi) in enumerate([(.9,.96),(1.04,1.1)]):
        for re_index,(rlo,rhi) in enumerate([(80.,100.),(100.,125.),(125.,150.),(150.,200.)]):
            for shape_index,shape in enumerate(['cylinder','square']):
                Re=float(np.round(rng.uniform(rlo+1.,rhi-1.),1));D=float(np.round(rng.uniform(dlo,dhi),3))
                side=-1 if (size_index+re_index+shape_index)%2==0 else 1
                cy=side*float(np.round(rng.uniform(.08,.39),3))
                cid=f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}'
                rows.append({'id':cid,'role':'training_supplement','shape':shape,'Re':Re,'cx':0.,'cy':cy,'D':D})
    seen={r['id'] for r in p['cases']+p['confirmation_cases']}
    fresh4=ROOT/'outputs/period_consistency_confirmation4_v1/preregister.json'
    if fresh4.exists():seen.update(r['id'] for r in json.loads(fresh4.read_text())['cases'])
    assert len(rows)==16 and len({r['id'] for r in rows})==16 and not (seen & {r['id'] for r in rows})
    return rows

def initialize_from_training_neighbor(case,row,p):
    # This is a CFD seed only, followed by >=60s corrected-solver relaxation and the original gate.
    neighbor_row=dict(row,role='fresh_confirmation')
    return ORIGINAL_INITIALIZE(case,neighbor_row,p)

ORIGINAL_INITIALIZE=rebuild.initialize

def prepare():
    assert shutil.disk_usage(ROOT).free>35*2**30
    OUT.mkdir(exist_ok=False);RAW.mkdir(exist_ok=False);CACHE.mkdir(exist_ok=False)
    p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    rows=new_cases(p)
    save(OUT/'preregister.json',{'registered_utc':now(),'seed':2026100401,'cases':rows,'count':16,
        'data_role':'TRAINING ADDITIONS ONLY; never a validation or independent-confirmation score',
        'selection':'2 shapes x4 Re intervals x2 narrow/thick D strata; seeded random inside; balanced y-sign; no model scores used',
        'original_train80_remain':True,'original_val8_remain':True,'already_seen32_12_4_remain_development_only':True,
        'reference':'unchanged repaired one-layer210x120 dt.005 native output.25; reequilibrate >=60s; unchanged Cd/Cl/St two-change stability thresholds',
        'initialization':'metadata nearest ORIGINAL TRAINING case native201 is a seed only; target is independently recomputed',
        'workers':4,'MPI_ranks_each':2,'storage':'isolated dirs; reversible verified compression for generated raw only',
        'script_sha256':sha(__file__),'base_preregister_sha256':sha(ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json'),
        'data_training_plan':'after all16 finish, original80 vs80+16 paired cycle-only continuation; no landmark change in this comparison'})
    save(OUT/'status.json',{'status':'prepared','total':16,'completed':[],'errors':[],'utc':now()})

def execute():
    if not OUT.exists():prepare()
    p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    c=json.loads((OUT/'preregister.json').read_text());assert sha(__file__)==c['script_sha256']
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.initialize=initialize_from_training_neighbor
    completed=[];errors=[];pending=[]
    for row in c['cases']:
        result=CACHE/(row['id']+'.json')
        if result.exists():
            data=json.loads(result.read_text());assert sha(CACHE/(row['id']+'.npz'))==data['cache_sha256'];completed.append(data)
        else:pending.append(row)
    save(OUT/'status.json',{'status':'CFD_running','total':16,'completed':completed,'errors':errors,'utc':now()})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(rebuild.one,row,p):row['id'] for row in pending}
        for future in as_completed(futures):
            try:completed.append(future.result())
            except Exception as error:errors.append({'id':futures[future],'error':repr(error)})
            save(OUT/'status.json',{'status':'CFD_running','total':16,'completed':completed,'errors':errors,'utc':now()})
    if errors:raise RuntimeError(errors)
    assert len(completed)==16
    save(OUT/'training_manifest.json',{'role':'training_only','cases':completed,'registered_cases_sha256':sha(OUT/'preregister.json'),
        'original_trainval_immutable':True,'native_dt':.25,'frames_each':165,'utc':now()})
    save(OUT/'status.json',{'status':'complete','total':16,'completed':completed,'errors':[],'utc':now()})
    print('CONTINUOUS_TRAINING_CFD_DONE',len(completed),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    try:
        if args.prepare_only:prepare()
        else:execute()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',{'error':repr(error),'utc':now()})
        raise
