#!/usr/bin/env python3
"""Generate eight sealed independent references; model scores wait for research freeze."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import corrected_benchmark_restart_v1 as rebuild
from research_v2_outer_support_cfd import initialize
from research_v2_common import OUT,rows,protected,now
from geometry_attack_v2 import save,sha

DEST=OUT/'independent8_cfd';RAW=ROOT/'data/raw/research_v2_independent8'
CACHE=ROOT/'data/processed/research_v2_independent8_SEALED_cases'

def design():
    rng=np.random.default_rng(2026100403);result=[]
    for shape in ['cylinder','square']:
        for sign in [-1,1]:
            for thin in [True,False]:
                Re=float(np.round(rng.uniform(90,190),1))
                cy=sign*float(np.round(rng.uniform(.55,.95),3))
                D=float(np.round(rng.uniform(.78,.86) if thin else rng.uniform(1.14,1.22),3))
                result.append({'id':f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}',
                    'shape':shape,'Re':Re,'cx':0.,'cy':cy,'D':D,'role':'independent_research_confirmation_ONLY'})
    assert len(result)==8 and len({q['id'] for q in result})==8
    assert not {q['id'] for q in result}.intersection(q['id'] for q in rows(True))
    return result

def main():
    protected();assert shutil.disk_usage(ROOT).free>30*2**30
    DEST.mkdir(exist_ok=False);RAW.mkdir(exist_ok=False);CACHE.mkdir(exist_ok=False)
    cases=design();p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    save(DEST/'preregister.json',{'utc':now(),'seed':2026100403,'cases':cases,
        'role':'INDEPENDENT confirmation after final research method/checkpoint freeze, never training or selection',
        'geometry':'both far position and thin/thick D together; continuous Re; two shapes x2 y-signs x2 D strata',
        'relation':'outside original80 support; combinations not claimed covered by convex hull of16 axis anchors',
        'reference':'unchanged repaired one-layer210x120 dt.005; same stability gate and .25 output',
        'workers':2,'MPI_ranks_each':2,'source_sha256':sha(__file__),
        'CFD_can_run_before_model_freeze':'references sealed; final checkpoint must be locked before startup-only predictions and future scoring',
        'no_case_dropping':True,'fallback_immutable':True})
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.initialize=initialize
    completed=[];errors=[];begin=time.perf_counter()
    save(DEST/'status.json',{'status':'running','total':8,'completed':[],'errors':[],'utc':now()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(rebuild.one,q,p):q['id'] for q in cases}
        for f in as_completed(tasks):
            if f.cancelled():continue
            try:completed.append(f.result())
            except Exception as e:
                errors.append({'id':tasks[f],'error':repr(e)})
                for task in tasks:task.cancel()
            protected();save(DEST/'status.json',{'status':'running' if not errors else 'failed',
                'total':8,'completed':completed,'errors':errors,'utc':now(),'wall_seconds':time.perf_counter()-begin})
    if errors:raise RuntimeError(errors)
    save(DEST/'SEALED_reference_manifest.json',{'utc':now(),'cases':cases,'results':completed,
        'cache':str(CACHE),'role':'INDEPENDENT confirmation ONLY','registered_sha256':sha(DEST/'preregister.json')})
    save(DEST/'status.json',{'status':'sealed_complete','utc':now(),'total':8,'completed':completed,'errors':[]})

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
