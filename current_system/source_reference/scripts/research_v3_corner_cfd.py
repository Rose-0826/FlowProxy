#!/usr/bin/env python3
"""Eight TRAINING corners, reflected signs already covered by exact C2 operator."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, shutil, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import corrected_benchmark_restart_v1 as rebuild
from research_v2_outer_support_cfd import initialize as original_initialize
from research_v2_common import OUT, rows, protected, now
from geometry_attack_v2 import save, sha

DEST=OUT/'next_round/corner_training_cfd'
RAW=ROOT/'data/raw/research_v3_corner_support'
CACHE=ROOT/'data/processed/research_v3_corner_support_cases'

def initialize(case,row,p):
    original_initialize(case,row,p)
    aux=case/'training_donor_seed.vtu';target=case/'training_donor_seed.xml'
    assert aux.exists() and not target.exists()
    digest=sha(aux);aux.rename(target);assert sha(target)==digest

def design():
    q=[]
    for shape in ['cylinder','square']:
        for Re in [80.,200.]:
            for D in [.75,1.25]:
                q.append({'id':f'{shape}_Re{Re:g}_cy1_D{D:g}',
                    'shape':shape,'Re':Re,'cx':0.,'cy':1.,'D':D,'role':'corner_support_TRAINING_ONLY'})
    assert len(q)==8 and not {x['id'] for x in q}.intersection(x['id'] for x in rows(True))
    return q

def main():
    protected();assert shutil.disk_usage(ROOT).free>30*2**30
    DEST.mkdir(parents=True,exist_ok=False);RAW.mkdir(exist_ok=False);CACHE.mkdir(exist_ok=False)
    p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text());cases=design()
    save(DEST/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'cases':cases,'data_role':'TRAINING ONLY',
        'design':'2 shapes x2 endpoint Re x2 diameter extremes at cy+1; joint y/D corners outside previous axial16',
        'negative_sign':'exact C2 projected model already evaluates reflected geometry/history every step; do not count reflection as additional independent CFD cases',
        'reference':'same repaired one-layer210x120,dt.005,165 frames at.25,same force stability gate',
        'initialization':'original TRAIN80 donor only; target CFD relaxation>=60s; auxiliary XML kept outside VTK scanner',
        'workers':3,'MPI_ranks_each':2,'no_case_dropping':True,'fallback_immutable':True,
        'later_comparison':'96 training versus96+8 corners using matched budgets; no new independent claim from these8'})
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.initialize=initialize
    done=[];errors=[];begin=time.perf_counter()
    save(DEST/'status.json',{'status':'running','utc':now(),'total':8,'completed':[],'errors':[]})
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(rebuild.one,q,p):q['id'] for q in cases}
        for f in as_completed(jobs):
            if f.cancelled():continue
            try:done.append(f.result())
            except Exception as e:
                errors.append({'id':jobs[f],'error':repr(e)})
                for job in jobs:job.cancel()
            protected();save(DEST/'status.json',{'status':'running' if not errors else 'failed','utc':now(),
                'total':8,'completed':done,'errors':errors,'wall_seconds':time.perf_counter()-begin})
    if errors:raise RuntimeError(errors)
    save(DEST/'training_manifest.json',{'utc':now(),'cases':cases,'results':done,'cache':str(CACHE),'data_role':'TRAINING ONLY','registered_sha256':sha(DEST/'preregister.json')})
    save(DEST/'status.json',{'status':'complete','utc':now(),'total':8,'completed':done,'errors':[],'wall_seconds':time.perf_counter()-begin})

if __name__=='__main__':
    try:main()
    except Exception as e:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
