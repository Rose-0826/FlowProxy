#!/usr/bin/env python3
"""New eight CFD references after sparse-observer method freeze; no dropping cases."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import corrected_benchmark_restart_v1 as rebuild
from research_v2_outer_support_cfd import initialize as donor_initialize
from research_v2_common import OUT,rows,protected,now
from geometry_attack_v2 import save,sha

DEST=OUT/'sparse_confirmation8_cfd'
RAW=ROOT/'data/raw/research_v4_sparse_confirmation8'
CACHE=ROOT/'data/processed/research_v4_sparse_confirmation8_cases'
WEIGHTS=OUT/'final_research_confirmation/selected_research_base_weights.pt'
OBSERVER=ROOT/'scripts/research_v4_sparse_observer.py'

def initialize(case,row,p):
    donor_initialize(case,row,p)
    a=case/'training_donor_seed.vtu';b=case/'training_donor_seed.xml';assert a.exists() and not b.exists()
    digest=sha(a);a.rename(b);assert sha(b)==digest

def design():
    rng=np.random.default_rng(2026100412);result=[]
    old={q['id'] for q in rows(True)}
    old.update(q['id'] for q in json.loads((OUT/'independent8_cfd/preregister.json').read_text())['cases'])
    for shape in ['cylinder','square']:
        for sign in [-1,1]:
            for thin in [True,False]:
                Re=float(np.round(rng.uniform(90,190),1));cy=sign*float(np.round(rng.uniform(.55,.95),3))
                D=float(np.round(rng.uniform(.78,.86) if thin else rng.uniform(1.14,1.22),3))
                result.append({'id':f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}','shape':shape,'Re':Re,'cx':0.,'cy':cy,'D':D,
                    'role':'NEW sparse-observer confirmation ONLY'})
    assert len(result)==8 and len({q['id'] for q in result})==8
    assert not old.intersection(q['id'] for q in result)
    return result

def main():
    protected();assert shutil.disk_usage(ROOT).free>30*2**30
    DEST.mkdir(exist_ok=False);RAW.mkdir(exist_ok=False);CACHE.mkdir(exist_ok=False)
    assert json.loads((OUT/'sparse_observer_pilot/report.json').read_text())['research_signal_passed']
    cases=design();p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    save(DEST/'observer_lock.json',{'utc':now(),'weights':str(WEIGHTS),'weights_sha256':sha(WEIGHTS),'observer_source_sha256':sha(OBSERVER),
        'TRAIN96_clock_prior_sha256':sha(OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json'),
        'TRAIN96_periods_sha256':sha(OUT/'outer_support_training/periods96.json'),
        'selected_method':'neural H4 template,4 true startup fields,causal phase/frequency EKF,4 Uy probes every2s before t40',
        'fixed_parameters':'unchanged sparse_observer_pilot preregister, no training or tuning on these8',
        'pilot_preregister_sha256':sha(OUT/'sparse_observer_pilot/preregister.json'),
        'comparison_methods':'autonomous research,neural H4 open,neuralEKF2s/4s,ordinary linearKF,nearest96 and aligned4 templates,same1%noise pair',
        'new_task_requires_observations':True,'autonomous40s_result_unchanged':True,'fallback_unchanged':True})
    save(DEST/'preregister.json',{'utc':now(),'seed':2026100412,'cases':cases,'source_sha256':sha(__file__),
        'role':'new independent confirmation of the sparse virtual-observation task after method freeze',
        'design':'continuous Re,combined far location and thin/wide diameter;2 shapes x2 y signs x2 D strata',
        'reference':'same repaired one-layer210x120,dt.005,.25 output,165 frames,same force stability gate',
        'workers':4,'MPI_ranks_each':2,'no_case_dropping':True,
        'inputs':'4 initial full fields plus76 Uy scalars for2s scheme,36 for4s; final2/4s without measurements; forecasts before current-time read',
        'primary':'neural_EKF_2s;4s andnoise robustness secondary,ordinary filter and CFD templates baselines',
        'independent_signal':'>=6/8 Ri40<1 and meanRi40<1,20% lower meanRi40 than both same-probe CFD template baselines; noisy loss<=1 case',
        'limits':'virtual sensors and existing stable initial fields; no measured hardware,not autonomous40s,not end-to-end solver elimination',
        'observer_lock_sha256':sha(DEST/'observer_lock.json')})
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.initialize=initialize
    complete=[];errors=[];begin=time.perf_counter()
    save(DEST/'status.json',{'status':'running','utc':now(),'total':8,'completed':[],'errors':[]})
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(rebuild.one,q,p):q['id'] for q in cases}
        for f in as_completed(jobs):
            if f.cancelled():continue
            try:complete.append(f.result())
            except Exception as e:
                errors.append({'id':jobs[f],'error':repr(e)})
                for j in jobs:j.cancel()
            protected();save(DEST/'status.json',{'status':'running' if not errors else 'failed','utc':now(),'total':8,
                'completed':complete,'errors':errors,'wall_seconds':time.perf_counter()-begin})
    if errors:raise RuntimeError(errors)
    save(DEST/'reference_manifest.json',{'utc':now(),'cases':cases,'results':complete,'cache':str(CACHE),
        'role':'independent sparse-observer confirmation ONLY','preregister_sha256':sha(DEST/'preregister.json')})
    save(DEST/'status.json',{'status':'complete','utc':now(),'total':8,'completed':complete,'errors':[],'wall_seconds':time.perf_counter()-begin})

if __name__=='__main__':
    try:main()
    except Exception as e:
        DEST.mkdir(exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
