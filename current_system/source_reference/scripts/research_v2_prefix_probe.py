#!/usr/bin/env python3
"""Finite decoder-window diagnosis on already-frozen self-generated trajectories."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,score,summaries,scales,protected,now
from corrected_benchmark_controls_v1 import dmd_periodic
from geometry_attack_v2 import save,sha

def run():
    protected();out=OUT/'prefix_probe';out.mkdir(exist_ok=False);(out/'predictions').mkdir();begin=time.perf_counter()
    lengths=[24,32,48,64]
    save(out/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),'observed':4,'source_operator':'reflection+velocityBC frozen formal weights',
        'source_predictions':str(OUT/'state_probe'),'windows':lengths,'DMD_rank':9,
        'purpose':'diagnose identifiability versus accumulated neural bias, finite exploratory window choices, all56 ALREADY-SEEN development'})
    registry=[]
    for q in rows():
        row=startup(q)
        with np.load(OUT/'state_probe/predictions'/('reflection_velocity_BC__'+q['id']+'.npz')) as a:direct=a['direct']
        for n in lengths:
            pred,diag=dmd_periodic(np.concatenate([row['fields'],direct[:n]]),row['times'][:n+4],row['times'][4:164],row['mask'],9,scales())
            pred[:n]=direct[:n]
            f=out/'predictions'/(f'prefix{n}__'+q['id']+'.npz');np.savez_compressed(f,prediction=pred)
            registry.append({'id':q['id'],'prefix_steps':n,'sha256':sha(f),'decoder':diag})
    save(out/'forecasts_frozen.json',{'utc':now(),'all_before_future_scoring':True,'registry':registry})
    report={}
    for n in lengths:
        part=[]
        for q in rows():
            with np.load(out/'predictions'/(f'prefix{n}__'+q['id']+'.npz')) as a:part.append(score(q,a['prediction'],startup(q)))
        report[str(n)]={'summary':summaries(part),'cases':part}
    protected();save(out/'report.json',report);save(out/'status.json',{'status':'complete','utc':now(),'wall_seconds':time.perf_counter()-begin})

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(OUT/'prefix_probe/failure.json',{'utc':now(),'error':repr(e)});raise
