#!/usr/bin/env python3
"""Expand TRAINING geometry support; keep prior competition data/model immutable."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy, json, shutil, sys, time
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import corrected_benchmark_restart_v1 as rebuild
from corrected_benchmark_storage_v1 import reconstruct
from cpu_data_prepare import frames,read_point_fields
from flowproxy.geometry import geometry_channels
from warmstart_iface import read_vectors,replace_internal
from geometry_attack_v2 import save,sha
from research_v2_common import OUT,protected,now

DEST=OUT/'outer_support_training_cfd'
RAW=ROOT/'data/raw/research_v2_outer_support'
CACHE=ROOT/'data/processed/research_v2_outer_support_cases'
SOURCE=ROOT/'data/raw/corrected_benchmark_restart_v1/cases'

def design():
    rows=[]
    for shape in ['cylinder','square']:
        for Re in [80.,200.]:
            for cy,D in [(-1.,1.),(1.,1.),(0.,.75),(0.,1.25)]:
                rows.append({'id':f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}',
                    'shape':shape,'Re':Re,'cx':0.,'cy':cy,'D':D,
                    'role':'outer_support_training_only'})
    assert len(rows)==16 and len({q['id'] for q in rows})==16
    return rows

def initialize(case,row,p):
    choices=[q for q in p['cases'] if q['role']=='train' and q['shape']==row['shape']]
    donor=min(choices,key=lambda q:(((q['Re']-row['Re'])/120)**2+
        ((q['cy']-row['cy'])/.8)**2+((q['D']-row['D'])/.2)**2,q['id']))
    source=SOURCE/donor['id']
    meta=json.loads((source/'dataset_source.json').read_text())
    items=frames(source);stamp,vtk=min(items,key=lambda a:abs(a[0]-meta['future_end']))
    assert abs(stamp-meta['future_end'])<1e-7
    seed=case/'training_donor_seed.vtu'
    payload=vtk.read_bytes() if vtk.exists() else reconstruct(Path(str(vtk)+'.dedup.json'))
    seed.write_bytes(payload)
    rebuild.run.command(case,'postProcess -func writeCellCentres -time 0','log.writeCellCentres',240)
    centers=read_vectors(case/'0/C');points,values=read_point_fields(seed)
    # Transport the donor body location/diameter; this is only a CFD seed.
    query=centers[:,:2].copy()
    for axis,key in [(0,'cx'),(1,'cy')]:
        query[:,axis]=donor[key]+(query[:,axis]-row[key])*donor['D']/row['D']
    mapped=LinearNDInterpolator(points,values,fill_value=np.nan)(query)
    bad=~np.isfinite(mapped).all(1)
    if bad.any():mapped[bad]=values[cKDTree(points).query(query[bad])[1]]
    assert np.isfinite(mapped).all()
    target=case/'201';target.mkdir()
    ub='internalField nonuniform List<vector>\n'+str(len(mapped))+'\n(\n'+\
        '\n'.join(f'({u:.10g} {v:.10g} 0)' for u,v,pv in mapped)+'\n);\n'
    pb='internalField nonuniform List<scalar>\n'+str(len(mapped))+'\n(\n'+\
        '\n'.join(f'{pv:.10g}' for u,v,pv in mapped)+'\n);\n'
    for name,body in [('U',ub),('p',pb)]:
        (target/name).write_text(replace_internal((case/'0'/name).read_text(),body))
    save(case/'initialization.json',{'donor':donor['id'],'donor_role':'original_train80',
        'time':stamp,'source_bytes_sha256':sha(seed),'nearest_fallback':int(bad.sum()),
        'meaning':'seed only; independent target CFD relaxation >=60s and same cycle-stability gate'})

def main():
    protected();assert shutil.disk_usage(ROOT).free>35*2**30
    DEST.mkdir(exist_ok=False);RAW.mkdir(exist_ok=False);CACHE.mkdir(exist_ok=False)
    p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    existing={q['id'] for q in p['cases']+p['confirmation_cases']}
    cases=design();assert not existing.intersection(q['id'] for q in cases)
    save(DEST/'preregister.json',{'utc':now(),'cases':cases,'data_role':'TRAINING ONLY',
        'design':'2 shapes x2 endpoint Re x4 outer geometry anchors; no observed flow label used in design',
        'support_change':'cy from +-0.4 to +-1.0; D from0.9--1.1 to0.75--1.25; old extreme32 becomes within expanded parameter support',
        'reporting':'any benefit belongs to expanded-data research task; never substitute into original80-only score',
        'CFD':'unchanged repaired one-layer210x120 dt.005; 165 outputs at.25; original force stability gate',
        'workers':3,'MPI_ranks_each':2,'fallback_immutable':True,'source_sha256':sha(__file__),
        'later_training':'matched original80 continuation versus80+16 outer anchors; same operator and task budget',
        'independent_confirmation':'new cases only after complete research method freeze'})
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.initialize=initialize
    completed=[];errors=[];begin=time.perf_counter()
    save(DEST/'status.json',{'status':'running','total':16,'completed':[],'errors':[],'utc':now()})
    with ThreadPoolExecutor(max_workers=3) as pool:
        future={pool.submit(rebuild.one,q,p):q['id'] for q in cases}
        for job in as_completed(future):
            if job.cancelled():continue
            try:completed.append(job.result())
            except Exception as e:
                errors.append({'id':future[job],'error':repr(e)})
                for other in future:other.cancel()
            protected()
            save(DEST/'status.json',{'status':'running' if not errors else 'failed',
                'total':16,'completed':completed,'errors':errors,'utc':now(),
                'wall_seconds':time.perf_counter()-begin})
    if errors:raise RuntimeError(errors)
    save(DEST/'training_manifest.json',{'cases':cases,'results':completed,'data_role':'TRAINING ONLY',
        'cache':str(CACHE),'registered_sha256':sha(DEST/'preregister.json'),'utc':now()})
    save(DEST/'status.json',{'status':'complete','total':16,'completed':completed,
        'errors':[],'utc':now(),'wall_seconds':time.perf_counter()-begin})

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
