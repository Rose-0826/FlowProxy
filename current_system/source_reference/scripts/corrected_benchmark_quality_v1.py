#!/usr/bin/env python3
"""Two known CFD anchors only; choose no model and use no AI score."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, json, shutil, copy, time, datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from build_true2d_case_v1 import build_case
import true2d_recheck_v1 as run
import cfd_numerics_analyze_v1 as stat
OUT=ROOT/'outputs/corrected_benchmark_quality_v1'
RAW=ROOT/'data/raw/corrected_benchmark_quality_v1'
run.ENV['HWLOC_COMPONENTS']='linux,stop'

def mpi(n=4):
    return f'mpirun --host localhost:{n} --mca plm isolated --bind-to none --stdin none --mca btl self,vader --mca pml ob1 --mca oob_tcp_if_include lo -np {n} icoFoam -parallel'

def prepare():
    assert not OUT.exists() and not RAW.exists()
    OUT.mkdir();RAW.mkdir()
    previous=json.loads((ROOT/'outputs/true2d_recheck_v1/preregister.json').read_text())
    p={'registered_utc':run.now(),'version':'corrected_benchmark_quality_v1',
       'cases':previous['cases'],'config':previous['config'],
       'reference':'existing repaired one-layer 210x120 background mesh dt=.005 native CFD',
       'variants':{'fine_dt00025':{'background':[210,120],'dt':.0025},
                   'finer_dt0005':{'background':[280,160],'dt':.005}},
       'source_time':302,'relax_end':362,'end':403,'dense_start':362,
       'statistics_window':[373,403],'workers':3,'MPI_ranks':4,
       'cycle_gate':{'Cd_mean':.005,'Cl_rms':.01,'St':.005},
       'time_gate':{'Cd_mean':.01,'Cl_rms':.03,'St':.01},
       'space_gate':{'Cd_mean':.02,'Cl_rms':.05,'St':.02},
       'field_difference_over_full_reference_rms':[.05,.05,.08],
       'interpretation':'predeclared internal sensitivity checks on two known anchors; no formal GCI or experimental validation; reference choice independent of AI scores',
       'sampling':'native .25 exact physical seconds, no temporal interpolation',
       'no_training_or_model_selection':True,'original_files_immutable':True,
       'script_sha256':sha(__file__)}
    save(OUT/'preregister.json',p)
    save(OUT/'status.json',{'status':'preparing','total':4,'completed':[],'errors':[]})
    for v in p['variants']:
        for row in p['cases']:
            cid=row['id'];target=RAW/v/cid
            source=ROOT/'data/raw/true2d_recheck_v1/fine_dt0005'/cid
            if v=='finer_dt0005':
                old=ROOT/'data/raw/cfd_numerics_round2_v1/third_dt0005'/cid
                build_case(old,target,.005,403,.25)
            else:
                target.mkdir(parents=True)
                for name in ['constant','system','0']:
                    shutil.copytree(source/name,target/name)
            target.joinpath('302').mkdir()
            for name in ['U','p']:
                shutil.copy2(target/'0'/name,target/'302'/name)
            run.configure(target,302,362,p['variants'][v]['dt'],10)
            run.command(target,f'mapFields -consistent -mapMethod cellPointInterpolate -sourceTime 302 "{source}"','log.mapFields_quality',180)
            target.joinpath('system/decomposeParDict').write_text('FoamFile {version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains 4;method scotch;\n')
            save(target/'source_manifest.json',{'source':str(source),'time':302,
                'source_fields_sha256':{n:sha(source/'302'/n) for n in ['U','p']},
                'target_mesh_sha256':{n:sha(target/'constant/polyMesh'/n) for n in ['points','faces','boundary']},
                'AI_initialization':False,'mapping':'cellPointInterpolate'})
    save(OUT/'status.json',{'status':'prepared','total':4,'completed':[],'errors':[]})
    print('QUALITY_PREPARED',flush=True)

def solve(v,row,p):
    begin=time.perf_counter();case=RAW/v/row['id']
    check=run.command(case,'checkMesh','log.checkMesh_quality',180)
    assert 'Mesh OK.' in check.read_text()
    from mesh_2d_repair_probe_v1 import points
    z=np.unique(np.round(points(case/'constant/polyMesh/points')[:,2],8))
    assert len(z)==2
    run.command(case,'decomposePar -time 302','log.decomposePar_quality',180)
    a=run.command(case,mpi(),'log.icoFoam_quality_relax',7200);run.assert_solver(a,362)
    run.configure(case,362,403,p['variants'][v]['dt'],.25)
    b=run.command(case,mpi(),'log.icoFoam_quality_dense',7200);run.assert_solver(b,403)
    (case/'log.icoFoam').write_text(a.read_text()+b.read_text())
    run.command(case,'reconstructPar -time "362:403"','log.reconstructPar_quality',1800)
    assert (case/'403/U').exists() and (case/'403/p').exists()
    run.command(case,'foamToVTK -useTimeName -time "362:403"','log.foamToVTK_quality',1800)
    from cpu_data_prepare import frames
    items=[(t,f) for t,f in frames(case) if 362-1e-7<=t<=403+1e-7]
    assert len(items)==165 and np.allclose(np.diff([t for t,f in items]),.25,atol=1e-8)
    result={'variant':v,'id':row['id'],'wall_seconds':time.perf_counter()-begin,'single_layer':True,'Mesh_OK':True,'native_uniform_frames':165,'normal_solver_end':True}
    print('QUALITY_CFD_DONE',v,row['id'],round(result['wall_seconds'],1),flush=True)
    return result

def analyze():
    p=json.loads((OUT/'preregister.json').read_text());stat.OUT=OUT;rows=[]
    for row in p['cases']:
        bank={}
        bank['fine_dt0005']=stat.case_stats(ROOT/'data/raw/true2d_recheck_v1/fine_dt0005'/row['id'],row,p['config'],[272,302])
        for v in p['variants']:
            bank[v]=stat.case_stats(RAW/v/row['id'],row,p['config'],p['statistics_window'])
        def compare(v,gate):
            d=stat.difference(bank['fine_dt0005'],bank[v])
            flags={'loads':all(d['loads_relative'][k]<tol for k,tol in gate.items())}
            for field in ['mean','rms']:
                flags[field]=all(x<tol for x,tol in zip(d[field+'_field_difference_over_full_reference_rms'],p['field_difference_over_full_reference_rms']))
            return {'difference':d,'checks':flags,'passed':all(flags.values())}
        stable={v:all(s[0]['last_cycle_relative_change'][k]<tol for k,tol in p['cycle_gate'].items()) for v,s in bank.items()}
        rows.append({'id':row['id'],'time':compare('fine_dt00025',p['time_gate']),'space':compare('finer_dt0005',p['space_gate']),
                     'stationarity':stable,'statistics':{v:s[0] for v,s in bank.items()}})
    passed=all(r['time']['passed'] and r['space']['passed'] and all(r['stationarity'].values()) for r in rows)
    report={'completed_utc':run.now(),'cases':rows,'all_internal_gates_passed':passed,'known_anchors':2,
            'chosen_reference':{'background':[210,120],'dt':.005,'sample_dt':.25,'single_layer':True} if passed else None,
            'selection_used_AI_scores':False,'formal_grid_convergence':False,'requires_further_refinement_if_failed':not passed}
    save(OUT/'report.json',report)
    print('QUALITY_ANALYZED',passed,flush=True)

def execute():
    p=json.loads((OUT/'preregister.json').read_text());done=[];errors=[]
    save(OUT/'status.json',{'status':'running','total':4,'completed':done,'errors':errors})
    with ThreadPoolExecutor(max_workers=p['workers']) as pool:
        tasks={pool.submit(solve,v,r,p):(v,r['id']) for v in p['variants'] for r in p['cases']}
        for f in as_completed(tasks):
            try:done.append(f.result())
            except Exception as e:errors.append({'job':tasks[f],'error':repr(e)})
            save(OUT/'status.json',{'status':'running','total':4,'completed':done,'errors':errors})
    save(OUT/'status.json',{'status':'failed' if errors else 'complete','total':4,'completed':done,'errors':errors})
    if errors:raise RuntimeError(errors)
    analyze()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--analyze',action='store_true');args=a.parse_args()
    prepare() if args.prepare else analyze() if args.analyze else execute()
