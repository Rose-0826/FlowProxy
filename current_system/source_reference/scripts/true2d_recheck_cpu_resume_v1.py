#!/usr/bin/env python3
"""Resume unstarted fine reference jobs; preserve launcher failures and all scores."""
from pathlib import Path
import sys,json,time,os,re
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import true2d_recheck_v1 as m

def run():
    parallel='--mpi-local' in sys.argv
    p=json.loads((m.OUT/'preregister.json').read_text())
    old=json.loads((m.OUT/'status.json').read_text())
    assert old['status']=='failed' and len(old['completed'])==2
    assert all(r['variant']=='coarse_dt002' for r in old['completed'])
    for filename in ['status.json','report.json']:
        target=m.OUT/(filename.replace('.json','_before_execution_amendment.json'))
        assert not target.exists();target.write_bytes((m.OUT/filename).read_bytes())
    for rel,digest in p['frozen_sha256'].items():assert m.sha(ROOT/rel)==digest
    m.ENV['HWLOC_COMPONENTS']='linux,stop'
    line=('mpirun --host localhost:4 --mca plm isolated --bind-to none --stdin none '
          '--mca btl self,vader --mca pml ob1 --mca oob_tcp_if_include lo -np 4 icoFoam -parallel') if parallel else 'icoFoam'
    m.save(m.OUT/'execution_amendment.json',{
        'utc':m.now(),'original_error':'MPI launcher remained idle; no Time lines, no solver ranks started',
        'scope':'execution environment only; same meshes, mapped initial fields, dt, field/force outputs and frozen models',
        'original_launch_failures_preserved':True,'actual_solver_line':line,
        'MPI_ranks':4 if parallel else 1,'no_new_prediction_method_or_model_selection':True,
        'script_sha256':m.sha(__file__)})
    completed=old['completed'].copy();errors=[]
    def one(row):
        cid=row['id'];case=m.RAW/'fine_dt0005'/cid;begin=time.perf_counter()
        assert '\nTime = ' not in (case/'log.icoFoam_phase1').read_text()
        numeric=[x.name for x in case.iterdir() if x.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?',x.name)]
        assert set(numeric)=={'0','201'},numeric
        initial={n:m.sha(case/'201'/n) for n in ['U','p']}
        m.save(m.OUT/f'fine_dt0005__{cid}_resumed_initialization.json',{
            'utc':m.now(),'native_time':201,'mapped_initial_sha256':initial,
            'no_actual_solver_step_before_resume':True,'MPI_ranks':4 if parallel else 1})
        m.configure(case,201,261,.005,10)
        first=m.command(case,line,'log.icoFoam_cpu_phase1');m.assert_solver(first,261)
        m.configure(case,261,302,.005,.25)
        second=m.command(case,line,'log.icoFoam_cpu_phase2');m.assert_solver(second,302)
        assert not (case/'log.icoFoam').exists()
        (case/'log.icoFoam').write_text(first.read_text()+second.read_text())
        if parallel:m.command(case,'reconstructPar -time "261:302"','log.reconstructPar',1200)
        assert (case/'302/U').exists() and (case/'302/p').exists()
        m.command(case,'foamToVTK -useTimeName -time "261:302"','log.foamToVTK',1200)
        items=[(t,f) for t,f in m.frames(case) if 261-1e-7<=t<=302+1e-7]
        ts=np.array([t for t,_ in items]);assert len(ts)>=164 and abs(ts[163]-ts[3]-40)<1e-7
        result={'variant':'fine_dt0005','id':cid,'seconds':time.perf_counter()-begin,
                'final_time':302,'mesh_OK':True,'VTK_frames':len(items),'startup_times':ts[:4].tolist(),
                'forecast_end':float(ts[163]),'no_active_children':True,'MPI_ranks':4 if parallel else 1,
                'solver_sha256':m.sha(case/'log.icoFoam'),
                'mesh_sha256':{n:m.sha(case/'constant/polyMesh'/n) for n in ['points','faces','boundary']}}
        print('TRUE2D_FINE_DONE',cid,round(result['seconds'],1),flush=True);return result
    m.save(m.OUT/'status.json',{'status':'running','utc':m.now(),'completed':completed,'errors':[],
                              'execution_amendment':'execution_amendment.json','pending_jobs':2})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,row):row for row in p['cases']}
        for f in as_completed(futures):
            try:completed.append(f.result())
            except Exception as e:errors.append({'variant':'fine_dt0005','id':futures[f]['id'],'error':repr(e)})
            m.save(m.OUT/'status.json',{'status':'running','utc':m.now(),'completed':completed,'errors':errors,
                                      'execution_amendment':'execution_amendment.json','pending_jobs':4-len(completed)-len(errors)})
    evaluations={v:json.loads((m.OUT/('evaluation_'+v)/'report.json').read_text())
                 for v in ['old_reference','coarse_dt002']}
    if not errors:
        evaluations['fine_dt0005']=m.evaluate('fine_dt0005',p)
        m.analyze(p,['coarse_dt002','fine_dt0005'])
        for rel,digest in p['frozen_sha256'].items():assert m.sha(ROOT/rel)==digest
    m.save(m.OUT/'report.json',{'utc':m.now(),'evaluations':evaluations,'completed':completed,'errors':errors,
                              'scope':'two previously examined anchors; no training or new geometry confirmation',
                              'frozen_weights_and_configuration_unchanged':not errors,
                              'execution_amendment':'execution_amendment.json'})
    m.save(m.OUT/'status.json',{'status':'complete' if not errors else 'failed','utc':m.now(),
                              'completed':completed,'errors':errors,'execution_amendment':'execution_amendment.json'})
    print('TRUE2D_RECHECK_CPU_COMPLETE',len(completed),len(errors),flush=True)

if __name__=='__main__':run()
