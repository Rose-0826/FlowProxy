#!/usr/bin/env python3
"""Two same-mesh time-step controls, nominated after the repair diagnostic."""
from pathlib import Path
import sys,json,time,re,shutil,copy
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import true2d_recheck_v1 as m

PARENT=ROOT/'outputs/true2d_recheck_v1'
PRAW=ROOT/'data/raw/true2d_recheck_v1'
OUT=ROOT/'outputs/true2d_time_control_v1';RAW=ROOT/'data/raw/true2d_time_control_v1'
LINE=('mpirun --host localhost:2 --mca plm isolated --bind-to none --stdin none '
      '--mca btl self,vader --mca pml ob1 --mca oob_tcp_if_include lo -np 2 icoFoam -parallel')

def run():
    assert not OUT.exists() and not RAW.exists();OUT.mkdir();RAW.mkdir()
    original=json.loads((PARENT/'preregister.json').read_text());p=copy.deepcopy(original)
    p.update({'utc':m.now(),'variants':{'coarse_dt0005':{'dt':.005,'MPI_ranks':2,'start':201,'end':302,
              'input_start':261,'score_window':[272,302]}},
              'purpose':'Separate dt sensitivity on identical repaired coarse mesh from spatial sensitivity at fixed dt=.005',
              'nominated_after_parent_scores':True,'known_anchors_not_independent_confirmation':True,
              'source':'repaired coarse native CFD t201; exact U/p copy, no AI initialization',
              'primary':'time: corrected coarse .02 vs same mesh .005; space: corrected coarse .005 vs fine .005',
              'budget':'two additional CFD controls only; no training or algorithm selection',
              'script_sha256':m.sha(__file__)})
    m.save(OUT/'preregister.json',p)
    sources=[]
    for row in p['cases']:
        cid=row['id'];source=PRAW/'coarse_dt002'/cid;case=RAW/'coarse_dt0005'/cid;case.mkdir(parents=True)
        for name in ['constant','system','0']:shutil.copytree(source/name,case/name)
        (case/'201').mkdir()
        for name in ['U','p']:shutil.copy2(source/'201'/name,case/'201'/name)
        m.configure(case,201,261,.005,10)
        (case/'system/decomposeParDict').write_text('FoamFile {version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains 2;method scotch;\n')
        sources.append({'case':cid,'source':str(source),
                        'mesh_sha256':{n:m.sha(source/'constant/polyMesh'/n) for n in ['points','faces','boundary']},
                        'initial_sha256':{n:m.sha(source/'201'/n) for n in ['U','p']}})
    m.save(OUT/'sources.json',{'cases':sources})
    m.ENV['HWLOC_COMPONENTS']='linux,stop';m.OUT=OUT;m.RAW=RAW
    completed=[];errors=[]
    def one(row):
        cid=row['id'];case=RAW/'coarse_dt0005'/cid;begin=time.perf_counter()
        check=m.command(case,'checkMesh','log.checkMesh',180);assert 'Mesh OK.' in check.read_text()
        m.command(case,'decomposePar -time 201','log.decomposePar',180)
        first=m.command(case,LINE,'log.icoFoam_phase1');m.assert_solver(first,261)
        m.configure(case,261,302,.005,.25)
        second=m.command(case,LINE,'log.icoFoam_phase2');m.assert_solver(second,302)
        (case/'log.icoFoam').write_text(first.read_text()+second.read_text())
        m.command(case,'reconstructPar -time "261:302"','log.reconstructPar',1200)
        assert (case/'302/U').exists() and (case/'302/p').exists()
        m.command(case,'foamToVTK -useTimeName -time "261:302"','log.foamToVTK',1200)
        items=[(t,f) for t,f in m.frames(case) if 261-1e-7<=t<=302+1e-7];ts=np.array([t for t,_ in items])
        assert len(ts)>=164 and abs(ts[163]-ts[3]-40)<1e-7
        r={'id':cid,'variant':'coarse_dt0005','seconds':time.perf_counter()-begin,'Mesh_OK':True,
           'startup_times':ts[:4].tolist(),'end':float(ts[163]),'MPI_ranks':2,'solver_log_sha256':m.sha(case/'log.icoFoam')}
        print('TIME_CONTROL_DONE',cid,round(r['seconds'],1),flush=True);return r
    m.save(OUT/'status.json',{'status':'running','completed':[],'errors':[]})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,r):r for r in p['cases']}
        for f in as_completed(futures):
            try:completed.append(f.result())
            except Exception as e:errors.append({'case':futures[f]['id'],'error':repr(e)})
            m.save(OUT/'status.json',{'status':'running','completed':completed,'errors':errors})
    if not errors:
        result=m.evaluate('coarse_dt0005',p)
        (OUT/'field_statistics').mkdir();m.stat.OUT=OUT/'field_statistics';rows=[];banks={}
        for row in p['cases']:
            cid=row['id'];bank={
                'coarse_dt002':m.stat.case_stats(PRAW/'coarse_dt002'/cid,row,p['config'],[171,201]),
                'coarse_dt0005':m.stat.case_stats(RAW/'coarse_dt0005'/cid,row,p['config'],[272,302]),
                'fine_dt0005':m.stat.case_stats(PRAW/'fine_dt0005'/cid,row,p['config'],[272,302])}
            banks[cid]={v:a[0] for v,a in bank.items()}
            rows.append({'id':cid,'time_step_only':m.stat.difference(bank['coarse_dt002'],bank['coarse_dt0005']),
                         'space_at_fixed_dt':m.stat.difference(bank['coarse_dt0005'],bank['fine_dt0005'])})
        quality={cid:{v:all(a['last_cycle_relative_change'][k]<limit for k,limit in p['cycle_quality_gate'].items())
                      for v,a in b.items()} for cid,b in banks.items()}
        m.save(OUT/'reference_statistics.json',{'statistics':banks,'comparisons':rows,'late_cycle_quality_gate':quality,
                                              'not_formal_grid_convergence':True,'diagnostic_not_model_selection':True})
        for rel,digest in p['frozen_sha256'].items():assert m.sha(ROOT/rel)==digest
        for s in sources:
            for n,h in s['mesh_sha256'].items():assert m.sha(Path(s['source'])/'constant/polyMesh'/n)==h
        m.save(OUT/'report.json',{'evaluation':result,'completed':completed,'errors':[],'no_training':True})
    m.save(OUT/'status.json',{'status':'complete' if not errors else 'failed','completed':completed,'errors':errors})
    print('TIME_CONTROL_FINISHED',len(completed),errors,flush=True)

if __name__=='__main__':run()
