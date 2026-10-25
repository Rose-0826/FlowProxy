#!/usr/bin/env python3
"""Resume only two unmapped fine meshes; preserve compatibility failure."""
from pathlib import Path
import sys,json,datetime,shutil,subprocess,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from aic_stop_control import stop_job,active_group_members
OUT=ROOT/'outputs/cfd_numerics_restart_v1';RAW=ROOT/'data/raw/cfd_numerics_restart_v1'

def run():
    original=json.loads((OUT/'status.json').read_text())
    assert original['status']=='failed' and len(original['completed'])==4
    for name in ['status.json','preregister.json']:
        target=OUT/(name[:-5]+'_initial_mapping_failure.json')
        assert not target.exists();shutil.copy2(OUT/name,target)
    protocol=json.loads((OUT/'preregister.json').read_text())
    amendment={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'reason':'Installed OpenFOAM v2406 mapFields supports mapNearest/interpolate/cellPointInterpolate, not cellVolumeWeight.',
      'change':'Use cellPointInterpolate on already generated fine meshes; no coarse reruns and no mesh regeneration.',
      'no_scoring_or_model_change':True,'script_sha256':sha(__file__)}
    save(OUT/'mapping_compatibility_amendment.json',amendment)
    protocol['start_from']='native coarse CFD201; exact copy on same mesh; mapFields cellPointInterpolate on refined mesh'
    protocol['mapping_compatibility_amendment']='mapping_compatibility_amendment.json'
    save(OUT/'preregister.json',protocol)
    done=original['completed'].copy();errors=[]
    def one(row):
        case=RAW/'fine_dt001'/row['id'];source=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/row['id']
        assert (case/'constant/polyMesh/points').exists()
        assert not (case/'261').exists()
        lines=['#!/bin/bash','set -eo pipefail',f'cd "{case}"',f'source "{ROOT}/cfd/env.sh"',
          f'mapFields -consistent -mapMethod cellPointInterpolate -sourceTime 201 "{source}" > log.mapFields_retry',
          'checkMesh > log.checkMesh_retry','icoFoam > log.icoFoam',
          'foamToVTK -useTimeName -time "221:261" > log.foamToVTK']
        script=case/'Allrun_numerics_resume';assert not script.exists();script.write_text('\n'.join(lines)+'\n')
        begin=time.perf_counter()
        with (case/'log.numerics_resume_driver').open('x') as log:
            proc=subprocess.Popen(['bash',str(script)],cwd=case,stdout=log,stderr=subprocess.STDOUT,
              start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
            try:
                rc=proc.wait(timeout=protocol['timeout'])
                if active_group_members(proc.pid):stop_job(proc);raise RuntimeError('Active children after finish')
            except BaseException:
                if proc.poll() is None:stop_job(proc)
                raise
        if rc!=0:raise RuntimeError('Fine resume failed '+row['id'])
        if 'End' not in (case/'log.icoFoam').read_text(errors='replace')[-1600:]:raise RuntimeError('No normal solver End')
        result={'id':row['id'],'variant':'fine_dt001','seconds':time.perf_counter()-begin,
          'no_active_group_members':True,'solver_log_sha256':sha(case/'log.icoFoam'),
          'fine_mesh_already_prepared':True,'mapping_method':'cellPointInterpolate'}
        print('FINE_DONE',row['id'],round(result['seconds'],1),flush=True);return result
    save(OUT/'status.json',{'status':'running','completed':done,'errors':[],'total':6,'initial_failures_preserved':True})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,row):row['id'] for row in protocol['cases']}
        for future in as_completed(futures):
            try:done.append(future.result())
            except Exception as e:errors.append({'case':futures[future],'error':repr(e)})
            save(OUT/'status.json',{'status':'running','completed':done,'errors':errors,'total':6,'initial_failures_preserved':True})
    save(OUT/'status.json',{'status':'failed' if errors else 'complete','completed':done,'errors':errors,'total':6,'initial_failures_preserved':True})
    if errors:raise RuntimeError('Fine retry failed; preserve records')
    print('NUMERICS_ALL_COMPLETE',flush=True)

if __name__=='__main__':run()
