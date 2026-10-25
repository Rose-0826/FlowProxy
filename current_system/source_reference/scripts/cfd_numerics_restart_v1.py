#!/usr/bin/env python3
"""Native CFD mesh/time sensitivity from existing settled states, never AI initial states."""
from pathlib import Path
import sys,json,copy,datetime,shutil,subprocess,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'cfd/scripts')]
from geometry_attack_v2 import sha,save
from generate_case import write_case
from aic_stop_control import stop_job,active_group_members
OUT=ROOT/'outputs/cfd_numerics_restart_v1';RAW=ROOT/'data/raw/cfd_numerics_restart_v1'


def prepare():
    if OUT.exists() or RAW.exists():raise FileExistsError('Numerics stage exists')
    OUT.mkdir();RAW.mkdir()
    p=json.loads((ROOT/'outputs/geometry_attack_confirmation_d1_v1/preregister.json').read_text());ids=['cylinder_Re110_cy-0.2_D1','square_Re170_cy0.2_D1']
    rows=[r for r in p['cases'] if r['id'] in ids];cfg=p['CFD']['config']
    protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':rows,'config':cfg,
        'start_from':'native coarse CFD201; exact copy on same mesh, mapFields cellVolumeWeight on refined mesh',
        'variants':{'coarse_dt002':{'nx':140,'ny':80,'dt':.02},'coarse_dt001':{'nx':140,'ny':80,'dt':.01},'fine_dt001':{'nx':210,'ny':120,'dt':.01}},
        'physical_window':[201,261],'scoring_window':[231,261],'statistics':'last3 complete lift periods; field mean and RMS with time integration; load mean/RMS and crossing frequency',
        'no_AI_in_this_experiment':True,'no_AI_model_retuning':True,'not_full_formal_grid_convergence':True,
        'workers':3,'timeout':2400,'script_sha256':sha(__file__)}
    save(OUT/'preregister.json',protocol)
    for variant,v in protocol['variants'].items():
        for row in rows:
            config=copy.deepcopy(cfg);config['cfd'].update({'end_time':261,'delta_t':v['dt'],'write_interval':.25,'n_cells_x':v['nx'],'n_cells_y':v['ny']})
            case=write_case(config,RAW/variant,Re=row['Re'],cx=row['cx'],cy=row['cy'],D=row['D'],shape=row['shape'])
            source=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/row['id'];assert (source/'201/U').exists()
            control=case/'system/controlDict';text=control.read_text().replace('startTime       0;', 'startTime       201;')
            # Generated whitespace varies across versions; match the dictionary token.
            import re
            text=re.sub(r'\bstartTime\s+0;', 'startTime 201;',text);text=re.sub(r'\bstartFrom\s+\w+;', 'startFrom startTime;',text);control.write_text(text)
            initial=case/'201';initial.mkdir()
            for name in ['U','p']:shutil.copy2(source/'201'/name,initial/name)
            lines=['#!/bin/bash','set -eo pipefail',f'cd "{case}"',f'source "{ROOT}/cfd/env.sh"']
            if variant=='fine_dt001':
                for name in ['U','p']:shutil.copy2(case/'0.orig'/name,initial/name)
                lines+=['blockMesh > log.blockMesh','snappyHexMesh -overwrite > log.snappyHexMesh',
                        f'mapFields -consistent -mapMethod cellVolumeWeight -sourceTime 201 "{source}" > log.mapFields']
            else:shutil.copytree(source/'constant/polyMesh',case/'constant/polyMesh')
            lines+=['checkMesh > log.checkMesh','icoFoam > log.icoFoam','foamToVTK -useTimeName -time "221:261" > log.foamToVTK']
            (case/'Allrun_numerics').write_text('\n'.join(lines)+'\n')
            save(OUT/(variant+'__'+row['id']+'_setup.json'),{'source':str(source),'source_U201_sha256':sha(source/'201/U'),
                'source_p201_sha256':sha(source/'201/p'),'target':str(case),'variant':variant,'config':config})
    print('NUMERICS_PREPARED',flush=True)


def run():
    p=json.loads((OUT/'preregister.json').read_text());done=[];errors=[]
    def one(variant,row):
        case=RAW/variant/row['id'];begin=time.perf_counter()
        with (case/'log.numerics_driver').open('x') as f:
            proc=subprocess.Popen(['bash',str(case/'Allrun_numerics')],cwd=case,stdout=f,stderr=subprocess.STDOUT,
                start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
            try:
                rc=proc.wait(timeout=p['timeout'])
                if active_group_members(proc.pid):stop_job(proc);raise RuntimeError('Active children after completion')
            except BaseException:
                if proc.poll() is None:stop_job(proc)
                raise
        if rc!=0:raise RuntimeError('Numerics failed '+variant+' '+row['id'])
        log=case/'log.icoFoam'
        if 'End' not in log.read_text(errors='replace')[-1600:]:raise RuntimeError('No normal solver End')
        result={'id':row['id'],'variant':variant,'seconds':time.perf_counter()-begin,'no_active_group_members':True,'solver_log_sha256':sha(log)}
        print('NUMERICS_DONE',variant,row['id'],round(result['seconds'],1),flush=True);return result
    save(OUT/'status.json',{'status':'running','completed':done,'errors':errors,'total':6})
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(one,v,r):(v,r['id']) for v in p['variants'] for r in p['cases']}
        for future in as_completed(futures):
            try:done.append(future.result())
            except Exception as e:errors.append({'job':futures[future],'error':repr(e)})
            save(OUT/'status.json',{'status':'running','completed':done,'errors':errors,'total':6})
    save(OUT/'status.json',{'status':'failed' if errors else 'complete','completed':done,'errors':errors,'total':6})
    if errors:raise RuntimeError('Numerics failed; preserve all results')
    print('NUMERICS_ALL_COMPLETE',flush=True)


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');args=a.parse_args()
    prepare() if args.prepare else run()
