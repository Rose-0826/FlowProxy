#!/usr/bin/env python3
"""Third mesh and second time refinement, with a frozen sensitivity gate."""
from pathlib import Path
import sys,json,copy,datetime,shutil,subprocess,os,time,re
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'cfd/scripts')]
from geometry_attack_v2 import save,sha
from generate_case import write_case
from aic_stop_control import stop_job,active_group_members
OUT=ROOT/'outputs/cfd_numerics_round2_v1';RAW=ROOT/'data/raw/cfd_numerics_round2_v1'

def prepare():
    assert not OUT.exists() and not RAW.exists();OUT.mkdir();RAW.mkdir()
    previous=json.loads((ROOT/'outputs/cfd_numerics_restart_v1/preregister.json').read_text())
    protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':previous['cases'],'config':previous['config'],
      'variants':{'fine_dt0005':{'nx':210,'ny':120,'dt':.005},'third_dt0005':{'nx':280,'ny':160,'dt':.005}},
      'source':'existing fine_dt001 native CFD t261; no AI initialization',
      'physical_window':[261,311],'scoring_window':[281,311],'statistics':'last3 complete positive lift cycles; original physical gauge and time integration',
      'mapping':'cellPointInterpolate on third mesh; exact copy on existing fine mesh',
      'gate':{'time_refinement':{'Cd_mean':.01,'Cl_rms':.03,'St':.01},
              'space_refinement_same_dt':{'Cd_mean':.02,'Cl_rms':.05,'St':.02},
              'field_difference_over_full_reference_rms':{'mean':[.05,.05,.08],'rms':[.05,.05,.08]},
              'late_cycle_change':{'Cd_mean':.005,'Cl_rms':.01,'St':.005}},
      'interpretation':'internal sensitivity gate, not official competition threshold or formal Richardson/GCI',
      'comparison':'time: old fine_dt001 vs new fine_dt0005; space: fine_dt0005 vs third_dt0005',
      'not_changing_AI_model_or_old_scores':True,'workers':3,'timeout':3600,'script_sha256':sha(__file__)}
    save(OUT/'preregister.json',protocol)
    for variant,v in protocol['variants'].items():
        for row in protocol['cases']:
            config=copy.deepcopy(protocol['config']);config['cfd'].update({'end_time':311,'delta_t':v['dt'],'write_interval':.25,'n_cells_x':v['nx'],'n_cells_y':v['ny']})
            source=ROOT/'data/raw/cfd_numerics_restart_v1/fine_dt001'/row['id'];assert (source/'261/U').exists()
            case=write_case(config,RAW/variant,Re=row['Re'],cx=row['cx'],cy=row['cy'],D=row['D'],shape=row['shape'])
            p=case/'system/controlDict';t=p.read_text();t=re.sub(r'\bstartTime\s+0;', 'startTime 261;',t);t=re.sub(r'\bstartFrom\s+\w+;', 'startFrom startTime;',t);p.write_text(t)
            initial=case/'261';initial.mkdir()
            same=variant=='fine_dt0005'
            for name in ['U','p']:shutil.copy2(source/'261'/name if same else case/'0.orig'/name,initial/name)
            lines=['#!/bin/bash','set -eo pipefail',f'cd "{case}"',f'source "{ROOT}/cfd/env.sh"']
            if same:shutil.copytree(source/'constant/polyMesh',case/'constant/polyMesh')
            else:lines+=['blockMesh > log.blockMesh','snappyHexMesh -overwrite > log.snappyHexMesh',f'mapFields -consistent -mapMethod cellPointInterpolate -sourceTime 261 "{source}" > log.mapFields']
            lines+=['checkMesh > log.checkMesh','icoFoam > log.icoFoam','foamToVTK -useTimeName -time "281:311" > log.foamToVTK']
            (case/'Allrun_numerics').write_text('\n'.join(lines)+'\n')
            save(OUT/(variant+'__'+row['id']+'_setup.json'),{'source':str(source),'target':str(case),
              'source_U261_sha256':sha(source/'261/U'),'source_p261_sha256':sha(source/'261/p'),'config':config})
    print('ROUND2_PREPARED',flush=True)

def run():
    p=json.loads((OUT/'preregister.json').read_text());done=[];errors=[]
    def one(variant,row):
        case=RAW/variant/row['id'];begin=time.perf_counter()
        with (case/'log.numerics_driver').open('x') as f:
            proc=subprocess.Popen(['bash',str(case/'Allrun_numerics')],cwd=case,stdout=f,stderr=subprocess.STDOUT,
              start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            try:
                rc=proc.wait(timeout=p['timeout'])
                if active_group_members(proc.pid):stop_job(proc);raise RuntimeError('Active children after completion')
            except BaseException:
                if proc.poll() is None:stop_job(proc)
                raise
        if rc:raise RuntimeError('Numerics failed '+variant+' '+row['id'])
        log=case/'log.icoFoam';text=log.read_text(errors='replace')
        assert 'End' in text[-1600:] and abs(float(re.findall(r'Time = (\S+)',text)[-1])-311)<1e-7
        result={'variant':variant,'id':row['id'],'seconds':time.perf_counter()-begin,'solver_log_sha256':sha(log),'no_active_group_members':True}
        print('ROUND2_DONE',variant,row['id'],round(result['seconds'],1),flush=True);return result
    save(OUT/'status.json',{'status':'running','completed':[],'errors':[],'total':4})
    with ThreadPoolExecutor(max_workers=p['workers']) as pool:
        fs={pool.submit(one,v,r):(v,r['id']) for v in p['variants'] for r in p['cases']}
        for future in as_completed(fs):
            try:done.append(future.result())
            except Exception as e:errors.append({'job':fs[future],'error':repr(e)})
            save(OUT/'status.json',{'status':'running','completed':done,'errors':errors,'total':4})
    save(OUT/'status.json',{'status':'failed' if errors else 'complete','completed':done,'errors':errors,'total':4})
    if errors:raise RuntimeError('Preserve failed round2')
    analyze()

def analyze():
    import cfd_numerics_analyze_v1 as a
    a.OUT=OUT;p=json.loads((OUT/'preregister.json').read_text());rows=[];stats={};g=p['gate']
    def assess(diff,loads):
        return {'loads':all(diff['loads_relative'][k]<v for k,v in loads.items()),
          'field_mean':all(x<v for x,v in zip(diff['mean_field_difference_over_full_reference_rms'],g['field_difference_over_full_reference_rms']['mean'])),
          'field_rms':all(x<v for x,v in zip(diff['rms_field_difference_over_full_reference_rms'],g['field_difference_over_full_reference_rms']['rms']))}
    for row in p['cases']:
        bank={}
        for variant in p['variants']:
            bank[variant]=a.case_stats(RAW/variant/row['id'],row,p['config'],p['scoring_window']);stats[variant+'__'+row['id']]=bank[variant][0]
        old=ROOT/'data/raw/cfd_numerics_restart_v1/fine_dt001'/row['id']
        bank['old_fine']=a.case_stats(old,row,p['config'],[231,261]);stats['old_fine__'+row['id']]=bank['old_fine'][0]
        dt=a.difference(bank['old_fine'],bank['fine_dt0005']);space=a.difference(bank['fine_dt0005'],bank['third_dt0005'])
        stability={k:all(v[0]['last_cycle_relative_change'][x]<y for x,y in g['late_cycle_change'].items()) for k,v in bank.items()}
        passed_time=assess(dt,g['time_refinement']);passed_space=assess(space,g['space_refinement_same_dt'])
        rows.append({'case':row['id'],'time_refinement':dt,'space_refinement_same_dt':space,'late_cycle_gate':stability,
          'time_gate':passed_time,'space_gate':passed_space,'internal_gate_passed':all(stability.values()) and all(passed_time.values()) and all(passed_space.values())})
    save(OUT/'sensitivity_report.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':rows,'statistics':stats,
      'all_internal_gates_passed':all(r['internal_gate_passed'] for r in rows),'not_formal_grid_convergence':True,
      'only_two_anchors':True,'not_AI_prediction_scores':True,'no_frozen_model_or_reference_replacement':True})
    print('ROUND2_ANALYZED',[(r['case'],r['internal_gate_passed']) for r in rows],flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else analyze() if args.analyze else run()
