#!/usr/bin/env python3
"""Bounded mesh-repair diagnostic with frozen models and two CFD anchors."""
from pathlib import Path
import sys, os, re, json, time, shutil, subprocess, datetime, threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from geometry_attack_v2 import save, sha, NativeSampler
from geometry_attack_evaluate_v1 import CFG, specs, Candidate
from build_true2d_case_v1 import build_case
from cpu_data_prepare import frames
from flowproxy.geometry import geometry_channels
from round2_complete40_v1 import frame_summary, summarize
from aic_stop_control import stop_job, active_group_members
import cfd_numerics_analyze_v1 as stat

OUT = ROOT/'outputs/true2d_recheck_v1'
RAW = ROOT/'data/raw/true2d_recheck_v1'
ENV = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
       'MKL_NUM_THREADS':'1', 'OMPI_ALLOW_RUN_AS_ROOT':'1', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM':'1'}
LOCK = threading.Lock()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def top_setting(text, key, value):
    text, n = re.subn(r'^'+key+r'\s+[^;]+;', f'{key} {value};', text, flags=re.MULTILINE)
    assert n == 1, (key, n)
    return text

def configure(case, start, end, dt, interval):
    p = case/'system/controlDict'; t = p.read_text()
    for k,v in [('startFrom','startTime'),('startTime',start),('endTime',end),
                ('deltaT',dt),('writeInterval',interval)]:
        t = top_setting(t,k,v)
    # Function-object writeInterval is a time-step count, not the field interval.
    t,n = re.subn(r'(\bforceCoeffs1\s*\{.*?\bwriteInterval\s+)[^;]+;',
                  r'\g<1>1;', t, flags=re.DOTALL)
    assert n == 1
    p.write_text(t)

def prepare():
    assert not OUT.exists() and not RAW.exists()
    OUT.mkdir(); RAW.mkdir()
    chosen = json.loads((ROOT/'outputs/cfd_numerics_restart_v1/preregister.json').read_text())['cases']
    bank = {s['name']:s for s in specs()}
    selected = [bank[n] for n in ['F0','K8','neural_periodic_closure']]
    frozen = {str(Path(s['path']).relative_to(ROOT)):sha(s['path']) for s in selected}
    for p in [ROOT/'outputs/self_trajectory_closure_v1/selected_val.json',
              ROOT/'outputs/startup_observability_v1/training_priors.json']:
        frozen[str(p.relative_to(ROOT))] = sha(p)
    p = {'utc':now(), 'cases':chosen, 'config':CFG, 'models':[s['name'] for s in selected],
         'frozen_sha256':frozen, 'variants':{
             'coarse_dt002':{'dt':.02,'MPI_ranks':1,'start':0,'end':201,
                              'initialization':'original uniform-inflow initial field; newly built single-layer coarse mesh',
                              'input_start':160,'score_window':[171,201]},
             'fine_dt0005':{'dt':.005,'MPI_ranks':4,'start':201,'end':302,
                            'initialization':'map repaired coarse native CFD t=201 to repaired fine mesh; no AI initialization',
                            'relaxation_seconds_before_startup':60,'input_start':261,'score_window':[272,302]}},
         'methods_unchanged':'F0, K8, selected neural-periodic closure; no training, tuning, phase alignment or pressure recentering',
         'primary':'coarse_dt002 versus old coarse reference, same dt=.02; frozen model transfer diagnostic only',
         'secondary':'fine_dt0005 versus corrected coarse, joint space/time sensitivity diagnostic; not formal grid convergence',
         'known_cases_not_independent_confirmation':True,
         'future_horizons_seconds':[2,8,20,40], 'initial_true_fields':4,
         'prediction_order':'save and hash all model predictions for a variant before reading corresponding future field values',
         'metric':'fluid mask >.5, raw p, channel relative L2 per frame; Ri max of channel time-mean/end divided by .05/.05/.08',
         'pass_condition':'strict Ri<1; averages do not prove all cases pass or every frame pass',
         'cycle_quality_gate':{'Cd_mean':.005,'Cl_rms':.01,'St':.005},
         'completion_required':'Mesh OK, exactly one cell layer, anchored final Time, End, native final U/p, all startup/future VTK',
         'no_changes_to_original_sources_or_model_weights':True,
         'budget':'two coarse solves and two fine solves only; timeout per solver phase 7200s',
         'field_output':'first phase every 10 physical seconds; final phase every .25 seconds; forces every solver step',
         'official_2D_requirement':'https://www.openfoam.com/documentation/tutorial-guide/2-incompressible-flow/2.1-lid-driven-cavity-flow',
         'script_sha256':sha(__file__)}
    save(OUT/'preregister.json',p)
    builds=[]
    for row in chosen:
        cid=row['id']; source=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/cid
        case=RAW/'coarse_dt002'/cid
        builds.append(build_case(source,case,.02,201,.25))
        configure(case,0,160,.02,10)
        fine=ROOT/'data/raw/true2d_reference_pilot_v1'/cid
        assert fine.exists() and not (fine/'201').exists() and not (fine/'processor0').exists()
        meta=json.loads((fine/'true2d_build.json').read_text())
        assert meta['mesh_OK'] and meta['exact_single_layer'] and meta['solver_not_run_by_builder']
        target=RAW/'fine_dt0005'/cid
        target.mkdir(parents=True)
        for name in ['constant','system','0']:
            shutil.copytree(fine/name,target/name)
        configure(target,201,261,.005,10)
        (target/'system/decomposeParDict').write_text(
            'FoamFile {version 2.0;format ascii;class dictionary;object decomposeParDict;}\n'
            'numberOfSubdomains 4;method scotch;\n')
        builds.append({'case':cid,'target':str(target),'source':str(fine),
                       'mesh_sha256':{n:sha(target/'constant/polyMesh'/n) for n in ['points','faces','boundary']},
                       'source_build_metadata_sha256':sha(fine/'true2d_build.json'),
                       'exact_single_layer':True,'fine_CFD_not_started':True})
    save(OUT/'builds.json',{'cases':builds})
    save(OUT/'status.json',{'status':'prepared','completed':[],'errors':[]})
    print('TRUE2D_RECHECK_PREPARED',flush=True)

def command(case, line, log, timeout=7200):
    path=case/log
    with path.open('x') as f:
        proc=subprocess.Popen(['bash','-c',f'set -eo pipefail; source "{ROOT}/cfd/env.sh"; '+line],
                              cwd=case,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=ENV)
        try:
            rc=proc.wait(timeout=timeout)
            if active_group_members(proc.pid):
                stop_job(proc); raise RuntimeError('Live children after process completion')
        except BaseException:
            if proc.poll() is None:stop_job(proc)
            raise
    if rc:raise RuntimeError(f'{case.name}/{log}: return {rc}: '+path.read_text(errors='replace')[-1200:])
    return path

def assert_solver(log, final):
    t=log.read_text(errors='replace')
    stamps=re.findall(r'^Time = (\S+)',t,re.MULTILINE)
    assert stamps and abs(float(stamps[-1])-final)<1e-7, (log,stamps[-1:] )
    assert re.search(r'^End\s*$',t,re.MULTILINE),log
    assert 'FOAM FATAL' not in t and 'nan' not in t.lower()

def solve(variant,row,p):
    cid=row['id']; v=p['variants'][variant]; case=RAW/variant/cid
    begin=time.perf_counter()
    check=command(case,'checkMesh','log.checkMesh_recheck',180)
    assert 'Mesh OK.' in check.read_text()
    if variant=='fine_dt0005':
        source=RAW/'coarse_dt002'/cid
        (case/'201').mkdir()
        for name in ['U','p']:shutil.copy2(case/'0'/name,case/'201'/name)
        source_hash={n:sha(source/'201'/n) for n in ['U','p']}
        command(case,f'mapFields -consistent -mapMethod cellPointInterpolate -sourceTime 201 "{source}"',
                'log.mapFields_repaired',180)
        command(case,'decomposePar -time 201','log.decomposePar',180)
        save(OUT/f'{variant}__{cid}_initialization.json',
             {'utc':now(),'source':str(source),'source_native_time':201,'source_hashes':source_hash,
              'mapping':'cellPointInterpolate','AI_initialization':False})
        line='mpirun -np 4 icoFoam -parallel'
    else:line='icoFoam'
    first=command(case,line,'log.icoFoam_phase1')
    first_end=160 if variant=='coarse_dt002' else 261
    assert_solver(first,first_end)
    configure(case,first_end,v['end'],v['dt'],.25)
    second=command(case,line,'log.icoFoam_phase2'); assert_solver(second,v['end'])
    (case/'log.icoFoam').write_text(first.read_text()+second.read_text())
    if variant=='fine_dt0005':
        command(case,'reconstructPar -time "261:302"','log.reconstructPar',1200)
    assert (case/str(v['end'])/'U').exists() and (case/str(v['end'])/'p').exists()
    command(case,f'foamToVTK -useTimeName -time "{v["input_start"]}:{v["end"]}"',
            'log.foamToVTK',1200)
    items=frames(case);lo=v['input_start']
    chosen=[(t,f) for t,f in items if lo-1e-7<=t<=lo+41+1e-7]
    assert len(chosen)>=164, (variant,cid,len(chosen))
    times=np.array([t for t,_ in chosen]);assert abs(times[163]-times[3]-40)<1e-7
    result={'variant':variant,'id':cid,'seconds':time.perf_counter()-begin,
            'final_time':v['end'],'mesh_OK':True,'VTK_frames':len(chosen),
            'startup_times':times[:4].tolist(),'forecast_end':float(times[163]),
            'no_active_children':True,'solver_sha256':sha(case/'log.icoFoam'),
            'mesh_sha256':{n:sha(case/'constant/polyMesh'/n) for n in ['points','faces','boundary']}}
    print('TRUE2D_CFD_DONE',variant,cid,round(result['seconds'],1),flush=True)
    return result

def evaluate(variant,p):
    import torch
    torch.set_num_threads(1)
    out=OUT/('evaluation_'+variant);out.mkdir();(out/'predictions').mkdir()
    for relative,digest in p['frozen_sha256'].items():assert sha(ROOT/relative)==digest
    bank={s['name']:s for s in specs()};candidates={n:Candidate(bank[n]) for n in p['models']}
    supports={};registry=[]
    for row in p['cases']:
        q={k:row[k] for k in ['shape','Re','D','cx','cy']}; geo=geometry_channels(CFG,**q)
        case=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/row['id'] if variant=='old_reference' else RAW/variant/row['id']
        lo=160 if variant=='old_reference' else p['variants'][variant]['input_start']
        items=[(t,f) for t,f in frames(case) if lo-1e-7<=t<=lo+41+1e-7][:164]
        assert len(items)==164
        ts=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
        initial=np.stack([sampler.sample(f) for _,f in items[:4]])
        supports[row['id']]={'geo':geo,'items':items,'times':ts,'case':case}
        for name,model in candidates.items():
            pred,elapsed=model.predict(q,initial,ts,case)
            file=out/'predictions'/f'{name}__{row["id"]}.npz'
            np.savez_compressed(file,prediction=pred)
            registry.append({'id':row['id'],'model':name,'file':str(file),'sha256':sha(file),
                             'initial_frame_files':[str(f) for _,f in items[:4]],
                             'initial_frame_hashes':[sha(f) for _,f in items[:4]],
                             'startup_times':ts[:4].tolist(),'network_and_extension_seconds':elapsed})
    save(out/'predictions_frozen.json',{'utc':now(),'rows':registry,
                                      'all_saved_before_corresponding_future_field_reads':True})
    rows=[]
    for cid,d in supports.items():
        g=d['geo'];m=g['mask'][None,None];sampler=NativeSampler(g['xx'],g['yy'],g['mask'])
        target=np.stack([sampler.sample(f) for _,f in d['items'][4:164]]).astype(np.float64)*m
        assert np.isfinite(target).all()
        for entry in [r for r in registry if r['id']==cid]:
            assert sha(entry['file'])==entry['sha256']
            pred=np.load(entry['file'])['prediction'].astype(np.float64)
            error=np.linalg.norm((pred-target)*m,axis=(-2,-1))/np.maximum(np.linalg.norm(target,axis=(-2,-1)),1e-8)
            hs={f'{sec}s':frame_summary(error[:n],d['times'][3],d['times'][n+3],sec,True)
                for sec,n in [(2,8),(8,32),(20,80),(40,160)]}
            rows.append({'id':cid,'model':entry['model'],'horizons':hs})
    summary={name:{h:summarize([r['horizons'][h] for r in rows if r['model']==name])
                    for h in ['2s','8s','20s','40s']} for name in p['models']}
    report={'variant':variant,'cases':rows,'summary':summary,'only_two_seen_geometry_anchors':True,
            'frozen_model_transfer_diagnostic_not_new_independent_confirmation':True,
            'no_grid_convergence_or_experimental_validation_claim':True}
    save(out/'report.json',report)
    print('TRUE2D_MODELS_DONE',variant,
          [(n,round(v['40s']['mean_R'],4),v['40s']['n_pass']) for n,v in summary.items()],flush=True)
    del candidates
    torch.cuda.empty_cache()
    return report

def analyze(p, completed_variants):
    folder=OUT/'field_statistics';folder.mkdir(exist_ok=True);stat.OUT=folder
    allstats={};comparisons=[]
    for row in p['cases']:
        cid=row['id'];bank={}
        old=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/cid
        bank['old_reference']=stat.case_stats(old,row,CFG,[171,201])
        for variant in completed_variants:
            bank[variant]=stat.case_stats(RAW/variant/cid,row,CFG,p['variants'][variant]['score_window'])
        allstats[cid]={k:v[0] for k,v in bank.items()}
        contrasts={'repair_at_same_coarse_dt':stat.difference(bank['old_reference'],bank['coarse_dt002'])}
        if 'fine_dt0005' in bank:
            contrasts['joint_space_time_refinement']=stat.difference(bank['coarse_dt002'],bank['fine_dt0005'])
        comparisons.append({'id':cid,**contrasts})
    quality={cid:{v:all(s['last_cycle_relative_change'][k]<limit for k,limit in p['cycle_quality_gate'].items())
                  for v,s in bank.items()} for cid,bank in allstats.items()}
    report={'statistics':allstats,'comparisons':comparisons,'late_cycle_quality_gate':quality,
            'phase_invariant_statistics_only':True,'not_formal_grid_convergence':True}
    save(OUT/'reference_statistics.json',report)
    print('TRUE2D_STATISTICS_DONE',quality,flush=True)
    return report

def run():
    p=json.loads((OUT/'preregister.json').read_text());completed=[];errors=[];evaluations={}
    save(OUT/'status.json',{'status':'running','completed':[],'errors':[]})
    # Old scores are reproduced with identical current code, never used to select a model.
    evaluations['old_reference']=evaluate('old_reference',p)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(solve,'coarse_dt002',row,p):('coarse_dt002',row) for row in p['cases']}
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for f in done:
                variant,row=pending.pop(f)
                try:
                    completed.append(f.result())
                    if variant=='coarse_dt002':
                        pending[pool.submit(solve,'fine_dt0005',row,p)]=('fine_dt0005',row)
                except Exception as e:
                    errors.append({'variant':variant,'id':row['id'],'error':repr(e)})
            for variant in p['variants']:
                if variant not in evaluations and sum(r['variant']==variant for r in completed)==len(p['cases']):
                    try:
                        evaluations[variant]=evaluate(variant,p)
                        if variant=='coarse_dt002':analyze(p,[variant])
                    except Exception as e:
                        errors.append({'stage':'evaluate','variant':variant,'error':repr(e)})
                        evaluations[variant]={'failed':True}
            save(OUT/'status.json',{'status':'running','utc':now(),'completed':completed,'errors':errors,
                                  'evaluated_variants':list(evaluations),'pending_jobs':len(pending)})
    if not errors:
        analyze(p,list(p['variants']))
        for relative,digest in p['frozen_sha256'].items():assert sha(ROOT/relative)==digest
    save(OUT/'report.json',{'utc':now(),'evaluations':evaluations,'completed':completed,'errors':errors,
                           'scope':'two known anchors; no training or fresh geometry confirmation',
                           'frozen_weights_and_configuration_unchanged':not errors})
    save(OUT/'status.json',{'status':'complete' if not errors else 'failed','utc':now(),
                          'completed':completed,'errors':errors,'evaluated_variants':list(evaluations)})
    print('TRUE2D_RECHECK_FINISHED',len(completed),len(errors),flush=True)

if __name__=='__main__':
    prepare() if '--prepare' in sys.argv else run()
