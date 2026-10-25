#!/usr/bin/env python3
"""Recover an auxiliary-VTK archival naming incident without repeating successful CFD."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip,json,os,re,subprocess,sys,threading,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,protected,now
from research_v2_outer_support_cfd import DEST,RAW,CACHE,initialize as original_initialize
from geometry_attack_v2 import save,sha
import corrected_benchmark_restart_v1 as rebuild

INC=OUT/'storage_incident';OLD_OUTER_PID=529609
FRESH_RAW=ROOT/'data/raw/research_v2_independent8';STOP=threading.Event();THREAD_ERRORS=[]

def running(pid,needle):
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    if (p/'stat').read_text().split()[2]=='Z':return False
    return needle.encode() in (p/'cmdline').read_bytes()

def repair_seed_files(root):
    changed=[]
    if not root.exists():return changed
    for case in (root/'cases').glob('*'):
        if not (case/'initialization.json').exists():continue
        p=case/'training_donor_seed.vtu'
        if not p.exists():continue
        assert case.resolve().is_relative_to(root.resolve())
        target=case/'training_donor_seed.xml';assert not target.exists()
        digest=sha(p);p.rename(target);assert sha(target)==digest
        changed.append({'old':str(p),'new':str(target),'sha256':digest,'utc':now(),
            'reason':'auxiliary original VTK bytes are XML; neutral extension excludes them from case/VTK-only dedup pass'})
    return changed

def keep_old_fresh_batch_safe():
    events=[];start=time.perf_counter()
    while not STOP.is_set():
        events.extend(repair_seed_files(FRESH_RAW))
        save(INC/'independent_batch_auxiliary_repairs.json',events)
        complete=OUT/'independent8_cfd/SEALED_reference_manifest.json'
        fail=OUT/'independent8_cfd/failure.json'
        if complete.exists():return
        if fail.exists():raise RuntimeError('Independent batch failed during incident recovery')
        if time.perf_counter()-start>14400:raise RuntimeError('Finite auxiliary repair limit')
        STOP.wait(10)

def guarded_fresh_repair():
    try:keep_old_fresh_batch_safe()
    except Exception as e:
        THREAD_ERRORS.append(repr(e))
        save(INC/'independent_auxiliary_repair_failure.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})

def initialize(case,row,p):
    original_initialize(case,row,p)
    seed=case/'training_donor_seed.vtu';target=case/'training_donor_seed.xml'
    assert seed.exists() and not target.exists()
    digest=sha(seed);seed.rename(target);assert sha(target)==digest

def recover_completed_case(row):
    case=RAW/'cases'/row['id'];source=json.loads((case/'dataset_source.json').read_text())
    assert source['quality']['passed'] and source['case']['id']==row['id']
    npz=CACHE/(row['id']+'.npz');assert sha(npz)==source['cache_sha256']
    with np.load(npz) as a:
        assert a['fields'].shape==(165,3,64,64) and np.isfinite(a['fields']).all()
        assert np.allclose(np.diff(a['times']),.25,atol=1e-8)
        assert abs(a['times'][163]-a['times'][3]-40)<1e-8
    native_start=int(source['window_start']);final=native_start+41
    dense=case/'log.icoFoam_dense'
    text=dense.read_text() if dense.exists() else gzip.open(Path(str(dense)+'.gz'),'rt').read()
    stamps=re.findall(r'^Time = (\S+)',text,re.MULTILINE)
    assert stamps and abs(float(stamps[-1])-final)<1e-7 and re.search(r'^End\s*$',text,re.MULTILINE)
    courant=[float(x) for x in re.findall(r'Courant Number mean:\s+\S+\s+max:\s+(\S+)',text)]
    assert courant and max(courant)<1
    repair_seed_files(RAW)
    n=rebuild.compress_generated_raw(case)
    result={'id':row['id'],'role':row['role'],'wall_seconds':None,
        'wall_seconds_note':'original completion bookkeeping failed after sampling; runtime not reconstructed or invented',
        'dense_start':native_start,'CFD_stable':True,'max_Courant_dense':max(courant),
        'uniform_frames':165,'St':source['quality']['St'],'cache_sha256':sha(npz),
        'losslessly_compressed_files':n,'storage_recovered_without_CFD_rerun':True}
    save(CACHE/(row['id']+'.json'),result);return result

def restore_outer_batch():
    # Wait until its one already-started extra solve has finished; never duplicate it.
    start=time.perf_counter();events=[]
    while running(OLD_OUTER_PID,'research_v2_outer_support_cfd.py'):
        events.extend(repair_seed_files(RAW));save(INC/'outer_auxiliary_repairs.json',events)
        if time.perf_counter()-start>10800:raise RuntimeError('Original outer batch did not finish')
        time.sleep(10)
    old_failure=DEST/'failure.json'
    if old_failure.exists():
        target=INC/'original_outer_failure.json';assert not target.exists();old_failure.rename(target)
    events.extend(repair_seed_files(RAW));save(INC/'outer_auxiliary_repairs.json',events)
    p=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    c=json.loads((DEST/'preregister.json').read_text());cases=c['cases']
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.initialize=initialize
    completed=[];pending=[]
    for row in cases:
        result=CACHE/(row['id']+'.json')
        if result.exists():
            z=json.loads(result.read_text());assert sha(CACHE/(row['id']+'.npz'))==z['cache_sha256'];completed.append(z)
        elif (RAW/'cases'/row['id']/'dataset_source.json').exists():
            completed.append(recover_completed_case(row))
        else:
            assert not (RAW/'cases'/row['id']).exists(),('unfinished case requires explicit recovery',row['id'])
            pending.append(row)
    errors=[];save(DEST/'status.json',{'utc':now(),'status':'running_recovered','total':16,'completed':completed,'errors':[],
        'storage_incident':str(INC),'physics_and_sampling_unchanged':True})
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(rebuild.one,row,p):row['id'] for row in pending}
        for f in as_completed(jobs):
            if f.cancelled():continue
            try:completed.append(f.result())
            except Exception as e:
                errors.append({'id':jobs[f],'error':repr(e),'traceback':traceback.format_exc()})
                for other in jobs:other.cancel()
            protected();save(DEST/'status.json',{'utc':now(),'status':'running_recovered' if not errors else 'failed',
                'total':16,'completed':completed,'errors':errors,'storage_incident':str(INC)})
    if errors:raise RuntimeError(errors)
    assert len(completed)==16
    save(DEST/'training_manifest.json',{'cases':cases,'results':completed,'data_role':'TRAINING ONLY',
        'cache':str(CACHE),'registered_sha256':sha(DEST/'preregister.json'),'utc':now(),
        'storage_recovery_source_sha256':sha(__file__),'CFD_and_sampling_not_changed':True})
    save(DEST/'status.json',{'utc':now(),'status':'complete','total':16,'completed':completed,'errors':[]})

def restart_dependent_jobs():
    # Previous jobs failed before creating training or selecting any model.
    for name in ['outer_support_training_failure.json']:
        p=OUT/name
        if p.exists():
            target=INC/name;assert not target.exists();p.rename(target)
    folder=OUT/'final_research_confirmation'
    if folder.exists():
        assert not (folder/'selected_research_lock.json').exists()
        assert not list((folder/'predictions').glob('*'))
        target=INC/'failed_confirmation_setup';assert not target.exists();folder.rename(target)
    assert not (OUT/'outer_support_training').exists()
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    jobs=[]
    for name in ['research_v2_outer_support_train.py','research_v2_final_confirmation.py']:
        log=(INC/(name+'.restart.log')).open('x')
        p=subprocess.Popen([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        log.close();jobs.append({'script':name,'pid':p.pid})
    save(INC/'dependent_jobs_restarted.json',{'utc':now(),'jobs':jobs})

def run():
    protected();INC.mkdir(exist_ok=False)
    save(INC/'incident.json',{'utc':now(),'cause':'new auxiliary training_donor_seed.vtu outside case/VTK triggered encode_vtk path assertion during archival',
        'CFD_data_integrity':'native solve, stability gates, sampling and cache checks complete before archival; reuse verified outputs',
        'repair':'preserve bytes, rename auxiliary XML to .xml; no solver, metric, model, pressure or split change',
        'old_outer_status':json.loads((DEST/'status.json').read_text()),'source_sha256':sha(__file__)})
    thread=threading.Thread(target=guarded_fresh_repair,daemon=True);thread.start()
    restore_outer_batch();restart_dependent_jobs()
    thread.join(timeout=14400);assert not thread.is_alive()
    assert not THREAD_ERRORS,THREAD_ERRORS
    protected();save(INC/'status.json',{'utc':now(),'status':'recovered','CFD_rerun_for_first_completed_cases':False,'fallback_unchanged':True})

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(INC/'failure.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});STOP.set();raise
