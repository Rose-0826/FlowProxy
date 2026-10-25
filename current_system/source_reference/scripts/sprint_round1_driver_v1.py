#!/usr/bin/env python3
"""Finite dependency scheduler for this one authorized experiment round."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from geometry_attack_v2 import save,sha
from cycle_landmarks_train_v1 import FORMAL,FORMAL_SHA
OUT=ROOT/'outputs/sprint_round1_v1'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def outcome(folder):
    if (folder/'failure.json').exists():return 'failed'
    if (folder/'completed_report.json').exists() or (folder/'training_manifest.json').exists():return 'complete'
    return 'running'

def worker_alive(folder):
    file=folder/'status.json'
    if not file.exists():return False
    s=json.loads(file.read_text())
    for worker in s.get('workers',[]):
        cmd=Path('/proc')/str(worker['pid'])/'cmdline'
        if cmd.exists() and b'cycle_landmarks_train_v1.py' in cmd.read_bytes():return True
    return False

def launch(label,script,args):
    log=OUT/(label+'.log')
    with log.open('x') as f:
        p=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=ENV)
    save(OUT/(label+'_launch.json'),{'utc':now(),'pid':p.pid,'script':script,'args':args,'log':str(log)})
    print('SPRINT_JOB_START',label,p.pid,flush=True)
    return p

def run():
    assert (OUT/'launch.json').exists() and sha(FORMAL)==FORMAL_SHA
    save(OUT/'driver.json',{'pid':os.getpid(),'utc':now(),'script_sha256':sha(__file__),
        'bounded_plan':'initial4 landmark/control trainings +16 TRAINING CFD; after dependencies4 original80/augmented96 trainings; development48 only for suites passing their preregistered gates',
        'no_final_blind_CFD_yet':True,'never_replace_formal_model_automatically':True})
    folders={'landmarks':ROOT/'outputs/cycle_landmarks_train_v1','CFD16':ROOT/'outputs/continuous_training_cfd_v1','data':ROOT/'outputs/continuous_data_train_v1'}
    started_data=False;checked_landmark=False;checked_data=False;extra_processes={};deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        landmark=outcome(folders['landmarks']);cfd=outcome(folders['CFD16'])
        if landmark!='running' and not worker_alive(folders['landmarks']) and not checked_landmark:
            checked_landmark=True
            if landmark=='complete' and json.loads((folders['landmarks']/'completed_report.json').read_text())['gate_passed']:
                extra_processes['landmark_development']=launch('landmark_development','cycle_sprint_development_v1.py',['--suite','landmarks'])
        development_running=any(p.poll() is None for name,p in extra_processes.items() if name.endswith('development'))
        if cfd=='complete' and checked_landmark and not worker_alive(folders['landmarks']) and not development_running and not started_data:
            extra_processes['data_training']=launch('data_training','cycle_landmarks_train_v1.py',['--suite','data'])
            started_data=True
        data=outcome(folders['data']) if started_data else 'waiting_for_dependencies'
        if started_data and data!='running' and not worker_alive(folders['data']) and not checked_data:
            checked_data=True
            if data=='complete' and json.loads((folders['data']/'completed_report.json').read_text())['gate_passed']:
                extra_processes['data_development']=launch('data_development','cycle_sprint_development_v1.py',['--suite','data'])
        jobs={name:('running' if proc.poll() is None else {'exit_code':proc.returncode}) for name,proc in extra_processes.items()}
        save(OUT/'status.json',{'status':'running','utc':now(),'landmarks':landmark,'training_CFD16':cfd,'data_training':data,'extra_jobs':jobs,
            'formal_model_unchanged':True,'final_independent_confirmation_not_started':True})
        if checked_landmark and cfd=='failed' and not development_running:break
        if checked_landmark and cfd=='complete' and checked_data and all(p.poll() is not None for p in extra_processes.values()):break
        time.sleep(20)
    else:raise TimeoutError('Finite round exceeded24h; preserving jobs/data and recording attention needed')
    assert sha(FORMAL)==FORMAL_SHA
    state=json.loads((OUT/'status.json').read_text());state.update(status='complete',utc=now())
    state['completed_with_failures']=any(state[k]=='failed' for k in ['landmarks','training_CFD16','data_training']) or any(p.returncode not in [None,0] for p in extra_processes.values())
    save(OUT/'completed_report.json',state);save(OUT/'status.json',state)
    print('SPRINT_ROUND_FINISHED',state['completed_with_failures'],flush=True)

if __name__=='__main__':
    try:run()
    except Exception as error:
        save(OUT/'driver_failure.json',{'utc':now(),'error':repr(error),'formal_model_preserved':True})
        raise
