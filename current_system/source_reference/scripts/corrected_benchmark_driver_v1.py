#!/usr/bin/env python3
"""Supervise the authorized restart and leave explicit status on any failure."""
from pathlib import Path
import sys,json,traceback,os
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
import corrected_benchmark_restart_v1 as restart
OUT=restart.OUT
def main():
    lock=json.loads((OUT/'implementation_lock.json').read_text())
    for path,digest in lock['code_sha256'].items():assert sha(ROOT/path)==digest,(path,'implementation changed after freeze')
    save(OUT/'driver.json',{'pid':os.getpid(),'started_utc':restart.run.now(),'command':str(Path(__file__).resolve()),'implementation_lock_sha256':sha(OUT/'implementation_lock.json')})
    restart.execute()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        prior=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
        save(OUT/'failure.json',{'utc':restart.run.now(),'error':repr(e),'traceback':traceback.format_exc(),'prior_status':prior,'original_artifacts_preserved':True})
        save(OUT/'status.json',{'status':'pipeline_failed','error':repr(e),'prior_phase':prior.get('status'),'completed':prior.get('completed',[]),'errors':prior.get('errors',[])})
        raise
