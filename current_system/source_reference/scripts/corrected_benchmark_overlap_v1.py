#!/usr/bin/env python3
"""Start training after immutable train80+val8 are complete, overlap CFD test32."""
from pathlib import Path
import sys,json,time,subprocess,traceback
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
OUT=ROOT/'outputs/corrected_benchmark_restart_v1'
CACHE=ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
TRAINVAL=ROOT/'data/processed/dataset_corrected_trainval_v1.h5'

def run():
    p=json.loads((OUT/'preregister.json').read_text());required=[r for r in p['cases'] if r['role'] in ['train','val']]
    assert len(required)==88
    save(OUT/'overlap_status.json',{'status':'waiting_for_train80_val8','required_cases':88,'test_used_for_training':False})
    while True:
        status=json.loads((OUT/'status.json').read_text())
        if status['status'] in ['pipeline_failed','data_failed','reference_gate_failed']:
            raise RuntimeError('CFD producer did not pass prerequisite: '+status['status'])
        ready=[r for r in required if (CACHE/(r['id']+'.json')).exists() and (CACHE/(r['id']+'.npz')).exists()]
        save(OUT/'overlap_status.json',{'status':'waiting_for_train80_val8','ready':len(ready),'required_cases':88,'test_used_for_training':False})
        if len(ready)==88:break
        time.sleep(20)
    tmp=TRAINVAL.with_suffix('.partial.h5');assert not tmp.exists() and not TRAINVAL.exists()
    with h5py.File(tmp,'x') as h:
        h.attrs['protocol_sha256']=sha(OUT/'preregister.json');h.attrs['sample_dt']=.25;h.attrs['single_layer']=True
        groups=h.create_group('cases')
        for r in required:
            meta=json.loads((CACHE/(r['id']+'.json')).read_text());assert meta['CFD_stable'] and sha(CACHE/(r['id']+'.npz'))==meta['cache_sha256']
            with np.load(CACHE/(r['id']+'.npz')) as a:
                assert a['fields'].shape==(165,3,64,64) and abs(a['times'][163]-a['times'][3]-40)<1e-8
                g=groups.create_group(r['id'])
                for k in ['fields','times','geom','mask','forces']:g.create_dataset(k,data=a[k],compression='gzip',compression_opts=1)
                for k in ['shape','Re','cx','cy','D']:g.attrs[k]=r[k]
                g.attrs['role']=r['role'];g.attrs['St']=meta['St']
        groups=h.create_group('splits')
        for role in ['train','val']:groups.create_dataset(role,data=np.array(p['splits'][role],dtype='S'))
    tmp.replace(TRAINVAL)
    save(OUT/'dataset_trainval_manifest.json',{'path':str(TRAINVAL),'sha256':sha(TRAINVAL),'train':80,'val':8,'test':0,'confirmation':0,'immutable_after_sealing':True,'CFD_source_cache_hashes_verified':True})
    save(OUT/'overlap_status.json',{'status':'training_started_with_CFD_overlap','train':80,'val':8,'test':0})
    with (OUT/'training_overlap.log').open('x') as f:
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/corrected_benchmark_train_v1.py'),'--all'],stdout=f,stderr=subprocess.STDOUT)
        rc=proc.wait()
    if rc:raise RuntimeError('Training process failed: '+str(rc))
    save(OUT/'overlap_status.json',{'status':'training_complete','model_locked':True,'test_used_for_training':False})
    print('OVERLAPPED_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    try:run()
    except BaseException as e:
        save(OUT/'overlap_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
