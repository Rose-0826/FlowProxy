from pathlib import Path
import sys, os, json, time, resource, hashlib
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
sys.path.insert(0, '/root/autodl-tmp/flowproxy/work/targeted_coverage_20261007')
import optimize as c
import numpy as np
OUT=c.OUT/'material_evidence'
OUT.mkdir(exist_ok=True)
cfg=c.check();c.e.setup();torch=c.e.torch
system=c.OUT/'selected_system.json'
torch.cuda.synchronize();begin=time.perf_counter()
model=c.previous.Predictor(system)
torch.cuda.synchronize();load_s=time.perf_counter()-begin
timing=[]
for shape in ['cylinder','square',*c.e.shapes.SPECS]:
    q=next(q for q in cfg['development'] if q['shape']==shape)
    row=c.startup(q);values=c.packets(q,row);seconds=[]
    for _ in range(3):
        torch.cuda.synchronize();start=time.perf_counter()
        pred,di=model.predict(row,values)
        torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
        assert pred.shape==(160,3,64,64) and np.isfinite(pred).all()
    timing.append(dict(id=q['id'],shape=shape,seconds=seconds,median_seconds=float(np.median(seconds)),input_budget=76,role='seen timing only'))
selected=json.loads(system.read_text());files={Path(selected['original_shapes']['checkpoint']),Path(selected['new_shapes']['checkpoint'])}
for banks in [selected['original_shapes']['banks'],selected['new_shapes']['old_banks'],selected['new_shapes']['new_banks']]:
    files.update(Path(z['path']) for z in banks.values())
c.save(OUT/'current_runtime.json',dict(utc=c.now(),system_sha256=c.sha(system),device=torch.cuda.get_device_name(0),CUDA_context_initialized_before_load=True,load_seconds=load_s,case_results=timing,deployment_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(files)],peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,protocol='Single process; eight supported contours; 3 repeats each; full160-frame inference incl bootstrap/library/startup/observer; excludes file IO, startup-field acquisition, measurement arrival and CUDA context initialization. No warm-up discarded. No isolated CFD speedup claim.'))
arrays={};records=[]
for i,qi in enumerate([0,6]):
    q=cfg['fresh_confirmation'][qi];q=dict(q,cache=str(c.CACHE));row=c.startup(q)
    with np.load(c.CACHE/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
    arrays[f'case{i}_truth']=truth[[31,79,159]];arrays[f'case{i}_mask']=row['mask']
    rec=dict(id=q['id'],q=q,seconds=[8,20,40],reference_sha256=c.sha(c.CACHE/(q['id']+'.npz')),registrations={})
    for method in ['parent','selected']:
        folder=c.OUT/'fresh'/method/'clean';target=folder/'predictions'/(q['id']+'.npz')
        reg=json.loads((folder/'registry'/(q['id']+'.json')).read_text())
        assert c.sha(target)==reg['file_sha256']
        with np.load(target) as z:p=z['prediction']
        assert hashlib.sha256(p.tobytes()).hexdigest()==reg['prediction_sha256']
        arrays[f'case{i}_{method}']=p[[31,79,159]];rec['registrations'][method]=reg
    records.append(rec)
np.savez_compressed(OUT/'current_field_snapshots.npz',**arrays)
c.save(OUT/'current_field_manifest.json',dict(cases=records,system_sha256=c.sha(system),export_only=True,training_or_CFD_started=False,field_npz_sha256=c.sha(OUT/'current_field_snapshots.npz'),coordinates=dict(x=[-8,20],y=[-8,8],grid=[64,64])))
c.check()
print('CURRENT_MATERIAL_EVIDENCE_READY',load_s,[r['median_seconds'] for r in timing],flush=True)
