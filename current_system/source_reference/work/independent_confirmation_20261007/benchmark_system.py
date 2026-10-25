"""Bounded full-system timing on seen cases; never selection or fresh8 scoring."""
from pathlib import Path
import os,sys,json,time,resource
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(Path(__file__).parent))
import campaign as c

def run():
    cfg=c.check();c.e.setup();torch=c.e.torch
    torch.cuda.synchronize();begin=time.perf_counter();model=c.Predictor(cfg['system_path']);torch.cuda.synchronize();load_s=time.perf_counter()-begin
    oldqs=c.e.old.queries();newqs=json.loads((c.e.OUT/'run_config.json').read_text())['design']['evaluation'];results=[]
    for shape in ['cylinder','square',*c.e.shapes.SPECS]:
        if shape in ['cylinder','square']:
            q=next(q for q in oldqs if q['shape']==shape);row=c.e.old.p.startup(q);values=c.e.old.packets(q,row)[0]
        else:
            q=next(q for q in newqs if q['shape']==shape);row=c.e.startup(q);values=c.e.packets(q,row)
        times=[]
        for repeat in range(3):
            torch.cuda.synchronize();start=time.perf_counter();pred,di=model.predict(row,values);torch.cuda.synchronize();times.append(time.perf_counter()-start)
            assert pred.shape==(160,3,64,64) and c.np.isfinite(pred).all()
        results.append(dict(id=q['id'],shape=shape,seconds=times,median_seconds=float(c.np.median(times)),input_budget=76,role='seen timing only'))
    selected=json.loads((c.OUT/'system_locked.json').read_text());files={Path(selected['original_shapes']['checkpoint']),Path(selected['new_shapes']['checkpoint'])}
    for bankset in [selected['original_shapes']['banks'],selected['new_shapes']['old_banks'],selected['new_shapes']['new_banks']]:
        files.update(Path(z['path']) for z in bankset.values())
    c.check();c.save(c.OUT/'full_system_runtime.json',dict(utc=c.now(),system_sha256=cfg['system_sha256'],device=torch.cuda.get_device_name(0),CUDA_context_initialized_before_load=True,load_both_networks_and_all_current_loader_libraries_seconds=load_s,case_results=results,deployment_files=[dict(file=str(p.relative_to(c.ROOT)),bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(files)],peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,protocol='one process, eight supported contours, 3 repeats each, full160-frame inference incl bootstrap/library/startup/observer; excludes packet/reference IO, startup-field acquisition, measurement arrival and CUDA-context initialization; concurrent finite CFD running on same host; not an isolated-CFD speedup or independent accuracy experiment'))
    print('FULL_SYSTEM_TIMING_COMPLETE',load_s,flush=True)

if __name__=='__main__':
    try:run()
    except Exception as exc:c.save(c.OUT/'runtime_failure.json',dict(utc=c.now(),error=repr(exc),traceback=__import__('traceback').format_exc()));raise
