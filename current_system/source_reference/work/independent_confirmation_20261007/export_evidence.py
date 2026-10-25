"""Finite completion/export stage; no scheduler, experiment, training or selection."""
from pathlib import Path
import os,sys,time,json,csv,hashlib,zipfile,shutil,math,traceback
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import campaign as c
DEST=c.OUT/'data_evidence'

def wait_complete():
    deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        s=json.loads((c.OUT/'overall_status.json').read_text())
        if s['stage']=='failure' or s['stage']=='CFD_failure':raise RuntimeError('Finite campaign failed; preserve data: '+str(s))
        h=c.OUT/'historical16/failure.json'
        if h.exists():raise RuntimeError('Historical regression failed: '+h.read_text())
        if s['stage']=='complete' and (c.OUT/'historical16/report.json').exists():return
        c.save(c.OUT/'export_status.json',dict(stage='waiting_existing_finite_jobs',utc=c.now(),campaign_stage=s['stage']))
        time.sleep(30)
    raise TimeoutError('Finite export exceeded 8h; no new job launched')

def validate(rows,summary):
    for horizon in ['8s','40s']:
        s=summary[horizon];v=[z['horizons'][horizon] for z in rows];r=[z['R'] for z in v]
        assert len(v)==s['n'] and sum(x<=1 for x in r)==s['n_pass']
        assert math.isclose(sum(r)/len(r),s['mean_R'],rel_tol=1e-6) and math.isclose(max(r),s['max_R'],rel_tol=1e-6)
        for z in v:
            computed=max(a/b for a,b in zip(z['mean']+z['end'],[.05,.05,.08]*2))
            assert math.isclose(computed,z['R'],rel_tol=1e-6)

def copy(path,relative):
    dest=DEST/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)

def export():
    cfg=c.check();extra=json.loads((c.OUT/'supplemental_input_lock.json').read_text())
    for p,d in extra['files'].items():assert c.sha(c.ROOT/p)==d
    clean=json.loads((c.OUT/'clean_report.json').read_text())['reports']
    noise=json.loads((c.OUT/'noise_report.json').read_text())
    history=json.loads((c.OUT/'historical16/report.json').read_text())['reports']
    data=json.loads((c.OUT/'dataset_manifest.json').read_text())
    receipt=json.loads((c.OUT/'verification_receipt.json').read_text())
    assert len(data['quality'])==8 and receipt['replays']==8 and receipt['all_future_poison_tests_passed']
    for quality in data['quality']:assert quality['mesh_pass'] and quality['single_layer'] and quality['cycles_pass'] and quality['max_Courant']<1 and quality['frames']==165
    for method,report in clean.items():validate(report['cases'],report['summary']['all']);assert len(report['cases'])==8
    for method,report in history.items():validate(report['cases'],report['summary']['all']);assert len(report['cases'])==16
    for mode,seeds in noise.items():
        assert len(seeds)==5
        for seed,report in seeds.items():validate(report['cases'],report['summary']['all']);assert len(report['cases'])==8
    c.save(c.OUT/'export_status.json',dict(stage='exporting',utc=c.now()))
    DEST.mkdir(exist_ok=True)
    for name in ['config.json','system_locked.json','training_identity.json','excluded_case_inventory.json','supplemental_input_lock.json','environment.json','preflight.json','dataset_manifest.json','clean_report.json','noise_report.json','verification_receipt.json','final_result.json','historical_identity_input.json']:
        copy(c.OUT/name,Path('records')/name)
    copy(c.OUT/'historical16/report.json','records/historical16_report.json')
    for sub in ['predictions','registry','historical16/predictions','historical16/registry']:
        for p in (c.OUT/sub).rglob('*'):
            if p.is_file():copy(p,p.relative_to(c.OUT))
    for p in Path(__file__).parent.glob('*.py'):copy(p,Path('code')/p.name)
    for p in cfg['source_files']:copy(c.ROOT/p,Path('fixed_source')/p)
    routes=[]
    for q in cfg['cases']:
        cid=q['id'];row=c.startup(q);v=c.packets(q,row)
        case=c.RAW/'cases'/cid
        for name in ['true2d_build.json','stability_history.json','dataset_source.json','initialization.json']:
            copy(case/name,Path('CFD_quality')/cid/name)
        for p in case.glob('log.*'):
            if 'checkMesh' in p.name or 'icoFoam_dense' in p.name:copy(p,Path('CFD_quality')/cid/p.name)
        copy(c.CACHE/(cid+'.npz'),Path('references/fresh8')/(cid+'.npz'))
        copy(c.CACHE/(cid+'.json'),Path('references/fresh8')/(cid+'.json'))
        dest=DEST/'inputs/fresh8'/(cid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,fields=row['fields'],geom=row['geom'],mask=row['mask'],times=row['times'],measurements=v)
        for seed in cfg['noise_seeds']:
            rng=np.random.default_rng(int(hashlib.sha256((str(seed)+':'+cid).encode()).hexdigest()[:8],16))
            noisy=v+rng.normal(size=v.shape)*.01*c.e.old.p.scales()[1]
            f=row['fields']+rng.normal(size=row['fields'].shape).astype(np.float32)*.01*np.asarray(c.e.old.p.scales(),np.float32)[None,:,None,None]*row['mask'][None,None]
            np.savez_compressed(dest.with_name(cid+f'_noise{seed}.npz'),startup_noisy=f,future_noisy=noisy)
        routes.append(dict(id=cid,known_shape=q['shape'],network='original' if q['shape'] in ['cylinder','square'] else 'shared_adapted',case_id_not_used_for_routing=True))
    for item in json.loads((c.OUT/'historical16/report.json').read_text())['source']:
        source=Path(item['cache_file']);assert c.sha(source)==item['cache_sha256']
        copy(source,Path('references/historical16')/source.name)
        copy(source.with_suffix('.json'),Path('references/historical16')/source.with_suffix('.json').name)
    allrows=[]
    for role,reports in [('fresh_independent8',clean),('historical_seen16',history)]:
        for method,report in reports.items():
            for row in report['cases']:allrows.append((role,method,'clean',0,row))
    for mode,seeds in noise.items():
        for seed,report in seeds.items():
            for row in report['cases']:allrows.append(('fresh_independent8','final_system',mode,int(seed),row))
    csvpath=DEST/'all_case_metrics.csv'
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['role','method','noise','seed','case','shape','Re','cy','D','seconds','R','pass','Ux_mean','Uy_mean','p_mean','Ux_end','Uy_end','p_end'])
        for role,method,mode,seed,row in allrows:
            for h in ['8s','40s']:
                q=row['q'];v=row['horizons'][h];w.writerow([role,method,mode,seed,row['id'],q['shape'],q['Re'],q['cy'],q['D'],h[:-1],v['R'],v['R']<=1,*v['mean'],*v['end']])
    stats={}
    for mode,seeds in noise.items():
        values=[r['summary']['all']['40s'] for r in seeds.values()]
        stats[mode]=dict(seeds=5,pass_range=[min(r['n_pass'] for r in values),max(r['n_pass'] for r in values)],
                         mean_R_average=float(np.mean([r['mean_R'] for r in values])),mean_R_range=[min(r['mean_R'] for r in values),max(r['mean_R'] for r in values)],worst_R_across_realizations=max(r['max_R'] for r in values))
    c.save(DEST/'summary.json',dict(primary=clean['final_system']['summary'],controls={m:r['summary'] for m,r in clean.items() if m!='final_system'},historical16=history['final_system']['summary'],noise=stats,scope='New parameters on all8 known contours; 8 cases only; historical16 separate seen regression; no candidate reselection',routes=routes,CSV_rows=len(allrows)*2))
    (DEST/'README.md').write_text('''# Fixed-system confirmation data

This data package verifies reference quality, per-case metrics, inputs and prediction records. It is not the complete model deployment package.

- fresh8: 8 parameter cases selected before target CFD, one for each supported fixed contour; no system reselection.
- historical16: omitted older batches1/2 evaluated with the current system; seen regression, not fresh confirmation.
- controls: same4 startup fields and76 future Uy scalars. No-residual retains neural bootstrap/clock; training-template control uses nearest TRAIN-only true harmonic template and startup/clock calibration. One concrete template implementation, not all ROMs.
- noise: 5 fixed Gaussian realizations per case for future-only1% and startup+future1%; strength uses TRAIN channel scales.
- inputs contain exactly4 fields and76 finite future scalars; future full fields are in references for scoring only.
- predictions and registries precede full-field scoring. All cases and failures are retained; no pass-rate promotion threshold.
- metric R: max of channel mean and last-frame relativeL2 divided by[.05,.05,.08]; each relativeL2 denominator is the reference fluid-field norm. R<=1 is an internal target.
- Run verify_metrics.py with Python+NumPy to reproduce reported R from reference/prediction arrays without a GPU or model.
''',encoding='utf-8')
    verifier=Path(__file__).parent/'verify_metrics.py';copy(verifier,'verify_metrics.py')
    copy(Path(__file__).parent/'verify_CFD_quality.py','verify_CFD_quality.py')
    c.save(c.OUT/'data_export_receipt.json',dict(utc=c.now(),status='EXPORTED',fresh_CFD=8,independent_clean_predictions=24,noise_predictions=80,historical_regression_predictions=48,CSV_rows=len(allrows)*2,dependency_check_pass=True))
    return DEST

def finalize():
    import subprocess
    subprocess.run([sys.executable,str(DEST/'verify_metrics.py'),str(DEST)],check=True)
    subprocess.run([sys.executable,str(DEST/'verify_CFD_quality.py'),str(DEST)],check=True)
    items=[]
    for p in sorted(DEST.rglob('*')):
        if p.is_file() and p.name!='manifest.json':items.append(dict(file=p.relative_to(DEST).as_posix(),bytes=p.stat().st_size,sha256=c.sha(p)))
    c.save(DEST/'manifest.json',dict(files=items,not_complete_model_deployment=True,utc=c.now()))
    archive=c.OUT/'AIC_fixed_system_confirmation_data_20261007.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,Path(DEST.name)/p.relative_to(DEST))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    c.save(c.OUT/'archive_receipt.json',dict(file=str(archive),bytes=archive.stat().st_size,sha256=c.sha(archive),manifest_files=len(items),CRC_pass=True,utc=c.now()))
    c.save(c.OUT/'export_status.json',dict(stage='complete',utc=c.now(),archive=str(archive),summary=json.loads((DEST/'summary.json').read_text())))
    print('DATA_EXPORT_VERIFIED',flush=True)

if __name__=='__main__':
    try:wait_complete();export();finalize()
    except Exception as exc:
        c.save(c.OUT/'export_failure.json',dict(utc=c.now(),error=repr(exc),traceback=traceback.format_exc()))
        c.save(c.OUT/'export_status.json',dict(stage='failure',error=repr(exc),utc=c.now()));raise
