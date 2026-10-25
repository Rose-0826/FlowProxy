"""Finite coverage/rollout campaign. No scheduler and no additional expansion."""
from pathlib import Path
import os, sys, json, time, copy, hashlib, shutil, traceback, fcntl, itertools, subprocess, csv, tarfile
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
ROOT = Path('/root/autodl-tmp/flowproxy')
WORK = ROOT/'work/targeted_coverage_20261007'
OUT = ROOT/'outputs/targeted_coverage_20261007'
RAW = ROOT/'data/raw/targeted_coverage_20261007'
CACHE = ROOT/'data/processed/targeted_coverage_20261007'
OLD_CONFIRM = ROOT/'outputs/independent_confirmation_20261007'
sys.path.insert(0, str(ROOT/'work/independent_confirmation_20261007'))
import campaign as previous
e = previous.e
old = e.old
save, sha, now = previous.save, previous.sha, previous.now
TARGETS = ['ellipse_a1p4', 'ellipse_a1p8', 'rectangle_a1p3', 'rectangle_a1p7']

def config(): return json.loads((OUT/'config.json').read_text())

def check():
    cfg=config()
    for name,digest in cfg['dependencies'].items():
        assert sha(ROOT/name)==digest, 'Pinned input/source changed: '+name
    return cfg

def query(q,cache,role):
    q=copy.deepcopy(q);q['cache']=str(cache);q['role']=role
    q['family']=e.shapes.SPECS[q['shape']]['family'] if q['shape'] in e.shapes.SPECS else q['shape']
    return q

def prepare():
    assert not (OUT/'config.json').exists(), 'Do not redesign an existing campaign'
    for d in [WORK,OUT,RAW,CACHE]:d.mkdir(parents=True,exist_ok=True)
    e.setup();inventory,scanned=previous.inventory();excluded={previous.key(x['q']) for x in inventory}
    train=[]
    for shape in TARGETS:
        for a,b,c in itertools.product(range(2),repeat=3):
            Re=[105.,175.][a];cy=[.08,.48][b]*(-1 if (a+b+c)%2 else 1);D=[1.025,1.175][c]
            q=dict(id=f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}',shape=shape,Re=Re,cx=0.,cy=cy,D=D,
                   family=e.shapes.SPECS[shape]['family'],role='additional_TRAIN')
            assert previous.key(q) not in excluded;excluded.add(previous.key(q));train.append(q)
    rng=np.random.default_rng(202610071);fresh=[]
    # 8 target cases plus four checks of the other supported profiles.
    for shape in [s for s in TARGETS for _ in range(2)]+['irregular_b1','irregular_b2','cylinder','square']:
        while True:
            Re=round(float(rng.uniform(95,185)),2);cy=round(float(rng.uniform(-.55,.55)),3)
            D=round(float(rng.uniform(.82,1.20) if shape in ['cylinder','square'] else rng.uniform(1.,1.20)),3)
            q=dict(id=f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}',shape=shape,Re=Re,cx=0.,cy=cy,D=D,
                   family=e.shapes.SPECS[shape]['family'] if shape in e.shapes.SPECS else shape,
                   role='new_fixed_system_confirmation')
            if abs(cy)>.05 and previous.key(q) not in excluded:
                excluded.add(previous.key(q));fresh.append(q);break
    dev=[query(q,q['cache'],'SEEN_development') for q in old.queries()]
    expcfg=json.loads((e.OUT/'run_config.json').read_text())
    dev += [query(q,e.CACHE,'SEEN_development') for q in expcfg['design']['evaluation']]
    dev += [query(q,previous.CACHE,'SEEN_development_after_previous_confirmation') for q in json.loads((OLD_CONFIRM/'config.json').read_text())['cases']]
    hist=json.loads((OLD_CONFIRM/'historical16/report.json').read_text())
    dev += [query(x['q'],Path(x['cache_file']).parent,'SEEN_historical_regression') for x in hist['source']]
    assert len(train)==32 and len(fresh)==12 and len(dev)==108 and len({q['id'] for q in dev})==108
    assert not ({q['id'] for q in train+fresh}&{q['id'] for q in dev})
    previouscfg=json.loads((OLD_CONFIRM/'config.json').read_text());dependencies={**previouscfg['dependencies']}
    parent=ROOT/'outputs/shape_expansion_20261006/selected_system.json'
    files=[parent,e.OUT/'training_dataset.json',e.OUT/'run_config.json',OLD_CONFIRM/'config.json',OLD_CONFIRM/'final_result.json',OLD_CONFIRM/'historical16/report.json']
    files += list(WORK.glob('*.py'))
    for q in dev:files += [Path(q['cache'])/(q['id']+'.npz'),Path(q['cache'])/(q['id']+'_startup.npz')]
    for p in files:dependencies[str(p.relative_to(ROOT))]=sha(p)
    cfg=dict(created_utc=now(),seed=202610071,parent_system_path=str(parent),parent_system_sha256=sha(parent),
             additional_train=train,fresh_confirmation=fresh,development=dev,workers=6,MPI_ranks_each=2,
             actual_CPU_quota_cores=14,dependencies=dependencies,excluded_parameters=len(inventory),
             training_count=156,newshape_training_count=56,
             arms=[dict(name='coverage_only',lr=1e-5,steps=1500,rollout32=False),
                   dict(name='coverage_rollout32',lr=1e-5,steps=1500,rollout32=True)],
             checkpoints=[750,1500],noise_seeds=[7001,7002,7003,7004,7005],
             scope='Known eight fixed contours. Old 108 are now seen development; previous fixed8 first scores retained unchanged.',
             selection='Choose parent or one of four fixed candidates on seen newshape18 only. Freeze before generating new12 CFD.',
             rollout='32 self-fed steps; gradient truncated into 8-step blocks. Only newshape TRAIN labels, Uy/near-wake emphasis.',
             inputs='4 startup full fields + 76 future Uy scalars. Every field emitted before same-time scalar reading.',
             storage='Server only for models/CFD. Retain all reference cases and failures; archive verified dominated branches only.',
             finite=True,no_automation=True)
    save(OUT/'config.json',cfg);save(OUT/'parent_system.json',json.loads(parent.read_text()))
    save(OUT/'excluded_parameter_inventory.json',dict(records=inventory,metadata_files=scanned))
    save(OUT/'overall_status.json',dict(stage='prepared',utc=now()))
    print('PREPARED_TRAIN32_FRESH12',flush=True)

def generate(cases,stage):
    cfg=check();previous.RAW=RAW;previous.CACHE=CACHE
    rebuild=previous.rebuild;rebuild.RAW=RAW;rebuild.CACHE=CACHE
    rebuild.write_case=previous.write_case;rebuild.geometry_channels=e.shapes.geometry_channels;rebuild.initialize=previous.initialize
    protocol=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    done=[];errors=[]
    def one(q):
        if shutil.disk_usage(ROOT).free<40*2**30:raise RuntimeError('Disk reserve reached; preserve every case')
        meta=CACHE/(q['id']+'.json')
        if meta.exists():
            info=json.loads(meta.read_text());assert info['CFD_stable'] and info['uniform_frames']==165 and info['cache_sha256']==sha(CACHE/(q['id']+'.npz'));return info
        assert not (RAW/'cases'/q['id']).exists(), 'Partial CFD preserved; explicit recovery required: '+q['id']
        return rebuild.one(q,protocol)
    save(OUT/'overall_status.json',dict(stage=stage,completed=0,total=len(cases),utc=now()))
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs={pool.submit(one,q):q for q in cases}
        for future in as_completed(jobs):
            if future.cancelled():continue
            try:done.append(future.result())
            except Exception as exc:
                errors.append(dict(id=jobs[future]['id'],error=repr(exc)))
                for pending in jobs:pending.cancel()
            save(OUT/(stage+'_status.json'),dict(completed=done,errors=errors,total=len(cases),utc=now()))
            save(OUT/'overall_status.json',dict(stage=stage if not errors else 'CFD_failure',completed=len(done),total=len(cases),errors=errors,utc=now()))
            print(stage,len(done),len(cases),'errors',len(errors),flush=True)
    assert not errors,errors
    audit=[]
    for q in cases:
        info=json.loads((CACHE/(q['id']+'.json')).read_text());case=RAW/'cases'/q['id']
        mesh=json.loads((case/'true2d_build.json').read_text());source=json.loads((case/'dataset_source.json').read_text())
        assert mesh['exact_single_layer'] and mesh['mesh_OK'] and source['quality']['passed']
        assert info['CFD_stable'] and info['max_Courant_dense']<1 and info['uniform_frames']==165
        with np.load(CACHE/(q['id']+'.npz')) as z:
            assert z['fields'].shape==(165,3,64,64) and np.isfinite(z['fields']).all() and np.allclose(np.diff(z['times']),.25)
        audit.append(dict(q=q,cache_sha256=sha(CACHE/(q['id']+'.npz')),max_Courant=info['max_Courant_dense'],dense_start=info['dense_start'],mesh_pass=True,cycle_pass=True))
    save(OUT/(stage+'_manifest.json'),dict(cases=cases,quality=audit,results=done,utc=now()))
    return audit

def startup(q):
    with np.load(Path(q['cache'])/(q['id']+'_startup.npz')) as z:
        row={k:z[k] for k in ['fields','geom','mask','times']}
    assert row['fields'].shape==(4,3,64,64)
    return dict(row,id=q['id'],q={k:q[k] for k in ['shape','Re','cx','cy','D']})

def packets(q,row):
    probes,_=old.p.probes(row);values=np.full((160,4),np.nan)
    with np.load(Path(q['cache'])/(q['id']+'.npz')) as z:
        for step in range(8,160,8):values[step-1]=[z['fields'][step+3,1,i,j] for i,j in probes]
    assert np.isfinite(values).sum()==76;return values

def summarize(rows):
    groups=['all']+sorted({q['q']['family'] for q in rows})
    return {g:{h:old.p.summarize([r['horizons'][h] for r in rows if g=='all' or r['q']['family']==g]) for h in ['8s','40s']} for g in groups}

def evaluate(system,queries,folder,mode='clean',seed=0):
    check();e.setup();model=previous.Predictor(system);rows=[];folder=Path(folder)
    for i,q in enumerate(queries):
        row=startup(q);values=packets(q,row)
        if mode!='clean':
            rng=np.random.default_rng(int(hashlib.sha256((str(seed)+':'+q['id']).encode()).hexdigest()[:8],16))
            values=values+rng.normal(size=values.shape)*.01*old.p.scales()[1]
            if mode=='startup_and_future1pct':row['fields']=row['fields']+rng.normal(size=row['fields'].shape).astype(np.float32)*.01*np.asarray(old.p.scales(),np.float32)[None,:,None,None]*row['mask'][None,None]
        target=folder/'predictions'/(q['id']+'.npz');registry=folder/'registry'/(q['id']+'.json')
        if target.exists() and registry.exists():
            registered=json.loads(registry.read_text());assert registered['file_sha256']==sha(target)
            with np.load(target) as z:pred=z['prediction']
            assert hashlib.sha256(pred.tobytes()).hexdigest()==registered['prediction_sha256']
        else:
            pred,di=model.predict(row,values);assert pred.shape==(160,3,64,64) and np.isfinite(pred).all()
            poisoned=values.copy();poisoned[79:]+=100.;other,_=model.predict(row,poisoned)
            assert np.array_equal(pred[:80],other[:80]),'Future scalar affected earlier output'
            target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,prediction=pred)
            save(registry,dict(id=q['id'],role=q['role'],prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),file_sha256=sha(target),system_sha256=sha(system),
                              fields_registered_before_fullfield_scoring=True,startup_fields=4,future_scalar_readings=76,future_poison_test=True,mode=mode,seed=seed,diagnostic=di))
        with np.load(Path(q['cache'])/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
        rows.append(dict(id=q['id'],q=q,horizons=old.p.metrics(pred,truth,row['mask'],row['times'])))
        save(folder/'status.json',dict(completed=i+1,total=len(queries),utc=now()))
    report=dict(cases=rows,summary=summarize(rows),role=queries[0]['role'],mode=mode,seed=seed,system=str(system),system_sha256=sha(system))
    save(folder/'report.json',report);del model
    import torch;torch.cuda.empty_cache()
    return report

def choose():
    cfg=check();candidates=json.loads((OUT/'candidate_index.json').read_text())['candidates']
    newqueries=[q for q in cfg['development'] if q['shape'] in e.shapes.SPECS];assert len(newqueries)==18
    parentfile=OUT/'development/parent/report.json'
    parent=json.loads(parentfile.read_text()) if parentfile.exists() else evaluate(OUT/'parent_system.json',cfg['development'],parentfile.parent)
    parent18=[r for r in parent['cases'] if r['q']['shape'] in e.shapes.SPECS]
    reports={'parent':dict(cases=parent18,summary=summarize(parent18))}
    for candidate in candidates:
        file=OUT/'development'/candidate['name']/'report.json'
        reports[candidate['name']]=json.loads(file.read_text()) if file.exists() else evaluate(candidate['system_path'],newqueries,file.parent)
    def score(name):
        a=reports[name]['summary']['all'];s=a['40s']
        return (-s['n_pass'],s['mean_R'],s['max_R'],a['8s']['mean_R'])
    winner=min(reports,key=score)
    system=json.loads((OUT/'parent_system.json').read_text()) if winner=='parent' else json.loads(Path(next(c['system_path'] for c in candidates if c['name']==winner)).read_text())
    save(OUT/'selected_system.json',system)
    save(OUT/'development_comparison.json',dict(reports=reports,parent_all108=parent,selected=winner,all_seen=True,role='Previous independent8 consumed only as seen development, never TRAIN. Original branch90 scores inherited from unchanged parent.'))
    save(OUT/'selection_lock.json',dict(selected=winner,system_sha256=sha(OUT/'selected_system.json'),candidates=candidates,development_only=True,fresh_CFD_not_generated=not (OUT/'CFD_fresh_manifest.json').exists(),utc=now()))
    # Verify exact preservation of the two original routes with predictions registered earlier.
    model=previous.Predictor(OUT/'selected_system.json');preserved=[]
    for shape in ['cylinder','square']:
        q=next(q for q in cfg['development'] if q['shape']==shape);row=startup(q);p,_=model.predict(row,packets(q,row))
        reg=json.loads((OUT/'development/parent/registry'/(q['id']+'.json')).read_text())
        assert hashlib.sha256(p.tobytes()).hexdigest()==reg['prediction_sha256'];preserved.append(q['id'])
    assert system['original_shapes']==json.loads((OUT/'parent_system.json').read_text())['original_shapes']
    save(OUT/'original_route_receipt.json',dict(unchanged_configuration=True,bitwise_replays=preserved,original90_scores_inherited_not_new_evaluation=True))
    print('SELECTED_ON_SEEN18',winner,flush=True)

def cleanup(candidates,reports,winner):
    removed=[];dominated=[]
    def stats(report):
        s=report['summary']['all'];return [s['40s']['n_pass'],-s['40s']['mean_R'],-s['40s']['max_R'],s['8s']['n_pass'],-s['8s']['mean_R'],-s['8s']['max_R']]
    for c in candidates:
        if c['name']==winner:continue
        a=stats(reports[c['name']])
        if any(all(b>=x for b,x in zip(stats(r),a)) and any(b>x for b,x in zip(stats(r),a)) for n,r in reports.items() if n!=c['name']):dominated.append(c)
    if dominated:
        paths=[]
        for c in dominated:
            paths.append(Path(c['checkpoint']));paths += list(Path(c['bank_folder']).rglob('*'))
        paths=sorted({p for p in paths if p.is_file()})
        assert all(p.resolve().is_relative_to(OUT.resolve()) for p in paths)
        hashes={str(p.relative_to(OUT)):sha(p) for p in paths};archive=OUT/'dominated_candidates.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for p in paths:tar.add(p,arcname=str(p.relative_to(OUT)))
        with tarfile.open(archive,'r:gz') as tar:
            for name,digest in hashes.items():assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest
        for p in paths:removed.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size));p.unlink()
        save(OUT/'cleanup_receipt.json',dict(archive=str(archive),sha256=sha(archive),removed=removed,dominated=[c['name'] for c in dominated],CFD_preserved=True))
    else:save(OUT/'cleanup_receipt.json',dict(removed=[],dominated=[],CFD_preserved=True))

def finish():
    cfg=check();lock=json.loads((OUT/'selection_lock.json').read_text());selected=OUT/'selected_system.json';assert sha(selected)==lock['system_sha256']
    qs=[query(q,CACHE,q['role']) for q in cfg['fresh_confirmation']]
    clean={};noise={}
    for name,system in [('parent',OUT/'parent_system.json'),('selected',selected)]:
        clean[name]=evaluate(system,qs,OUT/'fresh'/name/'clean')
        noise[name]={mode:{} for mode in ['future1pct','startup_and_future1pct']}
        for mode in noise[name]:
            for seed in cfg['noise_seeds']:noise[name][mode][str(seed)]=evaluate(system,qs,OUT/'fresh'/name/mode/str(seed),mode,seed)
    save(OUT/'fresh_comparison.json',dict(reports=clean,primary='selected',selected_before_fresh_CFD=True,no_reselection_after_fresh=True))
    save(OUT/'noise_report.json',noise)
    model=previous.Predictor(selected);replays=[]
    for q in qs:
        row=startup(q);p,_=model.predict(row,packets(q,row));reg=json.loads((OUT/'fresh/selected/clean/registry'/(q['id']+'.json')).read_text())
        assert hashlib.sha256(p.tobytes()).hexdigest()==reg['prediction_sha256'];replays.append(q['id'])
    check();assert sha(selected)==lock['system_sha256']
    comparison=json.loads((OUT/'development_comparison.json').read_text());cleanup(lock['candidates'],comparison['reports'],lock['selected'])
    allrows=[]
    for name,report in clean.items():
        for row in report['cases']:allrows.append((name,'clean',0,row))
    for name,modes in noise.items():
        for mode,seeds in modes.items():
            for seed,report in seeds.items():
                for row in report['cases']:allrows.append((name,mode,int(seed),row))
    with (OUT/'fresh_case_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['system','condition','seed','case','family','horizon','R','pass','Ux_mean','Uy_mean','p_mean','Ux_end','Uy_end','p_end'])
        for name,mode,seed,row in allrows:
            for horizon in ['8s','40s']:
                z=row['horizons'][horizon];w.writerow([name,mode,seed,row['id'],row['q']['family'],horizon,z['R'],z['R']<=1,*z['mean'],*z['end']])
    save(OUT/'verification_receipt.json',dict(replays=replays,clean_systems=2,cases_each=12,noise_systems=2,noise_conditions=2,noise_seeds=5,noise_cases_each=12,all264_future_poison_tests=True,
         startup_fields=4,future_scalar_readings=76,original_branch_unchanged=True,source_check_pass=True,selected_system_unchanged=True))
    save(OUT/'final_result.json',dict(utc=now(),selected=lock['selected'],fresh_summary={name:r['summary'] for name,r in clean.items()},
         noise_summary={name:{mode:{seed:r['summary'] for seed,r in seeds.items()} for mode,seeds in modes.items()} for name,modes in noise.items()},
         development_summary={name:r['summary'] for name,r in comparison['reports'].items()},TRAIN156=True,new_CFD44=True,
         old108_seen=True,known_contours_only=True,parent_first_fixed8_unchanged=True,no_reselection_after_fresh=True,models_on_server=True))
    save(OUT/'overall_status.json',dict(stage='complete',utc=now(),selected=lock['selected'],fresh=clean['selected']['summary']['all']['40s']))
    print('TARGETED_COVERAGE_COMPLETE',flush=True)

def run():
    handle=(OUT/'campaign.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=check()
    if (OUT/'final_result.json').exists():print('ALREADY_COMPLETE');return
    if not (OUT/'CFD_train_manifest.json').exists():generate(cfg['additional_train'],'CFD_train')
    check();save(OUT/'overall_status.json',dict(stage='training',utc=now()))
    for arm in cfg['arms']:
        done=OUT/arm['name']/'done.json'
        if done.exists():continue
        with (OUT/(arm['name']+'.log')).open('ab') as log:
            subprocess.run([sys.executable,'-u',str(WORK/'train_target.py'),'--arm',arm['name']],stdout=log,stderr=subprocess.STDOUT,check=True)
    from train_core import candidate_index
    candidate_index();choose()
    if not (OUT/'CFD_fresh_manifest.json').exists():generate(cfg['fresh_confirmation'],'CFD_fresh')
    save(OUT/'overall_status.json',dict(stage='fresh_evaluation_and_noise',utc=now()));finish()

if __name__=='__main__':
    try:
        if '--prepare' in sys.argv:prepare()
        else:run()
    except Exception as exc:
        save(OUT/'failure.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()))
        save(OUT/'overall_status.json',dict(stage='failure',utc=now(),error=repr(exc)));raise
