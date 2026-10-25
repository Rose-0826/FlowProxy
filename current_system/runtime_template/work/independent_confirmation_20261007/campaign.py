"""Fixed-system confirmation: eight new parameter cases, no training/selection."""
from pathlib import Path
import os, sys, json, time, hashlib, shutil, traceback, re, fcntl, copy
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path('/root/autodl-tmp/flowproxy')
WORK = ROOT/'work/independent_confirmation_20261007'
OUT = ROOT/'outputs/independent_confirmation_20261007'
RAW = ROOT/'data/raw/independent_confirmation_20261007'
CACHE = ROOT/'data/processed/independent_confirmation_20261007'
EXP = ROOT/'work/shape_expansion_20261006'
sys.path.insert(0, str(EXP))
import expansion_core as e
from system_entry import Predictor
from train_expansion import load_candidate
import corrected_benchmark_restart_v1 as rebuild
from research_v3_corner_cfd import original_initialize
import numpy as np

def save(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)

def now(): return e.old.p.now()
def sha(path): return e.old.p.digest(Path(path))
def key(q):
    return (q['shape'],round(float(q['Re']),6),round(float(q.get('cx',0)),6),
            round(abs(float(q['cy'])),6),round(float(q['D']),6))

def inventory():
    records={};files=0
    def walk(x, origin):
        if isinstance(x,dict):
            if isinstance(x.get('shape'),str) and all(n in x for n in ['Re','cy','D']):
                try:records.setdefault(key(x), dict(q={n:x.get(n,0) for n in ['shape','Re','cx','cy','D']},origin=origin))
                except (TypeError,ValueError):pass
            for v in x.values():walk(v,origin)
        elif isinstance(x,list):
            for v in x:walk(v,origin)
    # Include old/historical evaluation metadata, not just the current dev set.
    paths=list((ROOT/'outputs').rglob('*.json'))+list((ROOT/'data/raw').rglob('meta.json'))
    for p in paths:
        if OUT in p.parents:continue
        try:walk(json.loads(p.read_text()),str(p.relative_to(ROOT)));files+=1
        except (UnicodeError,json.JSONDecodeError,OSError):continue
    return list(records.values()),files

def prepare():
    assert not (OUT/'config.json').exists(), 'Existing confirmation must not be redesigned'
    for p in [WORK,OUT,RAW,CACHE]:p.mkdir(parents=True,exist_ok=True)
    e.setup();existing,scanned=inventory();excluded={key(x['q']) for x in existing}
    rng=np.random.default_rng(20261007);cases=[]
    shapes=['cylinder','square',*e.shapes.SPECS]
    for shape in shapes:
        added=shape in e.shapes.SPECS
        while True:
            Re=round(float(rng.uniform(95,185)),2)
            cy=round(float(rng.uniform(-.55,.55)),3)
            D=round(float(rng.uniform(1.,1.20) if added else rng.uniform(.82,1.20)),3)
            q=dict(id=f'{shape}_Re{Re:g}_cy{cy:g}_D{D:g}',shape=shape,Re=Re,cx=0.,cy=cy,D=D,
                   role='independent_fixed_system_confirmation',family=e.shapes.SPECS[shape]['family'] if added else shape)
            if key(q) not in excluded and abs(cy)>.05:
                excluded.add(key(q));cases.append(q);break
    system=ROOT/'outputs/shape_expansion_20261006/selected_system.json'
    selected=json.loads(system.read_text());files={system}
    original=selected['original_shapes'];new=selected['new_shapes']
    files.update([Path(original['checkpoint']),Path(new['checkpoint'])])
    for banks in [original['banks'],new['old_banks'],new['new_banks']]:
        for info in banks.values():files.update([Path(info['path']),Path(info['path']).with_suffix('.json')])
    files.update([ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json',
                  ROOT/'outputs/corrected_benchmark_restart_v1/training/warm_K1_seed33/configuration.json',
                  e.old.p.OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json'])
    train=e.old.metadata();files.update(e.old.template_path(z) for z in train)
    train24=json.loads((e.OUT/'training_dataset.json').read_text())['cases']
    for q in train24:files.update([e.CACHE/(q['id']+'.npz'),e.CACHE/(q['id']+'.json')])
    sources={Path(__file__).resolve()}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and Path(f).resolve().is_relative_to(ROOT) and str(f).endswith('.py'):sources.add(Path(f).resolve())
    for p in sources:files.add(p)
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    config=dict(created_utc=now(),seed=20261007,cases=cases,total=8,workers=8,MPI_ranks_each=2,
                system_path=str(system),system_sha256=sha(system),dependencies=frozen,
                source_files=[str(p.relative_to(ROOT)) for p in sorted(sources)],
                excluded_case_count=len(existing),excluded_metadata_files=scanned,
                methods=['final_system','no_residual_library','training_template_observer'],
                noise_seeds=[7001,7002,7003,7004,7005],
                scope='Eight supported fixed contours; new Re/cy/D parameters, not unseen-contour proof.',
                selection='Final system fixed before CFD. Controls diagnose contribution, never select a replacement.',
                inputs='4 full startup fields; 76 future Uy scalars, emit field before same-time read.',
                storage='All cases retained; quality failures stop evaluation; no dropping or success fabrication.')
    save(OUT/'excluded_case_inventory.json',dict(records=existing,metadata_files=scanned))
    save(OUT/'config.json',config)
    save(OUT/'system_locked.json',selected)
    save(OUT/'training_identity.json',dict(original100=train,new24=train24))
    save(OUT/'overall_status.json',dict(stage='prepared',total=8,utc=now()))
    print('PREPARED_FIXED8',json.dumps(cases),flush=True)

def check():
    cfg=json.loads((OUT/'config.json').read_text())
    for name,digest in cfg['dependencies'].items():assert sha(ROOT/name)==digest, 'Locked dependency changed: '+name
    return cfg

def write_case(cfg,out_dir,Re,cx,cy,D,shape,**kw):
    if shape in e.shapes.SPECS:return e.shapes.write_case(cfg,out_dir,Re,cx,cy,D,shape,**kw)
    return e.shapes.original_case.write_case(cfg,out_dir,Re,cx,cy,D,shape,**kw)

def initialize(case,q,protocol):
    proxy=dict(q)
    if q['shape'] in e.shapes.SPECS:proxy['shape']='square' if q['family']=='rectangle' else 'cylinder'
    original_initialize(case,proxy,protocol)
    seed=case/'training_donor_seed.vtu';dest=case/'training_donor_seed.xml'
    assert seed.exists() and not dest.exists();digest=sha(seed);seed.rename(dest);assert sha(dest)==digest
    info=json.loads((case/'initialization.json').read_text())
    info.update(target_shape=q['shape'],target_shape_spec=e.shapes.SPECS.get(q['shape']),
                meaning='TRAIN donor is numerical initialization only; true CFD re-equilibration on target geometry >=60s.')
    save(case/'initialization.json',info)

def generate(cfg):
    rebuild.RAW=RAW;rebuild.CACHE=CACHE;rebuild.write_case=write_case
    rebuild.geometry_channels=e.shapes.geometry_channels;rebuild.initialize=initialize
    protocol=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    done=[];errors=[];start=time.perf_counter()
    def one(q):
        if shutil.disk_usage(ROOT).free<40*2**30:raise RuntimeError('Disk reserve: preserve cases')
        ready=CACHE/(q['id']+'.json')
        if ready.exists():
            r=json.loads(ready.read_text());assert r['CFD_stable'] and r['uniform_frames']==165 and r['cache_sha256']==sha(CACHE/(q['id']+'.npz'));return r
        if (RAW/'cases'/q['id']).exists():raise RuntimeError('Partial CFD preserved; explicit recovery needed: '+q['id'])
        return rebuild.one(q,protocol)
    save(OUT/'overall_status.json',dict(stage='CFD_running',total=8,completed=0,utc=now()))
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs={pool.submit(one,q):q for q in cfg['cases']}
        for f in as_completed(jobs):
            if f.cancelled():continue
            try:done.append(f.result())
            except Exception as exc:
                errors.append(dict(id=jobs[f]['id'],error=repr(exc)))
                for j in jobs:j.cancel()
            save(OUT/'CFD_status.json',dict(stage='CFD_running' if not errors else 'CFD_failure',completed=done,total=8,errors=errors,wall_seconds=time.perf_counter()-start,utc=now()))
            save(OUT/'overall_status.json',dict(stage='CFD_running' if not errors else 'CFD_failure',completed=len(done),total=8,errors=errors,utc=now()))
            print('CFD_PROGRESS',len(done),len(errors),flush=True)
    if errors:raise RuntimeError(errors)
    assert len(done)==8
    audit=[]
    for q in cfg['cases']:
        case=RAW/'cases'/q['id'];r=json.loads((CACHE/(q['id']+'.json')).read_text())
        mesh=json.loads((case/'true2d_case_manifest.json').read_text()) if (case/'true2d_case_manifest.json').exists() else None
        if mesh is None:
            candidates=[p for p in case.glob('*.json') if 'exact_single_layer' in p.read_text()]
            assert len(candidates)==1;mesh=json.loads(candidates[0].read_text())
        assert mesh['exact_single_layer'] and mesh['mesh_OK']
        src=json.loads((case/'dataset_source.json').read_text())
        assert src['quality']['passed'] and r['CFD_stable'] and r['max_Courant_dense']<1 and r['uniform_frames']==165
        with np.load(CACHE/(q['id']+'.npz')) as z:assert z['fields'].shape==(165,3,64,64) and np.isfinite(z['fields']).all() and np.allclose(np.diff(z['times']),.25)
        audit.append(dict(id=q['id'],mesh_pass=True,single_layer=True,cycles_pass=True,frames=165,cache_sha256=sha(CACHE/(q['id']+'.npz')),max_Courant=r['max_Courant_dense'],dense_start=r['dense_start']))
    save(OUT/'dataset_manifest.json',dict(role='first independent fixed-system confirmation',cases=cfg['cases'],results=done,quality=audit,cache=str(CACHE),raw=str(RAW)))
    return audit

def startup(q):
    with np.load(CACHE/(q['id']+'_startup.npz')) as z:row={k:z[k] for k in ['fields','geom','mask','times']}
    assert row['fields'].shape==(4,3,64,64)
    return dict(row,id=q['id'],q={k:q[k] for k in ['shape','Re','cx','cy','D']})

def packets(q,row):
    ij,_=e.old.p.probes(row);values=np.full((160,4),np.nan)
    with np.load(CACHE/(q['id']+'.npz')) as z:
        for step in range(8,160,8):values[step-1]=[z['fields'][step+3,1,i,j] for i,j in ij]
    assert np.isfinite(values).sum()==76;return values

def nearest_template(q,row,cfg,model):
    recs=e.old.metadata() if q['shape'] in ['cylinder','square'] else json.loads((e.OUT/'training_dataset.json').read_text())['cases']
    def params(rec):return rec['q'] if 'q' in rec else rec
    rec=min([r for r in recs if params(r)['shape']==q['shape']],key=lambda r:float(np.sum((e.old.b.features(params(r))-e.old.b.features(q))**2)))
    sourceq=params(rec)
    if q['shape'] in ['cylinder','square']:
        with np.load(e.old.template_path(rec)) as z:C=z['truthC'];mask=z['mask']
    else:
        info=json.loads((e.CACHE/(rec['id']+'.json')).read_text())
        with np.load(e.CACHE/(rec['id']+'.npz')) as z:
            B=e.old.p.basis(z['times']-z['times'][3],info['St']/sourceq['D'])
            C=e.old.b.previous.w.constrain(np.linalg.lstsq(B,z['fields'].reshape(165,-1),rcond=1e-10)[0].reshape(9,3,64,64),z['mask']);mask=z['mask']
    canonical,_,_=e.old.b.positive(C,sourceq,mask)
    if q['cy']<0:canonical=e.old.b.mirrored(canonical)
    C=e.old.b.body.to_physical(canonical,q,row['mask'])
    bundle=model.parent if q['shape'] in ['cylinder','square'] else model.added
    f=float(bundle['priors'][q['shape']](e.old.p.features(q)[None])[0])/q['D']
    pa,fit=e.old.p.fit_startup(C,row,f,False);C=e.old.p.correction(C,row,pa,'hybrid8_pH1')
    return C,pa,dict(TRAIN_source=rec['id'],no_neural_network=True,geometry_only_nearest_selection=True,startup=fit)

def predict(method,model,q,row,values,cfg):
    if method=='final_system':return model.predict(row,values)
    if method=='no_residual_library':
        bundle=model.parent if q['shape'] in ['cylinder','square'] else model.added
        own,C,pa,boot=e.old.p.bootstrap(bundle['model'],row,bundle['priors'])
        obs,di=e.old.p.observe(C,pa[1],row,values,pa)
        return e.old.p.hybrid(own,obs),dict(bootstrap=boot,observer=di,residual_removed=True)
    C,pa,di=nearest_template(q,row,cfg,model);pred,od=e.old.p.observe(C,pa[1],row,values,pa)
    return pred,dict(template=di,observer=od)

def summarize(rows):
    groups=['all','cylinder','square','ellipse','rectangle','irregular']
    return {g:{h:e.old.p.summarize([z['horizons'][h] for z in rows if g=='all' or z['q']['family']==g]) for h in ['8s','40s']} for g in groups}

def evaluate(cfg,audit):
    e.setup();model=Predictor(cfg['system_path']);clean={m:[] for m in cfg['methods']};noise={mode:{str(seed):[] for seed in cfg['noise_seeds']} for mode in ['future1pct','startup_and_future1pct']}
    for i,q in enumerate(cfg['cases']):
        check();row=startup(q);values=packets(q,row)
        jobs=[(method,'clean',0,row,values) for method in cfg['methods']]
        for seed in cfg['noise_seeds']:
            digest=int(hashlib.sha256((str(seed)+':'+q['id']).encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(digest);noisy=values+rng.normal(size=values.shape)*.01*e.old.p.scales()[1]
            r=dict(row);r['fields']=row['fields']+rng.normal(size=row['fields'].shape).astype(np.float32)*.01*np.asarray(e.old.p.scales(),np.float32)[None,:,None,None]*row['mask'][None,None]
            jobs.extend([('final_system','future1pct',seed,row,noisy),('final_system','startup_and_future1pct',seed,r,noisy)])
        predictions=[]
        for method,mode,seed,inputrow,inputvalues in jobs:
            target=OUT/'predictions'/method/mode/str(seed)/(q['id']+'.npz');reg=OUT/'registry'/method/mode/str(seed)/(q['id']+'.json')
            if target.exists() and reg.exists():
                recorded=json.loads(reg.read_text());assert recorded['file_sha256']==sha(target)
                with np.load(target) as z:pred=z['prediction']
                assert hashlib.sha256(pred.tobytes()).hexdigest()==recorded['prediction_sha256']
            else:
                pred,di=predict(method,model,q,inputrow,inputvalues,cfg)
                assert pred.shape==(160,3,64,64) and np.isfinite(pred).all()
                poison=inputvalues.copy();poison[79:]+=100.;other,_=predict(method,model,q,inputrow,poison,cfg)
                assert np.array_equal(pred[:80],other[:80]), 'Future measurement affects earlier outputs'
                target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,prediction=pred)
                save(reg,dict(id=q['id'],method=method,noise_mode=mode,seed=seed,
                              prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),file_sha256=sha(target),
                              registered_before_full_field_score=True,startup_fields=4,future_scalar_readings=76,future_poison_test=True,diagnostic=di))
            predictions.append((method,mode,seed,pred))
        # Full reference fields are opened for metrics only after all predictions are registered.
        with np.load(CACHE/(q['id']+'.npz')) as z:truth=z['fields'][4:164]
        for method,mode,seed,pred in predictions:
            result=dict(id=q['id'],q=q,horizons=e.old.p.metrics(pred,truth,row['mask'],row['times']))
            if mode=='clean':clean[method].append(result)
            else:noise[mode][str(seed)].append(result)
        save(OUT/'overall_status.json',dict(stage='evaluation',completed=i+1,total=8,utc=now()))
        print('EVALUATED',i+1,flush=True)
    check()
    report={m:dict(cases=rows,summary=summarize(rows)) for m,rows in clean.items()}
    noise_report={mode:{seed:dict(cases=rows,summary=summarize(rows)) for seed,rows in seeds.items()} for mode,seeds in noise.items()}
    save(OUT/'clean_report.json',dict(reports=report,primary='final_system',no_model_reselection=True,role='independent new parameters on eight supported fixed contours'))
    save(OUT/'noise_report.json',noise_report)
    for q in cfg['cases']:
        row=startup(q);v=packets(q,row);pred,_=model.predict(row,v)
        recorded=json.loads((OUT/'registry/final_system/clean/0'/(q['id']+'.json')).read_text())
        assert hashlib.sha256(pred.tobytes()).hexdigest()==recorded['prediction_sha256']
    save(OUT/'verification_receipt.json',dict(utc=now(),clean_methods=3,cases_each=8,noise_conditions=2,noise_seeds=5,noise_cases_each=8,replays=8,future_budget=76,all_future_poison_tests_passed=True,dependencies_unchanged=True))
    save(OUT/'final_result.json',dict(utc=now(),primary_summary=report['final_system']['summary'],controls={m:r['summary'] for m,r in report.items() if m!='final_system'},CFD_quality=audit,noise_summary={mode:{seed:r['summary'] for seed,r in seeds.items()} for mode,seeds in noise_report.items()},current_version_not_reselected=True))
    save(OUT/'overall_status.json',dict(stage='complete',completed=8,total=8,utc=now(),primary=report['final_system']['summary']['all']['40s']))
    print('INDEPENDENT_FIXED8_COMPLETE',flush=True)

def run():
    lock=(OUT/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=check()
    if (OUT/'final_result.json').exists():print('ALREADY_COMPLETE');return
    audit=generate(cfg);check();evaluate(cfg,audit)

if __name__=='__main__':
    try:
        if '--prepare' in sys.argv:prepare()
        else:run()
    except Exception as exc:
        save(OUT/'failure.json',dict(utc=now(),error=repr(exc),traceback=traceback.format_exc()))
        save(OUT/'overall_status.json',dict(stage='failure',error=repr(exc),utc=now()));raise
