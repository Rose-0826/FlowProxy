#!/usr/bin/env python3
"""Versioned CFD rebuild. Original inputs stay immutable; full40, uniform dt."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys,os,re,json,time,copy,shutil,subprocess,gzip,hashlib,threading
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'cfd/scripts')]
from geometry_attack_v2 import save,sha,attrs,NativeSampler
from generate_case import write_case
from build_true2d_case_v1 import build_case
from corrected_benchmark_quality_v1 import mpi
import true2d_recheck_v1 as run
from cfd_numerics_analyze_v1 import forces,moment
from cpu_data_prepare import frames,read_point_fields
from flowproxy.geometry import geometry_channels
from warmstart_iface import read_vectors,replace_internal
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
OUT=ROOT/'outputs/corrected_benchmark_restart_v1'
RAW=ROOT/'data/raw/corrected_benchmark_restart_v1'
CACHE=ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
DATA=ROOT/'data/processed/dataset_corrected_benchmark_v1.h5'
run.ENV['HWLOC_COMPONENTS']='linux,stop'
PREP_LOCK=threading.Lock()

def prepare():
    assert not OUT.exists() and not RAW.exists() and not CACHE.exists() and not DATA.exists()
    OUT.mkdir();RAW.mkdir();CACHE.mkdir()
    old=json.loads((ROOT/'outputs/true2d_recheck_v1/preregister.json').read_text())
    splits=json.loads((ROOT/'configs/splits/body_120.json').read_text())
    with h5py.File(ROOT/'data/processed/dataset_long.h5','r') as f:
        rows=[{'id':cid,'role':role,**attrs(f['cases'][cid])} for role,ids in splits.items() for cid in ids]
    confirmations=copy.deepcopy(json.loads((ROOT/'outputs/round2/preregister.json').read_text())['cases'])
    for row in confirmations:row.update({'cx':0.,'role':'fresh_confirmation'})
    assert len(rows)==120 and len(confirmations)==12 and not set(r['id'] for r in rows)&set(r['id'] for r in confirmations)
    p={'registered_utc':run.now(),'version':'corrected_benchmark_restart_v1','config':old['config'],
       'splits':splits,'cases':rows,'confirmation_cases':confirmations,
       'reference_choice':'use repaired 210x120 background, dt=.005 only if corrected_benchmark_quality_v1 internal gates pass; otherwise halt before large data or training',
       'CFD':{'background':[210,120],'dt':.005,'sample_dt':.25,'single_layer':True,'MPI_ranks':2,'workers':5,
              'warm_source_time':201,'minimum_relaxation_seconds':60,'maximum_relaxation_seconds':240,'extension_seconds':30,'dense_window_seconds':41},
       'initialization':'old CFD native VTK at t201 is only a seed for NEW corrected CFD; same target for original120, metadata-nearest training case for fresh12; no AI initialization or claim of cold-start solver speedup',
       'stability_gate':{'Cd_mean':.005,'Cl_rms':.01,'St':.005,'last_two_consecutive_cycle_changes_required':True},
       'quality_failures':'record and halt; never silently drop hard geometries or change split',
       'fields':'physical Ux/Uy/p; same uniform64 grid and geometric fluid mask; raw pressure, no future reference shift',
       'future':'four true startup fields, then160 autonomous future frames; 40.0 physical seconds',
       'train':{'seeds':[11,22,33],'warm_initialization':'old frozen K8, same initialization within each K1/K8 pair',
                'from_scratch_pair_seed':11,'updates':2500,'checks':[0,250,500,750,1000,1250,1500,1750,2000,2250,2500],
                'batch_cases':4,'lr_warm':.0001,'lr_scratch':.001,'weight_decay':.0001,'clip_grad':1.,
                'K1_control':'teacher-forced predictions on the SAME eight training targets and task order; .5 mean + .5 endpoint',
                'K8':'eight self-fed predictions, full eight-step backprop; same target window and loss weighting',
                'loss_scaling':'channel RMS computed only from corrected train80',
                'selection':'minimum validation8 meanRi of fixed periodic closure; tie maximumRi then update; include step0; no test or confirmation use'},
       'closure':{'bootstrap_steps':48,'H':4,'frequency_method':'identified_shape_mean','blend_steps':0,
                  'frequency_prior':'new corrected train80 force-derived shape-mean St only; no validation/test priors',
                  'all_choices_fixed_from_previous_pipeline':True},
       'test_roles':'original32 already-seen development test; untouched12 is fresh confirmation, evaluated only after model/config lock',
       'original_protected_sha256':old['frozen_sha256'],
       'original_split_sha256':sha(ROOT/'configs/splits/body_120.json'),
       'confirmation_registry_sha256':sha(ROOT/'outputs/round2/preregister.json'),
       'storage':'lossless gzip for generated large raw files after sampling and byte-hash round-trip verification; preserve source originals',
       'minimum_free_disk_GB':12,'script_sha256':sha(__file__)}
    save(OUT/'preregister.json',p)
    save(OUT/'status.json',{'status':'waiting_reference_quality','completed':[],'errors':[],'total':120})
    print('RESTART_PREREGISTERED',flush=True)

def cycle_stats(case,row,end):
    a=forces(case);a=a[(a[:,0]>=end-40)&(a[:,0]<=end)]
    lift=a[:,2]-np.trapezoid(a[:,2],a[:,0])/(a[-1,0]-a[0,0]);cross=[]
    for k in np.flatnonzero((lift[:-1]<=0)&(lift[1:]>0)):
        cross.append(float(a[k,0]-lift[k]*(a[k+1,0]-a[k,0])/(lift[k+1]-lift[k])))
    if len(cross)<4:return {'passed':False,'reason':'fewer than three complete cycles','cycles':[]}
    cycles=[]
    for x,y in zip(cross[-4:-1],cross[-3:]):
        mu,rms,_=moment(a[:,0],a[:,1:],x,y)
        cycles.append({'start':x,'end':y,'Cd_mean':float(mu[0]),'Cl_rms':float(rms[1]),'St':row['D']/(y-x)})
    changes=[{k:abs(b[k]-a[k])/max(abs(a[k]),1e-8) for k in ['Cd_mean','Cl_rms','St']} for a,b in zip(cycles[:-1],cycles[1:])]
    passed=all(c[k]<v for c in changes for k,v in {'Cd_mean':.005,'Cl_rms':.01,'St':.005}.items())
    return {'passed':passed,'cycles':cycles,'changes':changes,'St':float(np.mean([c['St'] for c in cycles]))}

def initialize(case,row,p):
    cid=row['id'];sourcecid=cid
    if row['role']=='fresh_confirmation':
        choices=[r for r in p['cases'] if r['role']=='train' and r['shape']==row['shape']]
        sourcecid=min(choices,key=lambda r:((r['Re']-row['Re'])/120)**2+((r['cy']-row['cy'])/.8)**2+((r['D']-row['D'])/.2)**2)['id']
    source=ROOT/'data/raw/cases_long40'/sourcecid
    candidates=frames(source);stamp,vtk=min(candidates,key=lambda x:abs(x[0]-201))
    assert abs(stamp-201)<1e-7
    run.command(case,'postProcess -func writeCellCentres -time 0','log.writeCellCentres',240)
    centers=read_vectors(case/'0/C')
    pts,values=read_point_fields(vtk);interp=LinearNDInterpolator(pts,values,fill_value=np.nan)
    mapped=interp(centers[:,:2]);bad=~np.isfinite(mapped).all(1)
    if bad.any():mapped[bad]=values[cKDTree(pts).query(centers[bad,:2])[1]]
    assert np.isfinite(mapped).all()
    dest=case/'201';dest.mkdir()
    ub='internalField nonuniform List<vector>\n'+str(len(mapped))+'\n(\n'+'\n'.join(f'({u:.10g} {v:.10g} 0)' for u,v,pv in mapped)+'\n);\n'
    pb='internalField nonuniform List<scalar>\n'+str(len(mapped))+'\n(\n'+'\n'.join(f'{pv:.10g}' for u,v,pv in mapped)+'\n);\n'
    for name,b in [('U',ub),('p',pb)]:
        (dest/name).write_text(replace_internal((case/'0'/name).read_text(),b))
    save(case/'initialization.json',{'source_case':sourcecid,'source_VTK':str(vtk),'source_time':stamp,'source_sha256':sha(vtk),'target_cells':len(mapped),'nearest_fallback_cells':int(bad.sum()),'seed_only_not_new_reference_truth':True,'CFD_re_equilibration_required':True})

def compress_generated_raw(case):
    """Reversible storage only, within the new generated case; verify every byte."""
    case=case.resolve();assert case.is_relative_to(RAW.resolve())
    from corrected_benchmark_storage_v1 import encode_vtk
    records=[];vtk_records=[]
    selected=[]
    for f in case.rglob('*'):
        if not f.is_file() or f.suffix=='.gz':continue
        rel=f.relative_to(case)
        if f.suffix in ['.vtu','.vtp'] or f.name.startswith('log.') or (re.match(r'^\d+(\.\d+)?$',f.parent.name) and f.name in ['U','p','phi','C','Cx','Cy','Cz']):
            selected.append(f)
    for f in selected:
        if f.suffix in ['.vtu','.vtp']:
            r=encode_vtk(f,case);vtk_records.append({'original_path':str(f.relative_to(case)),'original_bytes':r['original_bytes'],'uncompressed_sha256':r['original_sha256'],'pointer':str(f.with_name(f.name+'.dedup.json').relative_to(case))});continue
        original_sha=sha(f);dest=f.with_name(f.name+'.gz');assert not dest.exists()
        with f.open('rb') as src,gzip.open(dest,'wb',compresslevel=1) as target:shutil.copyfileobj(src,target)
        digest=hashlib.sha256()
        with gzip.open(dest,'rb') as verify:
            for block in iter(lambda:verify.read(1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==original_sha
        records.append({'original_path':str(f.relative_to(case)),'original_bytes':f.stat().st_size,'uncompressed_sha256':original_sha,'gzip_path':str(dest.relative_to(case)),'gzip_bytes':dest.stat().st_size})
        f.unlink()
    save(case/'lossless_storage_manifest.json',{'restore':'native/log files: gzip -dk; VTK: python scripts/corrected_benchmark_storage_v1.py <file.dedup.json>; exact bytes checked before removing generated uncompressed representation','files':records,'VTK_files':vtk_records})
    return len(records)+len(vtk_records)

def one(row,p):
    begin=time.perf_counter();cid=row['id'];case=RAW/'cases'/cid
    if shutil.disk_usage(ROOT).free<12e9:raise RuntimeError('Disk reserve reached; preserve all generated files')
    with PREP_LOCK:
        cfg=copy.deepcopy(p['config']);cfg['cfd'].update({'n_cells_x':210,'n_cells_y':120,'delta_t':.005,'end_time':302,'write_interval':.25})
        source=write_case(cfg,RAW/'mesh_sources',Re=row['Re'],cx=row['cx'],cy=row['cy'],D=row['D'],shape=row['shape'])
        run.command(source,'blockMesh','log.blockMesh_restart',240)
        run.command(source,'snappyHexMesh -overwrite','log.snappyHexMesh_restart',480)
        build_case(source,case,.005,302,.25)
        initialize(case,row,p)
    run.configure(case,201,261,.005,10)
    (case/'system/decomposeParDict').write_text('FoamFile {version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains 2;method scotch;\n')
    run.command(case,'decomposePar -time 201','log.decomposePar_restart',240)
    records=[];current=201;end=261
    while True:
        run.configure(case,current,end,.005,10)
        f=run.command(case,mpi(2),f'log.icoFoam_relax_{end}',10800);run.assert_solver(f,end)
        c=cycle_stats(case,row,end);records.append({'end':end,**c})
        save(case/'stability_history.json',records)
        if c['passed']:break
        if end>=201+240:raise RuntimeError(f'{cid}: failed CFD stability gate; do not drop case')
        current=end;end+=30
    run.configure(case,end,end+41,.005,.25)
    f=run.command(case,mpi(2),'log.icoFoam_dense',10800);run.assert_solver(f,end+41)
    courants=[float(x) for x in re.findall(r'Courant Number mean:\s+\S+\s+max:\s+(\S+)',f.read_text())]
    assert courants and max(courants)<1., (cid,'dense Courant',max(courants) if courants else None)
    final_gate=cycle_stats(case,row,end+41);assert final_gate['passed'],(cid,final_gate)
    run.command(case,f'reconstructPar -time "{end}:{end+41}"','log.reconstructPar_restart',1800)
    assert (case/str(end+41)/'U').exists() and (case/str(end+41)/'p').exists()
    run.command(case,f'foamToVTK -useTimeName -time "{end}:{end+41}"','log.foamToVTK_restart',1800)
    items=[(t,f) for t,f in frames(case) if end-1e-7<=t<=end+41+1e-7]
    times=np.array([t for t,f in items]);assert len(items)==165 and np.allclose(np.diff(times),.25,atol=1e-8)
    geo=geometry_channels(p['config'],**{k:row[k] for k in ['shape','Re','cx','cy','D']})
    sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask']);fields=np.stack([sampler.sample(f) for t,f in items]);assert np.isfinite(fields).all()
    target=CACHE/(cid+'.npz');assert not target.exists()
    np.savez_compressed(target,fields=fields,times=times,geom=geo['geom'],mask=geo['mask'],forces=forces(case))
    np.savez_compressed(CACHE/(cid+'_startup.npz'),fields=fields[:4],times=times,geom=geo['geom'],mask=geo['mask'])
    native_hashes=[{'time':t,'sha256':sha(f)} for t,f in items]
    save(case/'dataset_source.json',{'case':row,'window_start':end,'initial_true_times':times[:4].tolist(),'future_end':float(times[163]),'future_seconds':float(times[163]-times[3]),'quality':final_gate,'native_VTK_manifest':native_hashes,'cache_sha256':sha(target)})
    n=compress_generated_raw(case)
    result={'id':cid,'role':row['role'],'wall_seconds':time.perf_counter()-begin,'dense_start':end,'CFD_stable':True,'max_Courant_dense':max(courants),'uniform_frames':165,'St':final_gate['St'],'cache_sha256':sha(target),'losslessly_compressed_files':n}
    save(CACHE/(cid+'.json'),result)
    print('CORRECTED_CASE_DONE',row['role'],cid,round(result['wall_seconds'],1),flush=True)
    return result

def rebuild_dataset(p):
    target=DATA.with_suffix('.partial.h5');assert not DATA.exists() and not target.exists()
    with h5py.File(target,'x') as h:
        h.attrs['protocol_sha256']=sha(OUT/'preregister.json');h.attrs['sample_dt']=.25;h.attrs['single_layer']=True
        groups=h.create_group('cases')
        for row in p['cases']:
            with np.load(CACHE/(row['id']+'.npz')) as a:
                g=groups.create_group(row['id'])
                for k in ['fields','times','geom','mask','forces']:g.create_dataset(k,data=a[k],compression='gzip',compression_opts=1)
                for k in ['shape','Re','cx','cy','D']:g.attrs[k]=row[k]
                g.attrs['role']=row['role'];g.attrs['St']=json.loads((CACHE/(row['id']+'.json')).read_text())['St']
        s=h.create_group('splits')
        for role,ids in p['splits'].items():s.create_dataset(role,data=np.array(ids,dtype='S'))
    target.replace(DATA)
    save(OUT/'dataset_manifest.json',{'data':str(DATA),'sha256':sha(DATA),'train':80,'val':8,'already_seen_test':32,'uniform_dt':.25,'true_support':4,'autonomous_future_steps':160,'future_seconds':40,'fresh_confirmation_in_dataset':False})

def protected(p):
    for path,digest in p['original_protected_sha256'].items():assert sha(ROOT/path)==digest
    assert sha(ROOT/'configs/splits/body_120.json')==p['original_split_sha256']

def execute():
    p=json.loads((OUT/'preregister.json').read_text());protected(p)
    quality=ROOT/'outputs/corrected_benchmark_quality_v1'
    while not (quality/'report.json').exists():
        s=json.loads((quality/'status.json').read_text())
        if s['status']=='failed':raise RuntimeError('Reference quality CFD failed; preserve status')
        time.sleep(30)
    q=json.loads((quality/'report.json').read_text())
    if not q['all_internal_gates_passed']:
        save(OUT/'status.json',{'status':'reference_gate_failed','completed':[],'errors':[],'report':str(quality/'report.json')})
        print('RESTART_HALTED_REFERENCE_GATE',flush=True);return
    save(OUT/'reference_locked.json',{'locked_utc':run.now(),'reference':q['chosen_reference'],'quality_report_sha256':sha(quality/'report.json'),'AI_scores_used':False})
    done=[];errors=[]
    save(OUT/'status.json',{'status':'building_corrected120','total':120,'completed':done,'errors':errors})
    with ThreadPoolExecutor(max_workers=5) as pool:
        fs={pool.submit(one,row,p):row['id'] for row in p['cases']}
        for f in as_completed(fs):
            if f.cancelled():continue
            try:done.append(f.result())
            except Exception as e:errors.append({'id':fs[f],'error':repr(e)})
            save(OUT/'status.json',{'status':'building_corrected120','total':120,'completed':done,'errors':errors})
            if errors:
                for other in fs:other.cancel()
    if errors:
        save(OUT/'status.json',{'status':'data_failed','total':120,'completed':done,'errors':errors});raise RuntimeError(errors)
    rebuild_dataset(p);protected(p)
    save(OUT/'status.json',{'status':'data_complete_training','total':120,'completed':done,'errors':[]})
    subprocess.run([sys.executable,str(ROOT/'scripts/corrected_benchmark_train_v1.py'),'--all'],check=True)
    assert (OUT/'selected_model_lock.json').exists()
    subprocess.run([sys.executable,str(ROOT/'scripts/corrected_benchmark_train_v1.py'),'--test'],check=True)
    confirm=[];errors=[]
    save(OUT/'confirmation_status.json',{'status':'running','total':12,'completed':confirm,'errors':errors})
    with ThreadPoolExecutor(max_workers=5) as pool:
        fs={pool.submit(one,row,p):row['id'] for row in p['confirmation_cases']}
        for f in as_completed(fs):
            try:confirm.append(f.result())
            except Exception as e:errors.append({'id':fs[f],'error':repr(e)})
            save(OUT/'confirmation_status.json',{'status':'running','total':12,'completed':confirm,'errors':errors})
    if errors:raise RuntimeError(errors)
    subprocess.run([sys.executable,str(ROOT/'scripts/corrected_benchmark_train_v1.py'),'--confirm'],check=True)
    protected(p)
    save(OUT/'status.json',{'status':'complete','total':120,'completed':done,'errors':[],'fresh_confirmation':12,'original_weights_and_split_unchanged':True})
    print('CORRECTED_BENCHMARK_RESTART_COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--smoke-case',default=None);args=a.parse_args()
    if args.prepare:prepare()
    elif args.smoke_case:
        p=json.loads((OUT/'preregister.json').read_text());row=next(r for r in p['cases'] if r['id']==args.smoke_case);one(row,p)
    else:execute()
