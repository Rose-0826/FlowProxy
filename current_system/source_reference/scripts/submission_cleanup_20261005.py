"""User-authorized cleanup: preserve labels/evidence; exclude defective configuration era."""
from pathlib import Path
import os,json,hashlib,shutil,datetime,gzip,base64
ROOT=Path('/root/autodl-tmp/flowproxy').resolve()
OUT=ROOT/'outputs/submission_cleanup_20261005'
OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe(p):
 p=Path(p); q=p.resolve()
 if p.is_symlink() or ROOT not in q.parents:raise RuntimeError('Unsafe target '+str(p))
 return q
def dump(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2))
invalid_raw=['cases','cars','pilot_long','pilot_long2','cases_long40','st_holdout','warmstart_iface','warmstart_pilot','frozen_confirmation_v1','geometry_attack_confirmation_v1','geometry_attack_confirmation_d1_v1','geometry_attack_confirmation_far_v1','hybrid_twoframe_confirmation_v1']
invalid_out=['checkpoints','round2','warmstart_round1','gpu_leadtime_pilot_2026-10-02','geometry_attack_evaluation_v1','geometry_attack_v1','geometry_attack_v2','geometry_attack_confirmation_v1','geometry_attack_confirmation_d1_v1','geometry_attack_confirmation_far_v1','frozen_confirmation_v1','hybrid_observation_budget_v1','hybrid_twoframe_history_v1','hybrid_twoframe_confirmation_v1','hybrid_refined_reference_v1','uniform128_attack_v1','cpu_round1']
explanation='旧版二维算例厚度方向未满足 empty 边界的单层网格约束，且旧时间步尚未通过后续参考质量核验。这是算例配置问题，不是 OpenFOAM 软件失效。旧数据、模型及其成绩不计入本次参赛成果。保留最小故障配置、核验记录与排除清单后删除大文件。正确配置下的负结果仍保留；重复 native/VTK 中间场在已校验完整采样标签后精简。精简后保留网格、边界、力历史、判稳与求解日志、首末原生场及首末 VTK；两例保留完整原生历史。中间原生场不再完整保存，但 64×64 场标签与全部力时间序列保留。'
(OUT/'排除与清理说明.md').write_text('# 排除与清理说明\n\n'+explanation+'\n',encoding='utf-8')
dump('STOP_RESEARCH.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'user_request':'停止攻关，归纳成果并清理磁盘','new_training_or_cfd':False})
protected=[]
for p in (ROOT/'data/processed').rglob('*'):
 if p.is_file() and (('corrected' in p.as_posix()) or ('research_v' in p.as_posix()) or ('continuous_training' in p.as_posix()) or ('period_consistency' in p.as_posix())):
  protected.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
for s in ['outputs/period_consistency_train_v1/cycle_seed11/selected.pt','outputs/research_v2_multidirection/final_research_confirmation/selected_research_base_weights.pt']:
 p=ROOT/s;protected.append({'path':s,'bytes':p.stat().st_size,'sha256':sha(p)})
dump('protected_before.json',protected)
manifest=[]
mf=(OUT/'deleted_files.jsonl').open('w')
def remove(p,reason):
 p=safe(p)
 files=[p] if p.is_file() else [f for f in p.rglob('*') if f.is_file()]
 total=0
 for f in files:
  safe(f); n=f.stat().st_size;total+=n
  mf.write(json.dumps({'path':str(f.relative_to(ROOT)),'bytes':n,'reason':reason},ensure_ascii=False)+'\n')
 mf.flush()
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
 manifest.append({'path':str(p.relative_to(ROOT)),'bytes':total,'reason':reason})
def archive_metadata(p):
 for f in p.rglob('*'):
  if not f.is_file() or f.stat().st_size>8*1024*1024:continue
  if 'VTK' in f.parts or '.blocks' in f.parts:continue
  if f.suffix in ['.json','.csv','.md'] or f.name.startswith('log.checkMesh') or (f.parent.name=='system' and f.name in ['controlDict','blockMeshDict','fvSchemes','fvSolution','snappyHexMeshDict']):
   d=OUT/'excluded_configuration_evidence'/f.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,d)
before=shutil.disk_usage(ROOT)
for group,names in [('data/raw',invalid_raw),('outputs',invalid_out)]:
 for name in names:
  p=ROOT/group/name
  if p.exists():archive_metadata(p);remove(p,'excluded_defective_configuration_era')
for p in (ROOT/'data/processed').iterdir():
 if p.is_file() and p.name in ['dataset_long.h5','dataset_long.splits.json','dataset_long40_tail_v1.h5','dataset_long40_v1.h5','geometry_warp_v1.h5','uniform128_native_v1.h5']:
  if p.suffix=='.json':archive_metadata(p.parent)
  remove(p,'excluded_defective_configuration_label')
# Reproducible duplicate forecasts. Keep every final independent evaluation, and selected pilot versions.
research=ROOT/'outputs/research_v2_multidirection'
for name in ['state_probe','clock_anchor_probe','operator_probe','template_probe','spectrum_guard_probe','prefix_probe','sparse_startupcal_development','sparse_confirmation8_startupcal_secondary']:
 p=research/name/'predictions'
 if p.exists():remove(p,'duplicate_exploratory_forecast_arrays_reports_kept')
for name in ['clock_probe96','clock_bias_calibration','clock_prior104_probe','diagnostics']:
 p=research/'next_round'/name/'predictions'
 if p.exists():remove(p,'duplicate_exploratory_forecast_arrays_reports_kept')
p=research/'next_stage_split_observer/residual_training_data'
if p.exists():
 for f in list(p.glob('*.npz')):remove(f,'derived_training_array_rebuildable_from_corrected_labels')
# A cache must be verified before deleting raw history. Keep endpoint native VTK donor sources.
hash_to_cache={x['sha256']:ROOT/x['path'] for x in protected if x['path'].endswith('.npz')}
retained_cases=[]
full_keep={'cylinder_Re100_cy0_D1','square_Re100_cy0_D1'}
for meta in list((ROOT/'data/raw').rglob('dataset_source.json')):
 c=meta.parent
 if c.name.endswith('_source'):continue
 j=json.loads(meta.read_text()); digest=j.get('cache_sha256')
 if digest not in hash_to_cache:
  retained_cases.append({'case':str(c.relative_to(ROOT)),'status':'untouched_no_exact_cache_hash'});continue
 cache=hash_to_cache[digest]
 if c.name in full_keep and 'corrected_benchmark_restart_v1' in c.parts:
  retained_cases.append({'case':str(c.relative_to(ROOT)),'status':'full_native_history_retained','cache':str(cache.relative_to(ROOT))});continue
 numeric=[]
 for p in c.iterdir():
  if p.is_dir():
   try:numeric.append((float(p.name),p))
   except ValueError:pass
 numeric.sort()
 first=float(j['window_start']);end=float(j['future_end'])
 keep_times={0.,first,first+.25,first+.5,first+.75,end,numeric[-1][0]}
 for t,p in numeric:
  if not any(abs(t-k)<1e-6 for k in keep_times):remove(p,'duplicated_native_intermediate_field_verified_cache_retained')
 for p in list(c.glob('processor*')):
  if p.is_dir():remove(p,'parallel_decomposition_duplicate_reconstructed_fields_retained')
 vtk=c/'VTK'
 if vtk.exists():
  keep=[]
  for p in vtk.iterdir():
   if not p.is_dir() or p.name=='.blocks':continue
   m=next(p.glob('*.dedup.json'),None)
   if m:
    d=json.loads(m.read_text());lit=base64.b64decode(d['parts'][0].get('literal','')).decode('utf-8',errors='replace')
    import re
    match=re.search(r"time=['\"]([^'\"]+)",lit)
    t=float(match.group(1)) if match else None
    if t is not None and any(abs(t-k)<1e-6 for k in keep_times):keep.append(p)
    else:remove(p,'duplicated_vtk_intermediate_field_verified_cache_retained')
   else:
    # Ordinary VTK: no automatic thinning when time cannot be verified.
    keep.append(p)
  referenced=set()
  for m in vtk.rglob('*.dedup.json'):
   d=json.loads(m.read_text());bd=(m.parent/d['block_directory']).resolve()
   referenced.update((bd/(part['block']+'.gz')).resolve() for part in d['parts'] if 'block' in part)
  blocks=vtk/'.blocks'
  if blocks.exists():
   for p in list(blocks.iterdir()):
    if p.is_file() and p.resolve() not in referenced:remove(p,'unreferenced_vtk_dedup_block')
   assert all(p.exists() for p in referenced),'Retained VTK references missing blocks'
 retained_cases.append({'case':str(c.relative_to(ROOT)),'status':'thinned','cache':str(cache.relative_to(ROOT)),'kept_times':sorted(keep_times)})
dump('retained_raw_cases.json',retained_cases)
for x in protected:
 p=ROOT/x['path']; assert p.exists() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
mf.close();after=shutil.disk_usage(ROOT)
dump('cleanup_report.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_free_bytes':before.free,'after_free_bytes':after.free,'actual_freed_bytes':after.free-before.free,'logical_deleted_bytes':sum(x['bytes'] for x in manifest),'operations':len(manifest),'protected_files_verified':len(protected),'protected_hashes_unchanged':True,'invalid_scores_excluded':True,'explanation':explanation,'groups':manifest})
print(json.dumps({'free_GiB':round(after.free/2**30,2),'freed_GiB':round((after.free-before.free)/2**30,2),'protected':len(protected),'raw_cases':len(retained_cases)}),flush=True)
