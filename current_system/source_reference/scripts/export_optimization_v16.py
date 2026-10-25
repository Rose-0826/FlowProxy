from pathlib import Path
import json,hashlib,datetime,tarfile,shutil,subprocess,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/aic_optimization_2026-10-02'
S=OUT/'snapshots';S.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
sources=['hybrid_observation_budget_v1','hybrid_twoframe_confirmation_v1','hybrid_twoframe_history_v1','cfd_numerics_round2_v1','hybrid_refined_reference_v1','mesh_2d_repair_probe_v1','true2d_reference_pilot_v1']
assert (ROOT/'outputs/cfd_numerics_round2_v1/sensitivity_report.json').exists()
assert json.loads((ROOT/'outputs/hybrid_refined_reference_v1/status.json').read_text())['status']=='complete'
for name in sources:
    src=ROOT/'outputs'/name
    for p in src.rglob('*'):
        rel=p.relative_to(src)
        if not p.is_file() or 'predictions' in rel.parts:continue
        if p.suffix not in ['.json','.csv','.log','.npz']:continue
        dst=S/name/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
scripts=['hybrid_observation_budget_v1.py','hybrid_twoframe_confirmation_v1.py','hybrid_twoframe_history_v1.py',
 'cfd_numerics_round2_v1.py','cfd_numerics_round2_completion_audit_v1.py','hybrid_refined_reference_v1.py',
 'cfd_numerics_analyze_v1.py','aic_evidence_animation.py','fourframe_new4_diagnostic.py','export_optimization_v16.py','mesh_2d_repair_probe_v1.py','build_true2d_case_v1.py']
for name in scripts:
    p=ROOT/'scripts'/name;dst=OUT/'code'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
expected={'outputs/checkpoints/fno_long/best.pt':'3276f4c7564d57543ae4f7f323202d63b47f919e6003b6857bd84832353cd9c4',
 'outputs/checkpoints/fno_k8_base/base.pt':'1430876e4c429efc3ec28548540640f6f12398be02b8b37c22de5881333feada',
 'outputs/startup_observability_v1/training_priors.json':'a280722c200af5c6eee022ee50ca221a7e952705193d3595abb0b81a4273a361',
 'outputs/self_trajectory_closure_v1/selected_val.json':'cf3368baa68328d6e36861d527a7f866afe44cad48cbcc418e01649a652216c9'}
frozen=[]
for file,digest in expected.items():
    got=sha(ROOT/file);assert got==digest,(file,got);frozen.append({'file':file,'sha256':got,'unchanged':True})
registries=[ROOT/'outputs'/n/'predictions_frozen.json' for n in ['hybrid_observation_budget_v1','hybrid_twoframe_history_v1','hybrid_refined_reference_v1']]
registries +=[ROOT/'outputs/hybrid_twoframe_confirmation_v1/evaluation/predictions_frozen.json',ROOT/'outputs/hybrid_twoframe_confirmation_v1/fourframe_posthoc_control/predictions_frozen.json']
preds=[]
for registry in registries:
    for r in json.loads(registry.read_text())['rows']:
        assert sha(r['file'])==r['sha256'];preds.append({'file':r['file'],'sha256':r['sha256'],'registry':str(registry)})
mesh=[]
for p in (ROOT/'data/raw/cfd_numerics_round2_v1').rglob('log.checkMesh'):
    text=p.read_text();match=re.search(r'^\s*cells:\s*(\d+)',text,re.MULTILINE)
    failed=re.findall(r'Failed (\d+) mesh checks',text)
    mesh.append({'case':str(p.parent),'cells':int(match[1]) if match else None,'mesh_OK':'Mesh OK.' in text,
      'failed_check_count':int(failed[-1]) if failed else 0,
      'non_aligned_empty_direction_edges':re.findall(r'Number of edges not aligned with or perpendicular to non-empty directions:\s*(\d+)',text),
      'checkMesh_sha256':sha(p)})
    dst=OUT/'mesh_logs'/(p.parent.parent.name+'__'+p.parent.name+'_checkMesh.log');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
assert not any(r['mesh_OK'] for r in mesh), 'Observed old mesh failures must remain visible'
save(OUT/'reference_quality_blocker.json',{'legacy_and_round2_refined_meshes_fail_2D_geometry_check':True,
  'original_single_layer_blockMesh_was_split_into_z_layers_by_snappy_refinement':True,
  'old_z_planes':[0,.025,.05,.075,.1],'corrected_probe_z_planes':[0,.1],
  'old_AI_scores_remain_only_fixed_discrete_reference_comparisons':True,
  'new_corrected_mesh_2s_runs_not_new_AI_scores':True,'physical_accuracy_claims_blocked_until_corrected_reference_is_validated':True})
for d in ['mesh_2d_repair_probe_v1','true2d_reference_pilot_v1']:
    for case in (ROOT/'data/raw'/d).iterdir():
        if case.name.endswith('_surface_source'):continue
        for p in case.iterdir():
            if p.is_file() and (p.name.startswith('log.') or p.name in ['true2d_build.json','Allrun_true2d']):
                dst=OUT/'mesh_repair_cases'/d/case.name/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
save(OUT/'frozen_and_prediction_audit.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen':frozen,'predictions':preds,'all_prediction_hashes_matched':True,'meshes':mesh,'heavy_prediction_files_retained_on_server_only':True})
owned=['cfd_numerics_round2_v1.py','cfd_numerics_round2_completion_audit_v1.py','hybrid_refined_reference_v1.py',
 'hybrid_twoframe_confirmation_v1.py','hybrid_twoframe_history_v1.py','aic_evidence_animation.py','fourframe_new4_diagnostic.py','mesh_2d_repair_probe_v1.py']
ps=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
active=[l for l in ps.splitlines() if any('/'+s in l for s in owned) or ('icoFoam' in l and 'icoFoam' in l.split()[1])]
assert not active,active
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
save(OUT/'resource_audit.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owned_active_processes':active,'gpu':gpu,'disk_free_bytes':shutil.disk_usage(ROOT).free,'no_new_training':True})
records=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='export_manifest.json':records.append({'file':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)})
save(OUT/'export_manifest.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,'not_full_raw_CFD_or_checkpoint_backup':True})
archive=Path('/tmp/aic_optimization_v16.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    tar.add(OUT,arcname=OUT.name)
    tar.add(ROOT/'outputs/aic_showcase_2026-10-02',arcname='aic_showcase_2026-10-02')
print('EXPORTED',len(records),archive.stat().st_size,sha(archive)[:32],flush=True)
print(sha(archive)[32:],flush=True)
