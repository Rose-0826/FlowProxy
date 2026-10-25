"""Curate one current line, retain compact evidence, remove obsolete bulk files."""
from pathlib import Path
import os,json,hashlib,gzip,shutil,tarfile,time,argparse
R=Path('/root/autodl-tmp/flowproxy')
A=R/'outputs/mainline_cleanup_20261005'
D=R/'release/MAINLINE_20261005'
LATEST='outputs/research_v2_multidirection/next_stage_split_observer/new_confirmation8_eval'
WEIGHT='outputs/research_v2_multidirection/final_research_confirmation/selected_research_base_weights.pt'
OLD=['eval','decision_20260930','stop_engineering_2026-10-02','noise_robustness_v1','noise_robustness_v2_B1','frozen_evidence_figure','frozen_evidence_figure_v2','solid_state_probe_v1','startup_observability_v1','self_trajectory_closure_v1','hybrid_controls_v1','nearest_cycle_template_v1','template_interpolation_v1','far_controls_v1','fno_training_closure_ablation_v1','hybrid_timing_noise_v1','aic_attack_2026-10-02','hybrid_start_phase_v1','aic_showcase_2026-10-02','cfd_numerics_restart_v1','cfd_numerics_round2_v1']
DROP_RAW=['cfd_numerics_restart_v1','cfd_numerics_round2_v1','mesh_2d_repair_probe_v1','true2d_reference_pilot_v1','true2d_recheck_v1','true2d_time_control_v1','continuous_training_cfd_v1','research_v2_hybrid_training_pilot','research_v3_corner_support']
DROP_PROCESSED=['continuous_training_cfd_v1_cases','research_v3_corner_support_cases']
KEEP_PROCESSED=['corrected_benchmark_restart_v1_cases','period_consistency_confirmation4_v1_cases','research_v2_outer_support_cases','research_v2_independent8_SEALED_cases','research_v4_sparse_confirmation8_cases','research_v5_split_confirmation8_cases']
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def within(p):
    assert p!=R and R in p.resolve().parents and not p.is_symlink(),str(p)
    return p
def add(p):
    rel=p.relative_to(R);t=D/rel;t.parent.mkdir(parents=True,exist_ok=True)
    if not t.exists():os.link(p,t)
def small(p):return p.suffix in ['.json','.jsonl','.csv','.md','.txt','.yaml','.yml','.py','.log'] and p.stat().st_size<12*1024**2
def compact_copy(p,kind):
    if p.is_symlink() or not p.is_file():return
    if small(p) or p.name.startswith('log.') and p.stat().st_size<1024**2 or p.name in ['controlDict','fvSchemes','fvSolution','transportProperties','blockMeshDict','snappyHexMeshDict','boundary'] and p.stat().st_size<1024**2:
        t=A/'evidence'/kind/p.relative_to(R);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
def treefiles(folder):
    for base,ds,fs in os.walk(folder,followlinks=False):
        ds[:]=[d for d in ds if not (Path(base)/d).is_symlink()]
        for n in fs:yield Path(base)/n

def prepare():
    assert R.resolve()==R
    A.mkdir(exist_ok=False);D.mkdir(exist_ok=False)
    before=shutil.disk_usage(R);deletes={}
    def remove(p,reason):
        within(p)
        if p.is_file():deletes[str(p.relative_to(R))]=reason
    for name in OLD:
        folder=R/'outputs'/name
        for p in treefiles(folder) if folder.exists() else []:
            compact_copy(p,'EXCLUDED_OLD_CONFIGURATION');remove(p,'EXCLUDED_OLD_CONFIGURATION')
    for name in DROP_RAW:
        folder=R/'data/raw'/name
        for p in treefiles(folder) if folder.exists() else []:
            compact_copy(p,'RAW_REFERENCE_OR_STOPPED_BRANCH');remove(p,'EXCLUDED_OLD_CONFIGURATION' if name.startswith('cfd_numerics') else 'SUPERSEDED_DIAGNOSTIC_OR_STOPPED_VALID_BRANCH')
    for name in DROP_PROCESSED:
        for p in treefiles(R/'data/processed'/name):
            compact_copy(p,'STOPPED_VALID_BRANCH');remove(p,'VALID_UNUSED_BY_SELECTED_TRAIN96')
    # Failed directions retain compact configuration/results. Full-field banks
    # and all nonselected checkpoints are no longer active competition artifacts.
    for p in treefiles(R/'outputs'):
        if A in p.parents or not p.is_file() or p.is_symlink():continue
        rel=str(p.relative_to(R))
        if rel in deletes:continue
        if p.suffix in ['.pt','.pth','.ckpt'] and rel!=WEIGHT:remove(p,'NONSELECTED_WEIGHTS_REPORT_RETAINED')
        if p.suffix in ['.npz','.npy','.h5','.hdf5']:
            keep=(rel.startswith(LATEST+'/') or '/measurements/' in rel or p.name.startswith('tasks_') or 'corrected_benchmark_quality_v1' in rel or p.name.endswith('_field_statistics.npz'))
            if not keep:remove(p,'OBSOLETE_PREDICTION_OR_DIAGNOSTIC_BANK_REPORT_RETAINED')
        if p.suffix in ['.gz','.zip','.tar'] and not any(k in rel for k in ['submission_cleanup_20261005','obsolete_raw_cleanup_20261003']):
            remove(p,'DUPLICATE_EXPORT_SUPERSEDED_BY_MAINLINE_ARCHIVE')
    for folder in [R/'outputs/submission_evidence_export_20261005',R/'outputs/submission_preview_20261005',R/'tmp']:
        for p in treefiles(folder) if folder.exists() else []:
            compact_copy(p,'DUPLICATE_EXPORT_OR_TEMP');remove(p,'DUPLICATE_EXPORT_OR_TEMP')
    # Compact all remaining historical reports in a single indexed archive.
    for p in treefiles(R/'outputs'):
        if A in p.parents or p.is_symlink() or not p.is_file():continue
        if small(p):compact_copy(p,'COMPACT_REPORTS')
    # Mainline reproducibility package: one weight, complete current labels,
    # same-input controls, causal sensor payloads, all latest frozen predictions.
    for folder in ['scripts','src','configs','cfd']:
        for p in treefiles(R/folder):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and small(p):add(p)
    for name in KEEP_PROCESSED:
        for p in treefiles(R/'data/processed'/name):
            if p.is_file():add(p)
    for name in ['dataset_corrected_trainval_v1.h5','dataset_corrected_benchmark_v1.h5']:add(R/'data/processed'/name)
    for folder in ['outputs/corrected_benchmark_restart_v1','outputs/period_consistency_train_v1','outputs/period_consistency_evaluate_v1','outputs/period_consistency_confirmation4_v1','outputs/corrected_benchmark_quality_v1','outputs/research_v2_multidirection']:
        for p in treefiles(R/folder):
            if p.is_file() and not p.is_symlink() and small(p):add(p)
    add(R/WEIGHT)
    for p in treefiles(R/LATEST):
        if p.is_file() and p.suffix=='.npz':add(p)
    for folder in ['outputs/research_v2_multidirection/sparse_observer_pilot/measurements','outputs/research_v2_multidirection/sparse_confirmation8_eval/measurements']:
        for p in treefiles(R/folder):
            if p.is_file():add(p)
    for p in (R/'outputs/research_v2_multidirection/outer_support_training').glob('tasks_*.npz'):add(p)
    old_release=R/'release/AIC2026_20261005'
    shutil.copy2(old_release/'reproduce_selected.py',D/'reproduce_selected.py')
    for name in ['requirements.txt','requirements-verified.txt']:
        if (old_release/name).exists():shutil.copy2(old_release/name,D/name)
    if (old_release/'licenses').exists():shutil.copytree(old_release/'licenses',D/'licenses')
    (D/'README.md').write_text('''# 唯一当前主线 / MAINLINE_20261005\n\nTRAIN96 神经空间先验、周期延拓、4点Uy因果观测、启动压力校准和前8秒自主预测保留。唯一正式权重为 final_research_confirmation/selected_research_base_weights.pt；训练来源 outer96 seed622 step750。\n\n复现：python reproduce_selected.py --case cylinder_Re94.7_cy0.747_D1.165 --score\n困难例：python reproduce_selected.py --case square_Re167.8_cy-0.743_D0.802 --score\n预测只读取目标启动4帧和因果稀疏Uy观测，--score在保存预测后读取全场真值。\n\n本包保留TRAIN96、已见DEV56、三批各8组的对应版本确认数据、最新8例全部冻结预测与关键对照。三个确认版本不能合并成最终24例成绩。最新8例：40秒4/8通过、平均R1.406、末端Uy7.002%、p4.674%，预定6/8和R<1门槛未达。\n\n历史代码/配置/报告是来源与失败方向的说明，属于REPORT_ONLY，不是可选候选；旧父权重已删除。历史训练入口可能仍引用已删除父权重，应从本复现入口验证当前模型。当前权重包含初始化之后的最终参数，可直接推断/从当前权重开展新实验，但本包不承诺重演已清理的每个历史训练分支。\n\n主线所需原生网格、边界、初始/终点场及两个数值质量参考算例留在aic原工作目录data/raw；大量中间原生时间帧已在此前清理。压缩包只包含采样标签，不是全部原生CFD中间历史备份。\n''',encoding='utf8')
    (A/'STATUS.md').write_text('''# 清理范围\n\n旧配置：厚度方向局部多层网格却使用empty边界，checkMesh报告非1D/2D。相关数据和成绩排除，配置/log与逐文件删除清单保留。\n\n正确配置上停止的路线不是无效实验：它们留下配置和结果，删除非选中权重、大型预测和无关原始场。当前只保留一个正式模型；自主分支和相同预算对照是该主线验收证据，必须保留。\n\n参赛材料保底包不更改。过去大量删除记录改为无损gzip保存。\n''',encoding='utf8')
    records=[]
    with gzip.open(A/'deletion_plan.jsonl.gz','wt') as f:
        for rel,reason in sorted(deletes.items()):
            p=R/rel;s=p.stat();rec={'path':rel,'bytes':s.st_size,'inode':s.st_ino,'nlink':s.st_nlink,'reason':reason}
            f.write(json.dumps(rec)+'\n')
    for p in treefiles(D):
        if p.is_file():records.append({'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':sha(p)})
    (D/'manifest.json').write_text(json.dumps(records,indent=2))
    protected=[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in treefiles(R/'data/processed') if p.is_file() and str(p.relative_to(R)) not in deletes]
    protected.append({'path':WEIGHT,'sha256':sha(R/WEIGHT)})
    (A/'protected_before.json').write_text(json.dumps(protected,indent=2))
    report={'disk_before':{'used':before.used,'free':before.free},'delete_files':len(deletes),'planned_logical_bytes':sum((R/r).stat().st_size for r in deletes),'protected_files':len(protected),'package_files':len(records),'package_logical_bytes':sum(r['bytes'] for r in records),'selected_weight_sha256':sha(R/WEIGHT),'manifest_sha256':sha(D/'manifest.json'),'plan_sha256':sha(A/'deletion_plan.jsonl.gz')}
    (A/'prepared.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

def verify():
    for q in json.loads((D/'manifest.json').read_text()):assert sha(D/q['path'])==q['sha256'],q['path']
    print('MAINLINE_MANIFEST_VERIFIED',flush=True)

def execute():
    verify();info=json.loads((A/'prepared.json').read_text());assert sha(A/'deletion_plan.jsonl.gz')==info['plan_sha256']
    before=shutil.disk_usage(R)
    with gzip.open(A/'deletion_plan.jsonl.gz','rt') as f:
        for line in f:
            q=json.loads(line);p=within(R/q['path']);assert p.stat().st_size==q['bytes'] and p.stat().st_ino==q['inode'];p.unlink()
    # Prior release was broader and contains extra weights. Its replacement has
    # verified per-file hashes and must already have a verified storage mirror.
    mirror=json.loads((A/'storage_mirror_verified.json').read_text());assert mirror['archive_sha256']==sha(R/'release/MAINLINE_20261005.tar.gz')
    for p in [R/'release/AIC2026_20261005',R/'release/AIC2026_20261005_有效代码数据模型.tar.gz']:
        within(p)
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    # Preserve large previous deletion audit byte-for-byte using compression.
    previous=R/'outputs/submission_cleanup_20261005';compressed=[]
    for name in ['cleanup_report.json','deleted_files.jsonl']:
        p=previous/name
        if not p.exists():continue
        expected=sha(p);out=p.with_name(name+'.gz')
        with p.open('rb') as src,gzip.open(out,'wb',compresslevel=6) as target:shutil.copyfileobj(src,target,1024**2)
        h=hashlib.sha256()
        with gzip.open(out,'rb') as f:
            while chunk:=f.read(1024**2):h.update(chunk)
        assert h.hexdigest()==expected
        compressed.append({'original':str(p.relative_to(R)),'compressed':str(out.relative_to(R)),'original_sha256':expected,'original_bytes':p.stat().st_size,'compressed_bytes':out.stat().st_size});p.unlink()
    for q in json.loads((A/'protected_before.json').read_text()):assert sha(R/q['path'])==q['sha256'],q['path']
    verify()
    # Remove empty directories from deleted trees only; no paths are inferred.
    for name in OLD+DROP_RAW+DROP_PROCESSED:
        for prefix in ['outputs','data/raw','data/processed']:
            folder=R/prefix/name
            if folder.exists():
                for b,ds,fs in os.walk(folder,topdown=False):
                    p=Path(b)
                    if not any(p.iterdir()):p.rmdir()
    after=shutil.disk_usage(R);result={'deleted_files':info['delete_files'],'protected_hashes_unchanged':True,'package_manifest_verified':True,'freed_bytes':after.free-before.free,'free_after':after.free,'used_after':after.used,'compressed_previous_audits':compressed,'utc':time.time()}
    (A/'deletion_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','verify','execute']);v=p.parse_args()
    {'prepare':prepare,'verify':verify,'execute':execute}[v.mode]()
