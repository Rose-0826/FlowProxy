from pathlib import Path
import json,shutil,hashlib
R=Path('/root/autodl-tmp/flowproxy');A=R/'outputs/mainline_cleanup_20261005'
core={'corrected_benchmark_quality_v1','corrected_benchmark_restart_v1','period_consistency_train_v1','period_consistency_evaluate_v1','period_consistency_confirmation4_v1','research_v2_multidirection','mainline_cleanup_20261005'}
moved=[]
for p in list((R/'outputs').iterdir()):
    if p.name in core or p.is_symlink():continue
    assert R in p.resolve().parents
    target=A/'historical_reports'/p.name;target.parent.mkdir(exist_ok=True)
    assert not target.exists()
    shutil.move(str(p),str(target));moved.append({'original':str(p),'archive':str(target),'role':'REPORT_OR_AUDIT_ONLY_NOT_AN_ACTIVE_CANDIDATE'})
(A/'historical_path_index.json').write_text(json.dumps(moved,ensure_ascii=False,indent=2))
weight=R/'outputs/research_v2_multidirection/final_research_confirmation/selected_research_base_weights.pt'
allweights=[p for p in R.rglob('*.pt') if p.is_file()]
assert len(allweights)==2 and len({(p.stat().st_dev,p.stat().st_ino) for p in allweights})==1
with weight.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest=='9b860db72359a0e874a7e5942ea2ccd14d2f1ea29339af49dd435a89128ac1aa'
(R/'CURRENT_MAINLINE.json').write_text(json.dumps({'model':'TRAIN96 spatial neural prior + causal sparse observer + startup pressure calibration + exact short8 preservation','weight':str(weight),'weight_sha256':digest,'other_active_weights':0,'hardlinked_release_is_same_model':True,'results':str(R/'outputs/research_v2_multidirection/next_stage_split_observer/new_confirmation8_eval/report.json'),'reproducer':str(R/'release/MAINLINE_20261005/reproduce_selected.py'),'data_roles':'TRAIN96; previously seen DEV56; 3 separate version-specific confirmation batches8, not pooled','old_configuration_scores_excluded':True,'history_index':str(A/'historical_path_index.json')},ensure_ascii=False,indent=2))
(R/'README_MAINLINE.md').write_text('''# 当前唯一主线\n\n入口与冻结权重在CURRENT_MAINLINE.json。只保留一份正式模型；release中的同名权重是同一文件的硬链接。当前最新8例40秒4/8通过，平均R1.406；不能写成全部过线。\n\n数据与参考：data/processed保留主线训练和评测所需标签；data/raw保留修正配置的对应原生网格、边界、初始/终点场和数值质量参考。其余训练权重、大型预测银行、无关CFD路线及旧配置数据已按删除清单清理。\n\noutputs/的period_consistency等目录只保留主线来源、配置、开发角色及历史结果的必要支撑，不再是可选候选；research_v2_multidirection中旧子路线同样只剩报告。代码保留历史名字是因为当前方法导入其中的纯辅助函数。历史训练入口不承诺可从已删除父权重继续执行。\n\n旧配置错误与旧数据清理记录、停止路线的报告和原路径索引在outputs/mainline_cleanup_20261005。配置错误对应的成绩排除；正确配置上的失败记录仍是有效实验记录。\n\n现有参赛材料仍作保底，release/competition_materials_20261005.zip不改。新研究建议另外归档，没有自动启动训练或CFD。\n''',encoding='utf8')
# Remove the duplicate prediction emitted for post-deletion verification; retain
# its JSON numerical proof. Reproduction/package contents otherwise unchanged.
for p in (R/'release/MAINLINE_20261005/reproduced').glob('*.npz'):p.unlink()
usage=shutil.disk_usage(R)
(A/'final_inventory.json').write_text(json.dumps({'free_bytes':usage.free,'used_bytes':usage.used,'single_unique_active_weight':True,'active_weight_sha256':digest,'moved_history_entries':len(moved),'retained_data_native_and_sampled':True},indent=2))
print((A/'final_inventory.json').read_text())
