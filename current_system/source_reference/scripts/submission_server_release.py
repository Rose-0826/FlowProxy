from pathlib import Path
import shutil,os,json,hashlib,datetime,subprocess
R=Path('/root/autodl-tmp/flowproxy').resolve();D=R/'release/AIC2026_20261005';D.mkdir(parents=True,exist_ok=True)
def include(p):
 s=R/p
 if not s.exists():return
 t=D/p;t.parent.mkdir(parents=True,exist_ok=True)
 if not t.exists():os.link(s,t)
# Preserve working source layout; helper imports retain their historical module names.
for folder in ['scripts','src','configs']:
 for p in (R/folder).rglob('*') if (R/folder).exists() else []:
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.yaml','.yml','.json','.txt']:include(p.relative_to(R))
for p in (R/'data/processed').rglob('*'):
 if not p.is_file():continue
 if any(k in str(p) for k in ['corrected','research_v2_outer_support_cases','research_v2_independent8_SEALED_cases','research_v4_sparse_confirmation8_cases','research_v5_split_confirmation8_cases','period_consistency_confirmation4']):include(p.relative_to(R))
for folder in ['outputs/corrected_benchmark_restart_v1','outputs/period_consistency_train_v1','outputs/period_consistency_evaluate_v1','outputs/period_consistency_confirmation4_v1','outputs/research_v2_multidirection']:
 for p in (R/folder).rglob('*'):
  if not p.is_file():continue
  if p.suffix in ['.json','.csv','.md','.txt'] and p.stat().st_size<12*1024**2:include(p.relative_to(R))
for p in ['outputs/period_consistency_train_v1/cycle_seed11/selected.pt','outputs/research_v2_multidirection/fallback_weights.pt','outputs/research_v2_multidirection/final_research_confirmation/selected_research_base_weights.pt']:include(p)
source=R/'outputs/research_v2_multidirection/next_stage_split_observer/new_confirmation8_eval'
for p in (source/'measurements').glob('*.npz'):include(p.relative_to(R))
for case in ['cylinder_Re94.7_cy0.747_D1.165','square_Re167.8_cy-0.743_D0.802']:
 for name in ['frozen_autonomous','v5_inference_split']:
  include((source/'predictions'/(name+'__'+case+'.npz')).relative_to(R))
shutil.copy2(R/'scripts/submission_reproduce_selected.py',D/'reproduce_selected.py')
(D/'requirements.txt').write_text('numpy\nscipy\nh5py\ntorch\nPyYAML\nmatplotlib\n')
readme='''# AIC 2026 有效成果复现包

本包仅提供修正真二维参考上的正式数据、权重、配置和评测；旧配置数据和成绩不计入。部分历史模块名仍作为辅助函数导入，不能据此把历史无效实验恢复为正式结果。

直接复现最新方案（需要CUDA；本次服务器为RTX3090）：
python reproduce_selected.py --case cylinder_Re94.7_cy0.747_D1.165 --score
python reproduce_selected.py --case square_Re167.8_cy-0.743_D0.802 --score

推断只读启动四帧、几何和独立Uy传感文件；--score才在预测保存后读取未来全场。第一个是通过例，第二个是困难例。所有8例完整成绩在对应report.json，预先配置和方法选择记录保留。

核心权重：outer96 seed622 step750，配精确反射与速度边界投影；最终融合用hybrid8_pH1。scripts包含原始实现和复现需要的助手。完整训练过程不是本次命令的内容，不会启动训练或CFD。

清理后完整64×64标签与力历史保留，但多数算例不再保有全部中间原生网格历史。文件使用硬链接归档，修改时先复制为新文件，避免原始证据被改动。上传网盘前可将本目录普通复制为独立文件。

软件权利与来源：团队绕流适配和流程代码、FNO文献、DMD文献、Kalman滤波文献分别注明。OpenFOAM及实际外部代码依其上游许可证；不得以文献引用代替软件授权，原始许可证请与源码一并核对保留。未宣称拥有第三方基础算法知识产权。
'''
(D/'README.md').write_text(readme,encoding='utf-8')
records=[]
for p in D.rglob('*'):
 if p.is_file():records.append({'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
(D/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print('RELEASE',str(D),'FILES',len(records),'LOGICAL_GiB',sum(a['bytes'] for a in records)/2**30,flush=True)
