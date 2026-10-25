# FlowProxy 当前正式重建系统

团队 AIC-2026-64154499。输入为已知几何、四帧完整Ux/Uy/p启动场与19次×4测点的76个后续Uy标量；输出40秒、160帧、64×64三通道场。两网络按五类八个已训练轮廓路由，启动时确定空间，运行中只以稀疏观测校正相位/频率。

## 运行

本目录公开代码、配置、摘要与SHA，不在Git中存权重、空间库和大数组。完整交付包由团队另行提供，当前没有公开下载链接。已有完整包时，先解压到独立文件夹（应能看到model_payload/和demo_inputs/），用Python3.12环境运行：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r current_system/requirements-cpu-libraries.txt
python current_system/verify_hashes.py
python current_system/restore_assets.py --package /path/to/extracted_complete_package
python current_system/run_demo.py --device cpu --all
```

`verify_hashes.py`核对公开源码/元数据清单；`restore_assets.py`按独立assets_manifest.json恢复并核对外部资源；预测器还会验证两权重八库SHA。Git仓库本身不能在缺少资源时从头推理。完整运行包可直接用自己的run_demo.py运行，不需要再拼装此目录。

CPU以最大绝对差<2e-4核对参考输出，GPU要求逐位一致。推理会生成runtime_cpu或runtime_cuda、replayed_predictions及功能回执。功能计时不能替代报告中RTX3090八轮廓正式基准。模型/空间库合计292,023,712字节。历史从头重训还需要服务器TRAIN156原始数据及历史初始化权重。

## 代码入口

- `run_demo.py`与`bootstrap_portable.py`：76标量正式推理和设备/路径迁移。
- `runtime_template/work/evidence_upgrade_20261007/lean_entry.py`：两网络Predictor与真实模型/库SHA检查。
- `runtime_template/src/flowproxy/models/fno.py`：FNO2d谱卷积实现。
- `runtime_template/work/spatial_observer_20261005/pilot.py`：目标启动附近的短期神经自举。
- `runtime_template/work/shape_expansion_20261006/expansion_core.py`：新增轮廓、自举与整体重建。
- `runtime_template/work/practical_targeted_20261006/target_core.py`：几何局部空间模式查询。
- `runtime_template/work/phase_transport_20261005/full_startup_space.py`：四帧全通道启动空间校准。
- `runtime_template/scripts/research_v4_sparse_observer.py`：输出当前帧后再吸收同刻观测的因果观察器。
- `runtime_template/scripts/research_v5_split_observer.py`：前8秒自主、8—12秒融合。
- `source_reference/`：保留训练、CFD及研发原始源码；历史绝对服务器路径不作为本机部署入口。`runtime_template/`通过bootstrap生成可迁移实际部署副本。
- `external_comparison/`：PIV适配及官方FLRONet紧凑配置，独立于正式三通道系统。

## 只读核验

完成资源恢复后，在仓库根目录运行：

```bash
python current_system/audit_cfd.py
python current_system/audit_piv.py
python current_system/audit_fresh38.py
python current_system/audit_CFD_quality_fresh38.py
python current_system/audit_clock_diagnostics.py
python current_system/verification/check_causality.py --device cpu
```

这些入口默认只输出终端，不写包内文件。clock审计重放固定启动上下文的480组观察器对照，不替代神经网络从头推理。另8例38标量真实推理可用`run_38_demo.py --device cpu --all`；正式系统仍为76标量。

## 结果与数据身份

`results/`保存可追溯轻量原报告和逐例CSV，`results/source_index.json`标明原路径及SHA。

- 冻结后12个已训练轮廓的新参数：40秒12/12，平均R0.633023，最坏0.935574；同批父版7/12，平均0.834115。
- 同12例事后控制：无未来更新7/12、平均1.031658；无空间库12/12、平均0.679472；仅更新相位12/12、平均0.629920。维数结果不单调。
- 另8例固定预算确认：76标量8/8、平均0.653000；38标量7/8、平均0.680589。与原12例分别报告。
- 公开PIV为另训两通道适配器、同次录制时间留出，Uy平均相对L2为46.882%，不作为正式三通道模型的零样本实测证明。

R≤1是内部工程目标，涵盖整段通道均值与末帧误差，非比赛官方评分。四帧完整启动场目前来自OpenFOAM；后续Uy为虚拟测点。CFD已有网格/周期/Courant检查，尚缺正式网格与时间步收敛及外部物理基准。

## 来源与许可证

团队部署代码与自有模型/空间库沿用MIT，见本目录LICENSE。独立external_comparison组件含FLRONet GPL-3.0代码，保留其LICENSE与README。DeepCFD来源见licenses；PIV为Shang/Tu、Zenodo DOI10.5281/zenodo.20765567、CC-BY-4.0，来源在原始PIV配置与报告中。第三方授权优先于团队许可，不将外部基线代码改称原创。
