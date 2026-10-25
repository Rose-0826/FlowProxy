# FlowProxy · 流相：周期绕流场重建

AIC 2026「AI+力学」开放式赛题，团队AIC-2026-64154499。正式系统通过神经空间先验、几何局部启动校准和稀疏相位反馈，重建稳定周期绕流。

## 当前入口

- [current_system/](current_system/README.md)：2026-10-25正式部署源码、复现步骤、观测预算和许可证。
- [代码/数据完整性清单](current_system/manifest_sha256.json)：公开源码与外部模型数据分别核验；大权重与原精度数组不提交到Git。
- [逐例结果与来源](current_system/results/)：冻结后新12例、同批事后控制、另8预算确认及独立PIV适配分别标识。
- [本轮核验记录](current_system/verification/checks_20261025.json)：源文件SHA、实际CPU重放、因果输入检查与默认只读审计。

正式输入是四帧完整Ux/Uy/p启动场和76个后续Uy标量，不能简写为无未来观测的自主预测。公开源码不包含完整模型资源；已有完整交付包时按current_system说明恢复，未发布的网盘地址不以占位链接代替。

## 原有工程

根目录的src/、scripts/、cfd/、autodl/与两个演示目录保留为早期数据生成、UNetEx/FNO和展示代码，不作为当前正式系统入口。9月技术状态保存在[历史README](docs/history/README_2026-09-23.md)。不改写旧实验成绩，不改变已有分支与数据策略。
