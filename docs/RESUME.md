# 简历与展示素材

项目名称建议：**Go2 强化学习行走与跨物理引擎部署** / **Go2 RL Locomotion and Cross-Engine Deployment**。

仅使用自己实际完成、能够讲清楚并复现的部分。基础任务与 PPO 来自 Isaac Lab/RSL-RL，不能写成从零实现 PPO 或提出了新算法。

## 中文项目描述

基于 Isaac Lab 与 RSL-RL PPO 搭建 Go2 行走训练及抗扰评估流程，采用 2,048 个并行环境开展继续训练；构建包含真实质心外力、跌倒追踪及恢复判据的评估协议，并使用独立种子、配对统计与多训练种子消融检验改进效果。完成 Newton/MJWarp 至 Isaac Sim/PhysX 的模型迁移，定位积分步长和指令接口差异，通过航向反馈与后髋动作补偿，在 0.5 m/s、30 秒平地演示中将最大横向偏差控制在 4.17 cm，最大偏航 1.82°。

可按岗位选用的两条：

- 构建四方向、多推力等级的抗扰评估与失败追踪；固定模型在 16 个配对评估种子下，左侧 2 BW 跌倒率由 53.81% 降至 16.31%，并保留其他方向恢复速度变慢的代价分析。
- 完成三组、八个训练种子的受控消融及 435,200 次矩阵评估，使用训练种子作为统计单位；如实报告主比较不显著，区分固定模型改进与算法层面的因果证据。

## English description

Built a Go2 locomotion training and robustness evaluation workflow using Isaac Lab and RSL-RL PPO, with 2,048 parallel environments, verified center-of-mass force injection, failure-aware recovery metrics, paired-seed inference, and a three-recipe/eight-training-seed ablation. Transferred a frozen policy from Newton/MJWarp to Isaac Sim/PhysX, diagnosed integration and command-interface differences, and achieved 4.17 cm maximum cross-track deviation and 1.82° maximum heading error in a 30 s, 0.5 m/s flat-ground demonstration using deployment corrections and external feedback.

## 面试时需要能解释

1. 为什么不跌倒不等于会行走？如何防止自动重置掩盖失败？
2. BW 是力还是冲量？为什么推扰方向要在施力开始时固定？
3. 为什么不能把几千个并行环境当成几千次独立训练？
4. 固定模型改善显著、消融不显著，两者为什么不矛盾？
5. 50 Hz 控制周期不变，积分步长为什么仍会影响跨引擎行为？
6. 航向反馈、后髋偏置各解决什么问题，哪些结论还没有验证？

发布后在简历项目标题旁添加实际 GitHub 仓库链接，不使用尚未创建的占位链接。
