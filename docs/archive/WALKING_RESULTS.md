> Historical report. Identifiers were redacted and links adapted for publication. Original machine paths and historical verification statements are not setup instructions.

# Go2 平地行走限时训练结果

日期：2026-10-22（Asia/Shanghai）。服务器：`<SSH_HOST>`。

新模型达到本次平地前进行走的验收标准：两个固定随机种子，共 192 次 30 秒试验全部通过，未发生跌倒。旧模型在相同测试协议下的行走通过率为 0%。该结论限于下述仿真条件。

## 时间与训练设置

- 训练前预估：训练 5–10 分钟，含检查和录像约 10–20 分钟。
- 训练进程实际耗时：356 秒，即 **5 分 56 秒**；训练循环约 345.84 秒；正常完成，退出码 0。
- 时间上限：595 秒发送中断，额外 5 秒后强制结束，即训练进程最多 600 秒。外层作业上限 660 秒，留出模型汇总时间。
- 使用第二张 RTX 5090，UUID `<GPU_UUID>`。
- 从先前 `model_199.pt` 恢复策略及优化器，2048 个并行环境，追加 1800 次 PPO 更新，每 25 次保存。
- 新训练采集 88,473,600 个环境步；最终保存 `model_1998.pt`。恢复时 RSL-RL 使用检查点记录的迭代编号，不能仅凭文件名差值推断累计更新次数。
- 保留官方 Go2 平地任务奖励、动作、观测、网络和物理设置；训练中的原有随机推扰继续启用。
- 软件：Isaac Lab v3.0.0、Newton/MJWarp、RSL-RL PPO、PyTorch 2.11.0+cu128；录像用 Newton OpenGL。不是完整 Isaac Sim/PhysX 链路的验证。

## 独立行走测试

种子 42 和 43 各运行 96 个并行环境，每档速度每个种子 32 次试验。每次持续 30 秒，速度等统计跳过最初 2 秒；跌倒判断覆盖完整 30 秒。

| 前进指令 | 试验次数 | 平均实际前进速度 | 平均平面速度误差 | 未跌倒 | 全部指标通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.3 m/s | 64 | 0.319 m/s | 0.053 m/s | 64/64 | 64/64 |
| 0.5 m/s | 64 | 0.545 m/s | 0.062 m/s | 64/64 | 64/64 |
| 0.8 m/s | 64 | 0.865 m/s | 0.082 m/s | 64/64 | 64/64 |

验收门槛在测试前写入脚本：每次试验无跌倒，平均平面速度误差小于 0.20 m/s，平均偏航角速度绝对值小于 0.30 rad/s；至少 95% 的统计时间内机身倾角不超过 30°且高度不低于 0.28 m；平均前进速度达到指令的 65%。整体通过率门槛为 90%，两组实测均为 100%。平面误差是机体坐标下速度误差向量的模长均值，不是位置误差或 RMSE。

测试条件：平地、恒定前进指令、零侧向/转向指令、无推扰、无观测噪声、从静止和标准关节姿态开始；保留随机初始朝向以及材质、机身质量和质心随机化。每个环境首次跌倒都会被记住，自动重置不能抹掉失败。

旧模型同协议种子 42 的 96 次试验：95 次未跌倒，但 **0 次通过行走验收**；平均前进速度约 0.001 m/s。因此本次改善包含真实前进能力，未把“站住不倒”计为成功。

## 模型与视频

服务器模型：

```text
${GO2_PROJECT_ROOT}/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-10-22_23-07-40_walk_bounded600/model_1998.pt
```

同一运行目录中保留中间检查点、任务/PPO 参数、TensorBoard 日志、导出的 `exported/policy.pt` 和 `exported/policy.onnx`。ONNX 权重在同目录的 `policy.onnx.data` 中，迁移 ONNX 模型时必须一起复制。

最终录像为该模型以 0.5 m/s 指令前进的跟随镜头，15 秒、1280×720、50 FPS、H.264、750 帧。已逐帧解码计数，抽取 6 帧检查；机器人保持在画面内。CUDA/OpenGL 直接互操作仍自动回退到拷贝模式，视频生成成功。

- 本地视频：`runtime-bridge/artifacts/go2-walking-demo.mp4`。
- 本地预览：`runtime-bridge/artifacts/go2-walking-preview.png`。
- 汇总：`runtime-bridge/artifacts/walking-summary.json`。
- 逐试验数据：`walking-evaluation-seed42.json`、`walking-evaluation-seed43.json`、`walking-baseline-evaluation.json`，位于同一 artifacts 目录。
- 视频 SHA256：`c38865af875c24551d26f381c97ed889c13bbe993f63af857c032c74072a5a17`。
- 远端报告和评测日志：`${GO2_PROJECT_ROOT}/reports/walking/`。

## 复现与验证

下面的命令用于重新评测现有模型，会短暂占用第二张显卡并写一个新的结果文件，不会训练或修改软件环境。需要先通过 Cisco VPN 登录第二台服务器；密码只在终端输入。

```bash
cd ${GO2_PROJECT_ROOT}/IsaacLab
export CUDA_VISIBLE_DEVICES=<GPU_UUID>
export OMP_NUM_THREADS=8
export WARP_CACHE_PATH=${GO2_PROJECT_ROOT}/cache/warp
timeout --signal=TERM --kill-after=5s 120s .venv/bin/python \
  ${GO2_PROJECT_ROOT}/experiments/walking/walking_evaluate.py \
  --checkpoint ${GO2_PROJECT_ROOT}/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-10-22_23-07-40_walk_bounded600/model_1998.pt \
  --output ${GO2_PROJECT_ROOT}/reports/walking/recheck-seed42.json --seed 42
```

已执行验证：

- 限时训练退出码 0，1800 次更新完成，有新检查点。
- `walking_evaluate.py`、`test_walking_metrics.py`、`walking_record.py` 和 `walking_transfer.py` 编译检查通过。
- `test_walking_metrics.py` 的两个测试通过：站着不动不能通过；速度正确但发生跌倒或低趴也不能通过。
- 旧策略基线、新策略种子 42、新策略种子 43 的真实仿真评测均完成，逐环境数据已保存。
- 视频完整帧数与分辨率检查通过，回传文件 SHA256 校验一致。
- 导出 JIT 模型在 CPU 上完成 48 维输入、12 维有限输出检查；ONNX 模型连同外部权重通过 `onnx.checker.check_model`。
- 收尾时两张显卡均为 0% 利用率，无 GPU 计算进程；第二张卡显存 15 MiB。驱动仍为 570.211.01，215 个依赖包兼容。私有源码副本仍只有之前安装阶段的 `pyproject.toml`、`uv.lock` 差异，本次未新增源码改动。

本地新增文件：`walking-preflight.sh`、`walking-read-eval-api.sh`、`walking-train-bounded.sh`、`walking-evaluate.py`、`test_walking_metrics.py`、`walking-run-evaluations.sh`、`walking-record.py/.sh`、`walking-transfer.py/.sh`、`walking-final-check.sh`、`walking-check-exports.sh` 及本报告；`RESULTS.txt` 补充本次结果。远端只在 `~/go2-rl/experiments/walking`、`reports/walking` 和新训练运行目录增加脚本、数据和模型；没有更改 IsaacLab 源码或依赖，也未改变系统驱动、已有 Miniconda/RoboTwin 环境或 VPN。

未验证：超过 30 秒的长时可靠性、更多随机种子、转向/侧移/后退、外部推扰、粗糙地形、实机部署。192/192 是此次有限样本结果，不能解释为任意条件下都不会跌倒。下一阶段可在这份模型上单独设计抗推扰对照实验。
