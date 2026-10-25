> Historical report. Identifiers were redacted and links adapted for publication. Original machine paths and historical verification statements are not setup instructions.

# Go2 仿真、短训练与视频检查结果

日期：2026-10-22。服务器：`<SSH_HOST>`。

已完成 Go2 资产加载、GPU 仿真、PPO 短训练、模型重新加载、无显示器录像及视频回传。

| 检查 | 实测结果 |
| --- | --- |
| 资产和仿真 | 官方 Go2 USD 成功加载；8 个并行环境运行 64 步，退出码 0 |
| PPO 训练 | 512 个环境、200 次迭代、2,457,600 个环境步，退出码 0 |
| 时间 | 训练循环 33.32 秒；含启动与保存的进程总耗时 45 秒；设置约 5 分钟上限 |
| 模型 | 保存 model_0/50/100/150/199.pt；录像加载 model_199.pt |
| 视频 | 1 个环境运行 500 步；H.264，1280×720，50 FPS，10 秒 |
| 视频验证 | 文件可解码，500 帧；抽取 6 帧检查画面，传输 SHA256 一致 |
| 收尾 | 两张卡均 0% 利用率，无 GPU 计算进程；第二张卡显存 15 MiB |
| 依赖 | uv pip check：215 个包兼容；近景录像脚本编译通过 |

本次实际栈为 Isaac Lab v3.0.0 + Newton/MJWarp 物理后端 + Newton OpenGL 渲染 + RSL-RL PPO。没有安装或验证完整 Isaac Sim/PhysX 运行链路。PyTorch 为 2.11.0+cu128，Python 为 3.12，驱动保留 570.211.01。

视频使用本次新训练的模型，没有替换成预训练模型。Go2 主要保持站立并有少量动作，尚未学会可靠的速度跟踪。训练末段速度跟踪误差仍较大。当前结果证明训练和录像流程可运行，不能作为稳定行走、抗推扰或实机部署的证据。

服务器输出目录：

```text
${GO2_PROJECT_ROOT}/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-10-22_22-44-07_shortcheck_200/
  model_199.pt
  exported/policy.pt
  exported/policy.onnx
  videos/play/clip_model_199_0001.mp4
${GO2_PROJECT_ROOT}/reports/short-check/
  smoke.log
  train.log
  training-summary.json
  video.log
  video-summary.json
  final-check.txt
${GO2_PROJECT_ROOT}/experiments/short-check/record_closeup.py
```

本地交付位于本文件旁的 `runtime-bridge/artifacts/`：

- `go2-ppo-200iter-demo.mp4`：最终近景视频，214723 字节。
- `go2-shortcheck-preview.png`：6 帧预览。
- `training-summary.json`、`video-summary.json`、`final-check.txt`：机器可读结果及收尾检查。

本地新增或调整的操作文件：`runtime-bridge.ps1`、`queue-runtime-job.ps1`、`runtime-smoke.sh`、`runtime-train-short.sh`、`runtime-record-video.sh`、`runtime-transfer-video.sh`、`runtime-final-check.sh` 和只读排查用 `runtime-read-*.sh`/`runtime-inspect-video.sh`。录像脚本只调整镜头，没有修改任务奖励、观测或物理参数。

本次只使用第二张 RTX 5090。没有启动长时间训练，没有更改系统驱动、Cisco VPN、已有 Miniconda/RoboTwin 环境或 shell 启动配置。密码由用户在交互终端输入，未读取或保存。

录像时 CUDA/OpenGL 直接互操作报不可用，渲染器自动回退到拷贝模式；最终视频生成成功。一次使用旧版 `--headless` 参数的录像启动被拒绝，后续改为当前版本支持的 EGL 无显示器渲染。默认镜头太远，最终改用独立近景脚本。首次近景参数覆盖遇到空配置对象，后续已用显式可视化配置解决。

未验证：长训练收敛、多随机种子表现、抗推扰、粗糙地形、实机部署；第一台服务器仍因账号认证未通过而未检查。下一阶段应先确定稳定行走基线，再开展有明确推力、持续时间、方向与成功判据的抗推扰实验。
