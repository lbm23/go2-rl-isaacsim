# Reproduction and verification

## What can be reproduced

- **Offline:** imported-file hashes, Python syntax, links, trial counts, paired-seed effect estimates, exact sign-flip p-values and Holm correction.
- **CPU:** controller/adapter, recovery protocol, curriculum sampling and analysis tests. Numerical suites need NumPy/SciPy/PyTorch; walking metric tests additionally import the recorded Isaac Lab entrypoint.
- **Compatible simulation installation:** frozen-policy walking, push evaluation, deployment diagnostics and the live straight-line scene.
- **Historical training:** code, budgets, seeds, plans and checkpoint identities are documented. Earlier continuation anchors, all intermediate training states and all ablation checkpoints are not bundled. Exact historical end-to-end replay is therefore not claimed.

Public copies preserve numerical outcomes but redact machine paths, server addresses and GPU UUIDs. `results/manifest.json` includes exported and original hashes. Historical hashes cannot validate redacted files or path-adjusted Python files; use the exported hashes for those. Original plan hashes are provenance references, not hashes of public redacted plans.

The public TorchScript export also redacts debug source paths. Its tensor and graph entries are unchanged, and CPU state/output equivalence to the original export was checked. Therefore its public file hash differs from the historical export hash while the policy weights remain the same.

## Offline and CPU checks

These commands do not start a simulator or consume GPU resources:

```bash
python tools/verify_repository.py
python tools/check_results.py
python tools/run_tests.py
```

Use an isolated analysis environment for additional dependencies; installation consumes disk/network and changes that environment only:

```bash
python -m venv .venv
# Activate .venv using your platform's standard activation command.
python -m pip install -r requirements-analysis.txt
python tools/run_tests.py --numerical
```

The exact historical percentile bootstrap intervals use the existing analysis code and fixed RNG plans. `check_results.py` recomputes counts, point estimates and exact tests; it does not regenerate those intervals or simulator trajectories. Historical uncertainty statements retain their original interpretation.

## Simulation environment

Recorded training installation: Isaac Lab local v3.0.0 checkout, revision `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`; Newton 1.6.1, Warp 1.17.0, MuJoCo Warp 3.12.0, RSL-RL 5.5.1, PyTorch 2.11.0+cu128. Deployment is isolated in Isaac Sim 6.2.0.0. See [training package inventory](../environment/training-packages.txt) and [simulation inventory](../environment/simulation-packages.txt).

Obtain upstream applications/assets using their official instructions and terms. The original training checkout had local dependency-manifest changes; their full patches are not included here. Package inventories alone do not recreate editable sources, CUDA/driver compatibility, runtime cache or missing optional extensions. The original server needed user-level matching graphics libraries for streaming; those are machine-specific and are not provided.

No installation, system-driver changes or retraining are performed by this repository's checks. The following examples are manual launch recipes and were adapted for portability; a fresh-machine simulator launch has not been verified.

## Evaluate the bundled checkpoint

On Linux with the recorded training environment installed at `$GO2_PROJECT_ROOT/IsaacLab`, set `GO2_PROJECT_ROOT` to this repository's absolute path. Choose an idle GPU yourself before simulation; these examples do not reserve one automatically. They write new reports and may load the GPU heavily.

```bash
export GO2_PROJECT_ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=0
export GO2_CHECKPOINT="$GO2_PROJECT_ROOT/deployment/isaacsim/artifact/model_7350.pt"
"$GO2_PROJECT_ROOT/IsaacLab/.venv/bin/python" experiments/walking/walking_evaluate.py \
  --checkpoint "$GO2_CHECKPOINT" --seed 940001 --output reports/new-walking.json
"$GO2_PROJECT_ROOT/IsaacLab/.venv/bin/python" experiments/push/push_experiment.py \
  --checkpoint "$GO2_CHECKPOINT" --seed 940001 --samples 64 --output-dir reports/new-push
```

The new seed above is an example, not a claim that it has never been used elsewhere. Freeze a complete independent seed plan before a confirmatory evaluation. Do not compare it directly with archived different-seed results as a paired experiment.

## Training callbacks

These are **optional future training commands**, included to document the workflow. They were not executed during repository preparation. They create model/log files and consume GPU time. Earlier base checkpoints must first be supplied and hash-verified using the recorded plans.

```bash
export PYTHONPATH="$GO2_PROJECT_ROOT/experiments/robust:$GO2_PROJECT_ROOT/experiments/left2"
export GO2_LEFT_REPORT_DIR="$GO2_PROJECT_ROOT/reports/new-left2"
mkdir -p "$GO2_LEFT_REPORT_DIR"
export GO2_LEFT_START_EPOCH=$(date +%s)
timeout --signal=INT --kill-after=5s 590s \
  "$GO2_PROJECT_ROOT/IsaacLab/.venv/bin/python" experiments/left2/left2_train.py \
  --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint "$GO2_BASE_4575" \
  --num_envs 2048 --max_iterations 4000 --seed 17 --logger tensorboard \
  --run_name left2_reproduction --viz none physics=newton_mjwarp agent.save_interval=25
```

The targeted curriculum is wall-time based, so different hardware need not reproduce the same updates. `robust_train.py` uses `GO2_ROBUST_REPORT_DIR`. The fixed-update ablation uses `GO2_AB_CASE`, `GO2_AB_BASE`, `GO2_AB_SEED`, `GO2_AB_GROUP` and the training CLI. Its callbacks enforce 2,400 updates and verify restored state; the archived experiment specification records randomization and budget choices. Low-level scripts are version-specific adapters, not a portable experiment scheduler.

## Isaac Sim diagnostics and live demo

Use the simulation environment's configured Python launcher, represented here by `$ISAAC_SIM_PYTHON`. It must already provide Isaac Sim, rendering and the native `isaacsim.robot.policy.examples` extension. Do not install simulation packages into the training environment.

These commands start physics/rendering and write diagnostic JSON under the demo directory. They leave actor weights unchanged:

```bash
export GO2_DEMO_ROOT="$GO2_PROJECT_ROOT/deployment/isaacsim"
"$ISAAC_SIM_PYTHON" "$GO2_DEMO_ROOT/go2_yaw_diagnostics.py" \
  --variant baseline --speed 0.5 --seconds 20
"$ISAAC_SIM_PYTHON" "$GO2_DEMO_ROOT/go2_yaw_diagnostics.py" \
  --variant feedback --speed 0.5 --seconds 30 --hip-bias 0.1
```

For the live UI, launch the configured Isaac Sim/WebRTC application, save valuable scene work, and execute `deployment/isaacsim/go2_straight_live.py` in its Script Editor. Set `GO2_DEMO_ROOT` in the application's environment, or execute the actual script file with its filename preserved (it uses `__file__` to locate artifacts). Pasting text without `__file__` requires setting it explicitly. This script backs up the stage and replaces it with a Go2 demo scene. Existing unsaved work should not be relied upon after replacement.

The baseline `go2_policy_demo.py` main block intentionally uses the original transfer configuration for diagnosis. Use the feedback diagnostic or live demo for the final corrected behavior. No guarantee is made for other policy-runner versions.
