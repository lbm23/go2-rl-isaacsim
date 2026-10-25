# Go2 model card

## Identity and lineage

- Robot/task: Unitree Go2, official Isaac Lab flat-ground velocity task.
- Algorithm: upstream RSL-RL PPO, project-specific training callbacks/curricula.
- Training physics: Newton/MJWarp. Deployment: Isaac Sim/PhysX.
- Observation/action dimensions: 48 / 12; policy period 0.02 s.
- Deployed checkpoint: `deployment/isaacsim/artifact/model_7350.pt`.
- Checkpoint SHA256: `ef2557038287f8cd9f8751bd9f00b33a862cf4f947f4336ce9d39b7bcd1d32d3`.
- TorchScript export: `deployment/isaacsim/artifact/policy.pt`; published and original checksums are listed in `results/manifest.json`. Source paths in seven debug records were redacted; tensor and graph entries were preserved byte-for-byte.
- Export verification: all eight actor tensors match the checkpoint, as recorded in `results/data/straight-results.json`.
- Publication verification: the redacted file loads successfully, with identical state tensors and exactly identical outputs for 128 fixed random observations compared with the original export.

Historical anchors:

| Checkpoint | SHA256 |
|---|---|
| model_1998 | `f6e4aefb206dfc78d6b78d50cde70b6c273dda44300fa38bc6379766fc5f1ca1` |
| model_4575 | `3dd78eba34c2ce81235c93a31daaca09151d089f3ac0f16df71c85e36b15b994` |
| model_7350 | `ef2557038287f8cd9f8751bd9f00b33a862cf4f947f4336ce9d39b7bcd1d32d3` |

## Intended use

Simulation research, portfolio demonstrations and analysis of the recorded experiments. The final deployment demonstration combines the frozen policy with bounded heading/cross-track feedback and rear hip action offsets; reported straightness does not describe raw policy output alone.

## Limitations

No hardware, rough-terrain or post-offset push validation. No guarantee of fall-free operation. Simulator pose is assumed available. Results compare fixed checkpoints; the multi-training-seed ablation did not establish a statistically significant advantage for the custom curriculum.

Only the final checkpoint/export is bundled. Earlier continuation anchors, all 24 ablation checkpoints and complete simulator trajectory traces are not bundled; hashes and trial/statistical outcomes remain available. Historical training cannot be reconstructed byte-for-byte from this repository alone.

## Loading and rights

Load only trusted model files. PyTorch checkpoints and TorchScript files are executable serialization formats, not passive documents. Verify the supplied checksum before loading. The project's code/model reuse license has not yet been selected by the owner; see `THIRD_PARTY.md`. Robot assets and simulator software must be obtained under their upstream terms.
