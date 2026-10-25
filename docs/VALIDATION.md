# Publication-copy validation

Date: 2026-10-25. These checks validate the curated copy; they do not constitute a new simulation or training run.

| Check | Where | Result |
|---|---|---|
| `python tools/verify_repository.py` | Local Python 3.13 | Imported checksums, Python syntax, JSON/gzip parsing, Markdown links, file types/sizes and text privacy patterns passed |
| `python tools/check_results.py` | Local Python 3.13 | Raw left-push trial counts, paired-seed effect, exact p-values, ablation means/Holm correction and saved deployment telemetry agree |
| `python tools/run_tests.py` | Local Python 3.13 | 12/12 tests passed, no extra packages |
| `python tools/run_tests.py --numerical --isaaclab` | Existing training environment, Python 3.12.15 | 46/46 tests passed across 9 isolated suites; no skipped tests |
| [GitHub CPU checks, run #1](https://github.com/lbm23/go2-rl-isaacsim/actions/runs/40203944880) | GitHub-hosted Ubuntu, Python 3.12 | Passed after publication: repository integrity, recorded-results checks and 12 basic CPU tests |

The remote check used a temporary copy of the curated Python sources and the existing environment. CUDA devices were hidden, no simulator loop was started, and no dependencies or model weights were modified. The numerical tests include a SciPy exact-test cross-check, incomplete-pair exclusion, curriculum-stage accounting, force removal, action/observation adaptation and fixed-update accounting.

The final exported checkpoint was copied from the existing server, with SHA256 checked before local writing. The final policy and YAML descriptors are included. A publication manifest records imported evidence and code hashes; documentation and utility scripts are tracked by Git rather than that import manifest.

TorchScript debug source paths were redacted in seven archive records using an equal-length replacement. Every non-debug archive entry, including tensor data and executable graph code, remained byte-identical. The published file was loaded in the existing CPU PyTorch environment; all state tensors and outputs on 128 fixed random observations matched the original exactly. See `results/validation/policy-export.json`.

## Practical limits

- Fresh-machine environment installation and full historical training replay were not tested. Earlier training anchors and all ablation checkpoints are not bundled.
- The path-adjusted simulation scripts were syntax-checked; the packaged copy was not relaunched in Isaac Sim. Saved videos and simulator measurements are the earlier project evidence.
- GitHub Actions covers offline checks and the 12 basic CPU tests. It does not run the numerical/simulator-dependent suites or simulate the robot; the 46-test result above comes from the existing training environment.
- Text scanning cannot establish third-party redistribution rights or guarantee that every binary is free of sensitive content. The included rendered previews were inspected for the project demonstration; applications, connection logs and robot assets are excluded.
- Bootstrap intervals remain the recorded historical analysis results; the offline headline checker recomputes point estimates and exact tests, not every confidence interval.
- No hardware or post-offset push validation; 0.3 m/s startup exceeds the 3° heading threshold.
