# Attribution and publication scope

This portfolio builds on the following upstream work:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): Go2 task, robot configuration, manager-based environment, training/playback entrypoints. Recorded source revision: `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` (local v3.0.0 checkout). Follow the license files at the specific upstream revision/component rather than assuming every component has one license.
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): PPO and on-policy training loop; recorded package `rsl-rl-lib==5.5.1`.
- [Newton](https://github.com/newton-physics/newton), MuJoCo Warp and NVIDIA Warp: training simulation backend.
- NVIDIA Isaac Sim: native runtime, policy runner, PhysX integration and rendering; recorded version 6.2.0.0.
- Unitree Go2 assets accessed through the official Isaac asset distribution; the robot USD files are not included.

Project-specific files implement force curricula, metric calculations, experiment comparison, seed-level statistics, deployment adaptation, controls and documentation. This attribution does not transfer ownership of upstream task definitions, PPO, simulators or robot designs to this project's author.

No upstream source trees, application installers, driver libraries, WebRTC client binaries or robot USD assets are redistributed here. Exported training descriptors may reference upstream classes/assets and remain subject to applicable upstream terms.

The project owner's license for original code and trained artifacts is pending. A public repository allows inspection through GitHub but should not be advertised as permissively licensed until the owner selects a license and verifies any derivative-file obligations. No blanket MIT/BSD license has been assigned automatically.
