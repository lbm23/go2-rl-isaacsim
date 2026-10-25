"""Record a bounded 0.5 m/s forward walk with a following camera."""
import sys

from isaaclab_rl.entrypoints.backends import play_rsl_rl
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

original_resolver = play_rsl_rl.resolve_task_config
original_creator = play_rsl_rl.create_isaaclab_env
original_playback = play_rsl_rl.run_playback
captured = {}


def configure(*args, **kwargs):
    env_cfg, agent_cfg = original_resolver(*args, **kwargs)
    env_cfg.episode_length_s = 30.0
    env_cfg.observations.policy.enable_corruption = False
    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (10000.0, 10000.0)
    command.ranges.lin_vel_x = (0.5, 0.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.debug_vis = False
    env_cfg.events.reset_base.params['pose_range'] = {}
    env_cfg.events.reset_base.params['velocity_range'] = {}
    env_cfg.events.reset_robot_joints.params['position_range'] = (1.0, 1.0)
    env_cfg.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
        eye=(1.6, -1.6, 1.1), lookat=(0.0, 0.0, 0.35),
        window_width=1280, window_height=720, headless=True,
        enable_markers=False, enable_live_plots=False,
    )
    return env_cfg, agent_cfg


def create_env(*args, **kwargs):
    env = original_creator(*args, **kwargs)
    captured['env'] = env.unwrapped
    return env


def playback(step, *, dt, args_cli, env_cfg):
    env = captured['env']

    def step_with_camera():
        x, y, z = env.scene['robot'].data.root_pos_w[0].tolist()
        env.sim.set_camera_view(eye=(x + 1.6, y - 1.6, z + 0.75), target=(x, y, z))
        step()

    original_playback(step_with_camera, dt=dt, args_cli=args_cli, env_cfg=env_cfg)


play_rsl_rl.resolve_task_config = configure
play_rsl_rl.create_isaaclab_env = create_env
play_rsl_rl.run_playback = playback
play_rsl_rl.run(sys.argv[1:])
