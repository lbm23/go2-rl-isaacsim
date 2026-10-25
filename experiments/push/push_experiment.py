"""Real external-force evaluation and bounded video playback for an existing policy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from isaaclab_rl.entrypoints.backends import play_rsl_rl
from push_protocol import (HORIZON, PUSH_DURATION, PUSH_START, TARGET_SPEED, conditions,
                           trial_metrics, world_directions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--direction', default='left')
    parser.add_argument('--level', type=float, default=1.0)
    parser.add_argument('--tag', default='evaluation')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = conditions()
    if args.video:
        selected = next(spec for spec in specs if spec['direction'] == args.direction and spec['level_bw'] == args.level)
        assignments = [selected]
    else:
        assignments = [spec for spec in specs for _ in range(args.samples)]
    captured = {}
    original_resolver = play_rsl_rl.resolve_task_config
    original_creator = play_rsl_rl.create_isaaclab_env
    original_video = play_rsl_rl.apply_video_recording

    def configure(*positional, **keyword):
        env_cfg, agent_cfg = original_resolver(*positional, **keyword)
        env_cfg.episode_length_s = HORIZON + 5.0
        env_cfg.observations.policy.enable_corruption = False
        command = env_cfg.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.resampling_time_range = (10000.0, 10000.0)
        command.ranges.lin_vel_x = (TARGET_SPEED, TARGET_SPEED)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.debug_vis = False
        env_cfg.events.reset_base.params['pose_range'] = {}
        env_cfg.events.reset_base.params['velocity_range'] = {}
        env_cfg.events.reset_robot_joints.params['position_range'] = (1.0, 1.0)
        if args.video:
            from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
            env_cfg.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
                eye=(1.6, -1.6, 1.1), lookat=(0.0, 0.0, 0.35),
                window_width=1280, window_height=720, headless=True,
                enable_markers=False, enable_live_plots=False,
            )
        return env_cfg, agent_cfg

    def create_env(*positional, **keyword):
        result = original_creator(*positional, **keyword)
        captured['env'] = result.unwrapped
        return result

    def configure_video(env_cfg, *positional, **keyword):
        original_video(env_cfg, *positional, **keyword)
        if args.video:
            for recorder in env_cfg.video_recorders:
                recorder.output_dir = str(output_dir / f'video-raw-{args.tag}')

    def execute(step, *, dt, args_cli, env_cfg):
        env = captured['env']
        assert bool(env.sim.physics_manager._supports_rigid_body_force_input)
        robot = env.scene['robot']
        body_ids, body_names = robot.find_bodies('base')
        assert len(body_ids) == 1, body_names
        masses = robot.data.body_mass.torch.sum(dim=1).clone()
        unit_xy = np.asarray([spec['unit_xy'] for spec in assignments], dtype=np.float32)
        levels = torch.tensor([spec['level_bw'] for spec in assignments], device=env.device)
        magnitudes = masses * 9.81 * levels
        count = len(assignments)
        assert env.num_envs == count
        steps = round(HORIZON / dt)
        onset = round(PUSH_START / dt)
        release = round((PUSH_START + PUSH_DURATION) / dt)
        trace = torch.zeros(steps, count, 7, device=env.device)
        ever_terminated = torch.zeros(count, dtype=torch.bool, device=env.device)
        force_vectors = torch.zeros(count, 3, device=env.device)
        forces = torch.zeros(count, 1, 3, device=env.device)
        torques = torch.zeros_like(forces)
        force_audit_error_N = None
        composer = robot.permanent_wrench_composer
        with torch.inference_mode():
            for index in range(steps):
                if index == onset:
                    q = robot.data.root_quat_w.torch
                    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                    directions = world_directions(yaw.cpu().numpy(), unit_xy)
                    force_vectors.copy_(torch.as_tensor(directions, device=env.device) * magnitudes[:, None])
                forces.zero_()
                if onset <= index < release:
                    forces[:, 0] = force_vectors
                    forces[ever_terminated] = 0.0
                # Set zeros too: this keeps the external-wrench path in the CUDA graph
                # and explicitly removes force at the end of the 0.2 s pulse.
                composer.set_forces_and_torques_index(forces=forces, torques=torques,
                                                     body_ids=body_ids, is_global=True)
                if index == onset:
                    force_b, _, _ = composer.get_forces_and_torques()
                    observed_norm = torch.linalg.vector_norm(wp.to_torch(force_b)[:, body_ids[0]], dim=-1)
                    expected_norm = torch.linalg.vector_norm(forces[:, 0], dim=-1)
                    force_audit_error_N = float((observed_norm - expected_norm).abs().max())
                    assert force_audit_error_N < 0.01, force_audit_error_N
                if args.video:
                    x, y, z = robot.data.root_pos_w.torch[0].tolist()
                    env.sim.set_camera_view(eye=(x + 1.6, y - 1.6, z + 0.75), target=(x, y, z))
                applied_norm = torch.linalg.vector_norm(forces[:, 0], dim=-1).clone()
                step()
                ever_terminated |= env.reset_terminated
                velocity = robot.data.root_lin_vel_b.torch
                height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                trace[index] = torch.stack((velocity[:, 0], velocity[:, 1],
                                            robot.data.root_ang_vel_b.torch[:, 2],
                                            robot.data.projected_gravity_b.torch[:, 2], height,
                                            env.reset_terminated.float(), applied_norm), dim=-1)
        # Clear and submit external force once more before closing the environment.
        composer.set_forces_and_torques_index(forces=torch.zeros_like(forces), torques=torques,
                                             body_ids=body_ids, is_global=True)
        robot.write_data_to_sim()
        trace_cpu = trace.cpu().numpy()
        mass_cpu = masses.cpu().numpy()
        vectors_cpu = force_vectors.cpu().numpy()
        rows = [trial_metrics(trace_cpu[:, trial], dt, spec, args.seed, trial,
                              mass_cpu[trial], vectors_cpu[trial])
                for trial, spec in enumerate(assignments)]
        result = {
            'checkpoint': args.checkpoint, 'seed': args.seed, 'dt': dt, 'horizon_s': HORIZON,
            'force_start_s': PUSH_START, 'force_duration_s': PUSH_DURATION,
            'force_application': 'Base body center of mass; real force in Newton/MJWarp; horizontal world vector frozen from heading at onset',
            'force_buffer_max_norm_error_N': force_audit_error_N,
            'conditions': '0.5 m/s forward, no other push or observation noise; random material/mass/COM retained; standard rest start',
            'rows': rows,
        }
        prefix = args.tag if args.video else f'seed{args.seed}'
        (output_dir / f'{prefix}.json').write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(output_dir / f'{prefix}-trace.npz', trace=trace_cpu, dt=dt,
                            masses=mass_cpu, force_world=vectors_cpu)
        print('PUSH_RUN_COMPLETE=' + json.dumps({'tag': prefix, 'trials': count,
                                                'force_buffer_error_N': force_audit_error_N,
                                                'fall_count': sum(row['fell_after_push'] for row in rows)}))

    play_rsl_rl.resolve_task_config = configure
    play_rsl_rl.create_isaaclab_env = create_env
    play_rsl_rl.apply_video_recording = configure_video
    play_rsl_rl.run_playback = execute
    argv = ['--task', 'Isaac-Velocity-Flat-UnitreeGo2', '--checkpoint', args.checkpoint,
            '--num_envs', str(len(assignments)), '--seed', str(args.seed),
            '--viz', 'newton_gl' if args.video else 'none', 'physics=newton_mjwarp']
    if args.video:
        argv += ['--video', '--video_length', str(round(HORIZON / 0.02))]
    play_rsl_rl.run(argv)


if __name__ == '__main__':
    main()
