"""Bounded flat-ground evaluation of an existing Go2 PPO policy.

Uses the official playback entrypoint. Writes metrics rather than training a model.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from isaaclab_rl.entrypoints.backends import play_rsl_rl


def summarize_trials(speed, error, yaw_error, vx, upright, height, failed):
    """Apply thresholds per trial before aggregating, so resets cannot mask falls."""
    successful = (
        (~failed)
        & (error < 0.20)
        & (yaw_error < 0.30)
        & (upright >= 0.95)
        & (height >= 0.95)
        & (vx >= 0.65 * speed)
    )
    groups = []
    for value in (0.3, 0.5, 0.8):
        mask = torch.isclose(speed, speed.new_tensor(value))
        groups.append({
            'target_speed_m_s': value,
            'trials': int(mask.sum()),
            'no_fall_rate': float((~failed[mask]).float().mean()),
            'success_rate': float(successful[mask].float().mean()),
            'mean_planar_error_m_s': float(error[mask].mean()),
            'mean_forward_speed_m_s': float(vx[mask].mean()),
            'mean_yaw_error_rad_s': float(yaw_error[mask].mean()),
            'upright_fraction': float(upright[mask].mean()),
            'height_ok_fraction': float(height[mask].mean()),
        })
    return successful, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    captured = {}
    original_resolver = play_rsl_rl.resolve_task_config
    original_creator = play_rsl_rl.create_isaaclab_env

    def configure(*positional, **keyword):
        env_cfg, agent_cfg = original_resolver(*positional, **keyword)
        env_cfg.episode_length_s = 35.0
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
        # Start at rest; retain randomized yaw, material, base mass and center of mass.
        env_cfg.events.reset_base.params['velocity_range'] = {}
        env_cfg.events.reset_robot_joints.params['position_range'] = (1.0, 1.0)
        return env_cfg, agent_cfg

    def create_env(*positional, **keyword):
        result = original_creator(*positional, **keyword)
        captured['env'] = result.unwrapped
        return result

    def evaluate(step, *, dt, args_cli, env_cfg):
        env = captured['env']
        robot = env.scene['robot']
        count = env.num_envs
        device = env.device
        speeds = torch.tensor([0.3, 0.5, 0.8], device=device).repeat_interleave(count // 3)
        assert speeds.numel() == count
        target = torch.zeros(count, 3, device=device)
        target[:, 0] = speeds
        failed = torch.zeros(count, dtype=torch.bool, device=device)
        planar_sum = torch.zeros(count, device=device)
        yaw_sum = torch.zeros_like(planar_sum)
        vx_sum = torch.zeros_like(planar_sum)
        upright_sum = torch.zeros_like(planar_sum)
        height_sum = torch.zeros_like(planar_sum)
        steps = round(30.0 / dt)
        warmup = round(2.0 / dt)
        with torch.inference_mode():
            for index in range(steps):
                env.command_manager.get_command('base_velocity').copy_(target)
                step()
                failed |= env.reset_terminated
                if index >= warmup:
                    velocity = robot.data.root_lin_vel_b
                    planar_sum += torch.linalg.vector_norm(velocity[:, :2] - target[:, :2], dim=-1)
                    yaw_sum += robot.data.root_ang_vel_b[:, 2].abs()
                    vx_sum += velocity[:, 0]
                    upright_sum += robot.data.projected_gravity_b[:, 2] < -math.cos(math.radians(30))
                    height_sum += robot.data.root_pos_w[:, 2] >= 0.28
        measured_steps = steps - warmup
        error = planar_sum / measured_steps
        yaw_error = yaw_sum / measured_steps
        vx = vx_sum / measured_steps
        upright = upright_sum / measured_steps
        height = height_sum / measured_steps
        successes, groups = summarize_trials(speeds, error, yaw_error, vx, upright, height, failed)
        summary = {
            'checkpoint': args.checkpoint, 'seed': args.seed,
            'duration_seconds_per_trial': 30.0, 'warmup_seconds': 2.0,
            'trials': count, 'no_fall_rate': float((~failed).float().mean()),
            'success_rate': float(successes.float().mean()),
            'stable_forward_walking_passed': bool(successes.float().mean() >= 0.90),
            'conditions': 'Flat ground, no push, no observation noise; random yaw/material/mass/COM retained',
            'thresholds': {
                'mean_planar_error_m_s_below': 0.20, 'mean_yaw_error_rad_s_below': 0.30,
                'upright_within_30deg_fraction_at_least': 0.95,
                'base_height_above_0_28m_fraction_at_least': 0.95,
                'mean_forward_speed_target_ratio_at_least': 0.65,
                'no_fall_during_entire_30_seconds': True,
                'aggregate_success_fraction_at_least': 0.90,
            },
            'groups': groups,
            'per_trial': {
                'target_speed': speeds.tolist(), 'mean_planar_error': error.tolist(),
                'mean_forward_speed': vx.tolist(), 'fell': failed.tolist(),
                'successful': successes.tolist(),
            },
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + '\n')
        print('GO2_WALKING_EVALUATION=' + json.dumps({k: v for k, v in summary.items() if k != 'per_trial'}))

    play_rsl_rl.resolve_task_config = configure
    play_rsl_rl.create_isaaclab_env = create_env
    play_rsl_rl.run_playback = evaluate
    play_rsl_rl.run([
        '--task', 'Isaac-Velocity-Flat-UnitreeGo2', '--checkpoint', args.checkpoint,
        '--num_envs', '96', '--seed', str(args.seed), '--viz', 'none', 'physics=newton_mjwarp',
    ])


if __name__ == '__main__':
    main()
