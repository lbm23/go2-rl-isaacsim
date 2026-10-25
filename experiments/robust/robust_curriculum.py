"""Direction-weighted external-force curriculum; no simulator source changes."""
import json
import math
from pathlib import Path

import torch

NAMES = ['forward', 'backward', 'left', 'right']
DIRECTION_PROBABILITIES = [0.15, 0.40, 0.15, 0.30]
STRENGTH_RANGES = [(0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


def promotion_allowed(successes, total, stage):
    return stage < 2 and total >= 128 and successes / total >= 0.90


def rotate_horizontal(yaw, vectors):
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((cosine * vectors[:, 0] - sine * vectors[:, 1],
                        sine * vectors[:, 0] + cosine * vectors[:, 1], torch.zeros_like(yaw)), dim=-1)


class ForceCurriculum:
    def __init__(self, env, output_dir):
        self.env = env
        self.robot = env.scene['robot']
        self.body_ids, _ = self.robot.find_bodies('base')
        assert len(self.body_ids) == 1
        assert env.sim.physics_manager._supports_rigid_body_force_input
        self.device = env.device
        self.count = env.num_envs
        self.dt = env.step_dt
        self.output_dir = Path(output_dir)
        self.steps = 0
        self.stages = [0, 0, 0, 0]
        self.stage_tensor = torch.zeros(4, dtype=torch.long, device=self.device)
        self.probabilities = torch.tensor(DIRECTION_PROBABILITIES, device=self.device)
        self.axes = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], device=self.device)
        self.age = torch.zeros(self.count, dtype=torch.long, device=self.device)
        self.direction = torch.zeros_like(self.age)
        self.episode_stage = torch.zeros_like(self.age)
        self.onset = torch.zeros_like(self.age)
        self.level = torch.zeros(self.count, device=self.device)
        self.pushed = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self.finished = torch.zeros_like(self.pushed)
        self.recovered = torch.zeros_like(self.pushed)
        self.failed = torch.zeros_like(self.pushed)
        self.hold = torch.zeros_like(self.age)
        self.force_vectors = torch.zeros(self.count, 3, device=self.device)
        self.forces = torch.zeros(self.count, 1, 3, device=self.device)
        self.torques = torch.zeros_like(self.forces)
        self.window_total = torch.zeros(4, device=self.device)
        self.window_success = torch.zeros_like(self.window_total)
        self.total = torch.zeros_like(self.window_total)
        self.success = torch.zeros_like(self.window_total)
        self.sampled_directions = torch.zeros_like(self.window_total)
        self.sampled_no_push = torch.zeros((), device=self.device)
        self.sampled_push = torch.zeros((), device=self.device)
        self.pulse_steps = round(0.2 / self.dt)
        self.recovery_steps = round(3.0 / self.dt)
        self.hold_steps = round(0.3 / self.dt)
        self.events = []
        self.reset(torch.ones(self.count, dtype=torch.bool, device=self.device))

    def reset(self, mask):
        # Resample all entries, then select; shapes stay fixed for cheap CUDA operations.
        directions = torch.multinomial(self.probabilities, self.count, replacement=True)
        pushing = torch.rand(self.count, device=self.device) < 0.70
        stages = self.stage_tensor[directions]
        level = 0.25 + stages.float() * 0.25 + torch.rand(self.count, device=self.device) * 0.25
        onset = torch.randint(round(3.0 / self.dt), round(8.0 / self.dt) + 1,
                              (self.count,), device=self.device)
        self.age[mask] = 0
        self.direction[mask] = directions[mask]
        self.episode_stage[mask] = stages[mask]
        self.pushed[mask] = pushing[mask]
        self.onset[mask] = onset[mask]
        self.level[mask] = level[mask]
        self.finished[mask] = False
        self.recovered[mask] = False
        self.failed[mask] = False
        self.hold[mask] = 0
        self.force_vectors[mask] = 0.0
        self.sampled_push += (mask & pushing).sum()
        self.sampled_no_push += (mask & ~pushing).sum()
        categories = torch.where(mask & pushing, directions + 1, 0)
        self.sampled_directions += torch.bincount(categories, minlength=5)[1:]

    def before_step(self):
        starting = self.pushed & (self.age == self.onset)
        q = self.robot.data.root_quat_w.torch
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        directions = rotate_horizontal(yaw, self.axes[self.direction])
        masses = self.robot.data.body_mass.torch.sum(dim=1)
        planned = directions * (masses * 9.81 * self.level)[:, None]
        self.force_vectors.copy_(torch.where(starting[:, None], planned, self.force_vectors))
        active = self.pushed & (self.age >= self.onset) & (self.age < self.onset + self.pulse_steps)
        self.forces[:, 0] = torch.where(active[:, None], self.force_vectors, 0.0)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.forces, torques=self.torques, body_ids=self.body_ids, is_global=True)

    def after_step(self):
        env = self.env
        terminated = env.reset_terminated
        done = terminated | env.reset_time_outs
        started = self.pushed & (self.age >= self.onset) & ~self.finished
        post = started & (self.age >= self.onset + self.pulse_steps)
        command = env.command_manager.get_command('base_velocity')
        velocity = self.robot.data.root_lin_vel_b.torch
        error = torch.linalg.vector_norm(velocity[:, :2] - command[:, :2], dim=-1)
        yaw_error = (self.robot.data.root_ang_vel_b.torch[:, 2] - command[:, 2]).abs()
        height = self.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        stable = (post & ~done & (error <= 0.25) & (yaw_error <= 0.5)
                  & (self.robot.data.projected_gravity_b.torch[:, 2] <= -math.cos(math.radians(25)))
                  & (height >= 0.25))
        self.hold.copy_(torch.where(stable, self.hold + 1, 0))
        self.recovered |= self.hold >= self.hold_steps
        self.failed |= started & terminated
        completed = started & (done | (self.age >= self.onset + self.pulse_steps + self.recovery_steps))
        successful = completed & self.recovered & ~self.failed & ~terminated
        categories = torch.where(completed, self.direction + 1, 0)
        success_categories = torch.where(successful, self.direction + 1, 0)
        counts = torch.bincount(categories, minlength=5)[1:]
        successes = torch.bincount(success_categories, minlength=5)[1:]
        # An episode may finish after its direction was promoted. Its old, easier
        # push must not provide evidence for another promotion at the new stage.
        current_stage_trial = self.episode_stage == self.stage_tensor[self.direction]
        gate_categories = torch.where(completed & current_stage_trial, self.direction + 1, 0)
        gate_success_categories = torch.where(successful & current_stage_trial, self.direction + 1, 0)
        self.window_total += torch.bincount(gate_categories, minlength=5)[1:]
        self.window_success += torch.bincount(gate_success_categories, minlength=5)[1:]
        self.total += counts
        self.success += successes
        self.finished |= completed
        self.age += 1
        self.reset(done)
        self.steps += 1
        if self.steps % 240 == 0:
            self.update_stages()
        if self.steps % 2400 == 0:
            self.save()

    def update_stages(self):
        totals = self.window_total.cpu().tolist()
        successes = self.window_success.cpu().tolist()
        for direction in range(4):
            if totals[direction] >= 128:
                rate = successes[direction] / totals[direction]
                if promotion_allowed(successes[direction], totals[direction], self.stages[direction]):
                    self.stages[direction] += 1
                    self.stage_tensor[direction] = self.stages[direction]
                    event = {'control_step': self.steps, 'direction': NAMES[direction],
                             'new_range_bw': STRENGTH_RANGES[self.stages[direction]],
                             'training_gate_recovery_rate': rate, 'window_trials': totals[direction]}
                    self.events.append(event)
                    print('FORCE_CURRICULUM_PROMOTION=' + json.dumps(event), flush=True)
                self.window_total[direction] = 0
                self.window_success[direction] = 0

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            'control_steps': self.steps, 'direction_order': NAMES,
            'direction_probabilities_given_push': DIRECTION_PROBABILITIES,
            'no_push_probability': 0.30, 'push_onset_range_s': [3.0, 8.0], 'push_duration_s': 0.2,
            'strength_ranges_bw': STRENGTH_RANGES, 'final_stages': self.stages,
            'completed_push_trials': self.total.cpu().tolist(),
            'training_gate_successes': self.success.cpu().tolist(),
            'sampled_push_episodes': int(self.sampled_push),
            'sampled_no_push_episodes': int(self.sampled_no_push),
            'sampled_push_direction_counts': self.sampled_directions.cpu().tolist(),
            'promotions': self.events,
            'training_gate': '90% of >=128 completed pushes per direction: recover within3s, error<=0.25m/s, yaw error<=0.5rad/s, tilt<=25deg, height>=0.25m for0.3s; survive until gate ends',
            'formal_evaluation_gate_unchanged': True,
        }
        (self.output_dir / 'curriculum.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
