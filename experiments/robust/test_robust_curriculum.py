"""Check curriculum progression and direction mapping without a simulator."""
import unittest
from types import SimpleNamespace

import torch

from robust_curriculum import (DIRECTION_PROBABILITIES, STRENGTH_RANGES,
                               ForceCurriculum, promotion_allowed, rotate_horizontal)


class CurriculumTests(unittest.TestCase):
    def test_promotion_requires_evidence_and_threshold(self):
        self.assertFalse(promotion_allowed(100, 100, 0))
        self.assertFalse(promotion_allowed(100, 128, 0))
        self.assertTrue(promotion_allowed(116, 128, 0))
        self.assertFalse(promotion_allowed(128, 128, 2))

    def test_weighting_and_caps(self):
        self.assertAlmostEqual(sum(DIRECTION_PROBABILITIES), 1.0)
        self.assertEqual(DIRECTION_PROBABILITIES, [0.15, 0.40, 0.15, 0.30])
        self.assertEqual(STRENGTH_RANGES[-1][1], 1.0)

    def test_old_stage_samples_cannot_support_new_stage(self):
        directions = torch.tensor([0, 1, 1, 3])
        episode_stage = torch.tensor([0, 0, 1, 0])
        current = torch.tensor([1, 1, 0, 0])
        eligible = episode_stage == current[directions]
        self.assertEqual(eligible.tolist(), [False, False, True, True])

    def test_body_directions_rotate_to_world_heading(self):
        yaw = torch.tensor([torch.pi / 2, 0.0])
        vectors = torch.tensor([[-1., 0.], [0., -1.]])
        expected = torch.tensor([[0., -1., 0.], [0., -1., 0.]])
        torch.testing.assert_close(rotate_horizontal(yaw, vectors), expected, atol=1e-6, rtol=1e-6)

    def test_force_is_removed_at_pulse_end(self):
        class Composer:
            def set_forces_and_torques_index(self, **kwargs):
                self.last = kwargs['forces'].clone()
        composer = Composer()
        quaternion = torch.zeros(4, 4)
        quaternion[:, 0] = 1
        robot = SimpleNamespace(
            find_bodies=lambda name: ([0], ['base']), permanent_wrench_composer=composer,
            data=SimpleNamespace(root_quat_w=SimpleNamespace(torch=quaternion),
                                 body_mass=SimpleNamespace(torch=torch.full((4, 1), 15.0))))
        env = SimpleNamespace(scene={'robot': robot}, num_envs=4, device='cpu', step_dt=0.02,
                              sim=SimpleNamespace(physics_manager=SimpleNamespace(_supports_rigid_body_force_input=True)))
        controller = ForceCurriculum(env, '.')
        controller.age[:] = torch.tensor([149, 150, 159, 160])
        controller.onset[:] = 150
        controller.pushed[:] = True
        controller.level[:] = 0.5
        controller.direction[:] = 0
        controller.force_vectors[:, 0] = 15 * 9.81 * 0.5
        controller.before_step()
        expected = torch.tensor([0., 73.575, 73.575, 0.])
        torch.testing.assert_close(composer.last[:, 0, 0], expected)
        controller.age[:] = 160
        controller.before_step()
        self.assertEqual(composer.last.count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
