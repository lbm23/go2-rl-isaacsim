import unittest
import torch
from types import SimpleNamespace
from left2_curriculum import Left2Curriculum,phase_at,sample_levels,DIRECTION_PROBABILITIES


class Left2Tests(unittest.TestCase):
    def test_fixed_clock_boundaries(self):
        self.assertEqual([phase_at(t) for t in [0,179.9,180,359.9,360,599]],[0,0,1,1,2,2])

    def test_rehearsal_weights(self):
        self.assertAlmostEqual(sum(DIRECTION_PROBABILITIES),1.)
        self.assertEqual(DIRECTION_PROBABILITIES,[.05,.25,.50,.20])

    def test_left_exposure_and_cap(self):
        torch.manual_seed(123)
        direction=torch.full((20000,),2)
        levels=[sample_levels(direction,p) for p in range(3)]
        self.assertTrue(all(float(x.min())>=.75 and float(x.max())<=2 for x in levels))
        self.assertLess(float(levels[0].mean()),float(levels[2].mean()))
        self.assertAlmostEqual(float((levels[2]==2).float().mean()),.25,delta=.02)
        self.assertGreater(float((levels[2]<1.25).float().mean()),.20)

    def test_backward_right_strength_preserved(self):
        for direction in [1,3]:
            values=sample_levels(torch.full((20000,),direction),2)
            self.assertGreaterEqual(float(values.min()),.5)
            self.assertLessEqual(float(values.max()),1.)
            self.assertAlmostEqual(float((values==1).float().mean()),.25,delta=.02)

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
        controller = Left2Curriculum(env, '.')
        controller.age[:] = torch.tensor([149, 150, 159, 160])
        controller.onset[:] = 150
        controller.pushed[:] = True
        controller.level[:] = 2.0
        controller.direction[:] = 0
        controller.force_vectors[:, 0] = 15 * 9.81 * 2.0
        controller.before_step()
        expected = torch.tensor([0., 294.3, 294.3, 0.])
        torch.testing.assert_close(composer.last[:, 0, 0], expected)
        controller.age[:] = 160
        controller.before_step()
        self.assertEqual(composer.last.count_nonzero().item(), 0)


if __name__=='__main__':
    unittest.main(verbosity=2)
