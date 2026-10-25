"""Check that standing still, falls and crouching cannot pass walking criteria."""
import importlib.util
import pathlib
import sys
import unittest

import torch


path = pathlib.Path(sys.argv[1])
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location('walking_evaluate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WalkingMetricsTests(unittest.TestCase):
    def test_standing_is_not_walking(self):
        speeds = torch.tensor([0.3, 0.5, 0.8])
        passed, _ = module.summarize_trials(speeds, speeds, speeds * 0, speeds * 0,
                                          torch.ones(3), torch.ones(3), torch.zeros(3, dtype=torch.bool))
        self.assertFalse(passed.any().item())

    def test_fall_and_crouch_fail_even_with_correct_speed(self):
        speeds = torch.tensor([0.3, 0.5, 0.8])
        passed, groups = module.summarize_trials(speeds, speeds * 0, speeds * 0, speeds,
                                               torch.ones(3), torch.tensor([1.0, 0.5, 1.0]),
                                               torch.tensor([True, False, False]))
        self.assertEqual(passed.tolist(), [False, False, True])
        self.assertEqual([group['success_rate'] for group in groups], [0.0, 0.0, 1.0])


unittest.main()
