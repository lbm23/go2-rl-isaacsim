"""Regression checks for pulse direction, recovery censoring and fall metrics."""
import unittest

import numpy as np

from push_protocol import conditions, first_stable_window, trial_metrics, world_directions


class PushProtocolTests(unittest.TestCase):
    def trace(self):
        data = np.zeros((750, 7), dtype=np.float32)
        data[:, 0] = 0.5
        data[:, 3] = -1.0
        data[:, 4] = 0.4
        return data

    def metrics(self, data):
        return trial_metrics(data, 0.02, conditions()[1], 42, 0, 15.0, [36.7875, 0.0, 0.0])

    def test_heading_rotation_and_force_levels(self):
        self.assertEqual(len(conditions()), 17)
        vectors = world_directions(np.array([np.pi/2, 0.0]), np.array([[1.0, 0.0], [0.0, -1.0]]))
        np.testing.assert_allclose(vectors, [[0, 1, 0], [0, -1, 0]], atol=1e-6)

    def test_transient_stability_does_not_count_as_recovery(self):
        mask = np.zeros(100, dtype=bool)
        mask[0:24] = True
        mask[50:75] = True
        self.assertAlmostEqual(first_stable_window(mask, 0.02), 1.0)

    def test_autoreset_cannot_hide_a_fall(self):
        data = self.trace()
        data[400, 5] = 1.0
        row = self.metrics(data)
        self.assertTrue(row['fell_after_push'])
        self.assertIsNone(row['recovery_time_s'])

    def test_no_recovery_is_not_reported_as_zero_seconds(self):
        data = self.trace()
        data[260:, 0] = 0.0
        self.assertIsNone(self.metrics(data)['recovery_time_s'])

    def test_short_impulse_and_sustained_tip(self):
        data = self.trace()
        data[250:260, 6] = 100.0
        data[280:290, 3] = 0.0
        row = self.metrics(data)
        self.assertAlmostEqual(row['commanded_impulse_Ns'], 20.0)
        self.assertTrue(row['fell_after_push'])

    def test_preexisting_fall_is_excluded(self):
        data = self.trace()
        data[200, 5] = 1.0
        row = self.metrics(data)
        self.assertFalse(row['eligible_at_push'])
        self.assertFalse(row['fell_after_push'])


if __name__ == '__main__':
    unittest.main()
