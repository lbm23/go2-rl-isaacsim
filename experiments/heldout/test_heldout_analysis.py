import unittest
import numpy as np
from heldout_analysis import sign_flip_p, paired_ci, holm, describe, completed_seed_pairs


class HeldoutTests(unittest.TestCase):
    def test_partial_pair_excluded_even_if_file_exists(self):
        progress=[dict(seed=1,stage='before',exit_code=0),dict(seed=1,stage='after',exit_code=0),
                  dict(seed=2,stage='before',exit_code=0),dict(seed=2,stage='after',exit_code=124)]
        self.assertEqual(completed_seed_pairs([1,2,3],progress,lambda seed,stage:True),[1])

    def test_exact_all_positive_and_swap(self):
        self.assertEqual(sign_flip_p([1]*4),2/16)
        self.assertEqual(sign_flip_p([-1]*4),2/16)
        self.assertEqual(sign_flip_p([0]*5),1.)

    def test_cluster_unit_not_environment_count(self):
        # Four seed-level effects remain four replicates regardless of 64 envs per seed.
        self.assertEqual(sign_flip_p([.1]*4),.125)
        self.assertGreater(sign_flip_p([.1]*4),.05)

    def test_ci_and_holm(self):
        idx=np.tile(np.arange(4),(100,1))
        np.testing.assert_allclose(paired_ci([1,1,1,1],idx),[1,1])
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_failures_not_zero_recovery(self):
        row=dict(eligible_at_push=True,fell_after_push=True,recovery_time_s=None,
                 postpush_mean_speed_error_before_fall_m_s=.8,recovery_observation_limit_s=9.8)
        result=describe([row])
        self.assertIsNone(result['median_recovery_s'])
        self.assertEqual(result['capped_recovery_mean_s'],9.8)
        self.assertEqual(result['recovery_rate'],0)

    def test_matches_scipy_exact(self):
        try:
            from scipy.stats import permutation_test
        except ImportError:
            self.skipTest('SciPy unavailable; no package installation')
        data=np.array([-.2,.1,.3,-.1,.2])
        ref=permutation_test((data,),np.mean,permutation_type='samples',n_resamples=np.inf).pvalue
        self.assertAlmostEqual(sign_flip_p(data),ref)


if __name__=='__main__':
    unittest.main(verbosity=2)
