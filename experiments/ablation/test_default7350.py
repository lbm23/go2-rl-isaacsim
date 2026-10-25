import unittest
import numpy as np
from default7350_compare import describe,audit_pair,crossed_ci


class ComparisonTests(unittest.TestCase):
    def test_failures_do_not_disappear_from_recovery_score(self):
        rows=[dict(eligible_at_push=True,fell_after_push=False,recovery_time_s=.6,
            postpush_mean_speed_error_before_fall_m_s=.1),dict(eligible_at_push=True,
            fell_after_push=True,recovery_time_s=None,postpush_mean_speed_error_before_fall_m_s=1.)]
        result=describe(rows)
        self.assertEqual(result['fall_rate'],.5);self.assertEqual(result['recovery_rate'],.5)
        self.assertAlmostEqual(result['median_recovery_s'],.6)
        self.assertAlmostEqual(result['capped_recovery_s'],5.2)
        self.assertAlmostEqual(result['survivor_speed_error_m_s'],.1)

    def test_pair_audit_rejects_changed_scenarios(self):
        rows=[dict(seed=930001,trial_id=i,direction='left',level_bw=2.,mass_kg=13.,nominal_force_N=255.06) for i in range(1088)]
        self.assertEqual(audit_pair(rows,rows)['maximum_mass_difference_kg'],0.)
        changed=[dict(r) for r in rows];changed[0]['mass_kg']=14.
        with self.assertRaises(AssertionError):audit_pair(rows,changed)
        changed=[dict(r) for r in rows];changed[0]['direction']='right'
        with self.assertRaises(AssertionError):audit_pair(rows,changed)

    def test_crossed_bootstrap_preserves_pairing_and_constant_effect(self):
        rng=np.random.default_rng(7);ref=np.arange(16)/100
        values=np.repeat(ref[None,:],8,axis=0)-.05
        result=crossed_ci(values,ref,rng.integers(0,8,(512,8)),rng.integers(0,16,(512,16)))
        self.assertAlmostEqual(result['change'],-.05)
        np.testing.assert_allclose(result['exploratory_ci95'],[-.05,-.05],atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
