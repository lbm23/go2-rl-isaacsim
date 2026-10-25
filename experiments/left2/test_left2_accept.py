import unittest
from left2_accept import decide


class AcceptanceTests(unittest.TestCase):
    def fixtures(self):
        return dict(primary=dict(fall_change_ci95_pp=[-20,-5],fall_p_exact_two_sided=.001),
            inferential_minimum_met=True,results=[dict(direction=d,level_bw=1.,fall_change_pp=.1,
            fall_change_ci95_pp=[-.2,.8],after=dict(fall_rate=.002)) for d in ['backward','right']])

    def test_preservation_required_even_if_left_improves(self):
        summary=self.fixtures();walks=[dict(stable_forward_walking_passed=True)]*2
        self.assertTrue(decide(summary,walks,10)['passed'])
        summary['results'][1]['fall_change_ci95_pp'][1]=3
        self.assertFalse(decide(summary,walks,10)['passed'])

    def test_no_claim_from_nonsignificance_or_missing_walking(self):
        summary=self.fixtures();summary['primary']['fall_p_exact_two_sided']=.12
        self.assertFalse(decide(summary,[],10)['passed'])
        self.assertFalse(decide(self.fixtures(),[dict(stable_forward_walking_passed=True)]*2,0)['passed'])


if __name__=='__main__':
    unittest.main(verbosity=2)
