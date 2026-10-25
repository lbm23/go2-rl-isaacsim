import unittest
import torch
from ablation_curriculum import recipes,phase_at_update
from ablation_budget import FixedBudget,state_equal
from ablation_analyze import sign_flip,holm,describe


class AblationTests(unittest.TestCase):
    def uniforms(self,n=20000):
        return torch.rand((9,n),generator=torch.Generator().manual_seed(300))

    def test_cycle_boundaries(self):
        self.assertEqual([phase_at_update(n) for n in [0,719,720,1439,1440,2400]],[0,0,1,1,2,2])

    def test_direction_weights_and_common_random_numbers(self):
        u=self.uniforms()
        for group,expected in [('A',[.25]*4),('B',[.05,.25,.50,.20])]:
            d,p,l,onset,phase=recipes(group,0,u)
            actual=torch.bincount(d,minlength=4)/len(d)
            torch.testing.assert_close(actual,torch.tensor(expected),atol=.015,rtol=0)
            self.assertAlmostEqual(float(p.float().mean()),.7,delta=.015)
            self.assertTrue(bool(((onset>=150)&(onset<=400)).all()))
        b=recipes('B',0,u);c=recipes('C',0,u)
        for field in [0,1,3]:torch.testing.assert_close(b[field],c[field])

    def test_pooled_distribution_exact_recipe_matching(self):
        u=self.uniforms();b=recipes('B',0,u)
        for phase,completed in [(0,0),(1,720),(2,1440)]:
            mask=b[4]==phase
            c=recipes('C',completed,u[:,mask])
            torch.testing.assert_close(b[2][mask],c[2])
        d,p,levels,_,phase=b
        left=levels[d==2]
        self.assertAlmostEqual(float((left==2).float().mean()),.1,delta=.015)

    def test_curriculum_caps_rehearsal_and_exact2(self):
        u=self.uniforms();u[0]=.5
        a=recipes('C',1440,u)
        self.assertTrue(bool((a[0]==2).all()))
        self.assertGreaterEqual(float(a[2].min()),.75)
        self.assertLessEqual(float(a[2].max()),2.)
        self.assertAlmostEqual(float((a[2]==2).float().mean()),.25,delta=.02)
        self.assertGreater(float((a[2]<1.25).float().mean()),.20)

    def test_exact_update_accounting(self):
        budget=FixedBudget(target=3,rollout_steps=24,num_envs=2048)
        for step in [24,48,72]:budget.after_update(step)
        self.assertEqual(budget.validate_final(72),147456)
        with self.assertRaises(AssertionError):budget.after_update(96)
        with self.assertRaises(AssertionError):FixedBudget().validate_final(24)
        with self.assertRaises(AssertionError):FixedBudget().after_update(23)

    def test_optimizer_and_normalizer_state_equality(self):
        state={'params':[torch.tensor([1.,2.])],'lr':.001,'normalizer':torch.zeros(3)}
        copy={'params':[torch.tensor([1.,2.])],'lr':.001,'normalizer':torch.zeros(3)}
        self.assertTrue(state_equal(state,copy));copy['lr']=.002
        self.assertFalse(state_equal(state,copy))

    def test_seed_inference_and_failure_aware_score(self):
        self.assertAlmostEqual(sign_flip([-1.]*8),2/256)
        self.assertEqual(sign_flip([0.]*8),1.)
        self.assertEqual(holm([.01,.04,.02]),[.03,.04,.04])
        result=describe([dict(eligible_at_push=True,fell_after_push=False,recovery_time_s=1.,
            postpush_mean_speed_error_before_fall_m_s=.1),dict(eligible_at_push=True,
            fell_after_push=True,recovery_time_s=None,postpush_mean_speed_error_before_fall_m_s=.2)])
        self.assertEqual(result['fall_rate'],.5)
        self.assertAlmostEqual(result['capped_recovery_s'],5.4)


if __name__=='__main__':unittest.main(verbosity=2)
