import copy
import math
import unittest

from go2_straight_controller import adapt_descriptor, adapt_env_config, straight_command, yaw_wxyz

class StraightControllerTests(unittest.TestCase):
    def test_heading_feedback_turns_toward_line(self):
        q=[math.cos(.1),0,0,math.sin(.1)]
        self.assertLess(straight_command(.5,[0,0,.36],q)[2],0)
        self.assertAlmostEqual(yaw_wxyz(q),.2)

    def test_lateral_feedback_returns_to_target(self):
        command=straight_command(.5,[1,.1,.36],[1,0,0,0])
        self.assertLess(command[1],0)
        self.assertLess(command[2],0)

    def test_commands_remain_inside_validated_limits(self):
        command=straight_command(.8,[0,100,.36],[0,0,0,1])
        self.assertLessEqual(abs(command[1]),.15)
        self.assertLessEqual(abs(command[2]),.6)

    def test_stand_does_not_move_sideways(self):
        self.assertEqual(straight_command(0,[0,.2,.36],[1,0,0,0]),[0,0,0])

    def test_stand_retains_heading_hold(self):
        q=[math.cos(.1),0,0,math.sin(.1)]
        command=straight_command(0,[0,.2,.36],q)
        self.assertEqual(command[:2],[0,0])
        self.assertLess(command[2],0)

    def test_comparison_mode_has_no_feedback(self):
        self.assertEqual(straight_command(.5,[0,.2,.36],[1,0,0,0],False),[.5,0,0])

    def test_invalid_state_is_rejected(self):
        with self.assertRaises(ValueError):
            straight_command(.5,[0,float('nan'),0],[1,0,0,0])
        with self.assertRaises(ValueError):
            straight_command(.5,[0,0,0],[0,0,0,0])
        with self.assertRaises(ValueError):
            straight_command(1,[0,0,0],[1,0,0,0])

    def test_only_rear_action_offsets_change(self):
        descriptor={'actions':[{'joint_names':['FL_hip_joint','RL_hip_joint','RR_hip_joint'],
            'offset':[.1,.1,-.1],'scale':.25}], 'observations':{'policy':['unchanged']}}
        original=copy.deepcopy(descriptor)
        result=adapt_descriptor(descriptor)
        self.assertEqual(descriptor,original)
        self.assertEqual(result['observations'],original['observations'])
        self.assertEqual(result['actions'][0]['offset'],[.1,.2,-.2])

    def test_missing_joint_is_rejected(self):
        with self.assertRaises(ValueError):
            adapt_descriptor({'actions':[{'joint_names':['RL_hip_joint'],'offset':[.1]}]})

    def test_new_actuator_limits_and_control_period_are_preserved(self):
        source={'sim':{'dt':.005},'decimation':4,'scene':{'robot':{'spawn':{
            'rigid_props':{'func':'metadata','linear_damping':0},
            'articulation_props':[{'func':'metadata','_usd_namespace':'physxArticulation'}]},
            'actuators':{'legs':{'actuator_effort_limit':23.7,'actuator_velocity_limit':30.1,
                                'joint_effort_limit':None,'stiffness':25}}}}}
        result=adapt_env_config(source)
        self.assertEqual(source['sim']['dt'],.005)
        self.assertEqual(result['sim']['dt']*result['decimation'],.02)
        self.assertEqual(result['scene']['robot']['actuators']['legs']['effort_limit'],23.7)
        self.assertEqual(result['scene']['robot']['actuators']['legs']['velocity_limit'],30.1)
        self.assertEqual(result['scene']['robot']['actuators']['legs']['stiffness'],25)

if __name__=='__main__':
    unittest.main()
