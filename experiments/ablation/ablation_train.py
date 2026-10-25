"""Isolated callbacks around the installed, unmodified RSL-RL training loop."""
import json
import math
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'robust'))
from isaaclab_rl.entrypoints.backends import train_rsl_rl
from rsl_rl.runners import OnPolicyRunner
from ablation_curriculum import AblationCurriculum,TARGET_UPDATES,ROLLOUT_STEPS
from ablation_budget import FixedBudget,state_equal

CASE=Path(os.environ['GO2_AB_CASE']);CASE.mkdir(parents=True,exist_ok=True)
BASE=os.environ['GO2_AB_BASE'];SEED=int(os.environ['GO2_AB_SEED'])
controllers=[];loaded_verified=[]
resolve_original=train_rsl_rl.resolve_task_config
create_original=train_rsl_rl.create_isaaclab_env
load_original=OnPolicyRunner.load
learn_original=OnPolicyRunner.learn


def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def configure(*args,**kwargs):
    env_cfg,agent_cfg=resolve_original(*args,**kwargs)
    env_cfg.events.push_robot=None;env_cfg.events.base_external_force_torque=None
    agent_cfg.init_at_random_ep_len=False
    assert agent_cfg.num_steps_per_env==24
    return env_cfg,agent_cfg


def create(*args,**kwargs):
    result=create_original(*args,**kwargs);env=result.unwrapped
    controller=AblationCurriculum(env,CASE);controllers.append(controller)
    step_original=env.step
    def step(action):
        controller.before_step();out=step_original(action);controller.after_step();return out
    env.step=step
    return result


def load(self,path,load_cfg=None,strict=True,map_location=None):
    assert str(Path(path).resolve())==str(Path(BASE).resolve())
    result=load_original(self,path,load_cfg=None,strict=True,map_location=map_location)
    reference=torch.load(path,map_location='cpu',weights_only=False)
    actual=self.alg.save()
    for key in ['actor_state_dict','critic_state_dict','optimizer_state_dict']:
        assert state_equal(reference[key],actual[key]),'Initial state mismatch: '+key
    assert self.current_learning_iteration==reference['iter']
    loaded_verified.append(True)
    atomic_json(CASE/'initial-state.json',dict(base=path,actor_critic_optimizer_exact=True,
        initial_iteration=self.current_learning_iteration,initial_lr=self.alg.learning_rate))
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    return result


def learn(self,num_learning_iterations,init_at_random_ep_len=False):
    assert num_learning_iterations==TARGET_UPDATES and not init_at_random_ep_len
    assert loaded_verified and len(controllers)==1
    c=controllers[0];assert c.count==2048 and c.steps==0
    budget=FixedBudget();update_original=self.alg.update
    def update():
        losses=update_original()
        assert all(math.isfinite(float(value)) for value in losses.values()),'Nonfinite loss'
        budget.after_update(c.steps);c.completed_updates=budget.completed
        if budget.completed%25==0:
            atomic_json(CASE/'progress.json',dict(completed_updates=budget.completed,
                target_updates=TARGET_UPDATES,control_steps=c.steps,transitions=c.steps*c.count,
                left_phase=c.stages[2],losses={k:float(v) for k,v in losses.items()}))
        return losses
    self.alg.update=update
    learn_original(self,num_learning_iterations,init_at_random_ep_len=False)
    transitions=budget.validate_final(c.steps)
    saved=self.alg.save();saved.update(iter=self.current_learning_iteration,
        infos={'ablation_completed_updates':budget.completed,'collected_transitions':transitions})
    path=CASE/'final.pt';temporary=CASE/'final.tmp.pt'
    torch.save(saved,temporary);temporary.replace(path)
    for state in [saved['actor_state_dict'],saved['critic_state_dict']]:
        assert all(not value.is_floating_point() or torch.isfinite(value).all() for value in state.values())
    c.save()
    atomic_json(CASE/'train-result.json',dict(group=os.environ['GO2_AB_GROUP'],training_seed=SEED,
       completed_updates=budget.completed,control_steps=c.steps,collected_transitions=transitions,
       checkpoint=str(path),initial_state_verified=True,final_iteration=self.current_learning_iteration))
    print('ABLATION_FIXED_BUDGET_COMPLETE',budget.completed,transitions,flush=True)


train_rsl_rl.resolve_task_config=configure
train_rsl_rl.create_isaaclab_env=create
OnPolicyRunner.load=load;OnPolicyRunner.learn=learn
try:
    train_rsl_rl.run(sys.argv[1:])
finally:
    for controller in controllers:controller.save()
