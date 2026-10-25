"""Pure helpers for exact update accounting and checkpoint-state checks."""
import torch


class FixedBudget:
    def __init__(self,target=2400,rollout_steps=24,num_envs=2048):
        self.target=target;self.rollout_steps=rollout_steps;self.num_envs=num_envs;self.completed=0

    def after_update(self,control_steps):
        assert self.completed<self.target,'Too many updates'
        assert control_steps==(self.completed+1)*self.rollout_steps,'Partial/mismatched rollout'
        self.completed+=1

    def validate_final(self,control_steps):
        assert self.completed==self.target,'Incomplete update budget'
        assert control_steps==self.target*self.rollout_steps
        return control_steps*self.num_envs


def state_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b
