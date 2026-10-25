"""Three predeclared force distributions with identical random-number budgets."""
import json
import os
from pathlib import Path
import torch
from robust_curriculum import ForceCurriculum

TARGET_UPDATES=2400
ROLLOUT_STEPS=24
WEIGHTED=[.05,.25,.50,.20]
UNIFORM=[.25]*4


def phase_at_update(completed):
    return 0 if completed<720 else (1 if completed<1440 else 2)


def recipes(group,completed,u):
    assert group in ('A','B','C') and u.shape[0]==9
    weights=UNIFORM if group=='A' else WEIGHTED
    thresholds=torch.tensor(weights,device=u.device,dtype=u.dtype).cumsum(0)
    directions=(u[0,:,None]>=thresholds[None,:]).sum(-1).clamp(max=3)
    if group=='C':
        phase=torch.full_like(directions,phase_at_update(completed))
    else:
        phase=(u[1]>=.3).long()+(u[1]>=.6).long()
    low=1.+.25*phase.float();high=low+.5
    high=torch.where(phase==2,2.,high)
    strong=low+(high-low)*u[4]
    strong=torch.where((phase==2)&(u[3]<1/3),2.,strong)
    left=torch.where(u[2]<.25,.75+.5*u[4],strong)
    retained=torch.where(u[5]<.25,1.,.5+.5*u[4])
    level=torch.where(directions==2,left,torch.where(directions==0,.25+.75*u[4],retained))
    return directions,u[8]<.70,level,(150+torch.floor(u[7]*251)).long(),phase


class AblationCurriculum(ForceCurriculum):
    def __init__(self,env,output_dir):
        self.group=os.environ['GO2_AB_GROUP'];self.training_seed=int(os.environ['GO2_AB_SEED'])
        self.completed_updates=0
        self.generator=torch.Generator(device=env.device).manual_seed(self.training_seed+1000003)
        self.recipe_phase=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.allocated_phase=torch.zeros(3,device=env.device)
        self.executed=torch.zeros((4,3),device=env.device)
        self.executed_exact2=torch.zeros((),device=env.device)
        self.executed_force_sum=torch.zeros(4,device=env.device)
        self.executed_level_sum=torch.zeros(4,device=env.device)
        super().__init__(env,output_dir)

    def reset(self,mask):
        u=torch.rand((9,self.count),generator=self.generator,device=self.device)
        directions,pushing,level,onset,phase=recipes(self.group,self.completed_updates,u)
        self.age[mask]=0;self.direction[mask]=directions[mask]
        self.pushed[mask]=pushing[mask];self.onset[mask]=onset[mask]
        self.level[mask]=level[mask];self.recipe_phase[mask]=phase[mask]
        self.episode_stage[mask]=self.stage_tensor[directions[mask]]
        self.finished[mask]=False;self.recovered[mask]=False;self.failed[mask]=False
        self.hold[mask]=0;self.force_vectors[mask]=0.
        selected=mask&pushing
        self.sampled_push+=selected.sum();self.sampled_no_push+=(mask&~pushing).sum()
        self.sampled_directions+=torch.bincount(torch.where(selected,directions+1,0),minlength=5)[1:]
        self.allocated_phase+=torch.bincount(torch.where(selected&(directions==2),phase+1,0),minlength=4)[1:]

    def before_step(self):
        phase=phase_at_update(self.completed_updates) if self.group=='C' else 0
        if self.stages[2]!=phase:
            self.stages[2]=phase;self.stage_tensor[2]=phase
            self.events.append(dict(completed_updates=self.completed_updates,left_phase=phase))
        starting=self.pushed&(self.age==self.onset)
        super().before_step()
        categories=torch.where(starting,self.direction*3+self.recipe_phase+1,0)
        self.executed+=torch.bincount(categories,minlength=13)[1:].reshape(4,3)
        self.executed_force_sum.scatter_add_(0,self.direction,self.forces[:,0].norm(dim=-1)*starting)
        self.executed_level_sum.scatter_add_(0,self.direction,self.level*starting)
        self.executed_exact2+=(starting&(self.direction==2)&(self.level==2.)).sum()

    def update_stages(self):
        pass  # Stage transitions use completed PPO cycles only, never recovery scores.

    def save(self):
        summary=super().save()
        summary.update(group=self.group,training_seed=self.training_seed,
          completed_updates=self.completed_updates,collected_control_steps=self.steps,
          collected_transitions=self.steps*self.count,
          direction_probabilities_given_push=UNIFORM if self.group=='A' else WEIGHTED,
          left_allocated_recipe_phase_counts=self.allocated_phase.cpu().tolist(),
          executed_pushes_by_direction_and_recipe_phase=self.executed.cpu().tolist(),
          executed_left_exact2=int(self.executed_exact2),
          executed_force_sum_N=self.executed_force_sum.cpu().tolist(),
          executed_level_sum_bw=self.executed_level_sum.cpu().tolist(),
          phase_changes=self.events,
          training_gate='Descriptive recovery only; not used for curriculum progression.',
          strength_definition='A/B pool phases 30/30/40 from the beginning; C uses update boundaries 720/1440; same conditional strength recipes')
        summary.pop('strength_ranges_bw',None);summary.pop('promotions',None)
        (self.output_dir/'curriculum.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
