"""Target 2 BW left while rehearsing backward/right and undisturbed walking."""
import os
import time
import torch
import robust_curriculum as base

DIRECTION_PROBABILITIES = [.05, .25, .50, .20]
LEFT_RANGES = [(1., 1.5), (1.25, 1.75), (1.5, 2.)]


def phase_at(elapsed_seconds):
    return 0 if elapsed_seconds < 180 else (1 if elapsed_seconds < 360 else 2)


def sample_levels(directions, phase):
    size=len(directions);device=directions.device
    uniform=lambda:torch.rand(size,device=device)
    lo,hi=LEFT_RANGES[phase]
    strong=lo+(hi-lo)*uniform()
    if phase==2:
        strong=torch.where(uniform()<1/3,2.,strong)
    left=torch.where(uniform()<.25,.75+.5*uniform(),strong)
    retained=torch.where(uniform()<.25,1.,.5+.5*uniform())
    forward=.25+.75*uniform()
    return torch.where(directions==2,left,torch.where(directions==0,forward,retained))


class Left2Curriculum(base.ForceCurriculum):
    def __init__(self,env,output_dir):
        self.wall_start=float(os.environ.get('GO2_LEFT_START_EPOCH',time.time()))
        self.configured_probabilities=False
        self.left_exact2=torch.zeros((),device=env.device)
        self.left_phase_samples=torch.zeros(3,device=env.device)
        super().__init__(env,output_dir)

    def reset(self,mask):
        if not self.configured_probabilities:
            self.probabilities=torch.tensor(DIRECTION_PROBABILITIES,device=self.device)
            self.configured_probabilities=True
        directions=torch.multinomial(self.probabilities,self.count,replacement=True)
        pushing=torch.rand(self.count,device=self.device)<.70
        phase=int(self.stage_tensor[2])
        level=sample_levels(directions,phase)
        onset=torch.randint(round(3/self.dt),round(8/self.dt)+1,(self.count,),device=self.device)
        self.age[mask]=0;self.direction[mask]=directions[mask]
        self.episode_stage[mask]=self.stage_tensor[directions[mask]]
        self.pushed[mask]=pushing[mask];self.onset[mask]=onset[mask];self.level[mask]=level[mask]
        self.finished[mask]=False;self.recovered[mask]=False;self.failed[mask]=False
        self.hold[mask]=0;self.force_vectors[mask]=0.
        selected=mask&pushing
        self.sampled_push+=selected.sum();self.sampled_no_push+=(mask&~pushing).sum()
        self.sampled_directions+=torch.bincount(torch.where(selected,directions+1,0),minlength=5)[1:]
        self.left_exact2+=(selected&(directions==2)&(level==2.)).sum()
        self.left_phase_samples[phase]+=(selected&(directions==2)).sum()

    def update_stages(self):
        phase=phase_at(time.time()-self.wall_start)
        if phase!=self.stages[2]:
            self.stages[2]=phase;self.stage_tensor[2]=phase
            event=dict(elapsed_seconds=time.time()-self.wall_start,control_step=self.steps,
                       left_phase=phase,left_range_bw=LEFT_RANGES[phase])
            self.events.append(event)
            print('LEFT2_PHASE='+__import__('json').dumps(event),flush=True)

    def save(self):
        summary=super().save()
        summary.update(direction_probabilities_given_push=DIRECTION_PROBABILITIES,
            curriculum_type='Fixed wall-time strength curriculum; no performance-driven phase selection',
            phase_boundaries_seconds=[180,360],left_ranges_bw=LEFT_RANGES,
            left_replay_probability=.25,left_replay_range_bw=[.75,1.25],
            left_last_phase_exact2_probability=.25,
            backward_right_distribution='75% uniform 0.5-1.0 BW, 25% exactly 1 BW',
            forward_range_bw=[.25,1.],sampled_left_exact2=int(self.left_exact2),
            sampled_left_phase_counts=self.left_phase_samples.cpu().tolist(),
            phase_changes=self.events,
            training_gate='Descriptive recovery within 3 s only; not used for phase transitions. Formal evaluation gate unchanged.')
        summary.pop('strength_ranges_bw',None);summary.pop('promotions',None)
        import json
        (self.output_dir/'curriculum.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
