"""Compare the unchanged policy in the original Newton/MJWarp environment."""
import json
import math
from pathlib import Path

import torch
from isaaclab_rl.entrypoints.backends import play_rsl_rl

ROOT=Path(__import__('os').environ.get('GO2_DEMO_ROOT', Path(__file__).resolve().parent))
captured={}
original_resolve=play_rsl_rl.resolve_task_config
original_create=play_rsl_rl.create_isaaclab_env

def configure(*args,**kwargs):
    cfg,agent=original_resolve(*args,**kwargs)
    cfg.episode_length_s=35.
    cfg.observations.policy.enable_corruption=False
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.rel_heading_envs=0.
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(10000.,10000.)
    cfg.commands.base_velocity.ranges.lin_vel_x=(.5,.5)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.debug_vis=False
    cfg.events.reset_base.params['pose_range']={}
    cfg.events.reset_base.params['velocity_range']={}
    cfg.events.reset_robot_joints.params['position_range']=(1.,1.)
    cfg.events.add_base_mass=None
    cfg.events.base_com=None
    return cfg,agent

def create(*args,**kwargs):
    env=original_create(*args,**kwargs)
    captured['env']=env.unwrapped
    return env

def evaluate(step,*,dt,args_cli,env_cfg):
    env=captured['env']
    robot=env.scene['robot']
    feet_names=['FL_foot','FR_foot','RL_foot','RR_foot']
    foot_ids=[robot.body_names.index(n) for n in feet_names]
    samples=[]
    with torch.inference_mode():
        for index in range(round(20/dt)):
            env.command_manager.get_command('base_velocity')[:]=torch.tensor([.5,0,0],device=env.device)
            step()
            q=robot.data.root_quat_w[0].cpu().tolist() # Lab 3.0 xyzw
            x,y,z,w=q
            yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            p=robot.data.root_pos_w[0].cpu().tolist()
            from isaaclab.utils.math import quat_apply_inverse
            feet=robot.data.body_pos_w[0,foot_ids]-robot.data.root_pos_w[0]
            qb=robot.data.root_quat_w[0].repeat(4,1)
            feet=quat_apply_inverse(qb,feet).cpu().tolist()
            samples.append({'t':index*dt,'p':p,'yaw':yaw,'feet_body':feet,
                'v_body':robot.data.root_lin_vel_b[0].cpu().tolist(),
                'joints':robot.data.joint_pos[0].cpu().tolist(),
                'terminated':bool(env.reset_terminated[0])})
    import numpy as np
    rear=np.array([s['feet_body'][2][1]-s['feet_body'][3][1] for s in samples])
    yaws=np.unwrap([s['yaw'] for s in samples])
    summary={'engine':'original_newton_mjwarp','duration':20.,'final_position':samples[-1]['p'],
        'final_yaw_deg':float(np.degrees(yaws[-1])),
        'max_abs_lateral_m':max(abs(s['p'][1]) for s in samples),
        'rear_width_mean':float(rear.mean()),'rear_width_min':float(rear.min()),
        'min_height':min(s['p'][2] for s in samples),
        'no_fall':not any(s['terminated'] for s in samples)}
    (ROOT/'diagnostic-original-newton.json').write_text(json.dumps({'summary':summary,
        'joint_names':robot.joint_names,'samples':samples}))
    print('GO2_NEWTON_RESULT',json.dumps(summary),flush=True)

play_rsl_rl.resolve_task_config=configure
play_rsl_rl.create_isaaclab_env=create
play_rsl_rl.run_playback=evaluate
play_rsl_rl.run(['--task','Isaac-Velocity-Flat-UnitreeGo2','--checkpoint',str(ROOT/'artifact/model_7350.pt'),
    '--num_envs','1','--seed','42','--viz','none','--train_env_cfg','physics=newton_mjwarp'])
