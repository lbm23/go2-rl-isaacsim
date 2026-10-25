"""Measure policy deployment and foot geometry without modifying learned weights."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__import__('os').environ.get('GO2_DEMO_ROOT', Path(__file__).resolve().parent))

def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def load_yaml(path):
    import yaml
    class Loader(yaml.SafeLoader):
        pass
    def unknown(loader, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)
    Loader.add_constructor(None, unknown)
    return yaml.load(Path(path).read_text(), Loader=Loader)

def resolved_config(variant):
    import yaml
    data = load_yaml(ROOT/'artifact/env.yaml')
    spawn=data['scene']['robot']['spawn']
    spawn['rigid_props'].pop('func',None)
    spawn['articulation_props']=next(p for p in spawn['articulation_props'] if p.get('_usd_namespace')=='physxArticulation')
    spawn['articulation_props'].pop('func',None)
    if variant != 'baseline':
        for actuator in data['scene']['robot']['actuators'].values():
            for new, old in [('actuator_effort_limit','effort_limit'), ('actuator_velocity_limit','velocity_limit'),
                             ('joint_effort_limit','effort_limit_sim'), ('joint_velocity_limit','velocity_limit_sim')]:
                if actuator.get(new) is not None:
                    actuator[old]=actuator[new]
    if variant in ('substep', 'solver', 'feedback'):
        data['sim']['dt']=0.0025
        data['decimation']=8
    if variant == 'solver':
        spawn['articulation_props']['solver_position_iteration_count']=16
        spawn['articulation_props']['solver_velocity_iteration_count']=4
    path=ROOT/'artifact'/f'env-diagnostic-{variant}.yaml'
    path.write_text(yaml.safe_dump(data,sort_keys=False))
    return path

def run(args):
    from isaacsim import SimulationApp
    app=SimulationApp({'headless': True, 'width':640,'height':480,'multi_gpu':False})
    try:
        import hashlib
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager, IsaacEvents
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.robot.policy.examples import PolicyArtifact, RobotPolicyRunner, get_go2_spec
        from isaacsim.robot.policy.examples.binding import derive_binding, bind_policy
        from go2_policy_demo import create_scene, read_pose

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.robot.policy.examples',True)
        # Reuse tested authoring; replace only the uninitialized runner's artifact config.
        runner=create_scene()
        runner.close()
        config_path=resolved_config(args.variant)
        artifact_dir=ROOT/'artifact'
        model_path=artifact_dir/'policy.pt'
        descriptor_path=artifact_dir/'IO_descriptors.yaml'
        if args.hip_bias:
            import yaml
            descriptor=yaml.safe_load(descriptor_path.read_text())
            for action in descriptor['actions']:
                for name, sign in [('RL_hip_joint',1),('RR_hip_joint',-1)]:
                    if name in action['joint_names']:
                        action['offset'][action['joint_names'].index(name)] += sign*args.hip_bias
            descriptor_path=artifact_dir/f'IO-hip-{args.hip_bias:.2f}.yaml'
            descriptor_path.write_text(yaml.safe_dump(descriptor))
        artifact=PolicyArtifact.from_files(model_path=str(model_path),env_config_path=str(config_path),
             descriptor_path=str(descriptor_path),backend='torchscript',
             model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
        runner=RobotPolicyRunner(get_go2_spec(name='Go2 diagnostic',engines={'newton':artifact,'physx':artifact},
               binding=None,default_spawn_position=(0,0,0.4),default_spawn_orientation=(1,0,0,0)),
               prim_path='/World/Go2',training_engine='newton')
        runner.spawn()
        SimulationManager.set_physics_dt(runner.physics_dt)
        timeline=omni.timeline.get_timeline_interface()
        timeline.play()
        app.update()
        command=[args.speed,0.,0.]
        runner.restart_from_default_state(command)
        joint_names=list(runner.articulation.dof_names)
        feet=RigidPrim(paths=[f'/World/Go2/{leg}_foot' for leg in ['FL','FR','RL','RR']])
        print('DIAGNOSTIC_JOINTS',joint_names,flush=True)
        print('DIAGNOSTIC_ACTUATORS',runner.env_config.actuator_model_specs(joint_names),flush=True)
        binding=bind_policy(derive_binding(artifact.load_descriptor()),robot_joint_space=joint_names)
        tick=[0]
        samples=[]
        first={'joint_names':joint_names,'actuator_specs':str(runner.env_config.actuator_model_specs(joint_names)),
               'joint_properties':str(runner.env_config.joint_properties(joint_names))}
        def step(dt, context):
            t=tick[0]*runner.physics_dt
            if args.variant=='feedback':
                p,q=read_pose(runner)
                yaw=yaw_of(q)
                command[1]=float(np.clip(-0.5*p[1],-0.15,0.15))
                command[2]=float(np.clip(-1.5*yaw-0.4*p[1],-0.6,0.6))
            runner.step(dt,command)
            tick[0]+=1
            if tick[0] % runner.decimation == 0:
                p,q=read_pose(runner)
                p=np.asarray(p);q=np.asarray(q)
                footpos,_=feet.get_world_poses()
                bodyfeet=(footpos.numpy()-p)@rotation(q)
                joints=runner.articulation.get_dof_positions().numpy()[0]
                velocity=runner.articulation.get_dof_velocities().numpy()[0]
                lin,ang=runner.articulation.get_velocities()
                samples.append({'t':t,'p':p.tolist(),'q':q.tolist(),'yaw':yaw_of(q),
                    'feet_body':bodyfeet.tolist(),'joints':joints.tolist(),'joint_vel':velocity.tolist(),
                    'v_body':(rotation(q).T@lin.numpy()[0]).tolist(), 'yaw_rate':float(ang.numpy()[0,2]),
                    'command':command.copy()})
        callback=SimulationManager.register_callback(step,IsaacEvents.POST_PHYSICS_STEP)
        while tick[0]*runner.physics_dt < args.seconds:
            app.update()
        timeline.pause()
        SimulationManager.deregister_callback(callback)
        z=np.array([s['p'][2] for s in samples])
        y=np.array([s['p'][1] for s in samples])
        yaws=np.unwrap([s['yaw'] for s in samples])
        rear=np.array([s['feet_body'][2][1]-s['feet_body'][3][1] for s in samples])
        front=np.array([s['feet_body'][0][1]-s['feet_body'][1][1] for s in samples])
        rear_3d=np.array([np.linalg.norm(np.array(s['feet_body'][2])-s['feet_body'][3]) for s in samples])
        v=np.array([s['v_body'][0] for s in samples if s['t']>2])
        summary={'variant':args.variant,'hip_bias':args.hip_bias,'speed_command':args.speed,'duration':args.seconds,
            'final_position':samples[-1]['p'],'min_height':float(z.min()),
            'final_yaw_deg':float(np.degrees(yaws[-1])),'max_abs_yaw_deg':float(np.degrees(np.abs(yaws).max())),
            'max_abs_lateral_m':float(np.abs(y).max()),'mean_forward_speed':float(v.mean()),
            'rear_width_min':float(rear.min()),'rear_width_mean':float(rear.mean()),
            'front_width_mean':float(front.mean()),'rear_foot_distance_min':float(rear_3d.min()),
            'no_fall':bool(z.min()>.28)}
        result={'summary':summary,'interface':first,'samples':samples}
        (ROOT/f'diagnostic-{args.variant}-{args.speed:.1f}-hip{args.hip_bias:.2f}.json').write_text(json.dumps(result))
        print('GO2_DIAGNOSTIC_RESULT',json.dumps(summary),flush=True)
        runner.close()
    finally:
        app.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['baseline','limits','substep','solver','feedback'],default='baseline')
    parser.add_argument('--speed',type=float,default=.5)
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--hip-bias',type=float,default=0.)
    run(parser.parse_args())
