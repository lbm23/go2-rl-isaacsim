"""Command and artifact adapters; no policy weights are changed."""
import copy
import math

HIP_STANCE_OFFSET = 0.10

def yaw_wxyz(q):
    if len(q) != 4 or not all(math.isfinite(float(v)) for v in q):
        raise ValueError('Invalid root quaternion')
    w,x,y,z=map(float,q)
    norm=math.sqrt(w*w+x*x+y*y+z*z)
    if norm < 1e-8:
        raise ValueError('Zero quaternion')
    w,x,y,z=(v/norm for v in (w,x,y,z))
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def straight_command(speed, position, quaternion, enabled=True):
    speed=float(speed)
    if not math.isfinite(speed) or not 0 <= speed <= .8:
        raise ValueError('Demo speed must be between 0 and 0.8 m/s')
    if len(position)!=3 or not all(math.isfinite(float(v)) for v in position):
        raise ValueError('Invalid robot position')
    yaw=yaw_wxyz(quaternion)
    if not enabled:
        return [speed,0.,0.]
    if speed == 0:
        return [0.,0.,max(-.3,min(.3,-1.5*yaw))]
    y=float(position[1])
    return [speed, max(-.15,min(.15,-.5*y)), max(-.6,min(.6,-1.5*yaw-.4*y))]

def adapt_env_config(original):
    data=copy.deepcopy(original)
    spawn=data['scene']['robot']['spawn']
    spawn['rigid_props'].pop('func',None)
    props=spawn['articulation_props']
    if isinstance(props,list):
        props=next(p for p in props if p.get('_usd_namespace')=='physxArticulation')
    props.pop('func',None)
    spawn['articulation_props']=props
    for actuator in data['scene']['robot']['actuators'].values():
        for current,legacy in [('actuator_effort_limit','effort_limit'),
            ('actuator_velocity_limit','velocity_limit'),('joint_effort_limit','effort_limit_sim'),
            ('joint_velocity_limit','velocity_limit_sim')]:
            if actuator.get(current) is not None:
                actuator[legacy]=copy.deepcopy(actuator[current])
    # Match the training integration resolution while retaining the 50 Hz actor.
    data['sim']['dt']=.0025
    data['decimation']=8
    return data

def adapt_descriptor(original,hip_offset=HIP_STANCE_OFFSET):
    if not 0 <= hip_offset <= .15:
        raise ValueError('Unvalidated rear hip offset')
    data=copy.deepcopy(original)
    changed=[]
    for action in data['actions']:
        for name,sign in [('RL_hip_joint',1),('RR_hip_joint',-1)]:
            if name in action['joint_names']:
                i=action['joint_names'].index(name)
                action['offset'][i]+=sign*hip_offset
                changed.append(name)
    if sorted(changed)!=['RL_hip_joint','RR_hip_joint']:
        raise ValueError('Both rear hip joints must be uniquely mapped')
    return data
