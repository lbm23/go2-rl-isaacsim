"""Continue PPO from a checkpoint using the isolated force curriculum wrapper."""
import os
import sys

from isaaclab_rl.entrypoints.backends import train_rsl_rl
from robust_curriculum import ForceCurriculum

original_resolver = train_rsl_rl.resolve_task_config
original_creator = train_rsl_rl.create_isaaclab_env
controllers = []


def configure(*args, **kwargs):
    env_cfg, agent_cfg = original_resolver(*args, **kwargs)
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    agent_cfg.init_at_random_ep_len = False
    return env_cfg, agent_cfg


def create_env(*args, **kwargs):
    result = original_creator(*args, **kwargs)
    raw = result.unwrapped
    controller = ForceCurriculum(raw, os.environ['GO2_ROBUST_REPORT_DIR'])
    controllers.append(controller)
    original_step = raw.step

    def step(action):
        controller.before_step()
        output = original_step(action)
        controller.after_step()
        return output

    raw.step = step
    return result


train_rsl_rl.resolve_task_config = configure
train_rsl_rl.create_isaaclab_env = create_env
try:
    train_rsl_rl.run(sys.argv[1:])
finally:
    for controller in controllers:
        controller.save()
