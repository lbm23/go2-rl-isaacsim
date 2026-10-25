"""Deploy the project's existing Go2 policy through Isaac Sim's native runner."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__import__('os').environ.get('GO2_DEMO_ROOT', Path(__file__).resolve().parent))

def runtime_config(artifact_dir, *, corrected=False):
    # Lab 3.0 exports schema application functions and per-engine schema lists;
    # the Sim 6.2 runner expects resolved USD property dictionaries.
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
    data = yaml.load((artifact_dir / 'env.yaml').read_text(), Loader=Loader)
    if corrected:
        from go2_straight_controller import adapt_env_config
        data = adapt_env_config(data)
        path = artifact_dir / 'env-straight-physx.yaml'
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path
    spawn = data['scene']['robot']['spawn']
    spawn['rigid_props'].pop('func', None)
    props = spawn['articulation_props']
    if isinstance(props, list):
        props = next(p for p in props if p.get('_usd_namespace') == 'physxArticulation')
    props.pop('func', None)
    spawn['articulation_props'] = props
    path = artifact_dir / 'env-runtime-physx.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path

def create_scene(*, corrected=False):
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.robot.policy.examples import PolicyArtifact, RobotPolicyRunner, get_go2_spec
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    SimulationManager.set_physics_sim_device('cpu')
    SimulationManager.set_physics_dt(0.005)
    stage.SetTimeCodesPerSecond(50)
    ground = UsdGeom.Cube.Define(stage, '/World/Ground')
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(200, 200, 0.1))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.3, 0.35)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(1.0)
    physics_material.CreateDynamicFrictionAttr(1.0)
    physics_material.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material, materialPurpose='physics')
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(1000)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr(2000)
    artifact_dir = ROOT / 'artifact'
    model_path = artifact_dir / 'policy.pt'
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    descriptor_path = artifact_dir / 'IO_descriptors.yaml'
    if corrected:
        import yaml
        from go2_straight_controller import adapt_descriptor
        descriptor = adapt_descriptor(yaml.safe_load(descriptor_path.read_text()))
        descriptor_path = artifact_dir / 'IO-straight.yaml'
        descriptor_path.write_text(yaml.safe_dump(descriptor))
    artifact = PolicyArtifact.from_files(
        model_path=str(model_path), env_config_path=str(runtime_config(artifact_dir, corrected=corrected)),
        descriptor_path=str(descriptor_path),
        backend='torchscript', model_sha256=digest)
    spec = get_go2_spec(
        name='Go2 project model_7350', engines={'newton': artifact, 'physx': artifact},
        default_spawn_position=(0, 0, 0.4), default_spawn_orientation=(1, 0, 0, 0),
        binding=None)
    runner = RobotPolicyRunner(spec, prim_path='/World/Go2', training_engine='newton')
    runner.spawn()
    SimulationManager.set_physics_dt(runner.physics_dt)
    assert runner.physics_dt * runner.decimation == .02
    return runner

def read_pose(runner):
    positions, orientations = runner.articulation.get_world_poses()
    return positions.numpy()[0].tolist(), orientations.numpy()[0].tolist()

if __name__ == '__main__':
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 640, 'height': 480, 'multi_gpu': False})
    try:
        import omni.kit.app
        import omni.timeline
        from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.robot.policy.examples', True)
        runner = create_scene()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        app.update()
        runner.restart_from_default_state([0.5, 0, 0])
        samples = []
        tick = [0]
        def step(dt, context):
            runner.step(dt, [0.5, 0, 0])
            tick[0] += 1
            if tick[0] % 40 == 0:
                pos, quat = read_pose(runner)
                samples.append({'time': tick[0] * 0.005, 'position': pos, 'orientation_wxyz': quat})
        callback = SimulationManager.register_callback(step, IsaacEvents.POST_PHYSICS_STEP)
        while tick[0] < 2400:
            app.update()
        timeline.pause()
        SimulationManager.deregister_callback(callback)
        heights = [s['position'][2] for s in samples]
        result = {'checkpoint': 'model_7350.pt', 'training_engine': 'newton_mjwarp',
                  'deployment_engine': SimulationManager.get_active_physics_engine(),
                  'duration_s': tick[0] * 0.005, 'command_m_s': [0.5, 0, 0],
                  'minimum_base_height_m': min(heights), 'final_position_m': samples[-1]['position'],
                  'forward_displacement_m': samples[-1]['position'][0],
                  'no_fall': min(heights) > 0.2,
                  'samples': samples}
        (ROOT / 'physx-test.json').write_text(json.dumps(result, indent=2))
        print('GO2_POLICY_RESULT', json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)
        runner.close()
    except BaseException:
        import traceback
        (ROOT / 'physx-test-error.txt').write_text(traceback.format_exc())
        raise
    finally:
        app.close()
