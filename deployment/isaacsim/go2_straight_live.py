"""Live Go2 straight-line evaluation with visible reference geometry and telemetry."""
import asyncio
import importlib
import importlib.util
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path

ROOT=Path(__import__('os').environ.get('GO2_DEMO_ROOT', Path(__file__).resolve().parent))
spec=importlib.util.spec_from_file_location('go2_straight_controller',str(ROOT/'go2_straight_controller.py'))
adapter=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=adapter
spec.loader.exec_module(adapter)
SESSION=sys.modules.setdefault('go2_straight_session',types.ModuleType('go2_straight_session'))

async def setup_straight_demo():
    import numpy as np
    import omni.kit.app
    import omni.timeline
    import omni.ui as ui
    import omni.usd
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.robot.policy.examples',True)
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf, UsdGeom, UsdShade, Sdf
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    try:
        app=omni.kit.app.get_app()
        context=omni.usd.get_context()
        timeline=omni.timeline.get_timeline_interface()
        stage=context.get_stage()
        backup=ROOT/('scene-before-straight-'+time.strftime('%Y%m%d-%H%M%S'))
        backup.mkdir()
        stage.GetRootLayer().Export(str(backup/'root.usda'))
        stage.GetSessionLayer().Export(str(backup/'session.usda'))
        old=getattr(SESSION,'state',None)
        if old is None:
            old=getattr(sys.modules.get('go2demo'),'GO2_DEMO',None)
        if old is not None:
            old['active']=False
            SimulationManager.deregister_callback(old['callback'])
            old['runner'].close()
            old['window'].destroy()
        timeline.stop()
        await app.next_update_async()
        await context.new_stage_async()
        policy_spec=importlib.util.spec_from_file_location('go2_policy_demo',str(ROOT/'go2_policy_demo.py'))
        go2_policy_demo=importlib.util.module_from_spec(policy_spec)
        sys.modules[policy_spec.name]=go2_policy_demo
        policy_spec.loader.exec_module(go2_policy_demo)
        runner=go2_policy_demo.create_scene(corrected=True)
        stage=context.get_stage()

        # Non-colliding visual guides; only the existing ground cube has collision.
        material=UsdShade.Material.Define(stage,'/World/GroundVisual')
        shader=UsdShade.Shader.Define(stage,'/World/GroundVisual/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.12,.15,.19))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.9)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/Ground')).Bind(material)

        def line(name,points,color,width=.015):
            curve=UsdGeom.BasisCurves.Define(stage,'/World/Guides/'+name)
            curve.CreateTypeAttr('linear')
            curve.CreateCurveVertexCountsAttr([len(points)])
            curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
            curve.CreateWidthsAttr([width])
            curve.SetWidthsInterpolation('constant')
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            return curve
        for x in range(-5,81):
            line(f'GridX{x+5}',[(x,-5,.008),(x,5,.008)],(.3,.34,.39),.012)
        for i in range(-5,6):
            line(f'GridY{i+5}',[(-5,i,.008),(80,i,.008)],(.3,.34,.39),.012)
        line('Target',[(-2,0,.015),(80,0,.015)],(.1,.4,1),.035)
        line('LaneLeft',[(-2,.3,.015),(80,.3,.015)],(.85,.85,.85),.018)
        line('LaneRight',[(-2,-.3,.015),(80,-.3,.015)],(.85,.85,.85),.018)
        line('Finish',[(15,-.6,.02),(15,.6,.02)],(.1,1,.3),.055)
        trail=line('ActualTrail',[(0,0,.025),(.001,0,.025)],(0,1,1),.025)
        heading=line('BodyHeading',[(0,0,.55),(.8,0,.55)],(1,.85,.05),.025)
        timeline.play()
        await app.next_update_async()
        runner.restart_from_default_state([.5,0,0])
        feet=RigidPrim(paths=[f'/World/Go2/{leg}_foot' for leg in ['FL','FR','RL','RR']])
        markers=[]
        for i,name in enumerate(['FL','FR','RL','RR']):
            sphere=UsdGeom.Sphere.Define(stage,'/World/Guides/Foot'+name)
            sphere.CreateRadiusAttr(.022)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*( (0.2,1,.3) if i<2 else (1,.45,.05) ))])
            markers.append(sphere.AddTranslateOp())
        control_path=ROOT/'straight-control.json'
        initial_control_mtime=control_path.stat().st_mtime_ns if control_path.exists() else None
        state={'active':True,'runner':runner,'feet':feet,'tick':0,'speed':.5,'enabled':True,
            'status':'walking','samples':[],'trail':[[0,0,.025],[.001,0,.025]],'min_height':100.,
            'max_yaw':0.,'max_lateral':0.,'camera':'side','recording':False,'record_frames':0,
            'last_control_mtime':initial_control_mtime,'command':[.5,0,0],'backup':str(backup),
            'completed_saved':False,'trial_speed':.5}
        SESSION.state=state

        def pose():
            p,q=go2_policy_demo.read_pose(runner)
            return p,q,adapter.yaw_wxyz(q)
        def save():
            p,q,yaw=pose()
            summary={'checkpoint':'model_7350.pt','engine':'physx','physics_dt':runner.physics_dt,
                'policy_hz':50,'rear_hip_offset_rad':adapter.HIP_STANCE_OFFSET,
                'feedback_enabled':state['enabled'],'state':state['status'],'simulation_seconds':state['tick']*.0025,
                'position_m':p,'heading_error_deg':math.degrees(yaw),'cross_track_m':p[1],
                'max_abs_heading_deg':state['max_yaw'],'max_abs_cross_track_m':state['max_lateral'],
                'minimum_height_m':state['min_height'],'speed_request_m_s':state['speed'],
                'trial_speed_m_s':state['trial_speed'],
                'actual_command':state['command'],'rear_width_m':state.get('rear_width'),
                'record_frames':state['record_frames'],'scene_backup':str(backup)}
            (ROOT/'straight-live-status.json').write_text(json.dumps(summary,indent=2))
            (ROOT/'straight-live-samples.json').write_text(json.dumps(state['samples']))

        def step(dt,physics_context):
            try:
                p,q,yaw=pose()
                if p[0]>=15 and state['speed']>0:
                    state['speed']=0.
                    state['status']='finished-15m-standing'
                command=adapter.straight_command(state['speed'],p,q,state['enabled'])
                state['command']=command
                runner.step(dt,command)
                state['tick']+=1
                if state['tick']%8==0:
                    p,q,yaw=pose()
                    footpos,_=feet.get_world_poses()
                    foot_world=footpos.numpy()
                    body_y=np.array([-math.sin(yaw),math.cos(yaw),0])
                    rear_width=float((foot_world[2]-foot_world[3])@body_y)
                    lin,ang=runner.articulation.get_velocities()
                    v=lin.numpy()[0]
                    state.update(rear_width=rear_width,vx=float(v[0]),foot_positions=foot_world.tolist())
                    state['min_height']=min(state['min_height'],p[2])
                    state['max_yaw']=max(state['max_yaw'],abs(math.degrees(yaw)))
                    state['max_lateral']=max(state['max_lateral'],abs(p[1]))
                    state['samples'].append({'t':state['tick']*.0025,'p':p,'yaw_deg':math.degrees(yaw),
                        'rear_width':rear_width,'vx':float(v[0]),'command':command})
                    if len(state['samples'])>3000:
                        state['samples'].pop(0)
                    if state['tick']%40==0:
                        previous=state['trail'][-1]
                        if math.hypot(p[0]-previous[0],p[1]-previous[1])>=.01:
                            state['trail'].append([p[0],p[1],.025])
                        if len(state['trail'])>1200:
                            state['trail'].pop(0)
                    if p[2]<.28 or abs(q[1])**2+abs(q[2])**2>.15:
                        state['status']='safety-paused'
                        timeline.pause()
                    if state['tick']%400==0:
                        save()
                    if state['status']=='finished-15m-standing' and not state['completed_saved']:
                        save()
                        completed={'summary':json.loads((ROOT/'straight-live-status.json').read_text()),
                            'samples':state['samples']}
                        (ROOT/'straight-completed.json').write_text(json.dumps(completed))
                        state['completed_saved']=True
            except Exception:
                timeline.pause()
                (ROOT/'straight-live-error.txt').write_text(traceback.format_exc())
                raise
        state['callback']=SimulationManager.register_callback(step,IsaacEvents.POST_PHYSICS_STEP)

        def walk():
            state['speed']=float(speed.model.get_value_as_float())
            state['trial_speed']=state['speed']
            state['status']='walking'
            timeline.play()
        def stand():
            state['speed']=0.
            state['status']='standing'
            timeline.play()
        def pause():
            state['status']='paused'
            timeline.pause()
            save()
        def reset():
            runner.restart_from_default_state([0,0,0])
            state.update(tick=0,samples=[],trail=[[0,0,.025],[.001,0,.025]],min_height=100.,max_yaw=0.,max_lateral=0.,completed_saved=False)
            walk()
        def toggle(model):
            state['enabled']=model.get_value_as_bool()
        def view(mode):
            state['camera']=mode

        window=ui.Window('Go2 | straight-line test',width=325,height=390)
        window.position_x=800
        window.position_y=250
        state['window']=window
        with window.frame:
            with ui.VStack(spacing=4):
                ui.Label('model_7350 | weights unchanged')
                ui.Label('400 Hz physics / 50 Hz policy')
                ui.Label('Rear hip stance offset: +/-0.10 rad')
                with ui.HStack(height=22):
                    check=ui.CheckBox(width=22)
                    check.model.set_value(True)
                    check.model.add_value_changed_fn(toggle)
                    ui.Label('Heading + line hold',word_wrap=True)
                ui.Label('Speed (m/s) | Finish: X=15m')
                speed=ui.FloatSlider(min=0.,max=.8,height=22)
                speed.model.set_value(.5)
                with ui.HStack(height=24,spacing=4):
                    ui.Button('Walk',width=75,clicked_fn=walk)
                    ui.Button('Stand',width=75,clicked_fn=stand)
                    ui.Button('Reset',width=75,clicked_fn=reset)
                    ui.Button('Pause',width=75,clicked_fn=pause)
                with ui.HStack(height=24,spacing=4):
                    ui.Button('Side',width=100,clicked_fn=lambda:view('side'))
                    ui.Button('Rear',width=100,clicked_fn=lambda:view('rear'))
                    ui.Button('Top',width=100,clicked_fn=lambda:view('top'))
                status_label=ui.Label('Starting...',word_wrap=True,height=100)
                ui.Label('Blue: target | Cyan: actual trajectory')
                ui.Label('Yellow: heading | Orange: rear feet')
                ui.Label('Grid spacing: 1m | White lane: +/-0.30m')
        print('GO2_STRAIGHT_LIVE_READY',flush=True)
        index=0
        last_record_time=-100.
        while state['active']:
            await app.next_update_async()
            index+=1
            if index%3:
                continue
            p,q,yaw=pose()
            if state['camera']=='side':
                eye=[p[0]+1.8,-3.2,1.9];target=[p[0],0,.25]
            elif state['camera']=='rear':
                eye=[p[0]-2.3,0,1.05];target=[p[0]+.25,0,.25]
            else:
                eye=[p[0]+.2,-.01,6];target=[p[0]+.2,0,0]
            set_camera_view(eye=eye,target=target)
            points=state['trail']
            trail.GetCurveVertexCountsAttr().Set([len(points)])
            trail.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
            heading.GetPointsAttr().Set([Gf.Vec3f(p[0],p[1],.55),
                Gf.Vec3f(p[0]+.7*math.cos(yaw),p[1]+.7*math.sin(yaw),.55)])
            for marker,point in zip(markers,state.get('foot_positions',[])):
                marker.Set(Gf.Vec3d(*point))
            status_label.text=(f"{state['status']} | t={state['tick']*.0025:.1f}s\n"
                f"X={p[0]:.2f}m | lateral={p[1]*100:+.1f}cm\n"
                f"Heading error={math.degrees(yaw):+.2f}deg\n"
                f"Speed={state.get('vx',0):.2f}m/s | height={p[2]:.2f}m\n"
                f"Rear foot width={state.get('rear_width',0)*100:.1f}cm")
            control_path=ROOT/'straight-control.json'
            if control_path.exists() and control_path.stat().st_mtime_ns!=state['last_control_mtime']:
                state['last_control_mtime']=control_path.stat().st_mtime_ns
                control=json.loads(control_path.read_text())
                action=control.get('action')
                if action=='walk':
                    speed.model.set_value(float(control.get('speed',.5)))
                    walk()
                elif action=='reset_walk':
                    speed.model.set_value(float(control.get('speed',.5)))
                    reset()
                elif action=='stand': stand()
                elif action=='pause': pause()
                elif action=='view': view(control.get('view','side'))
                elif action=='capture':
                    capture_viewport_to_file(get_active_viewport(),str(ROOT/('straight-'+state['camera']+'.png')))
                elif action=='record':
                    (ROOT/'recording').mkdir(exist_ok=True)
                    state['recording']=True
                    state['record_frames']=0
                    last_record_time=-100.
            if state['recording']:
                now=state['tick']*.0025
                if now-last_record_time>=.095:
                    frame=state['record_frames']
                    capture_viewport_to_file(get_active_viewport(),str(ROOT/'recording'/f'frame{frame:04d}.png'))
                    state['record_frames']+=1
                    last_record_time=now
                    if state['record_frames']>=80:
                        state['recording']=False
                        save()
    except BaseException:
        (ROOT/'straight-live-error.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),flush=True)
        raise

SESSION.task=asyncio.ensure_future(setup_straight_demo())
