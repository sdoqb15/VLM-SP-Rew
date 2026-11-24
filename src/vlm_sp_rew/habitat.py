"""Habitat-Sim continuous reconstruction adapter for R2R-adapted ObjectNav.

Only RGB reaches the policy. Native semantic instance IDs define trainer-side
visibility; route endpoints are never silently converted into object goals.
"""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
from vlm_sp_rew.reward import require, file_bytes


class HabitatController:
    def __init__(self, spec, run_root):
        import habitat_sim
        self.hs = habitat_sim
        self.spec = spec
        self.root = Path(spec['asset_root']).resolve()
        require(type(spec.get('gpu_device_id')) is int and spec['gpu_device_id'] >= 0,
                'Explicit allocated Habitat renderer index required')
        self.sim = None
        self.failed = False
        self.horizon = 0.

    def reset(self, scene, randomSeed):
        require(scene.get('backend') == 'habitat' and scene.get('coordinate_frame') == 'Habitat_Y_up',
                'Use a Habitat-native scene descriptor, not raw R2R viewpoint coordinates')
        for asset in scene['assets']:
            file_bytes(self.root, asset)
        pinned = {a['path'] for a in scene['assets']}
        require(scene['scene_id'] in pinned, 'Rendered reconstruction must be hash-pinned')
        require(scene['scene_dataset_config'] in pinned, 'Dataset config must be hash-pinned')
        hs = self.hs
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = str(self.root / scene['scene_id'])
        cfg.scene_dataset_config_file = str(self.root / scene['scene_dataset_config'])
        cfg.gpu_device_id = self.spec['gpu_device_id']
        cfg.enable_physics = False
        agent = hs.agent.AgentConfiguration()
        agent.height = float(self.spec.get('agent_height', 1.5))
        agent.radius = float(self.spec.get('agent_radius', .1))
        sensors = []
        for uuid, kind in [('rgb', hs.SensorType.COLOR), ('semantic', hs.SensorType.SEMANTIC)]:
            sensor = hs.CameraSensorSpec()
            sensor.uuid, sensor.sensor_type = uuid, kind
            sensor.resolution = [300, 300]
            sensor.position = [0., float(self.spec.get('camera_height', 1.25)), 0.]
            sensor.hfov = float(self.spec.get('horizontal_fov_degrees', 90.))
            sensors.append(sensor)
        agent.sensor_specifications = sensors
        agent.action_space = {
            'MoveAhead': hs.agent.ActionSpec('move_forward', hs.agent.ActuationSpec(amount=.25)),
            'RotateLeft': hs.agent.ActionSpec('turn_left', hs.agent.ActuationSpec(amount=90.)),
            'RotateRight': hs.agent.ActionSpec('turn_right', hs.agent.ActuationSpec(amount=90.)),
        }
        if self.sim is not None:
            self.sim.close()
        self.sim = hs.Simulator(hs.Configuration(cfg, [agent]))
        self.sim.seed(randomSeed)
        require(self.sim.pathfinder.is_loaded, 'A real navigable mesh is required')
        self.categories = scene['semantic_category_mapping']
        require(self.sim.semantic_scene.objects, 'ObjectNav requires native semantic instance annotations')
        self.horizon, self.failed = 0., False
        return self._event(self.sim.get_sensor_observations())

    def _event(self, observations):
        from habitat_sim.utils.common import quat_to_angle_axis
        state = self.sim.get_agent(0).get_state()
        angle, axis = quat_to_angle_axis(state.rotation)
        yaw = math.degrees(float(angle) * float(axis[1])) % 360
        pos = dict(zip(('x','y','z'), map(float, state.position)))
        objects = []
        semantic = np.asarray(observations['semantic'])
        for obj in self.sim.semantic_scene.objects:
            if obj is None or obj.category is None:
                continue
            native = obj.category.name()
            if native not in self.categories:
                continue
            instance = str(obj.id)
            # Habitat semantic pixel IDs correspond to the numeric suffix of object IDs.
            pixel_id = int(instance.rsplit('_', 1)[-1])
            objects.append(dict(objectId=instance, objectType=self.categories[native],
                position=dict(zip(('x','y','z'), map(float, obj.aabb.center))),
                visible=bool(np.any(semantic == pixel_id))))
        return SimpleNamespace(frame=np.asarray(observations['rgb'])[...,:3].copy(), metadata=dict(
            lastActionSuccess=not self.failed, errorMessage='collision' if self.failed else '',
            agent=dict(position=pos,rotation=dict(y=yaw)), objects=objects))

    def step(self, action, **kw):
        self.failed = False
        if action == 'GetReachablePositions':
            samples=[self.sim.pathfinder.get_random_navigable_point() for _ in range(20000)]
            event=self._event(self.sim.get_sensor_observations())
            event.metadata['actionReturn']=[dict(zip(('x','y','z'),map(float,x))) for x in samples if np.isfinite(x).all()]
            return event
        if action == 'GetShortestPathToPoint':
            query=self.hs.ShortestPath()
            query.requested_start=np.asarray([kw['position'][k] for k in ('x','y','z')])
            query.requested_end=np.asarray([kw['target'][k] for k in ('x','y','z')])
            found=self.sim.pathfinder.find_path(query)
            event=self._event(self.sim.get_sensor_observations())
            event.metadata['lastActionSuccess']=bool(found)
            event.metadata['actionReturn']=dict(corners=[dict(zip(('x','y','z'),map(float,x))) for x in query.points])
            return event
        if action == 'Teleport':
            from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs
            state = self.sim.get_agent(0).get_state()
            state.position = np.asarray([kw['position'][k] for k in ('x','y','z')])
            require(self.sim.pathfinder.is_navigable(state.position), 'Frozen start is not navigable')
            require(abs(float(kw.get('horizon',0))) < 1e-8, 'R2R-adapted four-action camera horizon must be zero')
            if 'rotation_xyzw' in kw:
                state.rotation = quat_from_coeffs(np.asarray(kw['rotation_xyzw'],dtype=float))
            else:
                state.rotation = quat_from_angle_axis(math.radians(kw['rotation']['y']), np.array([0,1,0]))
            state.sensor_states = {}
            self.sim.get_agent(0).set_state(state)
            obs = self.sim.get_sensor_observations()
        elif action == 'Pass':
            obs = self.sim.get_sensor_observations()
        else:
            require(action in ('MoveAhead','RotateLeft','RotateRight'), 'R2R-adapted has four task actions')
            obs = self.sim.step(action)
            self.failed = bool(self.sim.previous_step_collided)
        return self._event(obs)

    def stop(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
