"""Revised frozen-bank ObjectNav adapter. No downloads, model calls or GPU selection at import."""
from pathlib import Path
import copy, hashlib, json, math, os, shutil, time, uuid
import numpy as np
import elements
import embodied
from revision_protocol import (
    ARMS, require, finite, file_bytes, load_episode_bank, EpisodeSchedule,
    goal_one_hot, bind_field, SharedReward, requested_goal_state,
    path_length_metres, qualified_spl,
)

ACTIONS = ('MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Stop')

def spaces(vocabulary):
    obs = {
        'image': elements.Space(np.uint8, (96, 96, 3)),
        'goal': elements.Space(np.float32, (len(vocabulary),), 0, 1),
        'reward': elements.Space(np.float32, ()),
        'is_first': elements.Space(bool, ()),
        'is_last': elements.Space(bool, ()),
        'is_terminal': elements.Space(bool, ()),
    }
    return obs, {'action': elements.Space(np.int32, (), 0, len(ACTIONS))}

class FieldLookup:
    """Retain original nearest/previous-room rule; field is trainer-side only."""
    def __init__(self, field):
        self.points = field.get('points', [])
        require(self.points, 'Nonempty field required')
        for point in self.points:
            for axis in ('x', 'z'):
                finite(point['position'][axis], 'field.' + axis)
            finite(point['value'], 'field.value')
        self.last = None
    def value(self, position):
        ranked = sorted(enumerate(self.points), key=lambda item: (
            math.hypot(item[1]['position']['x'] - position['x'],
                       item[1]['position']['z'] - position['z']), item[0]))
        nearest = ranked[0][1]
        room = (self.last or nearest).get('room_label')
        chosen = nearest
        for _, point in ranked:
            distance = math.hypot(point['position']['x'] - position['x'],
                                  point['position']['z'] - position['z'])
            if distance <= .2:
                chosen = point
                break
            if point.get('room_label') == room:
                chosen = point if distance < 1.5 else nearest
                break
        self.last = chosen
        return float(chosen['value'])

def make_local_controller(spec, run_root):
    """Qualified H20 experience: Vulkan index is explicit, distinct from CUDA index."""
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    import shlex
    kwargs = dict(spec['controller_kwargs'])
    executable = Path(kwargs['local_executable_path']).resolve()
    digest = hashlib.sha256()
    with executable.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    require(digest.hexdigest() == spec['executable_sha256'], 'Engine SHA mismatch')
    index = spec.get('vulkan_device_index')
    require(type(index) is int and index >= 0, 'Explicit Vulkan device index required')
    require(spec.get('physical_gpu_uuid'), 'Record allocated physical renderer GPU UUID')
    require('gpu_device' not in kwargs and 'platform' not in kwargs,
            'Use explicit Vulkan index; platform fixed to CloudRendering')
    require(kwargs.get('width') == 300 and kwargs.get('height') == 300,
            'Revised default captures 300x300, policy 96x96; change protocol before changing size')
    class LocalController(Controller):
        @property
        def base_dir(self):
            result = Path(run_root) / 'engine_runtime'
            result.mkdir(parents=True, exist_ok=True)
            return str(result)
        @property
        def log_dir(self):
            result = Path(run_root) / 'simulator_logs'
            result.mkdir(parents=True, exist_ok=True)
            return str(result)
        def unity_command(self, *args, **opts):
            allocated = self.gpu_device
            self.gpu_device = None
            try:
                cmd = super().unity_command(*args, **opts)
            finally:
                self.gpu_device = allocated
            cmd = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
            return cmd + ['-force-device-index', str(index), '-logFile',
                          str(Path(self.log_dir) / ('Player-%s.log' % os.getpid()))]
    return LocalController(platform=CloudRendering, **kwargs)

class RevisedProcThorEnv(embodied.Env):
    """Fixed requested goal, exact scene/field binding and common four-arm reward."""
    @staticmethod
    def spaces_from_manifest(path, data_root=None):
        qualified = load_episode_bank(path, data_root)
        return spaces(qualified['goal_vocabulary'])

    def __init__(self, episode_manifest, data_root=None, split='train', seed=0,
                 arm='env', shared_reward_config=None, controller_config=None,
                 run_record_root=None, max_steps=200, shuffle=False,
                 controller_factory=None, allow_synthetic_fixture=False):
        require(arm in ARMS, 'Unknown arm')
        require(type(max_steps) is int and max_steps > 0, 'Positive max_steps required')
        self.qualified = load_episode_bank(episode_manifest, data_root)
        require(allow_synthetic_fixture or not self.qualified['bank'].get('synthetic', False),
                'Synthetic engineering bank cannot be used for training')
        self.root = Path(self.qualified['root'])
        self.schedule = EpisodeSchedule(self.qualified, split, seed, shuffle)
        self.arm, self.max_steps = arm, max_steps
        self.reward_config = json.loads(Path(shared_reward_config).read_text())
        self.rewarder = SharedReward(arm, self.reward_config)
        self.vocabulary = self.qualified['goal_vocabulary']
        self._obs_space, self._act_space = spaces(self.vocabulary)
        require(run_record_root is not None, 'Explicit writable run evidence directory required')
        self.run_root = Path(run_record_root) / ('env-' + uuid.uuid4().hex)
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.allow_synthetic_fixture = allow_synthetic_fixture
        self._guard()
        self.controller_spec = json.loads(Path(controller_config).read_text()) if controller_config else {}
        require(controller_factory is None or allow_synthetic_fixture,
                'Custom controller factory is restricted to explicit engineering tests')
        self.controller = ((controller_factory(self.controller_spec, self.run_root))
                           if controller_factory else make_local_controller(self.controller_spec, self.run_root))
        self.done = True
        self.closed = False

    @property
    def obs_space(self):
        return self._obs_space
    @property
    def act_space(self):
        return self._act_space
    def _guard(self):
        require(shutil.disk_usage(self.run_root).free >= 110 * 1024 ** 3,
                'Workvolume below 100GiB plus 10GiB margin; stop own writes')
        self.last_guard = time.monotonic()
    def _record(self, row):
        if time.monotonic() - self.last_guard >= 25:
            self._guard()
        with self.record_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    def _check_event(self, event, allocation=False):
        require(event.metadata.get('lastActionSuccess') is True,
                'Frozen allocation/action failed: ' + str(event.metadata.get('errorMessage', '')))
    def _state(self, event):
        position = event.metadata['agent']['position']
        state = requested_goal_state(event.metadata['objects'], self.episode['goal_category'], position)
        require(state['goal_instance_ids'] == sorted(self.episode['goal_instance_ids']),
                'Actual requested-goal instances differ from frozen scene allocation')
        return position, state
    def _observation(self, event, result=None, first=False):
        from PIL import Image
        result = result or {}
        rgb = np.asarray(event.frame)
        require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3, 'Actual RGB frame required')
        obs = {
            'image': np.asarray(Image.fromarray(rgb).resize((96, 96)), dtype=np.uint8),
            'goal': np.array(goal_one_hot(self.episode['goal_category'], self.vocabulary), np.float32),
            'reward': np.float32(result.get('reward', 0)),
            'is_first': np.bool_(first),
            'is_last': np.bool_(result.get('is_last', False)),
            'is_terminal': np.bool_(result.get('is_terminal', False)),
            'log/episode_reward': np.float32(self.episode_reward),
            'log/episode_length': np.int32(self.steps),
            'log/success': np.float32(result.get('success', False)),
            'log/executed_path_m': np.float32(path_length_metres(self.poses)),
            'log/spl_valid': np.bool_(False),
            'log/spl': np.float32(0),
        }
        if result.get('is_last') and self.episode.get('spl_reference'):
            reference = dict(self.episode['spl_reference'])
            require(reference.get('scene_sha256') == self.episode['scene']['sha256'] and
                    reference.get('goal_category') == self.episode['goal_category'] and
                    reference.get('start_position') == self.episode['start_position'] and
                    reference.get('start_yaw_degrees') == self.episode['start_yaw_degrees'] and
                    reference.get('success_distance_m') == self.reward_config['success_distance_m'],
                    'SPL reference must match the exact scene, goal, start and success rule')
            reference['executed_path_m'] = path_length_metres(self.poses)
            obs['log/spl'] = np.float32(qualified_spl(bool(result['success']), reference))
            obs['log/spl_valid'] = np.bool_(True)
        return obs

    def reset(self):
        return self._reset()
    def _reset(self):
        self.episode = self.schedule.next()
        self.record_path = self.run_root / ('episode-' + uuid.uuid4().hex + '.jsonl')
        try:
            self.field = bind_field(self.episode, self.arm, self.root)
            if self.field and self.arm == 'vlm' and not self.allow_synthetic_fixture:
                require(self.field.get('source_integrity_qualified') is True,
                        'Real VLM field requires qualified source annotation provenance')
            self.lookup = FieldLookup(self.field) if self.field else None
            self.steps, self.episode_reward = 0, 0.
            self.done = True
            scene = json.loads(file_bytes(self.root, self.episode['scene']))
            event = self.controller.reset(scene=scene, randomSeed=self.episode['seed'])
            self._check_event(event, allocation=True)
            event = self.controller.step(
                action='Teleport', position=self.episode['start_position'],
                rotation={'x': 0, 'y': self.episode['start_yaw_degrees'], 'z': 0},
                horizon=self.episode.get('start_horizon_degrees', 30), forceAction=False)
            self._check_event(event, allocation=True)
            position, state = self._state(event)
            for axis in ('x', 'y', 'z'):
                require(abs(position[axis] - self.episode['start_position'][axis]) <= .02,
                        'Actual allocated start differs from frozen start.' + axis)
            actual_yaw = event.metadata['agent']['rotation']['y']
            require(abs((actual_yaw - self.episode['start_yaw_degrees'] + 180) % 360 - 180) <= .1,
                    'Actual yaw differs from frozen yaw')
            self.poses = [dict(position)]
            self.rewarder.reset(state['distance'], self.lookup.value(position) if self.lookup else None)
            self._record(dict(event='reset', episode_id=self.episode['episode_id'],
                manifest_sha256=self.qualified['manifest_sha256'], episode=self.episode,
                arm=self.arm, reward_config=self.reward_config, controller_spec=self.controller_spec,
                pose=position, state=state, synthetic_engineering_fixture=self.allow_synthetic_fixture))
            self.done = False
            return self._observation(event, first=True)
        except Exception as error:
            self._record(dict(event='allocation_failure', episode_id=self.episode['episode_id'],
                              error_type=type(error).__name__, error=str(error)))
            raise

    def step(self, action):
        if bool(action.get('reset', False)):
            return self._reset()
        require(not self.done, 'Reset required before stepping an ended episode')
        choice = action.get('action')
        require(np.ndim(choice) == 0 and np.issubdtype(np.asarray(choice).dtype, np.integer),
                'Scalar integer action required')
        choice = int(choice)
        require(0 <= choice < len(ACTIONS), 'Action outside declared space')
        name = ACTIONS[choice]
        # Stop is a task decision; Pass refreshes metadata without asking Unity for an unsupported Stop action.
        event = self.controller.step(action='Pass' if name == 'Stop' else name)
        # Collisions/blocked movements remain legitimate transitions and are retained.
        position, state = self._state(event)
        self.steps += 1
        self.poses.append(dict(position))
        value = self.lookup.value(position) if self.lookup else None
        result = self.rewarder.step(name, state['eligible_visible_distance'], state['distance'],
                                  value, self.steps, self.max_steps)
        self.episode_reward += result['reward']
        self.done = result['is_last']
        obs = self._observation(event, result=result)
        self._record(dict(event='transition', episode_id=self.episode['episode_id'], step=self.steps,
                          action=name, engine_action='Pass' if name == 'Stop' else name,
                          action_success=bool(event.metadata.get('lastActionSuccess')),
                          action_error=event.metadata.get('errorMessage', ''), pose=position,
                          requested_goal_state=state, field_value=value, reward=result,
                          executed_path_m=path_length_metres(self.poses),
                          spl=float(obs['log/spl']) if obs['log/spl_valid'] else None,
                          rgb_sha256=hashlib.sha256(np.asarray(event.frame).tobytes()).hexdigest()))
        return obs
    def close(self):
        if not self.closed:
            self.controller.stop()
            self.closed = True
