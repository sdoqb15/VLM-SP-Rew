"""Integration tests use clearly synthetic houses and a fake controller, never scientific outcomes."""
from pathlib import Path
import copy, hashlib, json, os, tempfile, unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from revision_protocol import floor_identity, sha, load_episode_bank
from revised_env import RevisedProcThorEnv, FieldLookup, ACTIONS
from train_utils import make_env, make_agent, build_arg_parser, per_episode_callback
import elements

class FakeController:
    def __init__(self, spec, run_root):
        self.calls = []
        self.position = {'x': 0., 'y': .9, 'z': 0.}
        self.yaw = 0.
        self.closed = False
        self.fail_moves = False
        self.objects = [
            {'objectId': 'near-other-category', 'objectType': 'Television',
             'position': {'x': 0., 'y': .9, 'z': 0.}, 'visible': True},
            {'objectId': 'plant-1', 'objectType': 'HousePlant',
             'position': {'x': 2., 'y': .9, 'z': 0.}, 'visible': True},
        ]
    def event(self, success=True):
        return SimpleNamespace(frame=np.zeros((300, 300, 3), np.uint8), metadata={
            'lastActionSuccess': success, 'errorMessage': '' if success else 'blocked',
            'agent': {'position': dict(self.position), 'rotation': {'y': self.yaw}},
            'objects': copy.deepcopy(self.objects)})
    def reset(self, **kwargs):
        self.calls.append(('reset', kwargs))
        return self.event()
    def step(self, **kwargs):
        self.calls.append(('step', kwargs))
        if kwargs['action'] == 'Teleport':
            self.position = dict(kwargs['position'])
            self.yaw = kwargs['rotation']['y']
        elif kwargs['action'] == 'MoveAhead' and not self.fail_moves:
            self.position['x'] += .25
        return self.event(not(self.fail_moves and kwargs['action'] == 'MoveAhead'))
    def stop(self):
        self.closed = True

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        scene = {'rooms': [{'floorPolygon': [{'x': 0, 'z': 0}, {'x': 0, 'z': 3},
                  {'x': 3, 'z': 3}, {'x': 3, 'z': 0}]}]}
        raw = json.dumps(scene).encode()
        (self.root/'scene.json').write_bytes(raw)
        e = dict(episode_id='ENGINEERING_ONLY_plant', split='train',
                 scene=dict(path='scene.json', sha256=sha(raw)),
                 scene_identity_sha256=floor_identity(scene), goal_category='HousePlant',
                 goal_instance_ids=['plant-1'], seed=7,
                 start_position={'x': 0., 'y': .9, 'z': 0.}, start_yaw_degrees=0.,
                 fields={})
        for arm in ('geo', 'geo_vis', 'vlm'):
            field = dict(scene_sha256=sha(raw), goal_category='HousePlant', source_arm=arm,
                         frozen_before_training=True, synthetic=False,
                         source_integrity_qualified=True,
                         points=[{'position': {'x': 0, 'z': 0}, 'value': 0, 'room_label': 'a'},
                                 {'position': {'x': .25, 'z': 0}, 'value': 1, 'room_label': 'a'}])
            data=json.dumps(field).encode();(self.root/(arm+'.json')).write_bytes(data)
            e['fields'][arm]=dict(path=arm+'.json',sha256=sha(data))
        self.bank=dict(schema_version=1, synthetic=True, goal_vocabulary=['HousePlant','Television'],
                       episodes=[e])
        self.write_bank()
        self.reward=json.loads((Path(__file__).parents[1]/'shared_reward_revision_config.json').read_text())
        self.reward_path=self.root/'reward.json'
        self.reward_path.write_text(json.dumps(self.reward))
        self.envs=[]
    def write_bank(self):
        self.manifest=self.root/'bank.json'
        self.manifest.write_text(json.dumps(self.bank))
    def env(self,arm='env',**kwargs):
        options=dict(episode_manifest=self.manifest,shared_reward_config=self.reward_path,
                     run_record_root=self.root/'records',arm=arm,max_steps=3,
                     controller_factory=FakeController,allow_synthetic_fixture=True)
        options.update(kwargs)
        result=RevisedProcThorEnv(**options);self.envs.append(result);return result
    def tearDown(self):
        for env in self.envs:env.close()
        self.tmp.cleanup()
    def test_goal_is_requested_not_first_metadata_object(self):
        env=self.env();obs=env.reset()
        self.assertEqual(obs['goal'].tolist(),[1.,0.])
        stop=env.step({'action':np.int32(5)})
        self.assertEqual(float(stop['log/success']),0.)
        self.assertTrue(stop['is_terminal'])
        self.assertEqual(env.controller.calls[-1][1]['action'],'Pass')
    def test_success_uses_requested_category_and_valid_terminal(self):
        env=self.env();env.controller.objects[1]['position']['x']=.5
        env.reset();obs=env.step({'action':5})
        self.assertEqual(float(obs['log/success']),1.)
        self.assertAlmostEqual(float(obs['reward']),99.99,places=4)
        self.assertTrue(obs['is_last']);self.assertTrue(obs['is_terminal'])
        self.assertFalse(obs['log/spl_valid'])
    def test_four_arms_same_schedule_and_shared_reward_with_zero_coefficient(self):
        self.reward['lambda_shaping']=0
        self.reward_path.write_text(json.dumps(self.reward))
        outcomes=[];events=[]
        for arm in ('env','geo','geo_vis','vlm'):
            env=self.env(arm);reset=env.reset()
            outcomes.append([float(env.step({'action':0})['reward']),
                             float(env.step({'action':5})['reward'])])
            events.append([x for x in env.controller.calls if x[0]=='reset'][0][1])
        self.assertTrue(all(x==outcomes[0]for x in outcomes))
        self.assertTrue(all(x==events[0]for x in events))
        self.assertEqual(events[0]['randomSeed'],7)
    def test_reset_phi_applies_on_first_transition(self):
        env=self.env('vlm');env.reset();obs=env.step({'action':0})
        self.assertAlmostEqual(float(obs['reward']),.74,places=5)
        rows=[json.loads(x)for x in env.record_path.read_text().splitlines()]
        self.assertEqual(rows[0]['event'],'reset')
        self.assertTrue(rows[0]['synthetic_engineering_fixture'])
        self.assertEqual(rows[1]['reward']['field_reward'],.5)
    def test_timeout_truncates_not_terminal_and_blocks_further_steps(self):
        env=self.env(max_steps=1);env.reset();obs=env.step({'action':0})
        self.assertTrue(obs['is_last']);self.assertFalse(obs['is_terminal'])
        self.assertAlmostEqual(float(obs['reward']),-9.76,places=5)
        with self.assertRaises(ValueError):env.step({'action':0})
    def test_failed_movement_is_retained_transition(self):
        env=self.env();env.reset();env.controller.fail_moves=True
        obs=env.step({'action':0})
        self.assertFalse(obs['is_last'])
        row=json.loads(env.record_path.read_text().splitlines()[-1])
        self.assertFalse(row['action_success']);self.assertEqual(row['executed_path_m'],0)
    def test_goal_disappearance_fails_not_substituted(self):
        env=self.env();env.controller.objects.pop()
        with self.assertRaisesRegex(ValueError,'requested goal absent'):env.reset()
        row=json.loads(env.record_path.read_text().splitlines()[-1])
        self.assertEqual(row['event'],'allocation_failure')
    def test_field_scene_mismatch_fails_before_simulator_reset(self):
        spec=self.bank['episodes'][0]['fields']['vlm']
        p=self.root/spec['path'];field=json.loads(p.read_text());field['scene_sha256']='f'*64
        raw=json.dumps(field).encode();p.write_bytes(raw);spec['sha256']=sha(raw);self.write_bank()
        env=self.env('vlm')
        with self.assertRaisesRegex(ValueError,'different scene or goal'):env.reset()
        self.assertEqual(env.controller.calls,[])
    def test_actor_observation_and_log_separation_through_actual_driver(self):
        import embodied
        env=self.env()
        driver=embodied.Driver([lambda:env],parallel=False)
        seen=[];transitions=[]
        def policy(carry,obs,**kw):
            seen.append(set(obs));return carry,{'action':np.array([5],np.int32)},{}
        driver.on_step(lambda tran,worker:transitions.append(tran))
        try:driver(policy,steps=2)
        finally:driver.close()
        self.assertEqual(seen[0],{'image','goal','reward','is_first','is_last','is_terminal'})
        self.assertNotIn('pose',seen[0]);self.assertNotIn('field_value',seen[0])
        self.assertTrue(transitions[-1]['is_terminal'])
        self.assertIn('log/spl_valid',transitions[-1])
    def test_spaces_and_agent_factory_do_not_create_a_simulator(self):
        class FakeAgent:
            def __init__(self,obs_space,act_space,config):
                self.obs_space=obs_space;self.act_space=act_space
        config=elements.Config(dict(episode_manifest=str(self.manifest),data_root=None,agent={"engineering_stub": True},
            logdir=str(self.root/'agent'),seed=1,jax={'platform':'cpu'},batch_size=1,batch_length=2,
            replay_context=1,report_length=2,replica=0,replicas=1))
        with patch('dreamerv3.agent.Agent',FakeAgent),patch.object(RevisedProcThorEnv,'__init__',side_effect=AssertionError('No simulator for spaces')):
            agent=make_agent(config,RevisedProcThorEnv)()
        self.assertEqual(agent.obs_space['goal'].shape,(2,))
    def test_environment_factory_forwards_revision_and_worker_identity(self):
        class Capture:
            def __init__(self,**kwargs):self.kwargs=kwargs
        ctor=make_env(env_cls=Capture,revision_options=dict(arm='vlm',run_record_root='/declared/root'))
        self.assertEqual(ctor(3).kwargs,dict(arm='vlm',run_record_root='/declared/root/worker-3'))
    def test_revised_callback_does_not_report_heuristic_as_spl(self):
        class Logger:
            def __init__(self):self.values=[]
            def add(self,value,**kwargs):self.values.append(value)
        logger=Logger();cb=per_episode_callback([],logger)
        cb({'is_last':True,'log/success':0,'log/episode_reward':0,
            'log/episode_length':1,'log/spl_valid':False,'log/executed_path_m':.25},0)
        self.assertNotIn('spl',logger.values[-1])
        self.assertNotIn('legacy_heuristic_path_efficiency',logger.values[-1])
    def test_fixture_is_rejected_by_non_fixture_entry(self):
        with self.assertRaisesRegex(ValueError,'Synthetic engineering bank'):
            RevisedProcThorEnv(self.manifest)
    def test_explicit_cli_contract(self):
        args=build_arg_parser().parse_args(['--episode-manifest','bank.json','--arm','geo_vis','--jax-platform','cpu'])
        self.assertEqual(args.arm,'geo_vis');self.assertEqual(args.episode_manifest,'bank.json')

if __name__=='__main__':unittest.main()
