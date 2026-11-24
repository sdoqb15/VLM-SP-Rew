import sys,unittest,tempfile,json,hashlib,base64,types,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
import vlm_sp_rew.reward as R
import vlm_sp_rew.annotation_usage as A

def config():return dict(step_penalty=-1.,success_reward=100.,distance_coefficient=1.,horizon_penalty=0.,success_distance_m=1.,lambda_shaping=1.,tau=3.,w_hi=1.,w_lo=.5,r_up=1.,r_down=-1.)
class RewardTests(unittest.TestCase):
 def test_zero_coefficient_matches_environment_control(self):
  cfg=config();cfg['lambda_shaping']=0;models=[R.SharedReward(a,cfg)for a in R.ARMS]
  for m in models:m.reset(3,1)
  outputs=[m.step('MoveAhead',None,2,5,1,200)for m in models];self.assertEqual(len(set(x['reward']for x in outputs)),1)
 def test_first_transition_has_declared_field_difference(self):
  r=R.SharedReward('vlm',config());r.reset(2,1);x=r.step('MoveAhead',None,1.5,3,1,200);self.assertEqual(x['field_reward'],1);self.assertAlmostEqual(x['reward'],.5)
 def test_success_requires_visible_stop(self):
  r=R.SharedReward('env',config());r.reset(2);x=r.step('Stop',None,.5,None,1,200);self.assertFalse(x['success']);self.assertTrue(x['is_terminal'])
 def test_visible_target_does_not_suppress_paper_distance_reward(self):
  r=R.SharedReward('env',config());r.reset(2);x=r.step('Stop',.5,.5,None,1,200);self.assertTrue(x['success']);self.assertEqual(x['shared_distance_reward'],1.5);self.assertAlmostEqual(x['reward'],100.5)
 def test_timeout_bootstraps(self):
  r=R.SharedReward('env',config());r.reset(2);x=r.step('MoveAhead',None,2,None,200,200);self.assertTrue(x['truncated']);self.assertFalse(x['is_terminal']);self.assertAlmostEqual(x['reward'],-1.)
 def test_negative_distance_invalid(self):
  with self.assertRaises(ValueError):R.SharedReward('env',config()).reset(-1)
 def test_nonfinite_reward_config_invalid(self):
  cfg=config();cfg['tau']=float('nan')
  with self.assertRaises(ValueError):R.SharedReward('vlm',cfg)
 def test_goal_is_one_hot_not_target_coordinate(self):self.assertEqual(R.goal_one_hot('Mug',['HousePlant','Mug']),[0.,1.])
 def test_requested_category_any_visible_instance(self):
  objects=[dict(objectType='Mug',objectId='m1',visible=False,position=dict(x=.1,z=0)),dict(objectType='Mug',objectId='m2',visible=True,position=dict(x=.8,z=0)),dict(objectType='HousePlant',objectId='p',visible=True,position=dict(x=0,z=0))];x=R.requested_goal_state(objects,'Mug',dict(x=0,z=0));self.assertEqual(x['distance'],.1);self.assertEqual(x['eligible_visible_distance'],.8);self.assertEqual(x['goal_instance_ids'],['m1','m2'])
 def test_never_switch_missing_category(self):
  with self.assertRaises(ValueError):R.requested_goal_state([],'Mug',dict(x=0,z=0))
 def test_executed_path_includes_initial_translation(self):self.assertEqual(R.path_length_metres([dict(x=0,z=0),dict(x=3,z=4),dict(x=3,z=4)]),5)
 def test_heuristic_is_not_spl(self):
  with self.assertRaises(ValueError):R.qualified_spl(True,dict(units='steps',shortest_path_m=4,executed_path_m=5))
 def test_spl_requires_qualified_reference(self):self.assertEqual(R.qualified_spl(True,dict(units='metres',success_rule_matched=True,reference_kind='qualified_navigable_shortest_path',shortest_path_m=2,executed_path_m=4)),.5)

class BankTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);house=dict(rooms=[dict(floorPolygon=[dict(x=0,z=0),dict(x=2,z=0),dict(x=2,z=2)])]);raw=json.dumps(house).encode();(self.root/'scene.json').write_bytes(raw);self.case=dict(episode_id='e1',split='train',scene=dict(path='scene.json',sha256=R.sha(raw)),scene_identity_sha256=R.floor_identity(house),goal_category='Mug',goal_instance_ids=['Mug|1'],seed=1,start_position=dict(x=0,y=0,z=0),start_yaw_degrees=0);self.bank=dict(schema_version=1,goal_vocabulary=['Mug'],episodes=[self.case])
 def tearDown(self):self.tmp.cleanup()
 def save(self):
  p=self.root/'bank.json';p.write_text(json.dumps(self.bank));return p
 def test_shared_schedule_deterministic(self):
  q=R.load_episode_bank(self.save());a=R.EpisodeSchedule(q,'train',41,True);b=R.EpisodeSchedule(q,'train',41,True);self.assertEqual([a.next()for _ in range(3)],[b.next()for _ in range(3)])
 def test_same_geometry_cross_split_rejected(self):
  x=copy.deepcopy(self.case);x.update(episode_id='e2',split='test');self.bank['episodes'].append(x)
  with self.assertRaises(ValueError):R.load_episode_bank(self.save())
 def test_scene_corruption_rejected(self):
  p=self.save();(self.root/'scene.json').write_text('{}')
  with self.assertRaises(ValueError):R.load_episode_bank(p)
 def test_fake_geometry_identity_rejected(self):
  self.case['scene_identity_sha256']='a'*64
  with self.assertRaises(ValueError):R.load_episode_bank(self.save())
 def test_cross_goal_field_rejected(self):
  f=dict(scene_sha256=self.case['scene']['sha256'],goal_category='HousePlant',source_arm='vlm',frozen_before_training=True);raw=json.dumps(f).encode();(self.root/'field.json').write_bytes(raw);self.case['fields']={'vlm':dict(path='field.json',sha256=R.sha(raw))}
  with self.assertRaises(ValueError):R.bind_field(self.case,'vlm',self.root)
 def test_env_does_not_require_field_or_consume_vlm(self):self.assertIsNone(R.bind_field(self.case,'env',self.root))

class AnnotationTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);A.configure();A.CLIENT=None
 def tearDown(self):self.tmp.cleanup();A.configure();A.CLIENT=None
 def test_import_and_default_call_offline(self):
  with self.assertRaises(RuntimeError):A.request_annotation(self.root,'objects',[],300,.3)
  self.assertEqual(A.OPTIONS['calls'],0)
 def test_family_snapshot_refused(self):
  with self.assertRaises(ValueError):A.configure(True,'gpt-4o',1)
 def test_explicit_call_budget_required(self):
  with self.assertRaises(ValueError):A.configure(True,'synthetic-fixture-2000-01-01',0)
 def test_record_raw_provider_and_hashes_without_real_calls(self):
  class Response:
   choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='SYNTHETIC TEST OUTPUT'))]
   def model_dump(self,**kwargs):return dict(model='synthetic-fixture-2000-01-01',id='synthetic-no-network',usage=dict(prompt_tokens=0,completion_tokens=0))
  calls=[]
  def create(**kwargs):calls.append(kwargs);return Response()
  A.CLIENT=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)));A.configure(True,'synthetic-fixture-2000-01-01',1);image=b'synthetic-image-bytes-not-research';content=[dict(type='text',text='SYNTHETIC TEST PROMPT'),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(image).decode(),detail='high'))];A.request_annotation(self.root,'objects',content,300,.3);folders=list((self.root/'api_audit').iterdir());self.assertEqual(len(folders),1);p=folders[0];req=json.loads((p/'REQUEST.json').read_text());self.assertEqual(req['image_payload_hashes'][0]['sha256'],hashlib.sha256(image).hexdigest());self.assertTrue((p/'PROVIDER_RESPONSE.json').is_file());self.assertTrue((p/'COMPLETE.json').is_file())
  with self.assertRaises(RuntimeError):A.request_annotation(self.root,'objects',content,300,.3)
  self.assertEqual(len(calls),1)
if __name__=='__main__':unittest.main()
