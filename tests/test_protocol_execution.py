"""Protocol-level regression tests for the manuscript's actual runtime contracts."""
import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from vlm_sp_rew.environment import spaces,actions_for_benchmark
from vlm_sp_rew.protocol import field_config,reward_config,CHECKPOINTS,SEEDS
from vlm_sp_rew.preprocess import select_points
from vlm_sp_rew.geometry import build
from vlm_sp_rew.ablate_field import derive
from vlm_sp_rew.evaluate import evaluate
from vlm_sp_rew.reward import SharedReward

class ProtocolTests(unittest.TestCase):
    def test_native_action_spaces_and_rgb_only_interface(self):
        for benchmark,n in [('RoboTHOR',6),('ProcTHOR',6),('R2R-adapted',4)]:
            obs,act=spaces(['HousePlant'],benchmark)
            self.assertEqual(obs['image'].shape,(300,300,3));self.assertNotIn('goal',obs)
            self.assertEqual(len(actions_for_benchmark(benchmark)),n)
            self.assertEqual(int(act['action'].high),n)
            self.assertEqual(actions_for_benchmark(benchmark)[-1],'Stop')
    def test_benchmark_profiles_and_shared_override(self):
        self.assertEqual(field_config('RoboTHOR')['vlm_sp_rew_params']['N'],30)
        self.assertEqual(reward_config('RoboTHOR')['w_lo'],.1)
        self.assertEqual(reward_config('R2R-adapted')['r_up'],.5)
        for b in ('RoboTHOR','ProcTHOR','R2R-adapted'):
            self.assertEqual(field_config(b,True),field_config('ProcTHOR'))
            self.assertEqual(reward_config(b,True),reward_config('ProcTHOR'))
            SharedReward('vlm',reward_config(b))
        self.assertEqual(CHECKPOINTS[0],0);self.assertEqual(CHECKPOINTS[-1],400000)
        self.assertEqual(len(CHECKPOINTS),17);self.assertEqual(len(SEEDS),5)
    def test_sampling_does_not_relax_spacing_or_required_count(self):
        points=[dict(x=i,y=.9,z=0) for i in range(6)]
        a=select_points(points,3,2,7)
        self.assertEqual(a,select_points(points,3,2,7))
        with self.assertRaisesRegex(ValueError,'Not enough'):select_points(points,5,2,7)
    def test_geometry_control_uses_native_edges_and_visibility(self):
        g=dict(scene_sha256='a'*64,goal_category='HousePlant',goal_instances=[dict(object_id='plant',position=dict(x=2,y=.9,z=0))],
            points=[dict(position=dict(x=0,y=.9,z=0),goal_visible=False),dict(position=dict(x=1,y=.9,z=0),goal_visible=True)],edges=[[0,1],[1,0]])
        f=build(g,field_config('ProcTHOR'))
        self.assertEqual([x['graph_hops'] for x in f['points']],[2,1])
        self.assertEqual([x['value'] for x in f['points']],[2,5])
        self.assertEqual(f['source_arm'],'geo')
    def test_signal_ablation_preserves_allocation(self):
        f=dict(points=[dict(position=dict(x=0,z=0),graph_hops=8,value=2)],provenance={})
        audit=dict(points=[dict(position=dict(x=0,z=0),parsed=dict(goal_visible=True))])
        v=derive(f,audit,field_config('ProcTHOR'),'visibility');d=derive(f,audit,field_config('ProcTHOR'),'distance')
        self.assertEqual(v['points'][0]['value'],5);self.assertEqual(d['points'][0]['value'],1)
        self.assertEqual(f['points'][0]['value'],2)
    def test_evaluation_resets_recurrence_and_never_sends_log_metadata_to_policy(self):
        class Agent:
            initializations=0
            def init_policy(self,n):self.initializations+=1;return 0
            def policy(self,carry,obs,mode):
                self_seen.append((mode,set(obs)));return carry+1,dict(action=np.array([3])),{}
        class Env:
            episode=dict(episode_id='native-start-1',pair_id='pair1');closed=False
            def reset(self):return dict(image=np.zeros((300,300,3),np.uint8),is_last=np.bool_(False),log_secret=0,**{'log/pose':0})
            def step(self,action):return dict(is_last=True,**{'log/spl_valid':True,'log/spl':.5,'log/success':1})
            def close(self):self.closed=True
        self_seen=[];agent=Agent();env=Env()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'episodes.csv';evaluate(agent,env,1,p,dict(benchmark='R2R-adapted',group='Env',seed=288681,checkpoint_steps=0))
            rows=list(csv.DictReader(p.open()));self.assertEqual(rows[0]['record_kind'],'measured')
            self.assertIn('Env:288681:0:',rows[0]['episode_id'])
        self.assertEqual(agent.initializations,1);self.assertTrue(env.closed);self.assertNotIn('log/pose',self_seen[0][1])

if __name__=='__main__':unittest.main()
