"""Bounded CPU integration check of action counts and retained checkpoints.

Uses a generated controller fixture; does not launch Unity, a VLM, or real RL tasks.
"""
from pathlib import Path
import copy,json,os,sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.test_environment_integration import AdapterTests
from vlm_sp_rew import training_utils as t


def main():
    fixture=AdapterTests();fixture.setUp()
    try:
        root=fixture.root/'check';root.mkdir()
        args=t.build_arg_parser().parse_args(['--episode-manifest',str(fixture.manifest),
            '--shared-reward-config',str(fixture.reward_path),'--controller-config','engineering-fixture.json',
            '--run-record-root',str(root/'records'),'--logdir',str(root/'learner'),
            '--jax-platform','cpu','--steps','8','--checkpoint-every','4'])
        t.setup_gpu_env(platform='cpu')
        config=copy.deepcopy(t.load_dreamer_configs())
        config['agent']['dyn']['rssm'].update(deter=512,hidden=64,classes=4)
        config['agent']['enc']['simple']['depth']=4
        config['agent']['dec']['simple']['depth']=4
        for k in ('enc','dec'):
            config['agent'][k]['simple']['units']=64
        for k in ('rewhead','conhead','policy','value'):config['agent'][k]['units']=64
        class TestEnvironment:
            def __new__(cls,**kw):return fixture.env(max_steps=3,shuffle=False)
        from vlm_sp_rew.environment import ObjectNavEnv
        TestEnvironment.spaces_from_manifest=staticmethod(ObjectNavEnv.spaces_from_manifest)
        with patch.object(t,'load_dreamer_configs',return_value=config):
            t.run_training(args,TestEnvironment,t.per_episode_callback)
        index=json.loads((root/'learner/checkpoints.json').read_text())
        assert sorted(map(int,index))==[0,4,8],index
        for n,path in index.items():assert (Path(path)/'done').exists(),(n,path)
        records=[json.loads(line) for env in fixture.envs for path in env.run_root.glob('episode-*.jsonl') for line in path.read_text().splitlines()]
        executed=sum(x.get('event')=='transition' for x in records)
        resets=sum(x.get('event')=='reset' for x in records)
        assert executed==8 and resets>=2,(executed,resets)
        print(json.dumps(dict(check='checkpoint_and_action_counter',status='PASS',real_environment_steps=0,
            engineering_controller_actions=executed,reset_observations_excluded=resets,checkpoint_steps=list(map(int,index)))))
    finally:fixture.tearDown()


if __name__=='__main__':main()
