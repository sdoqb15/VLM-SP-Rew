"""Bounded CPU engineering check of the recovered Dreamer implementation, not an experiment."""
from pathlib import Path
import json, hashlib, os, tempfile, uuid
import elements
import numpy as np
from vlm_sp_rew.training_utils import load_dreamer_configs, make_agent, setup_gpu_env
from vlm_sp_rew.environment import ObjectNavEnv
from vlm_sp_rew.reward import floor_identity, sha

def main():
    setup_gpu_env(platform='cpu')
    root=Path(os.environ['TMPDIR'])/('dreamer-policy-smoke-'+uuid.uuid4().hex)
    root.mkdir(exist_ok=False)
    scene={'rooms':[{'floorPolygon':[{'x':0,'z':0},{'x':0,'z':2},{'x':2,'z':2},{'x':2,'z':0}]}]}
    raw=json.dumps(scene).encode();(root/'scene.json').write_bytes(raw)
    bank=dict(schema_version=1,synthetic=True,goal_vocabulary=['HousePlant','Television'],episodes=[
        dict(episode_id='SYNTHETIC_MODEL_INTERFACE_ONLY',split='train',
             scene={'path':'scene.json','sha256':sha(raw)},scene_identity_sha256=floor_identity(scene),
             goal_category='HousePlant',goal_instance_ids=['ENGINEERING_ONLY'],seed=7,
             start_position={'x':0,'y':.9,'z':0},start_yaw_degrees=0)])
    manifest=root/'bank.json';manifest.write_text(json.dumps(bank))
    config=elements.Config(load_dreamer_configs())
    # Existing upstream size1m architecture is used only to bound this CPU interface test.
    config=config.update({
        '.*\\.rssm':{'deter':512,'hidden':64,'classes':4},
        '.*\\.depth':4,'.*\\.units':64,
    })
    config=elements.Config({**dict(config), 'episode_manifest':str(manifest),'data_root':str(root)})
    config=config.update(dict(logdir=str(root/'logs'),seed=7,batch_size=1,batch_length=4,
                             report_length=4, replay_context=1))
    config=config.update({'jax.platform':'cpu','jax.compute_dtype':'float32',
        'jax.prealloc':False,'jax.debug':False,'jax.jit':True})
    agent=make_agent(config,ObjectNavEnv)()
    print(json.dumps(dict(stage='agent_initialized',parameters=len(agent.params))),flush=True)
    obs_space,act_space=ObjectNavEnv.spaces_from_manifest(manifest)
    obs={k:np.zeros((1,)+space.shape,space.dtype)for k,space in obs_space.items()}
    obs['is_first'][:]=True
    carry=agent.init_policy(1)
    carry,action,outs=agent.policy(carry,obs,mode='eval')
    assert np.asarray(action['action']).shape==(1,)
    assert 0<=int(action['action'][0])<6
    obs['is_first'][:]=False
    carry,action2,outs2=agent.policy(carry,obs,mode='eval')
    import jax
    report=dict(status='PASS',scope='Dreamer CPU initialization and two policy calls with 300x300 RGB and no goal channel; bounded size1m engineering check, no simulator or RL optimization',
                devices=[str(x)for x in jax.devices()],observations=list(obs_space),
                actions=[int(action['action'][0]),int(action2['action'][0])],
                parameter_arrays=len(agent.params),RL_environment_steps=0,
                optimizer_updates=0,paid_API_calls=0,simulator_calls=0)
    print(json.dumps(report),flush=True)
    (root/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
