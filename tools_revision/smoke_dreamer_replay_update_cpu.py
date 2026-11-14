"""Bounded synthetic CPU integration: real Dreamer + real Driver/replay + fake simulator."""
from pathlib import Path
import hashlib,json,os,uuid
import elements,numpy as np
import embodied
from train_utils import load_dreamer_configs,make_agent,make_replay,make_stream,setup_gpu_env
from revised_env import RevisedProcThorEnv
from tests_revision.test_environment_integration import AdapterTests

def main():
    setup_gpu_env(platform='cpu')
    root=Path(os.environ['TMPDIR'])/('dreamer-replay-update-'+uuid.uuid4().hex)
    root.mkdir(exist_ok=False)
    fixture=AdapterTests();fixture.setUp()
    driver=None;replay=None
    try:
        config=elements.Config(load_dreamer_configs()).update({
            '.*\\.rssm':{'deter':512,'hidden':64,'classes':4},
            '.*\\.depth':4,'.*\\.units':64})
        config=elements.Config({**dict(config),'episode_manifest':str(fixture.manifest),
                               'data_root':str(fixture.root)})
        config=config.update(dict(logdir=str(root/'logs'),seed=7,batch_size=1,batch_length=4,
                                 report_length=4,replay_context=1))
        config=config.update({'jax.platform':'cpu','jax.compute_dtype':'float32',
            'jax.prealloc':False,'jax.debug':False,'jax.jit':True,
            'replay.size':1024,'replay.chunksize':64})
        agent=make_agent(config,RevisedProcThorEnv)()
        print(json.dumps(dict(stage='agent_initialized')),flush=True)
        replay=make_replay(config)()
        env=fixture.env()
        driver=embodied.Driver([lambda:env],parallel=False)
        driver.on_step(replay.add)
        driver.reset(agent.init_policy)
        driver(lambda *args:agent.policy(*args,mode='train'),steps=16)
        print(json.dumps(dict(stage='driver_replay_collected',replay_sequences=len(replay))),flush=True)
        stream=iter(agent.stream(make_stream(config,replay,'train')))
        def digest_params():
            return {key:hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                    for key,value in agent.params.items()}
        before=digest_params()
        train_carry=agent.init_train(1)
        update_records=[]
        for _ in range(3):
            batch=next(stream)
            train_carry,outs,metrics=agent.train(train_carry,batch)
            if 'replay'in outs:replay.update(outs['replay'])
            update_records.append(dict(n_updates=int(agent.n_updates),returned_metrics=len(metrics),
                                       replay_updates_returned='replay'in outs))
        after=digest_params()
        changed=[key for key in before if before[key]!=after[key]]
        core_changed=[key for key in changed if key.startswith(('enc/','dyn/','pol/','rew/','con/','val/'))]
        assert int(agent.n_updates)==3
        assert core_changed,'Need real changed model weights, not only optimizer counters'
        assert all(np.isfinite(np.asarray(value)).all()for value in agent.params.values())
        report=dict(status='PASS',scope='SYNTHETIC ENGINEERING ONLY: actual Dreamer tiny upstream size1m config, actual Driver, actual replay/streams, 3 actual CPU gradient updates using fake-controller RGB/rewards. Not an ObjectNav efficacy experiment.',
                    real_simulator_calls=0,real_navigation_steps=0,paid_API_calls=0,GPU_used=False,
                    synthetic_driver_observations=16,synthetic_gradient_updates=3,
                    changed_core_arrays=len(core_changed),changed_core_parameter_names=core_changed,
                    update_records=update_records,all_parameter_arrays_finite=True)
        (root/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        if driver:driver.close()
        if replay and hasattr(replay,'close'):replay.close()
        fixture.tearDown()
if __name__=='__main__':main()
