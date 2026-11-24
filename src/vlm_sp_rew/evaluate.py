"""Evaluate frozen checkpoints on native frozen dev/test episodes."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from vlm_sp_rew.reward import load_episode_bank, require


def evaluate(agent, env, count, out, metadata):
    """A fresh recurrent state per episode; no learning and no field in the policy."""
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    columns=['record_kind','episode_id','benchmark','group','seed','pair_id','checkpoint_steps','success','spl']
    with out.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        try:
            for _ in range(count):
                obs=env.reset();carry=agent.init_policy(1)
                while not obs['is_last']:
                    actor={k:np.expand_dims(v,0) for k,v in obs.items() if not k.startswith('log/')}
                    carry,actions,_=agent.policy(carry,actor,mode='eval')
                    obs=env.step({'action':np.asarray(actions['action'])[0]})
                require(obs['log/spl_valid'], 'No SR/SPL export without the frozen navigable reference')
                writer.writerow(dict(metadata,record_kind='measured',episode_id='%s:%s:%s:%s'%(metadata['group'],metadata['seed'],metadata['checkpoint_steps'],env.episode['episode_id']),
                    pair_id=env.episode['pair_id'],success=int(obs['log/success']),spl=float(obs['log/spl'])))
                f.flush()
        finally:
            env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','checkpoint','episode-manifest','controller-config','run-record-root','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--data-root');p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--checkpoint-steps',type=int,required=True);p.add_argument('--group',required=True)
    p.add_argument('--jax-platform',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    from vlm_sp_rew.training_utils import setup_gpu_env, make_agent
    setup_gpu_env(platform=a.jax_platform)
    import elements
    from vlm_sp_rew.environment import ObjectNavEnv
    config=elements.Config.load(a.config).update(dict(episode_manifest=a.episode_manifest,data_root=a.data_root,
        logdir=str(Path(a.output).parent/'agent'),jax=dict(platform=a.jax_platform,compute_dtype='float32',
            policy_devices=[0],train_devices=[0],prealloc=False,jit=True,debug=False,precompile=False,profiler=False)))
    agent=make_agent(config,ObjectNavEnv)()
    elements.checkpoint.load(a.checkpoint,dict(agent=agent.load))
    q=load_episode_bank(a.episode_manifest,a.data_root)
    episodes=[e for e in q['bank']['episodes'] if e['split']==a.split]
    require(episodes,'No frozen evaluation episodes')
    # Evaluation uses only the deployed RGB interface; field reward is irrelevant to action selection.
    env=ObjectNavEnv(a.episode_manifest,a.data_root,a.split,config.seed,arm='env',
                    shared_reward_config=config.shared_reward_config,controller_config=a.controller_config,
                    run_record_root=a.run_record_root,max_steps=200,shuffle=False)
    evaluate(agent,env,len(episodes),a.output,dict(benchmark=q['bank']['benchmark'],group=a.group,
        seed=config.seed,checkpoint_steps=a.checkpoint_steps))


if __name__=='__main__':main()
