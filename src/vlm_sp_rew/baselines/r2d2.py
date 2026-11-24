"""Run the author Acme R2D2 implementation on the fixed ObjectNav contract."""
from pathlib import Path
import argparse,csv,importlib,json,os,pickle,shutil,subprocess,sys,time
import numpy as np

ACME_REVISION='1bc05bea97b94a2da7e5315b315118bdaa98f1f3'


def normalized_rgb(image):
    image=np.asarray(image)
    if image.shape!=(300,300,3)or image.dtype!=np.uint8:raise ValueError('Native300RGB uint8 required')
    return image.astype(np.float32)/np.float32(255)


def task_factory(options,split,seed):
    """Keep RGB and prior action at the policy boundary; never expose distance reward."""
    import dm_env
    from dm_env import specs
    from acme.wrappers.observation_action_reward import OAR
    from vlm_sp_rew.environment import ObjectNavEnv
    from vlm_sp_rew.baselines.common import ArraySpec
    class Task(dm_env.Environment):
        def __init__(self):
            self.env=ObjectNavEnv(options.episode_manifest,options.data_root,split,seed,arm='env',
                shared_reward_config=options.reward_config,controller_config=options.controller_config,
                run_record_root=options.run_record_root,max_steps=200,shuffle=split=='train',space_type=ArraySpec)
            self.actions=0;self.last_observation=None
        def observation_spec(self):
            return OAR(specs.BoundedArray((300,300,3),np.float32,0.,1.),specs.DiscreteArray(len(self.env.actions),np.int32),specs.Array((),np.float32))
        def action_spec(self):return specs.DiscreteArray(len(self.env.actions),np.int32)
        def reward_spec(self):return specs.Array((),np.float32)
        def discount_spec(self):return specs.BoundedArray((),np.float32,0.,1.)
        def reset(self):
            self.last_observation=self.env.reset()
            return dm_env.restart(OAR(normalized_rgb(self.last_observation['image']),np.int32(0),np.float32(0)))
        def step(self,action):
            self.last_observation=self.env.step(dict(action=np.int32(action)));self.actions+=1
            obs=OAR(normalized_rgb(self.last_observation['image']),np.int32(action),np.float32(0))
            reward=np.float32(self.last_observation['reward'])
            last=bool(self.last_observation['is_last']);terminal=bool(self.last_observation['is_terminal'])
            return dm_env.TimeStep(dm_env.StepType.LAST if last else dm_env.StepType.MID,reward,np.float32(0. if terminal else 1.),obs)
        def close(self):self.env.close()
    return Task()


def native_configuration():
    from acme.agents.jax import r2d2
    # Author discrete-control example, retaining its recurrent Q learner and replay.
    return r2d2.R2D2Config(burn_in_length=8,trace_length=40,sequence_period=20,
        min_replay_size=10000,batch_size=32,prefetch_size=1,samples_per_insert=1.,
        evaluation_epsilon=1e-3,learning_rate=1e-4,target_update_period=1200,variable_update_period=100)


def write_checkpoint(root,step,learner,contract):
    import jax
    target=Path(root)/('step-%09d.pkl'%step)
    if target.exists():return
    target.parent.mkdir(parents=True,exist_ok=True)
    state=jax.device_get(learner.save())
    incoming=sum(getattr(x,'nbytes',0) for x in jax.tree_util.tree_leaves(state))*2+1024**2
    if shutil.disk_usage(target.parent).free-incoming<110*1024**3:raise RuntimeError('Checkpoint would exhaust the operational storage margin')
    temp=target.with_suffix('.tmp')
    with temp.open('xb') as f:pickle.dump(dict(environment_actions=step,learner=state,acme_revision=ACME_REVISION,task_contract=contract),f,pickle.HIGHEST_PROTOCOL)
    temp.replace(target)


def train(options):
    import jax
    from acme.agents.jax import r2d2
    from acme.jax import experiments
    from vlm_sp_rew.protocol import validate_allocation
    validate_allocation(options.episode_manifest,options.data_root,shared=False)
    from vlm_sp_rew.baselines.common import task_contract
    tasks=[];holder={};loggers=[]
    def logger_factory(label,steps_key=None,task_instance=0):
        from acme.utils.loggers.csv import CSVLogger
        result=CSVLogger(directory_or_file=str(Path(options.logdir)/'metrics'),label=label,add_uid=False)
        loggers.append(result);return result
    def factory(seed):
        task=task_factory(options,'train',seed);tasks.append(task);return task
    class Builder(r2d2.R2D2Builder):
        def make_learner(self,*args,**kw):
            result=super().make_learner(*args,**kw);holder['learner']=result
            write_checkpoint(options.logdir,0,result,task_contract(tasks[0].env))
            return result
    run_module=importlib.import_module('acme.jax.experiments.run_experiment')
    original=run_module._LearningActor
    class TrainingBudgetReached(Exception):pass
    class SavingActor(original):
        def update(self):
            super().update();step=tasks[0].actions
            if step and(step%25000==0 or step==options.steps):write_checkpoint(options.logdir,step,holder['learner'],task_contract(tasks[0].env))
            if step==options.steps:raise TrainingBudgetReached
    run_module._LearningActor=SavingActor
    config=experiments.ExperimentConfig(builder=Builder(native_configuration()),network_factory=r2d2.make_atari_networks,
        environment_factory=factory,seed=options.seed,max_num_actor_steps=options.steps,checkpointing=None,logger_factory=logger_factory)
    try:experiments.run_experiment(config,num_eval_episodes=0)
    except TrainingBudgetReached:
        assert tasks[0].actions==options.steps
    finally:
        run_module._LearningActor=original
        for task in tasks:task.close()
        for logger in loggers:logger.close()


def evaluate(options):
    import jax
    from acme import specs
    from acme.agents.jax import r2d2
    # Load only checkpoints produced by this program, from a trusted local source.
    with Path(options.checkpoint).open('rb') as f:saved=pickle.load(f)
    if saved['acme_revision']!=ACME_REVISION:raise ValueError('R2D2 source/checkpoint revision mismatch')
    task=task_factory(options,options.split,options.seed)
    from vlm_sp_rew.baselines.common import check_task_contract
    check_task_contract(saved['task_contract'],task.env)
    env_spec=specs.make_environment_spec(task);networks=r2d2.make_atari_networks(env_spec)
    builder=r2d2.R2D2Builder(native_configuration());policy=builder.make_policy(networks,env_spec,evaluation=True)
    class Variables:
        def get_variables(self,names):return [saved['learner'].params for _ in names]
    actor=builder.make_actor(jax.random.PRNGKey(options.seed),policy,env_spec,variable_source=Variables())
    count=len(task.env.schedule.items)
    from vlm_sp_rew.export_metrics import FIELDS
    out=Path(options.output);out.parent.mkdir(parents=True,exist_ok=True)
    try:
        with out.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for _ in range(count):
                timestep=task.reset();actor.observe_first(timestep)
                while not timestep.last():
                    action=actor.select_action(timestep.observation);timestep=task.step(action);actor.observe(action,timestep)
                obs=task.last_observation
                if not obs['log/spl_valid']:raise ValueError('Qualified native SPL reference required')
                e=task.env.episode
                writer.writerow(dict(record_kind='measured',episode_id='R2D2:%d:%d:%s'%(options.seed,saved['environment_actions'],e['episode_id']),
                    benchmark=task.env.benchmark,group='R2D2',seed=options.seed,pair_id=e['pair_id'],checkpoint_steps=saved['environment_actions'],success=int(obs['log/success']),spl=float(obs['log/spl'])))
                f.flush()
    finally:task.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    for key in ('source-root','episode-manifest','controller-config','run-record-root'):p.add_argument('--'+key,required=True)
    p.add_argument('--data-root');p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int)
    p.add_argument('--logdir');p.add_argument('--checkpoint');p.add_argument('--output');p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--reward-config',default=str(Path(__file__).parents[1]/'configs/shared_reward.json'))
    a=p.parse_args();source=Path(a.source_root).resolve()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=ACME_REVISION:p.error('Use the pinned author Acme revision '+ACME_REVISION)
    if a.mode=='train' and(not a.logdir or not a.steps or a.steps<=0):p.error('Set --logdir and a positive --steps')
    if a.mode=='evaluate' and(not a.checkpoint or not a.output):p.error('Set --checkpoint and --output')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('Set CUDA_VISIBLE_DEVICES to the allocated GPU')
    from vlm_sp_rew.baselines.acme_local import prepare_local_source
    local_source=prepare_local_source(source,Path(a.run_record_root)/'_author_sources',ACME_REVISION)
    sys.path.insert(0,str(local_source));module=importlib.import_module('acme')
    if not Path(module.__file__).resolve().is_relative_to(local_source):p.error('Acme import did not resolve to the recorded local source')
    train(a) if a.mode=='train' else evaluate(a)


if __name__=='__main__':main()
