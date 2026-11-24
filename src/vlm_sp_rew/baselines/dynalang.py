"""Author Dynalang runner adapted to the fixed-goal RGB ObjectNav interface."""
from pathlib import Path
import argparse,csv,importlib,json,math,os,subprocess,sys
import numpy as np

DYNALANG_REVISION='5e3b93d4cdf3318537a97faa64a01219b82912a8'


def load_author(source):
    source=Path(source).resolve()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=DYNALANG_REVISION:raise ValueError('Use author Dynalang revision '+DYNALANG_REVISION)
    if 'embodied' in sys.modules:raise RuntimeError('Run the Dynalang entry in its separate author dependency environment')
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'dynalang'))
    native=importlib.import_module('embodied')
    if not Path(native.__file__).resolve().is_relative_to(source):raise ValueError('Dynalang must import its bundled embodied package')
    return native,importlib.import_module('dynalang.train'),importlib.import_module('dynalang.agent'),importlib.import_module('dynalang.nets')


def decoder_300(nets):
    """Keep the native six stride stages, reconstruct320 then crop to300."""
    original=nets.ImageDecoderResnet
    class Decoder(original):
        def __init__(self,shape,*args,**kw):
            self.output_shape=tuple(shape)
            if tuple(shape[:2])==(300,300):
                # Six strides:5*2**6=320. No resizing of the actor's300RGB.
                shape=(320,320,shape[-1]);kw['minres']=5
            super().__init__(shape,*args,**kw)
        def __call__(self,x):
            result=super().__call__(x)
            if self.output_shape[:2]==(300,300):result=result[:,10:310,10:310,:]
            return result
    nets.ImageDecoderResnet=Decoder
    return original


def make_task(native,runner,config,options,split,holder):
    from vlm_sp_rew.environment import ObjectNavEnv
    env=ObjectNavEnv(options.episode_manifest,options.data_root,split,options.seed,arm='env',
        shared_reward_config=options.reward_config,controller_config=options.controller_config,
        run_record_root=options.run_record_root,max_steps=200,shuffle=split=='train',space_type=native.Space)
    class Adapter(native.Env):
        def __init__(self):self.env=env;self.actions=0
        @property
        def obs_space(self):
            result={k:native.Space(v.dtype,v.shape,v.low,v.high)for k,v in env.obs_space.items()}
            result['is_read_step']=native.Space(bool)
            return result
        @property
        def act_space(self):
            result={k:native.Space(v.dtype,v.shape,v.low,v.high)for k,v in env.act_space.items()}
            result['reset']=native.Space(bool)
            return result
        def step(self,action):
            resetting=bool(action.get('reset',False));obs=env.step(action)
            if not resetting:self.actions+=1
            # Native logs remain in the common task's evidence, not encoder inputs.
            return dict({k:obs[k]for k in env.obs_space},is_read_step=np.bool_(False))
        def close(self):env.close()
    base=Adapter();holder['task']=base;holder['config']=config
    return runner.wrap_env(base,config)


def train(options,native,runner,agt):
    holder={};original_agent=agt.Agent;original_driver=native.Driver;original_make=runner.make_env
    class BudgetReached(Exception):pass
    class CaptureAgent:
        configs=original_agent.configs
        def __new__(cls,*args,**kw):
            result=original_agent(*args,**kw);holder['agent']=result;save_actions(0);return result
    def save_actions(step):
        cfg=holder['config'];root=Path(str(cfg.logdir))/'action_checkpoints';root.mkdir(parents=True,exist_ok=True)
        path=root/('step-%09d.ckpt'%step)
        if path.exists():return
        cp=native.Checkpoint(str(path),parallel=False);cp.agent=holder['agent'];cp.save()
        from vlm_sp_rew.baselines.common import task_contract
        path.with_suffix('.json').write_text(json.dumps(dict(environment_actions=step,source_revision=DYNALANG_REVISION,seed=options.seed,task_contract=task_contract(holder['task'].env)))+'\n')
    class Driver(original_driver):
        def __call__(self,policy,steps=0,episodes=0):
            observations=0;ended=0
            while observations<steps or ended<episodes:
                # Complete all upstream callbacks, including learner updates, before saving.
                super().__call__(policy,steps=1);observations+=1
                task=holder['task'];ended+=int(bool(task.env.done))
                n=task.actions
                if n and(n%25000==0 or n==options.steps):save_actions(n)
                if n==options.steps:raise BudgetReached
    def factory(config,**overrides):return make_task(native,runner,config,options,'train',holder)
    agt.Agent=CaptureAgent;native.Driver=Driver;runner.make_env=factory
    try:
        runner.main(['--configs','defaults','--task','objectnav_fixed','--logdir',options.logdir,'--seed',str(options.seed),
            '--use_wandb','False','--envs.amount','1','--envs.parallel','none','--envs.restart','False',
            '--run.steps','1000000000000000','--run.log_keys_video','image',
            '--encoder.cnn_keys','image','--decoder.cnn_keys','image','--encoder.mlp_keys','$^','--decoder.mlp_keys','$^',
            '--batch_size','4','--batch_length','16','--replay_size','50000','--jax.prealloc','False'])
    except BudgetReached:
        assert holder['task'].actions==options.steps
    finally:
        agt.Agent=original_agent;native.Driver=original_driver;runner.make_env=original_make
        if 'task' in holder:holder['task'].close()


def evaluate(options,native,runner,agt):
    import ruamel.yaml
    holder={};config=native.Config(ruamel.yaml.YAML(typ='safe').load(Path(options.config).read_text()))
    config=config.update({'envs.amount':1,'envs.parallel':'none','jax.prealloc':False,'use_wandb':False})
    env=make_task(native,runner,config,options,options.split,holder)
    step=native.Counter();agent=agt.Agent(env.obs_space,env.act_space,step,config)
    checkpoint=native.Checkpoint();checkpoint.agent=agent;checkpoint.load(options.checkpoint)
    receipt=json.loads(Path(options.checkpoint).with_suffix('.json').read_text())
    if receipt['source_revision']!=DYNALANG_REVISION or receipt['seed']!=options.seed:raise ValueError('Dynalang checkpoint identity mismatch')
    from vlm_sp_rew.baselines.common import check_task_contract
    check_task_contract(receipt['task_contract'],holder['task'].env)
    from vlm_sp_rew.export_metrics import FIELDS
    out=Path(options.output);out.parent.mkdir(parents=True,exist_ok=True)
    raw=holder['task'].env
    try:
        with out.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for _ in range(len(raw.schedule.items)):
                obs=env.step(dict(reset=True,action=np.zeros(env.act_space['action'].shape,env.act_space['action'].dtype)));carry=None
                while not obs['is_last']:
                    actions,carry=agent.policy({k:np.expand_dims(v,0)for k,v in obs.items()},carry,mode='eval')
                    obs=env.step(dict(reset=False,action=np.asarray(actions['action'])[0]))
                # Final reward/metric information is already retained by the common task.
                rows=[json.loads(line)for line in raw.record_path.read_text().splitlines()]
                last=rows[-1]
                if last['spl'] is None:raise ValueError('Qualified native SPL reference required')
                e=raw.episode;n=receipt['environment_actions']
                writer.writerow(dict(record_kind='measured',episode_id='Dynalang:%d:%d:%s'%(options.seed,n,e['episode_id']),benchmark=raw.benchmark,group='Dynalang',seed=options.seed,pair_id=e['pair_id'],checkpoint_steps=n,success=int(last['reward']['success']),spl=last['spl']))
                f.flush()
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','evaluate'])
    for key in ('source-root','episode-manifest','controller-config','run-record-root'):p.add_argument('--'+key,required=True)
    p.add_argument('--data-root');p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int)
    p.add_argument('--logdir');p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--output')
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--reward-config',default=str(Path(__file__).parents[1]/'configs/shared_reward.json'))
    a=p.parse_args()
    if a.mode=='train'and(not a.logdir or not a.steps or a.steps<10000):p.error('Set --logdir and a native training budget of at least10000 actions')
    if a.mode=='evaluate'and not all([a.config,a.checkpoint,a.output]):p.error('Set --config, --checkpoint and --output')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('Set CUDA_VISIBLE_DEVICES explicitly')
    native,runner,agt,nets=load_author(a.source_root);original=decoder_300(nets)
    try:train(a,native,runner,agt)if a.mode=='train'else evaluate(a,native,runner,agt)
    finally:nets.ImageDecoderResnet=original


if __name__=='__main__':main()
