"""Native AllenAct policies and PPO with the common fixed-goal ObjectNav task."""
from pathlib import Path
import copy,csv,hashlib,json
import gym
import numpy as np
import torch
from allenact.base_abstractions.task import Task,TaskSampler
from allenact.base_abstractions.sensor import Sensor,SensorSuite
from allenact.base_abstractions.preprocessor import Preprocessor,SensorPreprocessorGraph
from allenact.base_abstractions.experiment_config import ExperimentConfig,MachineParams
from allenact.base_abstractions.misc import RLStepResult
from allenact.utils.experiment_utils import Builder,TrainingPipeline,PipelineStage,TrainingSettings
from allenact.algorithms.onpolicy_sync.losses.ppo import PPO,PPOConfig
from vlm_sp_rew.environment import ObjectNavEnv,actions_for_benchmark
from vlm_sp_rew.baselines.common import ArraySpec
from vlm_sp_rew.reward import EpisodeSchedule,load_episode_bank


class RGBSensor(Sensor):
    def __init__(self):super().__init__('rgb',gym.spaces.Box(0,255,(300,300,3),dtype=np.uint8))
    def get_observation(self,env,task,*args,**kwargs):return task.latest['image']


def fixed_goal_encoder_input(encoder,inputs):
    """Retain the author's single-goal embedding internally, without a sensor."""
    observations,=inputs
    if encoder.goal_uuid in observations:raise ValueError('A fixed-goal policy accepts no external goal input')
    features=observations[encoder.resnet_uuid]
    index=torch.zeros((*features.shape[:-3],1),dtype=torch.int64,device=features.device)
    return ({**observations,encoder.goal_uuid:index},)


class CLIPResnetFeatures(Preprocessor):
    def __init__(self,weights,expected_sha256):
        super().__init__(['rgb'],'rgb_clip_resnet',gym.spaces.Box(-np.inf,np.inf,(2048,9,9),dtype=np.float32))
        self.weights=weights;self.expected_sha256=expected_sha256;self.model=None;self.device=torch.device('cpu')
    def to(self,device):
        self.device=torch.device(device)
        if self.model is not None:self.model.to(self.device)
        return self
    def process(self,obs,*args,**kw):
        if self.model is None:
            import clip
            digest=hashlib.sha256()
            with Path(self.weights).open('rb')as f:
                for chunk in iter(lambda:f.read(4*1024**2),b''):digest.update(chunk)
            if digest.hexdigest()!=self.expected_sha256:raise ValueError('Local CLIP RN50 weight hash mismatch')
            # Open each load by filename: older CLIP releases reuse an exhausted
            # file handle when a non-TorchScript state_dict falls through.
            try:
                state=torch.jit.load(self.weights,map_location='cpu').state_dict()
            except RuntimeError:
                state=torch.load(self.weights,map_location='cpu',weights_only=False)
            model=clip.model.build_model(state).to(self.device)
            self.model=model.visual.float().eval()
            for p in self.model.parameters():p.requires_grad_(False)
            if not hasattr(self.model,'layer4'):raise ValueError('The native EmbCLIP comparison uses CLIP RN50')
        rgb=obs['rgb'];leading=rgb.shape[:-3]
        if tuple(rgb.shape[-3:])!=(300,300,3):raise ValueError('Native300x300 RGB required')
        x=rgb.reshape(-1,300,300,3).permute(0,3,1,2).to(self.device,dtype=torch.float32)/255.
        mean=x.new_tensor([.48145466,.4578275,.40821073]).view(1,3,1,1)
        std=x.new_tensor([.26862954,.26130258,.27577711]).view(1,3,1,1)
        with torch.no_grad():
            x=(x-mean)/std;m=self.model
            x=m.relu1(m.bn1(m.conv1(x)));x=m.relu2(m.bn2(m.conv2(x)))
            x=m.relu3(m.bn3(m.conv3(x)));x=m.avgpool(x)
            x=m.layer4(m.layer3(m.layer2(m.layer1(x))))
        if tuple(x.shape[-3:])!=(2048,9,9):raise ValueError('Unexpected300RGB CLIP spatial feature shape')
        return x.reshape(*leading,2048,9,9)


class NavigationTask(Task):
    def __init__(self,env,initial,options):
        self.latest=initial;self.options=options;self.exported=False
        super().__init__(env,[RGBSensor()],copy.deepcopy(env.episode),200)
    @property
    def action_space(self):return gym.spaces.Discrete(len(self.env.actions))
    def render(self,mode='rgb',*args,**kwargs):return self.latest['image']
    def reached_terminal_state(self):return bool(self.env.done)
    def _step(self,action):
        self.latest=self.env.step(dict(action=int(action)))
        return RLStepResult(observation=self.get_observations(),reward=float(self.latest['reward']),done=bool(self.latest['is_last']),info={})
    def close(self):pass # The sampler owns and reuses the controller.
    def metrics(self):
        if not self.is_done():return {}
        result=dict(success=float(self.latest['log/success']),ep_length=self.num_steps_taken(),reward=self.cumulative_reward)
        if self.latest['log/spl_valid']:result['spl']=float(self.latest['log/spl'])
        if self.options.mode=='evaluate'and not self.exported:
            if not self.latest['log/spl_valid']:raise ValueError('Qualified native SPL reference required')
            from vlm_sp_rew.export_metrics import FIELDS
            path=Path(self.options.output);path.parent.mkdir(parents=True,exist_ok=True)
            new=not path.exists()
            with path.open('a',newline='')as f:
                w=csv.DictWriter(f,fieldnames=FIELDS)
                if new:w.writeheader()
                group='EmbCLIP-Codebook'if self.options.codebook else 'EmbCLIP'
                w.writerow(dict(record_kind='measured',episode_id='%s:%s:%s:%s'%(group,self.options.seed,self.options.checkpoint_steps,self.task_info['episode_id']),benchmark=self.env.benchmark,
                    group=group,seed=self.options.seed,pair_id=self.task_info['pair_id'],checkpoint_steps=self.options.checkpoint_steps,success=int(result['success']),spl=result['spl']))
            self.exported=True
        return result


class NavigationSampler(TaskSampler):
    def __init__(self,options,split,seed,**kw):
        self.options=options;self.split=split;self.seed=seed;self.last=None;self.sampled=0
        self.env=ObjectNavEnv(options.episode_manifest,options.data_root,split,seed,arm='env',shared_reward_config=options.reward_config,
            controller_config=options.controller_config,run_record_root=options.run_record_root,max_steps=200,shuffle=split=='train',space_type=ArraySpec)
    @property
    def length(self):return float('inf')if self.split=='train'else max(0,len(self.env.schedule.items)-self.sampled)
    @property
    def total_unique(self):return len(self.env.schedule.items)
    @property
    def last_sampled_task(self):return self.last
    @property
    def all_observation_spaces_equal(self):return True
    def next_task(self,force_advance_scene=False):
        if self.length==0:return None
        obs=self.env.reset();self.last=NavigationTask(self.env,obs,self.options);self.sampled+=1;return self.last
    def close(self):self.env.close()
    def reset(self):
        self.sampled=0;self.last=None
        self.env.schedule=EpisodeSchedule(self.env.qualified,self.split,self.seed,self.split=='train')
    def set_seed(self,seed):self.seed=seed;self.reset()


class NavigationConfig(ExperimentConfig):
    def __init__(self,options):
        self.options=options;self.sensors=[RGBSensor()]
        self.benchmark=load_episode_bank(options.episode_manifest,options.data_root)['bank']['benchmark']
    def tag(self):return 'FixedGoal-EmbCLIP-Codebook'if self.options.codebook else 'FixedGoal-EmbCLIP'
    def machine_params(self,mode,**kw):
        n=(self.options.actors if mode=='train'else 1)if mode==('train'if self.options.mode=='train'else'test')else 0
        graph=SensorPreprocessorGraph(SensorSuite(self.sensors).observation_spaces,
            [CLIPResnetFeatures(self.options.clip_weights,self.options.clip_sha256)]) if n else None
        return MachineParams(nprocesses=n,devices=[torch.device('cuda',0)],sensor_preprocessor_graph=graph)
    def create_model(self,**kw):
        from procthor_objectnav.models.resnet_gru import ResnetTensorNavActorCritic,ResnetTensorNavActorCriticCodebook
        cls=ResnetTensorNavActorCriticCodebook if self.options.codebook else ResnetTensorNavActorCritic
        external_space=kw['sensor_preprocessor_graph'].observation_spaces
        native_space=copy.deepcopy(external_space)
        native_space.spaces['fixed_goal']=gym.spaces.Discrete(1)
        model=cls(action_space=gym.spaces.Discrete(len(actions_for_benchmark(self.benchmark))),
            observation_space=native_space,goal_sensor_uuid='fixed_goal',
            rgb_resnet_preprocessor_uuid='rgb_clip_resnet',depth_resnet_preprocessor_uuid=None,
            hidden_size=512,goal_dims=32,add_prev_actions=True,auxiliary_uuids=[],visualize=False)
        # Preserve author parameter names, shapes and initialization. The native
        # one-entry embedding is a model parameter, never an environment channel.
        model.observation_space=external_space
        model.goal_visual_encoder.register_forward_pre_hook(fixed_goal_encoder_input)
        return model
    def make_sampler_fn(self,**kwargs):return NavigationSampler(**kwargs)
    def _sampler_args(self,split,process_ind,total_processes,seeds=None,**kw):
        if split!='train'and total_processes!=1:raise ValueError('Evaluation uses one pass over the frozen starts')
        return dict(options=self.options,split=split,seed=self.options.seed+process_ind if split=='train'else self.options.seed)
    def train_task_sampler_args(self,**kwargs):return self._sampler_args('train',**kwargs)
    def valid_task_sampler_args(self,**kwargs):return self._sampler_args('dev',**kwargs)
    def test_task_sampler_args(self,**kwargs):return self._sampler_args(self.options.split,**kwargs)
    def training_pipeline(self,**kw):
        remaining=self.options.steps;stages=[]
        # Preserve author32/64/128rollout progression; truncate its budget explicitly.
        for rollout in (32,64,128):
            if not remaining:break
            budget=min(remaining,10000000)if rollout<128 else remaining
            if budget%rollout:raise ValueError('Choose a budget divisible by the current native rollout length')
            stages.append(PipelineStage(loss_names=['ppo_loss'],max_stage_steps=budget,training_settings=TrainingSettings(num_steps=rollout)))
            remaining-=budget
        return TrainingPipeline(save_interval=25000,metric_accumulate_interval=1000,
            optimizer_builder=Builder(torch.optim.Adam,dict(lr=.0003)),num_mini_batch=1,update_repeats=4,max_grad_norm=.5,
            num_steps=128,named_losses=dict(ppo_loss=PPO(**PPOConfig)),gamma=.99,use_gae=True,gae_lambda=.95,
            advance_scene_rollout_period=None,pipeline_stages=stages)
