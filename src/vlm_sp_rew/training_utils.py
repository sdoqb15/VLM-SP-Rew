#!/usr/bin/env python3
"""
Training utility file.
"""

import os
import copy, json
from pathlib import Path
import argparse
import yaml
import numpy as np
import elements
import embodied
from functools import partial as bind

def setup_gpu_env(use_multi_gpu=False, selected_gpu=None, auto_select_gpu=False, platform='cuda'):
    """Respect explicit caller ownership; never auto-select another user's GPU."""
    if platform not in ['cpu', 'cuda']:
        raise ValueError('platform must be cpu or cuda')
    os.environ['JAX_PLATFORMS'] = platform
    os.environ['JAX_PLATFORM_NAME'] = platform
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if platform == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return [0], [0]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible:
        raise ValueError('Set CUDA_VISIBLE_DEVICES explicitly to GPUs assigned to this run')
    return [0], [0]


def make_env(split='train', max_houses=100, target_objects=None, max_steps=200,
             single_house=False, house_index=0, house_type=None, min_rooms=None, max_rooms=None,
             episode_order='random', sampling_positions_dir=None, enable_scene_reward=False,
             scene_values_dir=None, env_cls=None, revision_options=None):
    """Environment factory."""
    if env_cls is None:
        raise ValueError("env_cls must be provided to make_env")

    if revision_options is not None:
        def _make_revised(index):
            options = dict(revision_options)
            options['run_record_root'] = str(Path(options['run_record_root']) / ('worker-%d' % index))
            return env_cls(**options)
        return _make_revised

    def _make(index):
        return env_cls(split, max_houses, target_objects, max_steps,
                       single_house, house_index, house_type, min_rooms, max_rooms,
                       episode_order, sampling_positions_dir, enable_scene_reward,
                       scene_values_dir)
    return _make


def make_agent(config, env_cls):
    """Agent factory."""
    def _make():
        temp_env = None
        if getattr(config, 'episode_manifest', None):
            obs_space, act_space = env_cls.spaces_from_manifest(config.episode_manifest, config.data_root)
        else:
            temp_env = env_cls(
                split=config.split,
                max_houses=config.max_houses,
                target_objects=config.target_objects,
                max_steps=config.max_steps,
                single_house=config.single_house,
                house_index=config.house_index,
                house_type=getattr(config, 'house_type', None),
                min_rooms=getattr(config, 'min_rooms', None),
                max_rooms=getattr(config, 'max_rooms', None),
                episode_order=getattr(config, 'episode_order', 'random'),
                sampling_positions_dir=getattr(config, 'sampling_positions_dir', None),
                enable_scene_reward=getattr(config, 'enable_scene_reward', False),
                scene_values_dir=getattr(config, 'scene_values_dir', None)
            )

            notlog = lambda k: not k.startswith('log/')
            obs_space = {k: v for k, v in temp_env.obs_space.items() if notlog(k)}
            act_space = temp_env.act_space

        agent_config = elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        )

        from dreamerv3.agent import Agent as DreamerV3Agent
        agent = DreamerV3Agent(obs_space, act_space, agent_config)
        if temp_env is not None:
            temp_env.close()
        return agent
    return _make


def make_replay(config, name='replay', mode='train'):
    """Replay buffer factory."""
    def _make():
        batlen = config.batch_length if mode == 'train' else config.report_length
        consec = config.consec_train if mode == 'train' else config.consec_report
        capacity = config.replay.size if mode == 'train' else config.replay.size / 10
        length = consec * batlen + config.replay_context
        assert config.batch_size * length <= capacity

        directory = elements.Path(config.logdir) / name
        if config.replicas > 1:
            directory /= f'{config.replica:05}'
        kwargs = dict(
            length=length, capacity=int(capacity), online=config.replay.online,
            chunksize=config.replay.chunksize, directory=directory)

        if config.replay.fracs.uniform < 1 and mode == 'train':
            assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
                'Gradient scaling for low-precision vlm_sp_rew.train can produce invalid loss '
                'outputs that are incompatible with prioritized replay.')
            recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
            from embodied.core import selectors
            kwargs['selector'] = selectors.Mixture(dict(
                uniform=selectors.Uniform(),
                priority=selectors.Prioritized(**config.replay.prio),
                recency=selectors.Recency(recency),
            ), config.replay.fracs)

        return embodied.replay.Replay(**kwargs)
    return _make


def make_stream(config, replay, mode):
    """Data stream factory."""
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True)
    return stream


def make_logger(config):
    """Logger factory."""
    def _make():
        step = elements.Counter()
        logdir = config.logdir
        outputs = []

        outputs.append(elements.logger.TerminalOutput())

        for output in config.logger.outputs:
            if output == 'jsonl':
                outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
                outputs.append(elements.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'))
            elif output == 'tensorboard':
                outputs.append(elements.logger.TensorBoardOutput(logdir, config.logger.fps))
            elif output == 'scope':
                outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))

        return elements.Logger(step, outputs)
    return _make


def load_dreamer_configs():
    """Load DreamerV3 default configs."""
    config_path = Path(__file__).resolve().parents[1] / 'dreamerv3' / 'configs.yaml'
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    def fix_numeric_values(obj):
        if isinstance(obj, dict):
            return {k: fix_numeric_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_numeric_values(v) for v in obj]
        elif isinstance(obj, str):
            try:
                if 'e' in obj.lower() or obj.replace('.', '').replace('-', '').isdigit():
                    return float(obj)
            except ValueError:
                pass
            return obj
        else:
            return obj

    return fix_numeric_values(configs['defaults'])


def run_training(args, env_cls, per_episode_callback):
    """Run the vlm_sp_rew.train loop."""
    dreamer_configs = load_dreamer_configs()

    config_dict = copy.deepcopy(dreamer_configs)
    config_dict.update({
        'logdir': str(args.logdir),
        'seed': args.seed,
        'replica': 0,
        'replicas': 1,

        'split': args.split,
        'max_houses': args.max_houses,
        'target_objects': args.target_objects,
        'max_steps': args.max_episode_steps,
        'single_house': args.single_house,
        'house_index': args.house_index,
        'house_type': args.house_type,
        'min_rooms': args.min_rooms,
        'max_rooms': args.max_rooms,
        'episode_order': args.episode_order,
        'sampling_positions_dir': args.sampling_positions_dir,
        'enable_scene_reward': args.enable_scene_reward,
        'scene_values_dir': args.scene_values_dir,
        'episode_manifest': args.episode_manifest,
        'data_root': args.data_root,
        'arm': args.arm,
        'shared_reward_config': args.shared_reward_config,
        'controller_config': args.controller_config,
        'run_record_root': args.run_record_root,


        'batch_size': 4,
        'batch_length': 16,
        'report_length': 16,
        'consec_train': 1,
        'consec_report': 1,
        'replay_context': 1,
    })

    config_dict['jax'].update({
        'platform': args.jax_platform,
        'compute_dtype': 'float32',
        'policy_devices': [0],
        'train_devices': [0],
        'prealloc': False,
        'jit': True,
        'debug': True,
        'precompile': False,
        'profiler': False,
    })

    config_dict['run'].update({
        'steps': args.steps,
        'train_ratio': 8.0,
        'log_every': 32,
        'report_every': 300,
        'save_every': 900,
        'envs': 1,
        'eval_envs': 1,
        'eval_eps': 1,
        'report_batches': 1,
        'debug': True,
    })

    config_dict['replay'].update({
        'size': 5e4,
        'online': True,
        'fracs': {'uniform': 1.0, 'priority': 0.0, 'recency': 0.0},
        'prio': {'exponent': 0.8, 'maxfrac': 0.5, 'initial': float('inf'), 'zero_on_sample': True},
        'priosignal': 'model',
        'recexp': 1.0,
        'chunksize': 1024,
    })

    config_dict['logger'].update({
        'outputs': ['jsonl'],
        'timer': True,
        'fps': 15,
    })

    config = elements.Config(config_dict)

    logdir_path = elements.Path(config.logdir)
    logdir_path.mkdir(parents=True, exist_ok=True)
    config = config.update({'logdir': logdir_path})
    config.save(logdir_path / 'config.yaml')

    train_args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    episode_results = []

    print("Starting vlm_sp_rew.train...")

    agent = make_agent(config, env_cls)()
    replay = make_replay(config, 'replay', 'train')()
    logger = make_logger(config)()

    revision_options = None
    if config.episode_manifest:
        if not config.controller_config or not config.run_record_root:
            raise ValueError('Set controller-config and run-record-root')
        revision_options = dict(
            episode_manifest=config.episode_manifest, data_root=config.data_root,
            split='dev' if config.split == 'val' else config.split, seed=config.seed,
            arm=config.arm, shared_reward_config=config.shared_reward_config,
            controller_config=config.controller_config, run_record_root=config.run_record_root,
            max_steps=config.max_steps, shuffle=(config.episode_order == 'random'))
    fns = [bind(make_env(config.split, config.max_houses, config.target_objects,
                        config.max_steps, config.single_house, config.house_index,
                        config.house_type, config.min_rooms, config.max_rooms,
                        config.episode_order, config.sampling_positions_dir,
                        config.enable_scene_reward, config.scene_values_dir,
                        env_cls=env_cls, revision_options=revision_options), i)
           for i in range(train_args.envs)]
    driver = embodied.Driver(fns, parallel=not train_args.debug)

    driver.on_step(per_episode_callback(episode_results, logger))
    # Reset observations enter replay, but are not executed environment actions.
    driver.on_step(lambda tran, _: logger.step.increment() if not tran['is_first'] else None)
    driver.on_step(replay.add)

    stream_train = iter(agent.stream(make_stream(config, replay, 'train')))
    stream_report = iter(agent.stream(make_stream(config, replay, 'report')))

    carry_train = [agent.init_train(train_args.batch_size)]
    carry_report = agent.init_report(train_args.batch_size)

    batch_steps = train_args.batch_size * train_args.batch_length
    should_train = elements.when.Ratio(train_args.train_ratio / batch_steps)

    def trainfn(tran, worker):
        min_replay_size = train_args.batch_size * train_args.batch_length
        current_replay_size = len(replay)

        if current_replay_size < min_replay_size:
            if logger.step.value % 50 == 0:
                print(f"Collecting: {current_replay_size}/{min_replay_size}")
            return

        train_times = should_train(logger.step)
        if train_times > 0:
            print(f"Training - step {logger.step.value}")

            for _ in range(train_times):
                batch = next(stream_train)
                carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                if 'replay' in outs:
                    replay.update(outs['replay'])
                if 'priority' in outs:
                    replay.prioritize(outs['key'], outs['priority'])
                logger.add(mets, prefix='train')

    driver.on_step(trainfn)

    # Keep each declared step checkpoint, including initialization and final.
    cp = elements.Checkpoint(logdir_path / 'ckpt', keep=None, step=logger.step)
    cp.step = logger.step
    cp.agent = agent
    cp.replay = replay
    checkpoint_index = Path(str(logdir_path)) / 'checkpoints.json'
    if checkpoint_index.exists():
        saved=json.loads(checkpoint_index.read_text())
        require_step=max(map(int,saved))
        cp.load(saved[str(require_step)])
        if int(logger.step)!=require_step:
            raise ValueError('Checkpoint/index step mismatch')

    policy = lambda *args: agent.policy(*args, mode='train')
    driver.reset(agent.init_policy)

    should_log = elements.when.Clock(train_args.log_every)
    should_report = elements.when.Clock(train_args.report_every)
    checkpoint_index = Path(str(logdir_path)) / 'checkpoints.json'
    def save_step_checkpoint():
        path = Path(str(logdir_path)) / 'checkpoints' / ('step-%09d' % int(logger.step))
        if not (path / 'done').exists():
            cp.save(str(path))
        entries = json.loads(checkpoint_index.read_text()) if checkpoint_index.exists() else {}
        entries[str(int(logger.step))] = str(path.resolve())
        temp = checkpoint_index.with_suffix('.tmp')
        temp.write_text(json.dumps(entries, indent=2) + '\n')
        temp.replace(checkpoint_index)
    save_step_checkpoint()
    next_save = (int(logger.step) // args.checkpoint_every + 1) * args.checkpoint_every

    while logger.step < train_args.steps:
        # A single action at a time avoids overshooting fixed-budget checkpoints.
        driver(policy, steps=1)

        if should_report(logger.step) and len(replay):
            agg = elements.Agg()
            for _ in range(train_args.report_batches):
                carry_report, mets = agent.report(carry_report, next(stream_report))
                agg.add(mets)
            logger.add(agg.result(), prefix='report')

        if should_log(logger.step):
            logger.add(replay.stats(), prefix='replay')
            logger.write()

        if int(logger.step) >= next_save or int(logger.step) == args.steps:
            save_step_checkpoint()
            next_save += args.checkpoint_every

    driver.close()
    logger.close()

    print("Training completed")
    print(f"Training records: {config.logdir}")


def per_episode_callback(episode_results, logger):
    """Record simulator-defined success and qualified metric references only."""
    def callback(tran, worker):
        if not tran.get('is_last', False):
            return
        result = dict(success=float(tran['log/success']),
                      reward=float(tran['log/episode_reward']),
                      length=int(tran['log/episode_length']),
                      executed_path_m=float(tran['log/executed_path_m']),
                      spl_valid=bool(tran['log/spl_valid']))
        if result['spl_valid']:
            result['spl'] = float(tran['log/spl'])
        episode_results.append(result)
        logger.add(result, prefix='episode')
    return callback


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Train the manuscript RGB-only ObjectNav learner')
    parser.add_argument('--config', default=str(Path(__file__).parent / 'configs/paper.yaml'))
    parser.add_argument('--jax-platform', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--episode-manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--arm', choices=['env','geo','vlm'], default='env')
    parser.add_argument('--shared-reward-config', default=str(Path(__file__).parent / 'configs/shared_reward.json'))
    parser.add_argument('--controller-config')
    parser.add_argument('--run-record-root')
    parser.add_argument('--logdir', default='runtime/train')
    parser.add_argument('--split', choices=['train','val','test'], default='train')
    parser.add_argument('--steps', type=int, default=400000)
    parser.add_argument('--checkpoint-every', type=int, default=25000)
    parser.add_argument('--seed', type=int, default=288681)
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--episode-order', choices=['random','fixed'], default='random')
    # Internal defaults retained for the factory signature, not independent paper protocols.
    parser.set_defaults(max_houses=8, target_objects=['HousePlant'], single_house=False,
                        house_index=0, house_type=None, min_rooms=None, max_rooms=None,
                        sampling_positions_dir=None, enable_scene_reward=False, scene_values_dir=None)
    return parser


def setup_random_seed(seed):
    """Set random seed."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def print_training_config(args):
    print(json.dumps(dict(manifest=args.episode_manifest, arm=args.arm, seed=args.seed,
                          steps=args.steps, checkpoint_every=args.checkpoint_every,
                          logdir=args.logdir), indent=2))
