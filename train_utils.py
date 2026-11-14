#!/usr/bin/env python3
"""
Training utility file.
"""

import os
from pathlib import Path
import argparse
import yaml
import numpy as np
import elements
import embodied
from functools import partial as bind

successful_episodes = 0
total_episodes = 0


def get_gpu_memory_usage():
    """Query GPU memory usage."""
    try:
        import subprocess
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            gpu_idx = int(parts[0])
            used = int(parts[1])
            total = int(parts[2])
            usage_percent = (used / total) * 100
            gpus.append((gpu_idx, usage_percent, used, total))
        return gpus
    except Exception as e:
        print(f"Warning: failed to query GPU usage: {e}")
        return []


def select_best_gpu():
    """Select the least used GPU."""
    gpus = get_gpu_memory_usage()
    if not gpus:
        print("Using default GPU 0")
        return 0

    gpus.sort(key=lambda x: x[1])
    best_gpu = gpus[0]

    print("GPU usage:")
    for gpu_idx, usage, used, total in gpus:
        status = "" if usage < 20 else "" if usage < 50 else ""
        print(f"  GPU {gpu_idx}: {status} {usage:.1f}% ({used}MB/{total}MB)")

    print(f"Selected GPU {best_gpu[0]} (usage: {best_gpu[1]:.1f}%)")
    return best_gpu[0]


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
                'Gradient scaling for low-precision training can produce invalid loss '
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
    config_path = Path(__file__).parent / 'dreamerv3' / 'configs.yaml'
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
    """Run the training loop."""
    dreamer_configs = load_dreamer_configs()

    config_dict = dreamer_configs.copy()
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
        'outputs': ['jsonl', 'scope', 'tensorboard'],
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

    print("Starting training...")

    agent = make_agent(config, env_cls)()
    replay = make_replay(config, 'replay', 'train')()
    logger = make_logger(config)()

    revision_options = None
    if config.episode_manifest:
        if not config.controller_config or not config.run_record_root:
            raise ValueError('Revised experiments require controller-config and run-record-root')
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
    driver.on_step(lambda tran, _: logger.step.increment())
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

    cp = elements.Checkpoint(logdir_path / 'ckpt')
    cp.step = logger.step
    cp.agent = agent
    cp.replay = replay
    cp.load_or_save()

    policy = lambda *args: agent.policy(*args, mode='train')
    driver.reset(agent.init_policy)

    should_log = elements.when.Clock(train_args.log_every)
    should_report = elements.when.Clock(train_args.report_every)
    should_save = elements.when.Clock(train_args.save_every)

    while logger.step < train_args.steps:
        driver(policy, steps=10)

        if should_report(logger.step) and len(replay):
            agg = elements.Agg()
            for _ in range(train_args.report_batches):
                carry_report, mets = agent.report(carry_report, next(stream_report))
                agg.add(mets)
            logger.add(agg.result(), prefix='report')

        if should_log(logger.step):
            logger.add(replay.stats(), prefix='replay')
            logger.write()

        if should_save(logger.step):
            cp.save()

    driver.close()
    logger.close()

    print("Training completed")
    print(f"View results: tensorboard --logdir {config.logdir}")


def per_episode_callback(episode_results, logger):
    """Episode-end callback."""
    recent_rewards = []
    recent_successes = []
    recent_episode_results = []

    def callback(tran, worker):
        nonlocal recent_rewards, recent_successes, recent_episode_results
        global successful_episodes, total_episodes

        if not tran.get('is_last', False):
            return

        total_episodes += 1

        episode_reward = float(tran.get('log/episode_reward', 0.0))
        episode_length = int(tran.get('log/episode_length', 1))
        success = float(tran.get('log/success', 0.0))

        if success > 0.5:
            successful_episodes += 1

        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        recent_successes.append(1.0 if success > 0.5 else 0.0)
        if len(recent_successes) > 100:
            recent_successes.pop(0)

        revised_metrics = 'log/spl_valid' in tran
        spl_valid = bool(tran.get('log/spl_valid', False))
        shortest_path_length = int(tran.get('log/shortest_path_length', max(1, episode_length // 3)))

        episode_result = {
            'success': success > 0.5,
            'actual_path_length': episode_length,
            'shortest_path_length': shortest_path_length,
        }
        episode_results.append(episode_result)
        recent_episode_results.append(episode_result)
        if len(recent_episode_results) > 100:
            recent_episode_results.pop(0)

        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0

        official_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0

        average_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0

        # SPL = (1/N) * Σ(Si * li / max(pi, li))
        official_spl = 0.0
        if len(recent_episode_results) >= 10:
            try:
                spl_sum = 0.0
                for result in recent_episode_results:
                    if result['success']:
                        shortest_len = result['shortest_path_length']
                        actual_len = result['actual_path_length']
                        spl_sum += shortest_len / max(actual_len, shortest_len)
                official_spl = spl_sum / len(recent_episode_results)
            except Exception as e:
                print(f"Warning: SPL computation failed: {e}")
                official_spl = 0.0

        metrics = {
            'reward': episode_reward,
            'success': success,
            'success_rate': success_rate,
            'official_success_rate': official_success_rate,
            'legacy_heuristic_path_efficiency': official_spl,
            'average_reward': average_reward,
            'length': episode_length,
        }

        if revised_metrics:
            metrics.pop('legacy_heuristic_path_efficiency', None)
            metrics['spl_valid'] = float(spl_valid)
            metrics['executed_path_m'] = float(tran['log/executed_path_m'])
            if spl_valid:
                metrics['spl'] = float(tran['log/spl'])
        logger.add(metrics, prefix='episode')

        if total_episodes % 10 == 0:
            print(f"\nEpisode {total_episodes} Summary:")
            print(f"   Cumulative Success Rate: {success_rate:.3f} (all {total_episodes} episodes)")
            print(f"   Rolling Success Rate: {official_success_rate:.3f} (last {len(recent_successes)} episodes)")
            if revised_metrics:
                print('   Qualified SPL:', float(tran['log/spl']) if spl_valid else 'not available; no qualified path reference')
            else:
                print(f"   Legacy heuristic efficiency (not SPL): {official_spl:.3f} (last {len(recent_episode_results)} episodes)")
            print(f"   Average Reward: {average_reward:.3f} (last {len(recent_rewards)} episodes)")
            print(f"   Episode Length: {episode_length}")

    return callback


def build_arg_parser():
    """Create training argument parser."""
    parser = argparse.ArgumentParser(description='Train DreamerV3 on ProcTHOR')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'config.yaml'))
    parser.add_argument('--jax-platform', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--episode-manifest', help='Frozen scene/goal/pose/split manifest; enables revised adapter')
    parser.add_argument('--data-root', help='Root for manifest-relative scene and field artifacts')
    parser.add_argument('--arm', choices=['env', 'geo', 'geo_vis', 'vlm'], default='env')
    parser.add_argument('--shared-reward-config', default=str(Path(__file__).parent / 'shared_reward_revision_config.json'))
    parser.add_argument('--controller-config', help='Pinned local simulator binary and explicit Vulkan renderer configuration')
    parser.add_argument('--run-record-root', help='H20 workvolume for immutable episode evidence')

    parser.add_argument('--logdir', type=str, default='./logdir/dreamer/procthor_v1',
                        help='Log directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--max-houses', type=int, default=100,
                        help='Maximum number of houses to use')
    parser.add_argument('--steps', type=int, default=10000000, help='Training steps')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--max-episode-steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--target-objects', nargs='+',
                        default=['HousePlant'],
                        help='Target object types')
    parser.add_argument('--single-house', action='store_true',
                        help='Use single-house mode')
    parser.add_argument('--house-index', type=int, default=0,
                        help='House index for single-house mode')
    parser.add_argument('--house-type', type=str, default=None,
                        choices=['studio', 'apartment', 'house'],
                        help='Filter by house type (studio/apartment/house)')
    parser.add_argument('--min-rooms', type=int, default=None,
                        help='Min rooms')
    parser.add_argument('--max-rooms', type=int, default=None,
                        help='Max rooms')
    parser.add_argument('--episode-order', type=str, default='fixed',
                        choices=['random', 'fixed'],
                        help='Episode order: random or fixed')
    parser.add_argument('--sampling-positions-dir', type=str, default=None,
                        help='Sampling positions dir (for fixed mode)')
    parser.add_argument('--enable-scene-reward', action='store_true',
                        help='Enable scene-value reward')
    parser.add_argument('--scene-values-dir', type=str, default=None,
                        help='Scene values dir (contains value.txt)')
    return parser


def setup_random_seed(seed):
    """Set random seed."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def print_training_config(args):
    """Print training configuration."""
    print("Starting DreamerV3 training on ProcTHOR")
    print(f"Log directory: {args.logdir}")
    print(f"Target steps: {args.steps}")
    print(f"Split: {args.split}")

    if args.single_house:
        print(f"Single-house mode: house index {args.house_index}")
    else:
        print(f"Multi-house mode: {args.max_houses} houses")

    print(f"Target objects: {', '.join(args.target_objects)}")
    print(f"Episode order: {args.episode_order}")

    if args.sampling_positions_dir:
        print(f"Sampling positions dir: {args.sampling_positions_dir}")

    if args.enable_scene_reward:
        print("Scene-value reward: enabled")
        if args.scene_values_dir:
            print(f"Scene values dir: {args.scene_values_dir}")
        else:
            print("Warning: scene values dir not set")
    else:
        print("Scene-value reward: disabled")

    if len(args.target_objects) == 1:
        print(f"Single target: {args.target_objects[0]}")
    else:
        print(f"Multi-target: {len(args.target_objects)} objects")

    if args.house_type:
        print(f"House type filter: {args.house_type}")
    if args.min_rooms or args.max_rooms:
        room_filter = "Rooms: "
        if args.min_rooms:
            room_filter += f"{args.min_rooms}+"
        if args.max_rooms:
            if args.min_rooms:
                room_filter += " and "
            room_filter += f"≤{args.max_rooms}"
        print(f"{room_filter}")
