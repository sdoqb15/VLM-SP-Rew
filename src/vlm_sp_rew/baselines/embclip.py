"""EmbCLIP and EmbCLIP-Codebook using native author policies and AllenAct PPO."""
from pathlib import Path
import argparse,importlib,json,os,subprocess,sys

AUTHOR_REVISION='ed4feaf4944a01ad1d8b12a3dfc46f2c9cb4c992'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','evaluate'])
    for key in ('source-root','episode-manifest','controller-config','run-record-root','logdir','clip-weights','clip-sha256'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--data-root');p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int)
    p.add_argument('--ipc-root');p.add_argument('--actors',type=int,default=1);p.add_argument('--codebook',action='store_true');p.add_argument('--checkpoint');p.add_argument('--checkpoint-steps',type=int)
    p.add_argument('--output');p.add_argument('--task-contract');p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--reward-config',default=str(Path(__file__).parents[1]/'configs/shared_reward.json'))
    a=p.parse_args();source=Path(a.source_root).resolve()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=AUTHOR_REVISION:p.error('Use the pinned author source revision '+AUTHOR_REVISION)
    if a.actors<=0:p.error('Use a positive --actors count')
    if a.mode=='train'and(not a.steps or a.steps<=0):p.error('Set a positive --steps')
    if a.mode=='evaluate'and not all([a.checkpoint,a.output]):p.error('Set --checkpoint and --output')
    if a.mode=='evaluate'and Path(a.output).exists():p.error('Use a fresh output CSV')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('Set CUDA_VISIBLE_DEVICES explicitly')
    if len(a.clip_sha256)!=64:p.error('Set the full SHA256 of the local CLIP RN50 weights')
    sys.path.insert(0,str(source));os.environ['PYTHONPATH']=str(source)+os.pathsep+os.environ.get('PYTHONPATH','')
    from vlm_sp_rew.protocol import validate_allocation
    audit=validate_allocation(a.episode_manifest,a.data_root,shared=False)
    from vlm_sp_rew.reward import load_episode_bank
    bank=load_episode_bank(a.episode_manifest,a.data_root)
    from vlm_sp_rew.environment import actions_for_benchmark
    contract=dict(benchmark=audit['benchmark'],actions=list(actions_for_benchmark(audit['benchmark'])),
                  goal_vocabulary=bank['goal_vocabulary'],RGB=[300,300,3],goal_channel=False,max_episode_actions=200)
    record=Path(a.task_contract)if a.task_contract else Path(a.logdir)/'TASK_CONTRACT.json'
    if a.mode=='train':
        record.parent.mkdir(parents=True,exist_ok=True)
        if record.exists()and json.loads(record.read_text())!=contract:p.error('Existing output directory belongs to a different task contract')
        if not record.exists():record.write_text(json.dumps(contract,indent=2)+'\n')
        configuration=dict(arguments=vars(a),author_source_revision=AUTHOR_REVISION,manifest_sha256=audit['manifest_sha256'],source_files={n.name:__import__('hashlib').sha256(n.read_bytes()).hexdigest()for n in [Path(__file__),Path(__file__).with_name('allenact_tasks.py')]})
        with (Path(a.logdir)/'FIT_CONFIGURATION.json').open('x')as f:json.dump(configuration,f,indent=2);f.write('\n')
    elif not record.exists()or json.loads(record.read_text())!=contract:p.error('Supply the fitted model TASK_CONTRACT.json for this benchmark and fixed goal')
    if a.mode=='evaluate':
        import torch
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        actual=int(saved['total_steps'])
        if a.checkpoint_steps is not None and actual!=a.checkpoint_steps:p.error('Checkpoint action count mismatch')
        a.checkpoint_steps=actual
    from vlm_sp_rew.baselines.common import configure_local_ipc
    configure_local_ipc(a.ipc_root)
    from vlm_sp_rew.baselines.allenact_tasks import NavigationConfig
    from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
    # Use the author runner directly; no alternative PPO implementation is substituted.
    runner=OnPolicyRunner(config=NavigationConfig(a),output_dir=a.logdir,loaded_config_src_files=None,seed=a.seed,
        mode='train'if a.mode=='train'else'test',disable_tensorboard=True,disable_config_saving=True)
    if a.mode=='train':runner.start_train(try_restart_after_task_error=False)
    else:runner.start_test(a.checkpoint,max_sampler_processes_per_worker=1)


if __name__=='__main__':main()
