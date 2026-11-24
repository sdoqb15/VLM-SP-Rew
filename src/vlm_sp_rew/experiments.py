"""Prepare and launch the manuscript's Dreamer reward/field comparisons."""
from pathlib import Path
import argparse, json, subprocess, sys
from vlm_sp_rew.protocol import BENCHMARKS,SEEDS,validate_allocation,write_configs,field_config
from vlm_sp_rew.reward import require

EXPERIMENTS = {
    'shared-env':dict(arm='env',shared=True,coefficient=0.,field='none'),
    'shared-geo':dict(arm='geo',shared=True,coefficient=1.,field='geometry'),
    'shared-vlm':dict(arm='vlm',shared=True,coefficient=1.,field='full'),
    'weight-0.5':dict(arm='vlm',shared=True,coefficient=.5,field='full',benchmarks=['ProcTHOR']),
    'weight-2':dict(arm='vlm',shared=True,coefficient=2.,field='full',benchmarks=['ProcTHOR']),
    'benchmark-dreamer':dict(arm='env',shared=False,coefficient=0.,field='none'),
    'benchmark-vlm':dict(arm='vlm',shared=False,coefficient=1.,field='full'),
    'cross-scene-dreamer':dict(arm='env',shared=False,coefficient=0.,field='none',single_scene=True,benchmarks=['ProcTHOR','R2R-adapted']),
    'cross-scene-vlm':dict(arm='vlm',shared=False,coefficient=1.,field='full',single_scene=True,benchmarks=['ProcTHOR','R2R-adapted']),
    'visibility-only':dict(arm='vlm',shared=False,coefficient=1.,field='visibility',benchmarks=['ProcTHOR']),
    'distance-only':dict(arm='vlm',shared=False,coefficient=1.,field='distance',benchmarks=['ProcTHOR']),
    'llava-13b':dict(arm='vlm',shared=False,coefficient=1.,field='full',annotation='LLaVA-13B',benchmarks=['ProcTHOR']),
}


def prepare(name,benchmark,manifest,data_root,controller,out,seed,steps=None):
    exp=EXPERIMENTS[name];require(benchmark in exp.get('benchmarks',BENCHMARKS),'Benchmark not used by this experiment')
    require(seed in SEEDS,'Use the declared fitted seeds')
    audit=validate_allocation(manifest,data_root,exp['shared'])
    require(audit['benchmark']==benchmark,'Benchmark/manifest mismatch')
    if exp.get('single_scene'):require(audit['scene_counts']['train']==1,'Cross-scene fit uses one training scene')
    require(exp['shared'] or type(steps)is int and steps>0,'Set benchmark-specific training budget explicitly;400k belongs to the shared study')
    require(not exp['shared'] or steps in (None,400000),'Shared study uses400k')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    field,reward=write_configs(out/'configuration',benchmark,exp['shared'],exp['coefficient'])
    # Require the selected field variant/model before consuming GPU time.
    from vlm_sp_rew.reward import load_episode_bank,bind_field
    q=load_episode_bank(manifest,data_root)
    for episode in q['bank']['episodes']:
        if episode['split']!='train':continue
        value=bind_field(episode,exp['arm'],q['root'])
        if value:
            require(value.get('field_parameters')==field_config(benchmark,exp['shared'])['vlm_sp_rew_params'],'Use the field profile corresponding to this experiment, including benchmark-specific/shared settings')
        if value and exp['arm']=='vlm':
            require(value.get('signal_variant','full')==exp['field'],'Manifest field variant differs from experiment')
            if exp.get('annotation'):
                require(value.get('annotation_family')==exp['annotation'],'Use the separately frozen LLaVA-13B annotation field')
    command=[sys.executable,'-m','vlm_sp_rew.train','--episode-manifest',str(Path(manifest).resolve()),
             '--controller-config',str(Path(controller).resolve()),'--run-record-root',str(out/'records'),
             '--logdir',str(out/'learner'),'--shared-reward-config',str(reward),'--arm',exp['arm'],
             '--seed',str(seed),'--steps',str(400000 if exp['shared'] else steps),'--checkpoint-every','25000']
    if data_root:command+=['--data-root',str(Path(data_root).resolve())]
    (out/'run.json').write_text(json.dumps(dict(experiment=name,benchmark=benchmark,seed=seed,
        command=command,field_config=str(field),allocation=audit),indent=2)+'\n')
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    sub.add_parser('list')
    for op in ('prepare','train'):
        q=sub.add_parser(op)
        q.add_argument('--experiment',choices=EXPERIMENTS,required=True);q.add_argument('--benchmark',choices=BENCHMARKS,required=True)
        for key in ('episode-manifest','controller-config','out'):q.add_argument('--'+key,required=True)
        q.add_argument('--data-root');q.add_argument('--seed',type=int,choices=SEEDS,required=True);q.add_argument('--steps',type=int)
    a=p.parse_args()
    if a.operation=='list':print(json.dumps(EXPERIMENTS,indent=2));return
    command=prepare(a.experiment,a.benchmark,a.episode_manifest,a.data_root,a.controller_config,a.out,a.seed,a.steps)
    print(json.dumps(command))
    if a.operation=='train':subprocess.run(command,check=True)


if __name__=='__main__':main()
