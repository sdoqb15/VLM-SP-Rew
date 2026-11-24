"""Single source of manuscript experiment settings and scene allocation checks."""
from pathlib import Path
import copy, json
import yaml
from vlm_sp_rew.environment import actions_for_benchmark
from vlm_sp_rew.reward import load_episode_bank, require

BENCHMARKS = ('RoboTHOR', 'ProcTHOR', 'R2R-adapted')
SEEDS = (288681, 288682, 288683, 288684, 288685)
CHECKPOINTS = tuple(range(0,400001,25000))
PROFILES = {
 'RoboTHOR': dict(N=30,d_min=.8,tau_room=.4,delta_vis=1,delta_novis=3,
                  alpha_k=[4,3,2],beta_k=[1,0,-1],K_vis=2,tau=2,w_hi=1.,w_lo=.1,r_up=1.,r_down=-1.),
 'ProcTHOR': dict(N=50,d_min=1.,tau_room=.3,delta_vis=2,delta_novis=5,
                  alpha_k=[5,4,3],beta_k=[2,1,0,-1],K_vis=3,tau=3,w_hi=1.,w_lo=.5,r_up=1.,r_down=-1.),
 'R2R-adapted': dict(N=50,d_min=1.,tau_room=.2,delta_vis=2,delta_novis=5,
                  alpha_k=[6,5,4,3],beta_k=[2,1,0,-1],K_vis=4,tau=3,w_hi=1.,w_lo=.5,r_up=.5,r_down=-.5),
}


def field_config(benchmark, shared=False):
    require(benchmark in BENCHMARKS, 'Unknown benchmark')
    p = copy.deepcopy(PROFILES['ProcTHOR' if shared else benchmark])
    p['beta_tail_step'] = -1
    return dict(method_contract='paper_v1',vlm_sp_rew_params=p,
                stage_defaults=dict(graph_max_distance=1.5*p['d_min'],value_min_distance=p['d_min']))


def reward_config(benchmark, shared=False, coefficient=1.):
    p = field_config(benchmark,shared)['vlm_sp_rew_params']
    base = json.loads((Path(__file__).parent/'configs/shared_reward.json').read_text())
    base.update({k:p[k] for k in ('tau','w_hi','w_lo','r_up','r_down')})
    base['lambda_shaping'] = coefficient
    require(coefficient >= 0, 'Nonnegative shaping coefficient required')
    return base


def validate_allocation(path, data_root=None, shared=True):
    q = load_episode_bank(path,data_root)
    bank = q['bank']
    require(bank.get('benchmark') in BENCHMARKS, 'Explicit benchmark required')
    require(len(q['goal_vocabulary']) == 1, 'One fixed goal category per fitted policy; no goal channel')
    expected = dict(train=8,dev=2,test=8) if shared else None
    actual = {s:len({e['scene_identity_sha256'] for e in bank['episodes'] if e['split']==s})
              for s in ('train','dev','test')}
    if expected is not None:
        require(actual==expected, 'Shared protocol requires 8/2/8 disjoint scenes: '+str(actual))
        for split in ('dev','test'):
            pairs = {}
            for e in bank['episodes']:
                if e['split']==split:
                    require(e.get('pair_id'), 'Evaluation pair_id required')
                    pairs.setdefault(e['pair_id'],[]).append(e)
                    require(e.get('spl_reference'), 'Evaluation requires a navigable SPL reference')
            require(len(pairs)==expected[split], 'One fixed target pair per evaluation scene')
            require(all(len(v)==10 for v in pairs.values()), 'Ten frozen starts per evaluation scene/goal pair required')
    return dict(benchmark=bank['benchmark'],scene_counts=actual,
                actions=list(actions_for_benchmark(bank['benchmark'])),manifest_sha256=q['manifest_sha256'])


def write_configs(out, benchmark, shared=False, coefficient=1.):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    field=out/'field.yaml';reward=out/'reward.json'
    for p in (field,reward):require(not p.exists(),'Use a fresh configuration directory')
    field.write_text(yaml.safe_dump(field_config(benchmark,shared),sort_keys=False))
    reward.write_text(json.dumps(reward_config(benchmark,shared,coefficient),indent=2)+'\n')
    return field,reward
