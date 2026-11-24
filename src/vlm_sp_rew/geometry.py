"""Construct the VLM-free matched geometric field from native reachable-edge queries."""
import argparse, collections, hashlib, json
from pathlib import Path
import yaml
from vlm_sp_rew.field_contract import PaperParameters, value_from_hops, repair_unreachable_hops
from vlm_sp_rew.reward import require


def build(geometry, config):
    p=PaperParameters.from_config(config);points=geometry['points'];goal=len(points)
    require(len(geometry['goal_instances'])==1,'Fix one target instance')
    target=geometry['goal_instances'][0]['position'];adj=collections.defaultdict(list)
    nodes=[dict(id=i,position=x['position']) for i,x in enumerate(points)]+[dict(id=goal,position=target)]
    for a,b in geometry['edges']:
        require(0<=a<goal and 0<=b<goal,'Native sampled edge endpoint invalid');adj[a].append(b)
    for i,x in enumerate(points):
        import math
        if math.hypot(x['position']['x']-target['x'],x['position']['z']-target['z'])<=p.d_min:
            adj[i].append(goal);adj[goal].append(i)
    hops={goal:0};queue=collections.deque([goal])
    while queue:
        i=queue.popleft()
        for j in adj[i]:
            if j not in hops:hops[j]=hops[i]+1;queue.append(j)
    # No semantic room labels enter the geometric arm; unmatched components use
    # the same nearest-finite scalar inheritance, with one global region.
    hops,repairs=repair_unreachable_hops(nodes,hops,goal,{i:'geometry' for i in range(goal)})
    return dict(schema_version=1,source_arm='geo',frozen_before_training=True,synthetic=False,
        scene_sha256=geometry['scene_sha256'],goal_category=geometry['goal_category'],
        goal_instance_ids=[x['object_id'] for x in geometry['goal_instances']],
        source_integrity_qualified=True,method_contract='paper_v1',field_parameters=config['vlm_sp_rew_params'],paper_contract=dict(d_min=p.d_min),
        provenance=dict(native_geometry_sha256=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest(),repairs=repairs),
        points=[dict(id=i,position=x['position'],room_label='geometry',graph_hops=hops[i],
                     value=value_from_hops(x['goal_visible'],hops[i],p)) for i,x in enumerate(points)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('geometry','config','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();field=build(json.loads(Path(a.geometry).read_text()),yaml.safe_load(Path(a.config).read_text()))
    with Path(a.out).open('x') as f:json.dump(field,f,indent=2);f.write('\n')


if __name__=='__main__':main()
