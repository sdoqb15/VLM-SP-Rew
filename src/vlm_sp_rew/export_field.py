"""Export annotation graphs and semantic value fields from an eight-view cache."""
from pathlib import Path
import argparse, collections, hashlib, json, math, shutil, time
import yaml
from vlm_sp_rew.annotation_cache import audit_existing_cache
from vlm_sp_rew.reward import require, finite, sha, floor_identity, file_bytes
from vlm_sp_rew.graph import build_navigation_graph
from vlm_sp_rew.rooms import cluster_rooms_by_similarity, update_gpt4o_analysis_file
from vlm_sp_rew.paths import (
    add_goal_node_and_edges, build_adjacency_list, bfs_shortest_path,
)
from vlm_sp_rew.field_contract import PaperParameters, hops_from_paths, repair_unreachable_hops, value_from_hops

def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def export_cache(source_cache, scene_path, goal_metadata, goal_reference, config_path,
                 out, expected_snapshot=None, allow_parsed_only=False,
                 repair_all_unreachable=False, plots=False, paper_contract=True):
    source_cache, out = Path(source_cache).resolve(), Path(out).resolve()
    require(not out.exists(), 'Use a fresh output directory; never overwrite prior cache/results')
    require(out != source_cache and not out.is_relative_to(source_cache), 'Output must be outside immutable input cache')
    out.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(out.parent).free >= 110*1024**3, 'Preserve disk planning margin')
    last_guard = time.monotonic()
    def guard():
        nonlocal last_guard
        if time.monotonic()-last_guard >= 25:
            require(shutil.disk_usage(out).free >= 110*1024**3, 'Stop own writes before storage floor')
            last_guard = time.monotonic()
    # Strict input validation precedes derived writes.
    audit = audit_existing_cache(source_cache, goal_reference, expected_snapshot, allow_parsed_only)
    scene_raw = Path(scene_path).read_bytes()
    scene = json.loads(scene_raw)
    identity = floor_identity(scene)
    metadata_raw = Path(goal_metadata).read_bytes()
    metadata = json.loads(metadata_raw)
    require(metadata.get('scene_sha256') == sha(scene_raw), 'Goal metadata belongs to another scene')
    require(metadata.get('goal_reference_sha256') == audit['goal_reference']['sha256'],
            'Goal-reference image differs from frozen goal metadata')
    category = metadata.get('goal_category')
    require(isinstance(category, str) and category, 'Explicit requested category required')
    instances = metadata.get('goal_instances', [])
    require(instances and len({o['object_id'] for o in instances}) == len(instances),
            'Actual unique goal instance IDs/positions required')
    instances = sorted(instances, key=lambda item:item['object_id'])
    for item in instances:
        for axis in ('x','y','z'):
            finite(item['position'][axis], 'goal.position.'+axis)
    # Optional source capture record must match all actual image bytes, scene, goal and instances.
    capture_qualified = False
    if metadata.get('capture_record'):
        capture = json.loads(file_bytes(source_cache, metadata['capture_record']))
        require(capture.get('scene_sha256') == sha(scene_raw) and capture.get('goal_category') == category,
                'Capture identity differs from scene or goal')
        require(sorted(capture.get('goal_instance_ids',[])) == sorted(o['object_id']for o in instances),
                'Capture target instances mismatch')
        expected = {v['path']:v['sha256']for point in audit['points']for v in point['views']}
        actual = {v['path']:v['sha256']for v in capture.get('views',[])}
        require(actual == expected and len(actual)==len(capture['views']),
                'Capture/image inventory must be exact')
        require(capture.get('simulator_version') and capture.get('UTC') and
                type(capture.get('seed')) is int and capture.get('controller_configuration'),
                'Capture must retain actual simulator, time, seed and controller configuration')
        capture_qualified = True
    config_raw = Path(config_path).read_bytes()
    config = yaml.safe_load(config_raw)
    params = config['vlm_sp_rew_params']
    require(paper_contract, 'This exporter implements the current paper contract')
    paper = PaperParameters.from_config(config)
    if paper:
        require(len(instances)==1, 'Paper fixed-goal mode requires one explicit goal instance; no implicit first/all-instance change')
        require(not repair_all_unreachable, 'The paper defines its original all-unreachable inheritance; do not mix the legacy repair flag')
    graph_radius = float(config['stage_defaults']['graph_max_distance'])
    anchor_radius = float(config['stage_defaults']['value_min_distance'])
    require(graph_radius>0 and anchor_radius>0, 'Positive graph and anchor radii required')
    if paper:
        require(abs(graph_radius-1.5*paper.d_min)<1e-10, 'Paper graph radius must match original1.5*d_min')
    out.mkdir(exist_ok=False)
    dump(out/'INPUT_AUDIT.json', audit)
    dump(out/'GOAL_METADATA_COPY.json', metadata)
    derived = out/'derived_annotations'
    derived.mkdir()
    try:
        grouped = collections.defaultdict(list)
        for point in audit['points']:
            guard()
            directory = derived/point['dir_name'];directory.mkdir()
            (directory/'gpt4o_analysis.txt').write_bytes((source_cache/point['dir_name']/'gpt4o_analysis.txt').read_bytes())
            grouped[point['parsed']['room_label']].append(point)
        refined_rooms = {}
        for room, points in sorted(grouped.items()):
            positions = [dict(objects=p['parsed']['visible_objects'],
                              position=(p['position']['x'],p['position']['z']))for p in points]
            actual_labels = cluster_rooms_by_similarity(
                positions, similarity_threshold=params['tau_room'], distance_weight=0.0 if paper else .3)
            labels = {label:index+1 for index,label in enumerate(sorted(set(map(int,actual_labels))))}
            for point,label in zip(points,actual_labels):
                refined = room+'_'+str(labels[int(label)])
                refined_rooms[point['dir_name']] = refined
                update_gpt4o_analysis_file(derived/point['dir_name']/'gpt4o_analysis.txt', refined)
        nodes, edges = build_navigation_graph(derived, graph_radius)
        require(len(nodes)>=2, 'At least two actually captured points required')
        require(len(nodes)==len(audit['points']), 'No captured point may silently disappear')
        for index,node in enumerate(nodes):
            node['id'] = index
            x,z = node['position']
            node['position'] = {'x':float(x),'z':float(z)}
        target = instances[0]['position']
        nodes, edges, goal_id = add_goal_node_and_edges(
            nodes, edges, (target['x'],target['z']), anchor_radius)
        nodes[goal_id]['dir_name']='GOAL_SET_VIRTUAL'
        # The paper fixes one explicitly identified object instance for each run.
        connected = {(edge['from'],edge['to'])for edge in edges}
        for node in nodes[:goal_id]:
            distances=[math.hypot(node['position']['x']-o['position']['x'],
                                  node['position']['z']-o['position']['z'])for o in instances]
            distance = min(distances)
            if distance<=anchor_radius:
                for a,b,direction in [(node['id'],goal_id,'TO_GOAL_SET'),(goal_id,node['id'],'FROM_GOAL_SET')]:
                    if (a,b)not in connected:
                        edges.append(dict(**{'from':a,'to':b},distance=distance,direction=direction))
                        connected.add((a,b))
        edges = sorted(edges,key=lambda e:(e['from'],e['to']))
        adjacency = build_adjacency_list(nodes, edges)
        paths = {n['id']:bfs_shortest_path(adjacency,n['id'],goal_id)for n in nodes}
        positions = hops_from_paths(paths, goal_id)
        repair = [n['id'] for n in nodes[:goal_id] if n['id'] not in positions]
        room_ids = {n['id']:refined_rooms[n['dir_name']] for n in nodes[:goal_id]}
        positions, paper_repairs = repair_unreachable_hops(nodes, positions, goal_id, room_ids)
        graph = dict(nodes=[{k:v for k,v in n.items()if k!='analysis_file'}for n in nodes],
                     edges=edges,goal_node_id=goal_id,goal_instances=instances,
                     graph_edges_are_not_qualified_simulator_shortest_paths=True)
        dump(out/'GRAPH.json',graph)
        shortest = dict(goal_node_id=goal_id,
                        path_position_definition='Manuscript unweighted graph hops; virtual goal0 excluded from field',
                        average_positions={str(i):dict(dir_name=nodes[i]['dir_name'],position=float(p))
                                           for i,p in sorted(positions.items())},
                        original_paths={str(i):path for i,path in paths.items()},
                        repairs=paper_repairs)
        dump(out/'PATH_POSITIONS.json',shortest)
        visible = {p['dir_name']:p['parsed']['goal_visible'] for p in audit['points']}
        values = {n['id']:dict(value=value_from_hops(visible[n['dir_name']], positions[n['id']], paper)) for n in nodes[:goal_id]}
        require(len(values)==goal_id,'Every sampled point needs a finite value')
        synthetic = bool(metadata.get('synthetic',False))
        qualified = audit['source_integrity_qualified'] and capture_qualified and not synthetic
        field = dict(schema_version=1,source_arm='vlm',scene_sha256=sha(scene_raw),
                     scene_identity_sha256=identity,goal_category=category,
                     annotation_family='LLaVA-13B' if any('llava' in m.lower() for m in audit['requested_model_snapshots']) else 'GPT-4o',
                     goal_instance_ids=[o['object_id']for o in instances],
                     frozen_before_training=True,synthetic=synthetic,
                     source_integrity_qualified=qualified,
                     annotation_record_integrity_qualified=audit['source_integrity_qualified'],
                     capture_inventory_qualified=capture_qualified,
                     points=[dict(id=i,position=nodes[i]['position'],value=float(values[i]['value']),
                                  room_label=refined_rooms[nodes[i]['dir_name']],
                                  graph_path_position=float(positions[i]),
                                  position_repaired=i in repair)
                             for i in range(goal_id)],
                     provenance=dict(scene_sha256=sha(scene_raw),goal_metadata_sha256=sha(metadata_raw),
                                     config_sha256=sha(config_raw),input_cache_sha256=sha(json.dumps(audit,sort_keys=True).encode()),
                                     goal_anchor_policy='one_explicit_fixed_goal_instance',
                                     data_source='annotation_cache'))
        if paper:
            field['method_contract']='paper_v1'
            field['field_parameters']=params
            field['paper_contract']=dict(d_min=paper.d_min, graph_distance='unweighted hops, NOT hops+1',
                value_rule='original uniform alpha/beta distance-band equation',
                room_refinement='Jaccard only, average link, cut1-tau_room within original room labels',
                repaired_values_do_not_certify_graph_connections=True,
                validation_scope='field_export')
            for point in field['points']:
                point['graph_hops']=point['graph_path_position']
        dump(out/'FIELD.json',field)
        if plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
            for edge in edges:
                if edge['from']<edge['to'] and edge['to']<goal_id:
                    a,b=nodes[edge['from']]['position'],nodes[edge['to']]['position']
                    axes[0].plot([a['x'],b['x']],[a['z'],b['z']],color='.7',linewidth=.8)
            for ax in axes:
                ax.scatter([o['position']['x']for o in instances],[o['position']['z']for o in instances],
                           marker='*',s=100,color='red',label='declared goal instances')
                ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('z (m)')
            axes[0].scatter([n['position']['x']for n in nodes[:goal_id]],
                            [n['position']['z']for n in nodes[:goal_id]],s=10)
            scatter=axes[1].scatter([p['position']['x']for p in field['points']],
                                   [p['position']['z']for p in field['points']],
                                   c=[p['value']for p in field['points']],cmap='viridis')
            fig.colorbar(scatter,ax=axes[1],label='sign-shaping field value')
            fig.suptitle('Test data: graph and value field' if synthetic
                         else 'Annotation graph and semantic value field')
            fig.savefig(out/'GRAPH_AND_FIELD.png',dpi=160);fig.savefig(out/'GRAPH_AND_FIELD.pdf')
            plt.close(fig)
        # Fresh input bytes must still match every pre-export source hash.
        for item in audit['all_source_files']:
            guard()
            require(sha((source_cache/item['path']).read_bytes())==item['sha256'],'Input cache changed during export')
        require(sha(Path(scene_path).read_bytes())==sha(scene_raw), 'Scene changed during export')
        require(sha(Path(goal_metadata).read_bytes())==sha(metadata_raw), 'Goal metadata changed during export')
        require(sha(Path(config_path).read_bytes())==sha(config_raw), 'Config changed during export')
        require(sha(Path(goal_reference).read_bytes())==audit['goal_reference']['sha256'], 'Reference changed during export')
        if metadata.get('capture_record'):
            file_bytes(source_cache,metadata['capture_record'])
        files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
               for p in sorted(out.rglob('*'))if p.is_file()]
        complete=dict(status='COMPLETE',source_integrity_qualified=qualified,
                      artifact_type='annotation_graph_value_field_export',synthetic=synthetic,files=files,
                      no_API_simulator_or_training_calls=True,
                      original_source_cache_not_modified=True)
        dump(out/'COMPLETE.json',complete)
        return field,complete
    except BaseException as error:
        dump(out/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),
                                    no_missing_data_fabricated=True))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source-cache','scene','goal-metadata','goal-reference','out']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs/paper.yaml'))
    parser.add_argument('--expected-snapshot')
    parser.add_argument('--allow-parsed-only',action='store_true',help='Legacy diagnostic only: field remains unqualified')
    parser.add_argument('--repair-all-unreachable',action='store_true',help='Explicit revised repair of non-isolated disconnected components')
    parser.add_argument('--plots',action='store_true')
    parser.add_argument('--paper-contract',action='store_true',default=True,help='Use the manuscript graph, value and room equations')
    args=parser.parse_args()
    field,complete=export_cache(args.source_cache,args.scene,args.goal_metadata,args.goal_reference,
        args.config,args.out,args.expected_snapshot,args.allow_parsed_only,args.repair_all_unreachable,args.plots,args.paper_contract)
    print(json.dumps(dict(points=len(field['points']),source_integrity_qualified=complete['source_integrity_qualified'],
                          no_training_or_API_calls=True)))
if __name__=='__main__':main()
