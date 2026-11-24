"""Qualify native navigable SPL references against the fixed-instance stop rule."""
from pathlib import Path
import argparse, copy, hashlib, json, math, shutil, time
from vlm_sp_rew.reward import require, finite, load_episode_bank, file_bytes, requested_goal_state, path_length_metres
from vlm_sp_rew.environment import make_local_controller, actions_for_benchmark


def reference_controller_spec(policy_spec):
    """A separate continuous-navmesh oracle; never mutate the policy controller."""
    reference=copy.deepcopy(policy_spec)
    if reference.get('backend')!='habitat':
        reference['controller_kwargs']['snapToGrid']=False
    return reference


def native_reference(controller, episode, benchmark, call, native_goal_positions=None):
    """Minimum native path to a visible eligible stop pose in the declared region.

    THOR enumerates the native reachable lattice. Habitat consumes the native
    ObjectNav goal-viewpoint set, with its exact source recorded in the bank.
    """
    target=episode['goal_instance_ids'];require(len(target)==1,'One fixed target instance required')
    category=episode['goal_category'];start=episode['start_position'];yaw=episode['start_yaw_degrees']
    horizon=episode.get('start_horizon_degrees',0 if benchmark=='R2R-adapted' else 30)
    def state(event):
        return requested_goal_state(event.metadata['objects'],category,event.metadata['agent']['position'],target[0])
    def eligible(event):
        s=state(event);return s['eligible_visible_distance'] is not None and s['eligible_visible_distance']<=1.
    def teleport(pos,angle):
        e=call(action='Teleport',position=pos,rotation=dict(x=0,y=angle,z=0),horizon=horizon,forceAction=False)
        require(e.metadata['lastActionSuccess'],'Reference teleport failed')
        actual=e.metadata['agent']['position']
        require(math.dist([pos[k]for k in ('x','y','z')],[actual[k]for k in ('x','y','z')])<.02,'Reference pose differs from requested pose')
        return e
    initial=teleport(start,yaw);state(initial)
    if eligible(initial):
        return dict(shortest_path_m=0.,eligible_positions=1,reachable_candidates=0,endpoint=start,path_corners=[start],start_already_successful=True)
    target_object=next(x for x in initial.metadata['objects'] if x['objectId']==target[0])
    target_pos=target_object['position']
    if native_goal_positions is None:
        require(benchmark!='R2R-adapted','R2R-adapted requires the native ObjectNav goal-viewpoint source')
        query=call(action='GetReachablePositions');require(query.metadata['lastActionSuccess'],'Reachable lattice query failed')
        positions=query.metadata['actionReturn']
    else:positions=native_goal_positions
    require(isinstance(positions,list) and positions,'Native candidate stop positions required')
    unique={tuple(finite(pos[k],k)for k in ('x','y','z')):pos for pos in positions}
    candidates=[pos for _,pos in sorted(unique.items()) if math.dist([pos[k]for k in ('x','y','z')],[target_pos[k]for k in ('x','y','z')])<=1.]
    def visible_pose_at(pos):
        found=False;visible_pose=None
        for angle in [(yaw+90*i)%360 for i in range(4)]:
            event=teleport(pos,angle)
            if eligible(event):found=True;visible_pose=copy.deepcopy(event.metadata['agent']);break
            if 'LookUp' in actions_for_benchmark(benchmark):
                for direction in ('LookUp','LookDown'):
                    teleport(pos,angle)
                    for _ in range(8):
                        event=call(action=direction)
                        if not event.metadata['lastActionSuccess']:break
                        if eligible(event):found=True;visible_pose=copy.deepcopy(event.metadata['agent']);break
                    else:raise ValueError('Camera-range enumeration exceeded the declared finite bound')
                    if found:break
            if found:break
        return visible_pose if found else None
    valid=[];best=None;rejected=[]
    for pos in candidates:
        visible_pose=visible_pose_at(pos)
        found=visible_pose is not None
        if not found:continue
        verified_pos=pos;projected=False
        if math.dist([start[k]for k in ('x','y','z')],[pos[k]for k in ('x','y','z')])<.001:
            # Turning/tilting can reveal the goal without translating; SPL path cost is zero.
            corners=[dict(start)]
        else:
            path=call(action='GetShortestPathToPoint',position=start,target=pos,allowedError=.01)
            if not path.metadata['lastActionSuccess']:continue
            returned=path.metadata['actionReturn'];corners=returned['corners']
            if returned.get('status','PathComplete')!='PathComplete':
                rejected.append(dict(position=pos,reason='Native path is not complete',status=returned['status']));continue
        require(isinstance(corners,list)and corners,'Native shortest path has no corners')
        require(math.hypot(corners[0]['x']-start['x'],corners[0]['z']-start['z'])<=.02,
                'Native path endpoints differ from allocated start/stop region: '+json.dumps(dict(start=start,stop=pos,first=corners[0],last=corners[-1])))
        if math.hypot(corners[-1]['x']-pos['x'],corners[-1]['z']-pos['z'])>.02:
            # A native complete path can project a requested lattice point onto
            # the navmesh. Use that endpoint only after actual pose/visibility
            # checks, with the same body height and fixed-target success rule.
            verified_pos=dict(x=corners[-1]['x'],y=pos['y'],z=corners[-1]['z'])
            try:visible_pose=visible_pose_at(verified_pos)
            except ValueError as error:
                if not str(error).startswith(('Reference teleport failed','Reference pose differs')):raise
                rejected.append(dict(position=pos,projected_position=verified_pos,reason=str(error)));continue
            if visible_pose is None:
                rejected.append(dict(position=pos,projected_position=verified_pos,reason='Native path endpoint is outside visible fixed-target stop region'));continue
            projected=True
        length=path_length_metres(corners);valid.append(dict(position=verified_pos,requested_lattice_position=pos,length_m=length,visible_pose=visible_pose,native_projection_verified=projected))
        if best is None or length<best['shortest_path_m']:
            best=dict(shortest_path_m=length,endpoint=verified_pos,path_corners=corners,visible_pose=visible_pose,native_projection_verified=projected)
    require(best is not None,'No complete native path to visible stop region for the fixed target; rejected path endpoints: '+json.dumps(rejected))
    best.update(eligible_positions=len(valid),reachable_candidates=len(positions),candidate_stop_region=valid,rejected_path_candidates=rejected)
    return best


def qualify(manifest,data_root,controller_config,out,max_calls=10000,max_seconds=900):
    q=load_episode_bank(manifest,data_root);root=Path(q['root']);bank=copy.deepcopy(q['bank']);benchmark=bank['benchmark']
    require(max_calls>0 and max_seconds>0,'Set positive qualification resource limits')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(out.parent).free>110*1024**3,'Preserve operational storage margin')
    out.mkdir(exist_ok=False);spec=json.loads(Path(controller_config).read_text())
    oracle_spec=reference_controller_spec(spec)
    controller=make_local_controller(oracle_spec,out);began=time.monotonic();calls=0;receipt=out/'native_queries.jsonl'
    def call(**kw):
        nonlocal calls
        require(calls<max_calls and time.monotonic()-began<max_seconds,'Native reference qualification resource bound reached')
        require(shutil.disk_usage(out).free>110*1024**3,'Stop own writes before storage margin')
        calls+=1;event=controller.step(**kw)
        row=dict(index=calls,action=kw,success=bool(event.metadata['lastActionSuccess']),error=event.metadata.get('errorMessage',''))
        if kw['action']in ('GetReachablePositions','GetShortestPathToPoint'):row['action_return']=event.metadata.get('actionReturn')
        with receipt.open('a') as f:f.write(json.dumps(row)+'\n')
        return event
    try:
        for episode in bank['episodes']:
            if episode['split']=='train':continue
            require(calls<max_calls and time.monotonic()-began<max_seconds,'Native reference qualification resource bound reached')
            scene=json.loads(file_bytes(root,episode['scene']))
            if scene.get('backend')=='robothor':
                require(scene['engine_sha256']==spec['executable_sha256'],'Preset engine mismatch')
                for asset in scene['assets']:file_bytes(root,asset)
                scene=scene['scene_id']
            calls+=1;event=controller.reset(scene=scene,randomSeed=episode['seed'])
            require(event.metadata['lastActionSuccess'],'Reference scene reset failed')
            goal_positions=None;source=None
            if benchmark=='R2R-adapted':
                source=episode['native_objectnav_goal_region']
                region=json.loads(file_bytes(root,source))
                require(region['scene_sha256']==episode['scene']['sha256'] and
                    region['goal_instance_ids']==episode['goal_instance_ids'] and region['success_distance_m']==1.,'Native ObjectNav region identity/stop-rule mismatch')
                require(region['reference_kind']=='native_objectnav_goal_viewpoints' and region['coordinate_frame']=='Habitat_Y_up','Native ObjectNav viewpoints required; R2R route endpoints are not object goals')
                goal_positions=region['positions']
            result=native_reference(controller,episode,benchmark,call,goal_positions)
            episode['spl_reference']=dict(reference_kind='qualified_navigable_shortest_path',units='metres',success_rule_matched=True,
                shortest_path_m=result['shortest_path_m'],scene_sha256=episode['scene']['sha256'],goal_category=episode['goal_category'],
                goal_instance_ids=episode['goal_instance_ids'],start_position=episode['start_position'],start_yaw_degrees=episode['start_yaw_degrees'],
                start_horizon_degrees=episode.get('start_horizon_degrees',0 if benchmark=='R2R-adapted'else 30),
                success_distance_m=1.,native_goal_region_source=source,
                stop_region_convention='native ObjectNav annotated viewpoints and verified native path projections' if source else 'native reachable lattice and verified native path projections, allowed yaw and camera actions')
            (out/(episode['episode_id'].replace('/','_')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        bank['artifact_root']=str(root.resolve())
        (out/'bank.json').write_text(json.dumps(bank,indent=2)+'\n')
        (out/'COMPLETE.json').write_text(json.dumps(dict(input_manifest_sha256=q['manifest_sha256'],output_bank_sha256=hashlib.sha256((out/'bank.json').read_bytes()).hexdigest(),native_API_calls=calls,wall_seconds=time.monotonic()-began,controller_config=spec,reference_controller_config=oracle_spec,reference_metric='native continuous-navmesh path length to the verified visible fixed-target stop region'),indent=2)+'\n')
    except BaseException as error:
        (out/'FAILURE.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error),native_API_calls=calls,wall_seconds=time.monotonic()-began))+'\n');raise
    finally:controller.stop()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('episode-manifest','controller-config','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--data-root');p.add_argument('--max-api-calls',type=int,default=10000);p.add_argument('--max-seconds',type=int,default=900)
    a=p.parse_args();qualify(a.episode_manifest,a.data_root,a.controller_config,a.out,a.max_api_calls,a.max_seconds)


if __name__=='__main__':main()
