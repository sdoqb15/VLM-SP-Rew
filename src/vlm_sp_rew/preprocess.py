"""Capture the fixed target and eight-view sampling cache from a native scene bank."""
from pathlib import Path
import argparse, datetime, hashlib, json, math, random, shutil, time
import numpy as np
from PIL import Image
from vlm_sp_rew.reward import load_episode_bank, file_bytes, require
from vlm_sp_rew.environment import make_local_controller
from vlm_sp_rew.protocol import field_config


def select_points(candidates, count, spacing, seed):
    order=list(candidates);random.Random(seed).shuffle(order);chosen=[]
    for pos in order:
        if all(math.hypot(pos['x']-q['x'],pos['z']-q['z'])>=spacing for q in chosen):
            chosen.append(dict(pos))
            if len(chosen)==count:break
    require(len(chosen)==count,'Not enough reachable points at the declared spacing; record failure without relaxing the protocol')
    return chosen


def capture(manifest, data_root, episode_id, controller_config, goal_reference, out, shared=False):
    q=load_episode_bank(manifest,data_root);bank=q['bank'];root=Path(q['root'])
    episode=next(e for e in bank['episodes'] if e['episode_id']==episode_id)
    require(len(episode['goal_instance_ids'])==1,'Fix exactly one target instance')
    params=field_config(bank['benchmark'],shared)['vlm_sp_rew_params']
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(out.parent).free>110*1024**3,'Preserve operational disk margin')
    out.mkdir(exist_ok=False);last_guard=time.monotonic()
    def guard():
        nonlocal last_guard
        if time.monotonic()-last_guard>=25:
            require(shutil.disk_usage(out).free>110*1024**3,'Stop capture before disk floor')
            last_guard=time.monotonic()
    spec=json.loads(Path(controller_config).read_text());controller=make_local_controller(spec,out)
    try:
        scene=json.loads(file_bytes(root,episode['scene']))
        if scene.get('backend')=='robothor':
            require(scene['engine_sha256']==spec['executable_sha256'],'Preset descriptor/engine mismatch')
            for asset in scene['assets']:file_bytes(root,asset)
            scene=scene['scene_id']
        event=controller.reset(scene=scene,randomSeed=episode['seed'])
        require(event.metadata['lastActionSuccess'],'Native scene reset failed')
        target=[x for x in event.metadata['objects'] if x['objectId']==episode['goal_instance_ids'][0]]
        require(len(target)==1 and target[0]['objectType']==episode['goal_category'],'Fixed instance missing or category mismatch')
        reachable=controller.step(action='GetReachablePositions')
        require(reachable.metadata['lastActionSuccess'],'Native reachable-point query failed')
        points=select_points(reachable.metadata['actionReturn'],params['N'],params['d_min'],episode['seed'])
        views=[];point_records=[]
        from vlm_sp_rew.annotation_cache import DIRECTIONS
        for position in points:
            guard();directory=out/('pos_%.6f_%.6f'%(position['x'],position['z']));directory.mkdir()
            visible=False
            for k,direction in enumerate(DIRECTIONS):
                event=controller.step(action='Teleport',position=position,rotation=dict(x=0,y=k*45,z=0),horizon=0,forceAction=False)
                require(event.metadata['lastActionSuccess'],'Eight-view teleport failed; no missing direction permitted')
                rgb=np.asarray(event.frame);require(rgb.shape==(300,300,3) and rgb.dtype==np.uint8,'Native 300x300 RGB required')
                image=directory/(direction+'_'+str(k*45)+'deg.jpg');Image.fromarray(rgb).save(image,quality=95)
                views.append(dict(path=image.relative_to(out).as_posix(),sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
                visible |= any(x['objectId']==target[0]['objectId'] and x['visible'] for x in event.metadata['objects'])
            point_records.append(dict(position=position,goal_visible=bool(visible),dir_name=directory.name))
        edges=[]
        for i,a in enumerate(points):
            for j,b in enumerate(points[:i]):
                guard();distance=math.hypot(a['x']-b['x'],a['z']-b['z'])
                if distance>1.5*params['d_min']:continue
                result=controller.step(action='GetShortestPathToPoint',position=a,target=b,allowedError=.05)
                if not result.metadata['lastActionSuccess']:continue
                corners=result.metadata['actionReturn']['corners']
                length=sum(math.dist([x[k] for k in ('x','y','z')],[y[k] for k in ('x','y','z')]) for x,y in zip(corners,corners[1:]))
                if length<=1.5*params['d_min']+.01:edges.extend([[i,j],[j,i]])
        capture_record=dict(scene_sha256=episode['scene']['sha256'],goal_category=episode['goal_category'],
            goal_instance_ids=episode['goal_instance_ids'],views=views,seed=episode['seed'],
            simulator_version=spec.get('simulator_version','ai2thor-5.0.0' if spec.get('backend')!='habitat' else spec['habitat_sim_version']),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_configuration=spec)
        record=out/'CAPTURE.json';record.write_text(json.dumps(capture_record,indent=2)+'\n')
        metadata=dict(scene_sha256=episode['scene']['sha256'],goal_category=episode['goal_category'],
            goal_reference_sha256=hashlib.sha256(Path(goal_reference).read_bytes()).hexdigest(),
            goal_instances=[dict(object_id=target[0]['objectId'],position=target[0]['position'])],
            capture_record=dict(path=record.name,sha256=hashlib.sha256(record.read_bytes()).hexdigest()))
        (out/'GOAL_METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (out/'GEOMETRY.json').write_text(json.dumps(dict(points=point_records,edges=edges,goal_instances=metadata['goal_instances'],
            scene_sha256=episode['scene']['sha256'],goal_category=episode['goal_category']),indent=2)+'\n')
    except BaseException as error:
        (out/'FAILURE.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error)))+'\n')
        raise
    finally:controller.stop()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('episode-manifest','episode-id','controller-config','goal-reference','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--data-root');p.add_argument('--shared',action='store_true');a=p.parse_args()
    capture(a.episode_manifest,a.data_root,a.episode_id,a.controller_config,a.goal_reference,a.out,a.shared)


if __name__=='__main__':main()
