"""Shared manuscript reward and episode contracts; no simulator/model import or execution."""
from pathlib import Path
import json,hashlib,math,random,collections
ARMS=('env','geo','vlm')

def require(ok,message):
 if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def finite(x,name):
 require(type(x)in(int,float)and math.isfinite(x),name+' must be finite numeric');return float(x)

def file_bytes(root,spec):
 p=Path(spec['path']);require(not p.is_absolute()and'..'not in p.parts,'Use dataset-relative paths');root=Path(root).resolve();p=(root/p).resolve();require(p.is_relative_to(root),'Source escapes declared dataset root');raw=p.read_bytes();require(sha(raw)==spec['sha256'],'Artifact SHA mismatch: '+str(p));return raw

def floor_identity(scene):
 # Preset/reconstruction descriptors retain the native scene and asset identities.
 if scene.get('backend') in ('robothor','habitat'):
  require(isinstance(scene.get('scene_id'),str) and scene['scene_id'],'Native scene ID required')
  assets=scene.get('assets',[])
  require(assets and all(isinstance(a.get('sha256'),str)and len(a['sha256'])==64 for a in assets),'Pinned native scene/engine assets required')
  if scene['backend']=='habitat':require(scene.get('coordinate_frame')=='Habitat_Y_up','Habitat-native coordinates required; do not reuse raw R2R Z-up poses')
  if scene['backend']=='robothor':
   require(isinstance(scene.get('engine_sha256'),str)and len(scene['engine_sha256'])==64,'Pinned preset engine hash required')
   ident=dict(backend='robothor',scene_id=scene['scene_id'],engine_sha256=scene['engine_sha256'])
  else:
   geometry=[a['sha256']for a in assets if a['path']==scene['scene_id']]
   require(len(geometry)==1,'Native rendered geometry must be hash-pinned')
   ident=dict(backend='habitat',geometry_sha256=geometry[0])
  return sha(json.dumps(ident,sort_keys=True).encode())
 polygons=[]
 for room in scene.get('rooms',[]):
  points=room.get('floorPolygon',[])
  if points:polygons.append(sorted((round(finite(p['x'],'floor.x'),6),round(finite(p['z'],'floor.z'),6))for p in points))
 require(polygons,'Scene cache lacks actual floor geometry; other benchmarks need an explicitly qualified adapter')
 return sha(json.dumps(sorted(polygons),separators=(',',':')).encode())

def load_episode_bank(path,data_root=None):
 path=Path(path);raw=path.read_bytes();bank=json.loads(raw);root=Path(data_root or path.parent);require(bank.get('schema_version')==1,'schema_version1 required');episodes=bank.get('episodes');require(isinstance(episodes,list)and episodes,'Real frozen episodes required; no empty or placeholder schedule');ids=set();split_scenes=collections.defaultdict(set);vocab=bank.get('goal_vocabulary');require(isinstance(vocab,list)and vocab==sorted(set(vocab))and all(isinstance(x,str)and x for x in vocab),'One shared sorted goal vocabulary required')
 for e in episodes:
  ident=e.get('episode_id');require(isinstance(ident,str)and ident not in ids,'Unique episode_id required');ids.add(ident);require(e.get('split')in['train','dev','test'],'Declare train/dev/test purpose separately from official source split');scene=file_bytes(root,e['scene']);require(e.get('scene_identity_sha256')and len(e['scene_identity_sha256'])==64,'Record scene identity/floor geometry SHA; rotated/mirrored identity limits must be stated');require(e['scene_identity_sha256']==floor_identity(json.loads(scene)),'Scene identity does not match cached floor geometry');split_scenes[e['split']].add(e['scene_identity_sha256']);require(e['goal_category']in vocab,'Goal absent from frozen vocabulary');require(isinstance(e.get('goal_instance_ids'),list)and e['goal_instance_ids'],'Retain actual simulator goal instance IDs');require(type(e.get('seed'))is int,'Actual episode seed required');require(set(e.get('start_position',{}))=={'x','y','z'},'Fixed x/y/z start required')
  for k,v in e['start_position'].items():finite(v,'start.'+k)
  finite(e.get('start_yaw_degrees'),'start yaw')
 for a,b in [('train','dev'),('train','test'),('dev','test')]:require(not(split_scenes[a]&split_scenes[b]),'Scene leakage between '+a+' and '+b)
 return dict(bank=bank,root=str(root),manifest_sha256=sha(raw),goal_vocabulary=vocab)

def goal_one_hot(goal,vocabulary):
 require(goal in vocabulary,'Unknown requested goal');return [float(x==goal)for x in vocabulary]

class EpisodeSchedule:
 def __init__(self,qualified,split,seed=0,shuffle=False):
  self.items=[e for e in qualified['bank']['episodes']if e['split']==split];require(self.items,'No episodes in requested split');self.seed=seed;self.shuffle=shuffle;self.epoch=-1;self.position=0;self.order=[]
 def next(self):
  if not self.order or self.position==len(self.order):
   self.epoch+=1;self.position=0;self.order=sorted(self.items,key=lambda e:e['episode_id'])
   if self.shuffle:random.Random(self.seed+self.epoch).shuffle(self.order)
  e=self.order[self.position];self.position+=1;return dict(e)

def bind_field(episode,arm,root):
 require(arm in ARMS,'Unsupported primary arm')
 if arm=='env':return None
 spec=episode.get('fields',{}).get(arm);require(spec is not None,'Missing per-scene field for '+arm);raw=file_bytes(root,spec);field=json.loads(raw);require(field.get('scene_sha256')==episode['scene']['sha256']and field.get('goal_category')==episode['goal_category'],'Field belongs to a different scene or goal');require(field.get('source_arm')==arm,'Incorrect field source/control label');require(field.get('frozen_before_training')is True,'Freeze field before vlm_sp_rew.train');require(not field.get('synthetic',False),'Synthetic engineering field cannot enter a real experiment');return field

class SharedReward:
 """The manuscript's common environment reward plus its regional sign reward."""
 def __init__(self,arm,config):
  require(arm in ARMS,'Unknown arm');self.arm=arm;self.config=dict(config)
  for k in ['step_penalty','success_reward','distance_coefficient','success_distance_m','lambda_shaping','tau','w_hi','w_lo','r_up','r_down']:finite(config.get(k),k)
  for k,expected in [('step_penalty',-1.),('success_reward',100.),('distance_coefficient',1.),('success_distance_m',1.)]:require(config[k]==expected,'Manuscript constant differs: '+k)
  require(not config.get('distance_clipping_enabled',False)and not config.get('visibility_suppression',False)and config.get('horizon_penalty',0)==0,'Paper reward has no clipping, visibility suppression or extra horizon penalty')
  require(config['lambda_shaping']>=0,'Nonnegative shaping coefficient required');self.previous_distance=None;self.previous_value=None
 def reset(self,distance,field_value=None):
  self.previous_distance=finite(distance,'initial fixed-target distance');require(self.previous_distance>=0,'Distance cannot be negative');self.previous_value=None if self.arm=='env'else finite(field_value,'initial field value')
 def step(self,action,eligible_visible_distance,distance,field_value,step_count,max_steps):
  c=self.config;distance=finite(distance,'current fixed-target distance');require(distance>=0 and self.previous_distance is not None,'Reset before computing a nonnegative distance');eligible=eligible_visible_distance is not None and finite(eligible_visible_distance,'visible fixed-target distance')<=c['success_distance_m'];success=action=='Stop'and eligible;last=action=='Stop'or step_count>=max_steps;terminal=action=='Stop'
  progress=c['distance_coefficient']*(self.previous_distance-distance)
  reward=c['step_penalty']+progress+c['success_reward']*float(success);field_reward=0.
  if self.arm!='env':
   value=finite(field_value,'current field value');require(self.previous_value is not None,'Initialize the field at reset');weight=c['w_hi']if max(value,self.previous_value)>=c['tau']else c['w_lo'];direction=c['r_up']if value>self.previous_value else c['r_down']if value<self.previous_value else 0.;field_reward=c['lambda_shaping']*weight*direction;reward+=field_reward;self.previous_value=value
  self.previous_distance=distance;return dict(reward=reward,shared_distance_reward=progress,field_reward=field_reward,success=success,is_last=last,is_terminal=terminal,truncated=last and not terminal)

def requested_goal_state(objects,goal,position,instance_id=None):
 candidates=[o for o in objects if o['objectType']==goal and(instance_id is None or o['objectId']==instance_id)];require(candidates,'Frozen requested goal absent; record failed allocation, never switch target');distance=lambda o:math.sqrt(sum((o['position'].get(a,0)-position.get(a,0))**2 for a in('x','y','z')));return dict(distance=min(distance(o)for o in candidates),eligible_visible_distance=min((distance(o)for o in candidates if o.get('visible',False)),default=None),goal_instance_ids=sorted(o['objectId']for o in candidates))

def path_length_metres(poses):
 require(poses,'Initial pose required');return sum(math.sqrt(sum((b.get(k,0)-a.get(k,0))**2 for k in('x','y','z')))for a,b in zip(poses,poses[1:]))

def qualified_spl(success,reference):
 require(type(success)is bool,'Boolean success required');require(reference.get('units')=='metres'and reference.get('success_rule_matched')is True and reference.get('reference_kind')=='qualified_navigable_shortest_path','No SPL from Euclidean distance or heuristic step counts');l=finite(reference.get('shortest_path_m'),'reference');p=finite(reference.get('executed_path_m'),'executed');require(l>=0 and p>=0,'Nonnegative path lengths required');return float(success)*(1. if max(l,p)==0 else l/max(l,p))
