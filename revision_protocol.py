"""Shared contracts for the revised study; no simulator/model import or execution."""
from pathlib import Path
import json,hashlib,math,random,collections
ARMS=('env','geo','geo_vis','vlm')

def require(ok,message):
 if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def finite(x,name):
 require(type(x)in(int,float)and math.isfinite(x),name+' must be finite numeric');return float(x)

def file_bytes(root,spec):
 p=Path(spec['path']);require(not p.is_absolute()and'..'not in p.parts,'Use dataset-relative paths');root=Path(root).resolve();p=(root/p).resolve();require(p.is_relative_to(root),'Source escapes declared dataset root');raw=p.read_bytes();require(sha(raw)==spec['sha256'],'Artifact SHA mismatch: '+str(p));return raw

def floor_identity(scene):
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
 spec=episode.get('fields',{}).get(arm);require(spec is not None,'Missing per-scene field for '+arm);raw=file_bytes(root,spec);field=json.loads(raw);require(field.get('scene_sha256')==episode['scene']['sha256']and field.get('goal_category')==episode['goal_category'],'Field belongs to a different scene or goal');require(field.get('source_arm')==arm,'Incorrect field source/control label');require(field.get('frozen_before_training')is True,'Freeze field before training');require(not field.get('synthetic',False),'Synthetic engineering field cannot enter a real experiment');return field

class SharedReward:
 def __init__(self,arm,config):
  require(arm in ARMS,'Unknown arm');self.arm=arm;self.config=dict(config)
  for k in ['step_penalty','success_reward','distance_progress_clip','horizon_penalty','success_distance_m','lambda_shaping','tau','w_hi','w_lo','r_up','r_down']:finite(config.get(k),k)
  require(config['distance_progress_clip']>=0 and config['success_distance_m']>0,'Valid distance limits required');self.previous_distance=None;self.previous_value=None
 def reset(self,distance,field_value=None):
  self.previous_distance=finite(distance,'initial requested-goal distance');require(self.previous_distance>=0,'Distance cannot be negative');self.previous_value=None if self.arm=='env'else finite(field_value,'initial field value')
 def step(self,action,eligible_visible_distance,distance,field_value,step_count,max_steps):
  c=self.config;distance=finite(distance,'current requested-goal distance');require(distance>=0,'Distance cannot be negative');eligible=eligible_visible_distance is not None and finite(eligible_visible_distance,'visible eligible distance')<=c['success_distance_m'];success=action=='Stop'and eligible;last=action=='Stop'or step_count>=max_steps;terminal=action=='Stop';reward=c['step_penalty'];progress=0.
  if self.previous_distance is not None and not eligible:
   cap=c['distance_progress_clip'];progress=max(-cap,min(cap,self.previous_distance-distance));reward+=progress
  if success:reward+=c['success_reward']
  elif action!='Stop'and step_count>=max_steps:reward+=c['horizon_penalty']
  field_reward=0.
  if self.arm!='env':
   value=finite(field_value,'current field value');require(self.previous_value is not None,'Initialize field value at reset to implement equation from first transition');weight=c['w_hi']if max(value,self.previous_value)>=c['tau']else c['w_lo'];direction=c['r_up']if value>self.previous_value else c['r_down']if value<self.previous_value else 0.;field_reward=c['lambda_shaping']*weight*direction;reward+=field_reward;self.previous_value=value
  self.previous_distance=distance;return dict(reward=reward,shared_distance_reward=progress,field_reward=field_reward,success=success,is_last=last,is_terminal=terminal,truncated=last and not terminal)

def requested_goal_state(objects,goal,position):
 candidates=[o for o in objects if o['objectType']==goal];require(candidates,'Frozen requested goal absent; record failed allocation, never switch category');distance=lambda o:math.hypot(o['position']['x']-position['x'],o['position']['z']-position['z']);return dict(distance=min(distance(o)for o in candidates),eligible_visible_distance=min((distance(o)for o in candidates if o.get('visible',False)),default=None),goal_instance_ids=sorted(o['objectId']for o in candidates))

def path_length_metres(poses):
 require(poses,'Initial pose required');return sum(math.hypot(b['x']-a['x'],b['z']-a['z'])for a,b in zip(poses,poses[1:]))

def qualified_spl(success,reference):
 require(type(success)is bool,'Boolean success required');require(reference.get('units')=='metres'and reference.get('success_rule_matched')is True and reference.get('reference_kind')=='qualified_navigable_shortest_path','No SPL from Euclidean distance or heuristic step counts');l=finite(reference.get('shortest_path_m'),'reference');p=finite(reference.get('executed_path_m'),'executed');require(l>=0 and p>=0,'Nonnegative path lengths required');return float(success)*(1. if max(l,p)==0 else l/max(l,p))
