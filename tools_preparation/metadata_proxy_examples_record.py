"""Transparent metadata-derived examples, NOT real VLM labels or nav evidence."""
from pathlib import Path
import hashlib,json,math,random,sys,datetime
R=Path('/mnt/chenxuda/nav_rl_research/publication-prep/access-2026-28868-minimal-revision-20261109/12_factual_metadata_submission_r6_20261114')
D=R/'reconstructed-scenes';O=R/'supplement/metadata_proxy_examples';assert not O.exists();O.mkdir()
sys.path.insert(0,'/run/nav_rl_research/access-2026-28868-code-recovery-20261114/r5-paper-contract-code')
from paper_contract_r5 import PaperParameters,value_from_hops
cfg=dict(N=50,d_min=1.0,tau_room=0.3,delta_vis=2,delta_novis=5,K_vis=3,alpha_k=[5,4,3],beta_k=[2,1,0,-1],beta_tail_step=-1,tau=3,w_hi=1,w_lo=.5,r_up=1,r_down=-1)
def inside(x,z,p):
 result=False;j=len(p)-1
 for i in range(len(p)):
  xi,zi=p[i]['x'],p[i]['z'];xj,zj=p[j]['x'],p[j]['z']
  if(zi>z)!=(zj>z)and x<(xj-xi)*(z-zi)/(zj-zi)+xi:result=not result
  j=i
 return result
def objects(a):
 for x in a:
  yield x
  yield from objects(x.get('children',[]))
rows=json.loads((D/'REPRESENTATIVE_REPLACEMENT_CASES.json').read_text())['cases'];summary=[]
for c in rows:
 if c['benchmark']!='ProcTHOR-10K':continue
 h=json.loads((D/c['house_metadata_path']).read_text());rooms=h['rooms'];rng=random.Random(288680+c['public_dataset_index']);points=[]
 # Deterministic geometry-only Poisson-style rejection; no fabricated rendered views.
 candidates=[]
 for room in rooms:
  poly=room['floorPolygon'];xs=[x['x']for x in poly];zs=[x['z']for x in poly]
  for _ in range(5000):
   x=rng.uniform(min(xs),max(xs));z=rng.uniform(min(zs),max(zs))
   if inside(x,z,poly):candidates.append((x,z,room))
 rng.shuffle(candidates)
 for x,z,room in candidates:
  if all(math.hypot(x-p['x'],z-p['z'])>=1.0 for p in points):points.append(dict(id='p%03d'%len(points),x=x,z=z,room_id=room['id'],room_type=room['roomType']))
  if len(points)==50:break
 assert len(points)>4
 target=c['goal_position'];target_room=next((r for r in rooms if inside(target['x'],target['z'],r['floorPolygon'])),None)
 edges=[];adj={p['id']:[] for p in points}
 for i,p in enumerate(points):
  for q in points[i+1:]:
   if p['room_id']!=q['room_id']or math.hypot(p['x']-q['x'],p['z']-q['z'])>1.5:continue
   poly=next(r['floorPolygon']for r in rooms if r['id']==p['room_id'])
   if not all(inside(p['x']+(q['x']-p['x'])*s/12,p['z']+(q['z']-p['z'])*s/12,poly)for s in range(1,12)):continue
   edges.append([p['id'],q['id']]);adj[p['id']].append(q['id']);adj[q['id']].append(p['id'])
 # Cross-room edges are deliberately not guessed: door collision/visibility needs rendering.
 target_room_id=target_room['id']if target_room else None
 anchor=min((p for p in points if p['room_id']==target_room_id),key=lambda p:math.hypot(p['x']-target['x'],p['z']-target['z']),default=min(points,key=lambda p:math.hypot(p['x']-target['x'],p['z']-target['z'])))
 hops={anchor['id']:1};todo=[anchor['id']]
 for pid in todo:
  for q in adj[pid]:
   if q not in hops:hops[q]=hops[pid]+1;todo.append(q)
 labels=[];field=[];allobjects=list(objects(h['objects']))
 for p in points:
  room=next(r for r in rooms if r['id']==p['room_id']);cats=sorted({x['id'].split('|')[0]for x in allobjects if x.get('position')and inside(x['position']['x'],x['position']['z'],room['floorPolygon'])})
  labels.append(dict(point_id=p['id'],position=dict(x=p['x'],y=h['metadata']['agent']['position']['y'],z=p['z']),room_type_metadata=room['roomType'],object_categories_in_room_metadata=cats,target_visible=None,navigability_by_direction=None,actual_image_paths=[],actual_VLM_response=None,model_snapshot=None,query_date=None,source='OFFICIAL_HOUSE_METADATA_ONLY_NOT_VLM_OR_RENDERED_VISIBILITY'))
  # Only calculate a numeric field where the same-room geometry supplies a finite path.
  d=hops.get(p['id']);visible_proxy=p['room_id']==target_room_id
  if d is None:value=None
  else:
   k=min(3,math.ceil(d/2))if visible_proxy else math.ceil(d/5)
   value=[5,4,3][k-1]if visible_proxy else 3-k
  field.append(dict(point_id=p['id'],geometry_hops=d,within_target_room_proxy=visible_proxy,value_proxy=value,usable_as_real_VLM_field=False))
 out=O/c['case_id'];out.mkdir()
 files={'point_metadata.json':dict(status='METADATA_DERIVED_NOT_VLM_ANNOTATIONS',points=labels),'geometry_graph_proxy.json':dict(status='WITHIN_ROOM_FLOOR_POLYGON_GRAPH_ONLY_NOT_NATIVE_NAVIGABILITY',edges=edges,anchor=anchor['id'],cross_room_edges_not_invented=True),'value_field_proxy.json':dict(status='ILLUSTRATIVE_GEOMETRY_AND_ROOM_PROXY_NOT_VLM_VALUE_FIELD',values=field,unknown_unreachable_values_not_fabricated=True,parameters=cfg),'provenance.json':dict(case=c,source_house_SHA256=c['house_metadata_SHA256'],source='Official ProcTHOR-10K metadata; no rendered frames, API calls or learned predictions.',caveat='Room membership is not visual recognition; same-room is not target visibility. Graph ignores furniture collisions and does not infer doors. These are transparent schema/preparation examples only, never empirical VLM/nav evidence.')}
 for n,x in files.items():(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 summary.append(dict(case_id=c['case_id'],scene_id=c['scene_id'],goal_instance_id=c['goal_instance_id'],sampled_points=len(points),edges=len(edges),finite_proxy_values=sum(x['value_proxy']is not None for x in field),files={n:hashlib.sha256((out/n).read_bytes()).hexdigest()for n in files}))
(O/'README.txt').write_text('METADATA-DERIVED PREPARATION EXAMPLES ONLY.\nThese four real ProcTHOR scenes have actual house/target/room/object metadata.\nNo RGB images, VLM calls, VLM labels or native navigability were produced.\nVisibility, eight-direction navigability and VLM response/model/date remain null.\nGraph edges only connect same-room samples through floor polygons.\nNumeric values are explicitly named geometry/room proxies; unreachable values stay null.\nReplace with actual rendered/cached VLM labels before an empirical paper claim.\n')
(O/'MANIFEST.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='METADATA_DERIVED_EXAMPLES_NOT_EXPERIMENTAL_VLM_DATA',cases=summary,total_points=sum(x['sampled_points']for x in summary),actual_VLM_calls=0,native_renderings=0,actual_navigation_results=0),indent=2)+'\n')
print(json.dumps(dict(cases=len(summary),total_points=sum(x['sampled_points']for x in summary),actual_VLM_calls=0,not_empirical_data=True)))
