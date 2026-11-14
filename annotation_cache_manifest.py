"""Freeze actual existing eight-view annotation inputs; never invent missing records."""
from pathlib import Path
import json,hashlib,collections,math,re
from PIL import Image
DIRECTIONS=('N','NE','E','SE','S','SW','W','NW')

def require(ok,message):
 if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def entry(path,root):
 path=Path(path);raw=path.read_bytes();return dict(path=path.relative_to(root).as_posix(),bytes=len(raw),sha256=sha(raw))

def parse_annotation(raw):
 keys={}
 for line in raw.splitlines():
  if ':'not in line:continue
  k,v=line.split(':',1);k=k.strip();v=v.strip()
  if k in ['Objects','Room','Target_visible']or k.startswith('Navigable_'):
   require(k not in keys,'Duplicate annotation field '+k);keys[k]=v
 needed={'Objects','Room','Target_visible'}|{'Navigable_'+x for x in DIRECTIONS};require(needed<=set(keys),'Missing annotation fields '+str(sorted(needed-set(keys))));require(bool(keys['Room']),'Missing room label')
 for k in ['Target_visible']+['Navigable_'+x for x in DIRECTIONS]:require(keys[k].lower()in['yes','no'],'Nonbinary annotation '+k)
 return dict(room_label=keys['Room'],visible_objects=[x.strip()for x in keys['Objects'].split(',')if x.strip()],goal_visible=keys['Target_visible'].lower()=='yes',navigability={d:keys['Navigable_'+d].lower()=='yes'for d in DIRECTIONS})

def audit_existing_cache(root,goal_reference,expected_snapshot=None,allow_parsed_only=False):
 root=Path(root).resolve();goal_reference=Path(goal_reference).resolve();goal_raw=goal_reference.read_bytes();goal_sha=sha(goal_raw);points=[];all_files=[];snapshots=set();partial=False;provider_records=[]
 dirs=sorted(p for p in root.glob('pos_*')if p.is_dir());require(dirs,'No actual point directories; no complete dataset can be claimed')
 for p in dirs:
  parts=p.name.split('_');require(len(parts)==3,'Point directory must be pos_X_Z');x,z=map(float,parts[1:]);require(math.isfinite(x)and math.isfinite(z),'Nonfinite point coordinate');views=[]
  for k,d in enumerate(DIRECTIONS):
   f=p/(d+'_'+str(k*45)+'deg.jpg');require(f.is_file(),'Missing actual view '+str(f));raw=f.read_bytes()
   with Image.open(f)as image:require(image.mode=='RGB'and image.width>0 and image.height>0,'RGB image required');image.load();shape=[image.height,image.width,3]
   item=entry(f,root);item.update(heading=k*45,shape=shape);views.append(item);all_files.append(item)
  text=p/'gpt4o_analysis.txt';require(text.is_file(),'Missing actual annotation '+str(text));parsed=parse_annotation(text.read_text());all_files.append(entry(text,root));stage_records=[]
  for kind in ['objects_room_visibility','navigability']:
   complete=sorted((p/'api_audit').glob(kind+'_*'+'/COMPLETE.json'))
   if not complete:
    require(allow_parsed_only,'No immutable provider provenance for '+str(p)+' '+kind);partial=True;continue
   require(len(complete)==1,'Multiple successful calls: freeze an explicit annotation selection rather than silently choosing labels');folder=complete[0].parent;request=json.loads((folder/'REQUEST.json').read_text());done=json.loads((folder/'COMPLETE.json').read_text());provider=json.loads((folder/'PROVIDER_RESPONSE.json').read_text());response=(folder/'RESPONSE_TEXT.txt').read_text();require(sha(response.encode())==done['text_sha256'],'Response text was modified');require(provider['choices'][0]['message']['content']==response,'Text differs from frozen provider output');model=request['requested_model_snapshot'];snapshots.add(model);require(re.search(r'\d{4}-\d{2}-\d{2}',model),'Dated annotation model snapshot required for qualified release');require(request.get('kind')==kind,'Request kind mismatch')
   if expected_snapshot:require(model==expected_snapshot,'Unexpected annotation model snapshot')
   require(done['requested_model']==model and done['returned_model']==provider.get('model'),'Provider model provenance mismatch');expected=[v['sha256']for v in views]+([goal_sha]if kind=='objects_room_visibility'else[]);require([x['sha256']for x in request['image_payload_hashes']]==expected,'Input image/reference hashes mismatch')
   required={'Objects','Room','Target_visible'}if kind=='objects_room_visibility'else{'Navigable_'+d for d in DIRECTIONS};require(required<=set(parse_lines(response)),'Raw provider response lacks required fields')
   require(done.get('provider_response_id')==provider.get('id')and done.get('usage')==provider.get('usage'),'Provider ID/usage differs from completion metadata')
   for k,v in parse_lines(response).items():
    if k in ['Objects','Room','Target_visible']or k.startswith('Navigable_'):require(v==parse_lines(text.read_text()).get(k),'Derived parsed text differs from raw provider response: '+k)
   files=[entry(f,root)for f in [folder/'REQUEST.json',folder/'PROVIDER_RESPONSE.json',folder/'RESPONSE_TEXT.txt',folder/'COMPLETE.json']];all_files.extend(files);record=dict(point=p.name,kind=kind,requested_snapshot=model,returned_model=provider.get('model'),provider_id=provider.get('id'),request_UTC=request['UTC'],usage=provider.get('usage'),files=files);stage_records.append(record);provider_records.append(record)
  points.append(dict(dir_name=p.name,position=dict(x=x,z=z),parsed=parsed,views=views,provider_records=stage_records))
 require(len(snapshots)<=1,'Multiple requested model snapshots in one frozen field')
 return dict(schema_version=1,source_cache=str(root),goal_reference=dict(path=str(goal_reference),bytes=len(goal_raw),sha256=goal_sha),points=points,all_source_files=all_files,provider_records=provider_records,requested_model_snapshots=sorted(snapshots),source_integrity_qualified=not partial,parsed_only_legacy=partial,not_annotation_accuracy_or_navigation_efficacy=True)

def parse_lines(text):
 result={}
 for line in text.splitlines():
  if ':'in line:
   k,v=line.split(':',1);k=k.strip()
   if k in ['Objects','Room','Target_visible']or k.startswith('Navigable_'):require(k not in result,'Duplicate field in raw response');result[k]=v.strip()
 return result
