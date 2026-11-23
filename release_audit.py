"""Record annotation requests, provider responses and usage with explicit API enablement."""
from pathlib import Path
import os,json,datetime,hashlib,base64,uuid,re,shutil
OPTIONS=dict(execute=False,model=None,max_calls=0,calls=0);CLIENT=None

def configure(execute=False,model=None,max_calls=0):
 if execute:
  if not model or not re.search(r'\d{4}-\d{2}-\d{2}',model):raise ValueError('An actual dated model snapshot is required; family alias is insufficient')
  if type(max_calls)is not int or max_calls<=0:raise ValueError('Set an explicit positive max API-call budget')
 OPTIONS.update(execute=bool(execute),model=model,max_calls=max_calls,calls=0)

def request_annotation(position_dir,kind,content,max_tokens,temperature):
 if not OPTIONS['execute']:raise RuntimeError('Offline default: annotation API execution was not enabled')
 if OPTIONS['calls']>=OPTIONS['max_calls']:raise RuntimeError('Declared annotation call budget exhausted')
 root=Path(position_dir);audit=root/'api_audit';audit.mkdir(exist_ok=True)
 if shutil.disk_usage(root).free<110*1024**3:raise RuntimeError('Preserve110GiB operational disk floor')
 messages=[];images=[]
 for item in content:
  item=dict(item)
  if item.get('type')=='image_url':
   data=item['image_url']['url'];prefix,sep,encoded=data.partition(',')
   if not sep or not prefix.startswith('data:image/'):raise ValueError('Expected supplied local image, not an unrecorded remote URL')
   raw=base64.b64decode(encoded,validate=True);spec=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),detail=item['image_url'].get('detail'));images.append(spec);item=dict(type='image_sha256',**spec)
  messages.append(item)
 ident=kind+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8];target=audit/ident;target.mkdir()
 def write(name,value):
  p=target/name
  with p.open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
 request=dict(schema_version=1,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,requested_model_snapshot=OPTIONS['model'],messages_without_duplicated_image_payload=[dict(role='user',content=messages)],image_payload_hashes=images,max_tokens=max_tokens,temperature=temperature,source='provider_annotation_request')
 write('REQUEST.json',request);OPTIONS['calls']+=1
 try:
  global CLIENT
  if CLIENT is None:
   from openai import OpenAI
   key=os.environ.get('OPENAI_API_KEY')
   if not key:raise RuntimeError('Set OPENAI_API_KEY in environment; no key is included in release config')
   CLIENT=OpenAI(api_key=key, max_retries=0, timeout=60.0)
  response=CLIENT.chat.completions.create(model=OPTIONS['model'],messages=[dict(role='user',content=content)],max_tokens=max_tokens,temperature=temperature)
  raw=response.model_dump(mode='json');write('PROVIDER_RESPONSE.json',raw);text=response.choices[0].message.content
  if not isinstance(text,str)or not text.strip():raise ValueError('Empty model text')
  with(target/'RESPONSE_TEXT.txt').open('x',encoding='utf-8')as f:f.write(text)
  write('COMPLETE.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),requested_model=OPTIONS['model'],returned_model=raw.get('model'),provider_response_id=raw.get('id'),usage=raw.get('usage'),text_sha256=hashlib.sha256(text.encode()).hexdigest(),model_identifier_source='provider_response'))
  return text
 except BaseException as e:
  message=re.sub(r'sk-[A-Za-z0-9_-]+','[redacted]',str(e));write('FAILURE.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error_type=type(e).__name__,message=message[:1000],not_automatically_retried=True));raise
