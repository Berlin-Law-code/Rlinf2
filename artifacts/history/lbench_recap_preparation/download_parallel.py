from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib, requests, time
from huggingface_hub import hf_hub_url,get_hf_file_metadata
base=Path('/root/autodl-tmp/models/siglip2-so400m-patch14-224')
meta=get_hf_file_metadata(hf_hub_url('google/siglip2-so400m-patch14-224','model.safetensors'))
parts=base/'.parallel_parts';parts.mkdir(exist_ok=True)
chunk=64*1024*1024
ranges=[(i,s,min(s+chunk,meta.size)-1) for i,s in enumerate(range(0,meta.size,chunk))]
def fetch(item):
 i,start,end=item;p=parts/f'{i:04d}'
 if p.exists() and p.stat().st_size==end-start+1:return i
 for attempt in range(5):
  try:
   offset=p.stat().st_size if p.exists() else 0
   if offset==end-start+1:return i
   resume=start+offset
   with requests.get(meta.location,headers={'Range':f'bytes={resume}-{end}'},stream=True,timeout=(30,60)) as r:
    r.raise_for_status()
    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {resume}-{end}/{meta.size}':raise ValueError('Server did not honor byte range')
    with p.open('ab') as f:
     for b in r.iter_content(1024*1024):f.write(b)
   if p.stat().st_size!=end-start+1:raise ValueError('Incomplete byte range')
   return i
  except Exception as exc:
   if attempt==4:raise RuntimeError(f'Part {i} failed: {type(exc).__name__}') from None
   time.sleep(2)
with ThreadPoolExecutor(max_workers=32) as ex:
 for n,future in enumerate(as_completed([ex.submit(fetch,item) for item in ranges]),1):
  print('completed',n,'/',len(ranges),'part',future.result(),flush=True)
h=hashlib.sha256();output=base/'model.parallel.safetensors'
with output.open('wb') as out:
 for i,_,_ in ranges:
  with (parts/f'{i:04d}').open('rb') as f:
   while b:=f.read(8*1024*1024):h.update(b);out.write(b)
if h.hexdigest()!=meta.etag:raise RuntimeError('SHA256 does not match model repository')
output.replace(base/'model.safetensors')
print('Verified SHA256',h.hexdigest(),flush=True)
