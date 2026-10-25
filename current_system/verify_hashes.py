from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;items=json.loads((P/'manifest_sha256.json').read_text(encoding='utf-8'));errors=[]
for z in items:
 p=P/z['path'];h=hashlib.sha256()
 if not p.is_file():errors.append(z['path']);continue
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 if h.hexdigest()!=z['sha256'] or p.stat().st_size!=z['bytes']:errors.append(z['path'])
assert not errors,errors;print('ALL_FILE_HASHES_VERIFIED',len(items))
