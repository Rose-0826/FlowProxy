from pathlib import Path
import json,hashlib,shutil,tarfile,time
R=Path('/root/autodl-tmp/flowproxy');D=R/'release/MAINLINE_20261005'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for cid in ['cylinder_Re94.7_cy0.747_D1.165','square_Re167.8_cy-0.743_D0.802']:
    q=json.loads((D/'reproduced'/(cid+'.json')).read_text())
    assert q['max_abs_difference_from_frozen_prediction']==0.0
    assert not q['future_full_field_used_for_prediction'] and q['scalar_future_readings']==76
    (D/'reproduced'/(cid+'.npz')).unlink()
for p in D.rglob('__pycache__'):shutil.rmtree(p)
records=[]
for p in D.rglob('*'):
    if p.is_file() and p.name!='manifest.json':records.append({'path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':sha(p)})
(D/'manifest.json').write_text(json.dumps(records,indent=2))
archive=R/'release/MAINLINE_20261005.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tf:tf.add(D,arcname=D.name,recursive=True)
with tarfile.open(archive,'r:gz') as tf:
    members={x.name:x for x in tf.getmembers() if x.isfile()}
    for q in records:
        assert members[D.name+'/'+q['path']].size==q['bytes']
result={'path':str(archive),'bytes':archive.stat().st_size,'archive_sha256':sha(archive),'manifest_sha256':sha(D/'manifest.json'),'files':len(records),'logical_bytes':sum(a['bytes'] for a in records),'two_cases_exact_reproduction':True,'utc':time.time()}
(R/'outputs/mainline_cleanup_20261005/archive_verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
