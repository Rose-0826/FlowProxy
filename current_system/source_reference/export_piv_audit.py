from pathlib import Path
import numpy as np,json,shutil,tarfile,hashlib
O=Path('/root/autodl-tmp/flowproxy/outputs/evidence_upgrade_20261007');A=O/'piv_audit_export';A.mkdir(exist_ok=True)
report=json.loads((O/'piv/report.json').read_text())
with np.load(O/'piv/prepared_velocity.npz') as z:
 np.savez_compressed(A/'normalization.npz',mask=z['mask'],mean=z['mean'],scale=z['scale'])
 for s in report['config']['teststarts']:np.savez_compressed(A/f'truth_{s}.npz',fields=z['fields'][s+4:s+164])
for row in report['cases']:
 p=O/'piv/test_predictions'/row['method']/(str(row['start_frame'])+'.npz');assert hashlib.sha256(p.read_bytes()).hexdigest()==row['file_sha256'];t=A/'test_predictions'/row['method']/p.name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
with tarfile.open(O/'piv_audit.tar.gz','w:gz',compresslevel=1) as t:
 for p in A.rglob('*'):
  if p.is_file():t.add(p,arcname=str(p.relative_to(A)))
print('PIV_AUDIT', (O/'piv_audit.tar.gz').stat().st_size,hashlib.sha256((O/'piv_audit.tar.gz').read_bytes()).hexdigest(),flush=True)
