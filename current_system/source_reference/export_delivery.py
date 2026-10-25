from pathlib import Path
import json,hashlib,numpy as np,tarfile
R=Path('/root/autodl-tmp/flowproxy');O=R/'outputs/evidence_upgrade_20261007';P=O/'delivery_package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for device in ['cuda','cpu']:
 d=json.loads((P/f'standalone_verification_{device}.json').read_text());assert d['status']=='PASSED' and len(d['results'])==12 and not d['original_workspace_file_reads']
with np.load(O/'piv/prepared_velocity.npz') as z:
 s=6200;truth=z['fields'][s+4:s+164]*z['scale'][None,:,None,None]+z['mean'][None]
 with np.load(O/'piv/test_predictions/adapted_mechanisms/6200.npz') as p:prediction=p['prediction']*z['scale'][None,:,None,None]+z['mean'][None]
 np.savez_compressed(P/'evidence/piv/continuous_demo.npz',truth=truth,prediction=prediction,mask=z['mask'],x=z['x'],y=z['y'],probes=z['probes'],times=np.arange(1,161)*.05)
 provenance={'source':'https://zenodo.org/records/20765567','native_test_start':s,'units':'m/s','dt':.05,'frames':160,'mask':'TRAIN-derived valid-vector support','transformation':'normalized field * TRAIN RMS + TRAIN mean; no temporal interpolation','raw_dataset_md5':'4cc876439c48f7970afa477ea17d217a','export_sha256':sha(P/'evidence/piv/continuous_demo.npz')}
 (P/'evidence/piv/continuous_demo.json').write_text(json.dumps(provenance,indent=2))
items=[]
with tarfile.open(O/'verified_delivery.tar.gz','w:gz',compresslevel=2) as t:
 for p in sorted(P.rglob('*')):
  if not p.is_file() or any(x in ['runtime_cpu','runtime_cuda','replayed_predictions','__pycache__'] for x in p.relative_to(P).parts):continue
  t.add(p,arcname=str(p.relative_to(P)));items.append(str(p.relative_to(P)))
print('ARCHIVE',len(items),(O/'verified_delivery.tar.gz').stat().st_size,sha(O/'verified_delivery.tar.gz'),flush=True)
