from pathlib import Path
import json,urllib.request,hashlib,time,sys
ROOT=Path('/root/autodl-tmp/flowproxy');OUT=ROOT/'outputs/evidence_upgrade_20261007';RAW=ROOT/'data/experimental/zenodo_20765567'
OUT.mkdir(exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
meta=json.load(urllib.request.urlopen(urllib.request.Request('https://zenodo.org/api/records/20765567',headers={'User-Agent':'AIC flow reconstruction experiment'}),timeout=60))
(RAW/'source_metadata.json').write_text(json.dumps(meta,indent=2));file=meta['files'][0];assert file['key']=='cylinder_vel.mat'
target=RAW/file['key'];tmp=target.with_suffix('.partial');started=time.time()
if not target.exists():
 request=urllib.request.Request(file['links']['self'],headers={'User-Agent':'AIC flow reconstruction experiment'})
 h=hashlib.md5();count=0
 with urllib.request.urlopen(request,timeout=120) as response,tmp.open('wb') as f:
  while True:
   chunk=response.read(4*1024*1024)
   if not chunk:break
   f.write(chunk);h.update(chunk);count+=len(chunk)
   if count//(100*1024**2)!=(count-len(chunk))//(100*1024**2):print('DOWNLOAD_MB',count/1024**2,flush=True)
 assert count==file['size'] and 'md5:'+h.hexdigest()==file['checksum']
 tmp.replace(target)
else:
 h=hashlib.md5()
 with target.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 assert 'md5:'+h.hexdigest()==file['checksum']
import scipy.io as sio
try:description=sio.whosmat(target)
except NotImplementedError:
 import h5py
 with h5py.File(target) as f:description=[(k,f[k].shape,str(f[k].dtype)) for k in f]
receipt=dict(status='PUBLIC_EXPERIMENTAL_DATA_ACQUIRED',source='https://zenodo.org/records/20765567',DOI='10.5281/zenodo.20765567',license=meta['metadata'].get('license'),bytes=file['size'],md5=h.hexdigest(),variables=description,Re=413,sample_frequency_Hz=20,modalities=['Ux','Uy'],pressure_truth_available=False,finite_length_cylinder_midspan_2D_PIV=True,scope='External experimental dataset; outside frozen main-system low-Re/3-channel full-domain input conditions. No main-system accuracy claim until compatible evaluation is demonstrated.',seconds=time.time()-started)
(OUT/'PIV_acquisition.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
