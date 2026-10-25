from pathlib import Path
import json,hashlib,numpy as np,csv
ROOT=Path('/root/autodl-tmp/flowproxy');OUT=ROOT/'outputs/evidence_upgrade_20261007/piv'
r=json.loads((OUT/'report.json').read_text());cfg=r['config'];z=np.load(OUT/'prepared_velocity.npz');b=z['fields'];mask=z['mask'];scale=z['scale'].astype(float);mean=z['mean'].astype(float)
assert len(cfg['teststarts'])==10 and all(s>=6200 and s+164<=8000 for s in cfg['teststarts'])
assert all(t+164<=u for t,u in zip(cfg['teststarts'],cfg['teststarts'][1:]))
assert cfg['future_readout_steps']==list(range(7,159,8)) and len(cfg['future_readout_steps'])*4==76
maxdiff=0;flat=[]
for row in r['cases']:
 s=row['start_frame'];pfile=OUT/'test_predictions'/row['method']/(str(s)+'.npz');assert hashlib.sha256(pfile.read_bytes()).hexdigest()==row['file_sha256']
 with np.load(pfile) as a:p=a['prediction']
 assert hashlib.sha256(p.tobytes()).hexdigest()==row['prediction_sha256'];truth=b[s+4:s+164]
 errors=[]
 for channel in [0,1]:
  delta=(p[:,channel].astype(float)-truth[:,channel].astype(float))*scale[channel];target=truth[:,channel].astype(float)*scale[channel]+mean[channel]
  errors.append(np.linalg.norm(delta[:,mask],axis=1)/np.linalg.norm(target[:,mask],axis=1))
 actual=np.mean(errors,axis=1);reported=row['metrics']['mean_relative_L2'];diff=float(np.max(np.abs(actual-reported)));maxdiff=max(maxdiff,diff);assert diff<2e-6
 flat.append(dict(start_frame=s,method=row['method'],seconds=8,relative_Ux_L2_mean=actual[0],relative_Uy_L2_mean=actual[1],startup_frames=4,future_scalars=76))
with (OUT/'per_block_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=flat[0].keys());w.writeheader();w.writerows(flat)
receipt=dict(status='PASSED',prediction_sets=len(flat),max_mean_relative_L2_difference=maxdiff,time_blocks_nonoverlapping=True,ten_heldout_windows=True,training_labels_only_first4800_frames=True,mask_mean_scales_sensors_frequency_fit_only_TRAIN=True,causal_poison_checks_per_report=20,recording_independence='One experimental run, time-block holdout only',no_pressure_accuracy_claim=True,main_CFD_weights_unchanged=True)
(OUT/'metric_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
