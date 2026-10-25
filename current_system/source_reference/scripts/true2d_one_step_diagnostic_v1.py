from pathlib import Path
import sys,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_evaluate_v1 import Candidate,specs,CFG
from geometry_attack_v2 import NativeSampler,save,sha
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
O=ROOT/'outputs/true2d_recheck_v1';p=json.loads((O/'preregister.json').read_text());torch.set_num_threads(1)
models={s['name']:Candidate(s) for s in specs() if s['name'] in ['F0','K8']};rows=[];parity=[]
for variant in ['old_reference','coarse_dt002','fine_dt0005']:
 for q in p['cases']:
  cid=q['id'];case=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/cid if variant=='old_reference' else ROOT/'data/raw/true2d_recheck_v1'/variant/cid
  lo=160 if variant=='old_reference' else p['variants'][variant]['input_start']
  items=[(t,f) for t,f in frames(case) if lo-1e-7<=t<=lo+41+1e-7][:164]
  g=geometry_channels(CFG,**{k:q[k] for k in ['shape','Re','D','cx','cy']});sampler=NativeSampler(g['xx'],g['yy'],g['mask'])
  a=np.stack([sampler.sample(f) for _,f in items]).astype(np.float32);mask=g['mask'][None,None]
  target=a[4:164].astype(np.float64);past=np.stack([a[i:i+4] for i in range(160)])
  outputs={'copy_last':a[3:163]}
  for n,candidate in models.items():
   generated=[]
   with torch.inference_mode():
    for k in range(0,160,16):
     h=torch.from_numpy(past[k:k+16]).to('cuda');geom=torch.from_numpy(g['geom']).to('cuda')[None].expand(len(h),-1,-1,-1)
     generated.append(candidate.model(torch.cat([geom,h.reshape(len(h),12,64,64)],1))[:,:3].cpu().numpy())
   outputs[n]=np.concatenate(generated)
  for n,pr in outputs.items():
   e=np.linalg.norm((pr.astype(np.float64)-target)*mask,axis=(-2,-1))/np.maximum(np.linalg.norm(target*mask,axis=(-2,-1)),1e-8)
   rows.append({'variant':variant,'case':cid,'model':n,'per_channel_mean_relative_L2':e.mean(0).tolist(),'maximum_per_channel':e.max(0).tolist(),'actual_true_history_each_step':True})
  folder=O/('evaluation_'+variant)/'predictions';K=np.load(folder/f'K8__{cid}.npz')['prediction'];H=np.load(folder/f'neural_periodic_closure__{cid}.npz')['prediction']
  delta=float(np.max(np.abs(K[:48]-H[:48])));assert delta<2e-6
  parity.append({'variant':variant,'case':cid,'first48_K8_closure_max_absolute_difference':delta})
  print('TEACHER_DIAGNOSTIC',variant,cid,flush=True)
summary={v:{n:np.mean([r['per_channel_mean_relative_L2'] for r in rows if r['variant']==v and r['model']==n],axis=0).tolist() for n in ['copy_last','F0','K8']} for v in ['old_reference','coarse_dt002','fine_dt0005']}
save(O/'one_step_diagnostic.json',{'cases':rows,'summary':summary,'first48_parity':parity,'not_autonomous_prediction':True,'no_training_or_primary_replacement':True,'posthoc_diagnostic_only':True,'script_sha256':sha(__file__)})
print('TEACHER_DIAGNOSTIC_FINISHED',summary,flush=True)
