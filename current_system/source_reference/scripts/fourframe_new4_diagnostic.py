from pathlib import Path
import sys,json,datetime
import numpy as np,torch
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from hybrid_observation_budget_v1 import predict
from geometry_attack_evaluate_v1 import Candidate,specs,CFG
from geometry_attack_v2 import save,sha,NativeSampler
from cpu_data_prepare import frames
from flowproxy.geometry import geometry_channels
from self_trajectory_closure_v1 import build_priors
from round2_complete40_v1 import frame_summary,summarize
OUT=ROOT/'outputs/hybrid_twoframe_confirmation_v1/fourframe_posthoc_control';OUT.mkdir(exist_ok=False);(OUT/'predictions').mkdir()
spec=next(s for s in specs() if s['name']=='K8');torch.set_num_threads(1);model=Candidate(spec);priors=build_priors()
rows=json.loads((OUT.parent/'preregister.json').read_text())['cases']
save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'posthoc_control_after_twoframe_scores':True,'fourframe_primary_unchanged':True,'no_selection_or_training':True,'weights_sha256':sha(spec['path']),'script_sha256':sha(__file__)})
registry=[];lookup={}
for r in rows:
 cid=r['id'];q={k:r[k] for k in ['shape','Re','D','cx','cy']};g=geometry_channels(CFG,**q);items=frames(ROOT/'data/raw/hybrid_twoframe_confirmation_v1'/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(g['xx'],g['yy'],g['mask'])
 support=np.stack([sampler.sample(p) for _,p in items[:4]])
 pred,diag=predict(model,q,support,times,g['mask'],'true4',priors)
 f=OUT/'predictions'/f'{cid}.npz';np.savez_compressed(f,prediction=pred)
 registry.append({'id':cid,'file':str(f),'sha256':sha(f),'diagnostic':diag});lookup[cid]=(g,items,times)
save(OUT/'predictions_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':registry})
report=[]
for entry in registry:
 cid=entry['id'];g,items,times=lookup[cid];sampler=NativeSampler(g['xx'],g['yy'],g['mask']);target=np.stack([sampler.sample(p) for _,p in items[4:164]]).astype(np.float64);pred=np.load(entry['file'])['prediction'].astype(np.float64);mask=g['mask'][None,None]
 err=np.linalg.norm((pred-target)*mask,axis=(-2,-1))/np.maximum(np.linalg.norm(target*mask,axis=(-2,-1)),1e-8)
 report.append({'id':cid,'horizons':{f'{sec}s':frame_summary(err[:n],times[3],times[n+3],sec,True) for sec,n in [(2,8),(8,32),(20,80),(40,160)]}})
summary={h:summarize([r['horizons'][h] for r in report]) for h in ['2s','8s','20s','40s']}
save(OUT/'report.json',{'cases':report,'summary':summary,'posthoc_only':True});print('FOURFRAME_DIAGNOSTIC',summary['40s'])
