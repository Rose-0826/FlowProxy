#!/usr/bin/env python3
"""Probe consistency of solid-cell padding in frozen autonomous FNO rollouts."""
import datetime
import hashlib
import json
import sys
from collections import deque
from pathlib import Path
import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from flowproxy.models import build_model
from flowproxy.paths import load_config
from flowproxy.round2.metrics import per_frame_rel
from round2_complete40_v1 import MODELS,DATA,frame_summary,summarize,sha_file

OUT=ROOT/'outputs/solid_state_probe_v1'

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def main():
    torch.set_num_threads(1);device='cuda';cfg=load_config(ROOT/'configs/default.yaml')
    OUT.mkdir(exist_ok=False)
    splits=json.loads((ROOT/'configs/splits/body_120.json').read_text())
    save(OUT/'preregister.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'no_training':True,'sets':['val8_development','old_test32_seen'],
        'operation':'at every autonomous step multiply all three channels by the known binary fluid mask; feed this stored state onward',
        'physical_interpretation':'consistency of zero padding inside solids, not a pressure boundary condition',
        'fluid_predictions_unchanged_at_that_step':True,'inputs':'four true startup fields; geometry and no future fields',
        'horizons':[2,8,20,40],'source_data_sha256':sha_file(DATA),
        'checkpoint_sha256':{k:sha_file(v) for k,v in MODELS.items()},'script_sha256':sha_file(__file__)})
    report={'models':{}}
    for name,path in MODELS.items():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=build_model(saved['model_name'],16,3,cfg).to(device);model.load_state_dict(saved['model']);model.eval()
        cases={};summary={}
        with h5py.File(DATA,'r') as h,torch.inference_mode():
            for role in ['val','test']:
                cases[role]=[]
                for cid in splits[role]:
                    g=h['cases'][cid];geom=torch.from_numpy(g['geom'][...])[None].to(device)
                    mask=torch.from_numpy(g['mask'][...]).to(device)
                    hist=torch.from_numpy(g['fields'][:4]).to(device)
                    window=deque(hist.unbind(0),maxlen=4);pred=[];first_body=None
                    for n in range(160):
                        raw=model(torch.cat([geom,torch.stack(tuple(window)).reshape(1,12,64,64)],1))[0,:3]
                        if n==0:
                            body=mask<.5; z=raw[:,body]
                            first_body={'cells':int(body.sum()),'mean_abs':z.abs().mean(1).cpu().tolist(),
                                        'max_abs':z.abs().max(1).values.cpu().tolist(),
                                        'startup_max_abs':hist[:,:,body].abs().max().item()}
                        v=raw*mask[None];window.append(v);pred.append(v)
                    pred=torch.stack(pred)
                    # Truth used only by scoring after completion of the autonomous prediction.
                    true=torch.from_numpy(g['fields'][4:164]).to(device)
                    err=per_frame_rel(pred,true,mask).cpu().numpy();times=g['times'][...]
                    hs={}
                    for sec,n in [(2,8),(8,32),(20,80),(40,160)]:
                        hs[f'{sec}s']=frame_summary(err[:n],float(times[3]),float(times[n+3]),sec,True)
                    cases[role].append({'id':cid,'horizons':hs,'initial_body_prediction':first_body})
                summary[role]={key:summarize([x['horizons'][key] for x in cases[role]]) for key in ['2s','8s','20s','40s']}
                print('PROBE',name,role,'R8',round(summary[role]['8s']['mean_R'],4),'pass8',summary[role]['8s']['n_pass'],
                      'R40',round(summary[role]['40s']['mean_R'],3),'pass40',summary[role]['40s']['n_pass'],flush=True)
        report['models'][name]={'cases':cases,'summary':summary}
        save(OUT/'report.json',report)
        del model;torch.cuda.empty_cache()
    save(OUT/'summary.json',{k:v['summary'] for k,v in report['models'].items()})

if __name__=='__main__':main()
