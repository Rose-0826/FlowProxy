#!/usr/bin/env python3
"""Fixed synthetic support-noise diagnostic; no retraining or confirmation labels."""
from pathlib import Path
from collections import deque
import hashlib,json,sys
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from flowproxy.models import build_model
from flowproxy.paths import load_config
from flowproxy.round2.metrics import per_frame_rel
from round2_complete40_v1 import MODELS,frame_summary,summarize,horizon_selection
OUT=ROOT/'outputs/noise_robustness_v1'


def save(path,d):path.write_text(json.dumps(d,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    split=json.loads((ROOT/'configs/splits/body_120.json').read_text())
    groups={'val8_development':split['val'], 'test_position16_already_seen':[
        cid for cid in split['test'] if abs(float(cid.split('_cy')[1].split('_D')[0]))==.8]}
    ids=groups['val8_development']+groups['test_position16_already_seen']
    protocol={'task':'synthetic initial observation noise sensitivity, not independent confirmation',
              'groups':groups,'levels_rms_fraction':[.001,.01],'noise_seeds':[11,22,33],
              'noise':'independent Gaussian on 4 support frames; training80 channel RMS times level; fluid only',
              'horizons':[2,8],'no_retraining':True,'no_confirmation12_used':True,
              'same_noise_for_F0_and_K8':True,'MC_replicates_are_not_new_cases':True,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(OUT/'preregister.json',protocol)
    with h5py.File(ROOT/'data/processed/dataset_long.h5','r') as h:
        squares=np.zeros(3);count=0
        for cid in split['train']:
            fields=h['cases'][cid]['fields'][...];fluid=h['cases'][cid]['mask'][...]>.5
            squares+=np.square(fields[:,:,fluid].astype(np.float64)).sum(axis=(0,2))
            count+=fields.shape[0]*fluid.sum()
        scales=np.sqrt(squares/count+1e-8).astype(np.float32)
        geom=np.stack([h['cases'][cid]['geom'][...].astype(np.float32) for cid in ids])
        mask=np.stack([h['cases'][cid]['mask'][...].astype(np.float32) for cid in ids])
        support=np.stack([h['cases'][cid]['fields'][:4].astype(np.float32) for cid in ids])
        truth=np.stack([h['cases'][cid]['fields'][4:36].astype(np.float32) for cid in ids])
        clocks=np.stack([h['cases'][cid]['times'][...].astype(np.float32) for cid in ids])
    cfg=load_config(ROOT/'configs/default.yaml');device=torch.device('cuda')
    torch.set_num_threads(1)
    geom_t=torch.from_numpy(geom).to(device);mask_t=torch.from_numpy(mask).to(device)
    truth_t=torch.from_numpy(truth).to(device)
    report={'protocol':protocol,'training_rms_ux_uy_p':scales.tolist(),'models':{}}
    for name,path in MODELS.items():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=build_model(saved['model_name'],saved['in_channels'],saved['out_channels'],cfg).to(device)
        model.load_state_dict(saved['model']);model.eval()
        blocks={}
        for level in [0.,.001,.01]:
            rows=[]
            for seed in ([11] if level==0 else [11,22,33]):
                rng=np.random.default_rng(np.random.SeedSequence([seed,int(level*1e6)]))
                noise=(rng.normal(size=support.shape).astype(np.float32)*scales[None,None,:,None,None]*level*mask[:,None,None])
                observed=support+noise
                input_errors=[]
                for c in range(3):
                    num=np.linalg.norm(noise[:,:,c].reshape(len(ids),-1),axis=1)
                    den=np.maximum(np.linalg.norm(support[:,:,c].reshape(len(ids),-1),axis=1),1e-8)
                    input_errors.append(num/den)
                input_errors=np.stack(input_errors,axis=1)
                with torch.inference_mode():
                    fields=torch.from_numpy(observed).to(device)
                    window=deque(fields.unbind(1),maxlen=4);pred=[]
                    for _ in range(32):
                        hist=torch.stack(tuple(window),dim=1).reshape(len(ids),12,64,64)
                        v=model(torch.cat([geom_t,hist],dim=1))[:,:3];window.append(v);pred.append(v)
                    prediction=torch.stack(pred,dim=1)
                    errors=[]
                    for i in range(len(ids)):errors.append(per_frame_rel(prediction[i],truth_t[i],mask_t[i]).cpu().numpy())
                for i,cid in enumerate(ids):
                    horizons={}
                    for seconds in [2,8]:
                        chosen,complete=horizon_selection(clocks[i],seconds)
                        horizons[f'{seconds}s']=frame_summary(errors[i][chosen],float(clocks[i][3]),float(clocks[i][4+chosen[-1]]),seconds,complete)
                    rows.append({'id':cid,'seed':seed,'input_relative_ux_uy_p':input_errors[i].tolist(),'horizons':horizons})
            aggregates={}
            for group,wanted in groups.items():
                aggregates[group]={}
                for key in ['2s','8s']:
                    selected=[{'id':r['id'],'seed':r['seed'],**r['horizons'][key]} for r in rows if r['id'] in wanted]
                    s=summarize(selected)
                    s['case_count']=len(wanted);s['noise_draws_per_case']=1 if level==0 else 3
                    s['cases_all_noise_draws_pass']=sum(all(r['horizons'][key]['R']<1 for r in rows if r['id']==cid) for cid in wanted)
                    aggregates[group][key]=s
            blocks[str(level)]={'summary':aggregates,'cases':rows}
            print(name,level,json.dumps({g:{'R8':b['8s']['mean_R'],'all_draws_pass_cases':b['8s']['cases_all_noise_draws_pass']} for g,b in aggregates.items()}),flush=True)
        report['models'][name]=blocks
        del model,saved
    save(OUT/'report.json',report)
    save(OUT/'summary.json',{'training_rms_ux_uy_p':scales.tolist(),
          'models':{name:{level:b['summary'] for level,b in blocks.items()} for name,blocks in report['models'].items()},
          'is_synthetic_noise_not_independent_confirmation':True})


if __name__=='__main__':main()
