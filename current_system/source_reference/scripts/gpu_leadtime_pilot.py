#!/usr/bin/env python3
"""Bounded direct lead-time FNO pilot, original train80/val8 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/gpu_leadtime_pilot_2026-10-02'
BASE = ROOT / 'outputs/checkpoints/fno_k8_base/base.pt'
DATA = ROOT / 'data/processed/dataset_long.h5'
PERIODS = [1., 2., 4., 8., 16., 32.]
UPDATES = 2000
BATCH = 8
HORIZONS = [2., 8., 20.]
CHECKS = [0, 500, 1000, 1500, 2000]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding='utf-8')


def tasks(seed):
    rng = np.random.default_rng(seed)
    cases, starts, strides = [], [], []
    for _ in range(UPDATES):
        pick = rng.choice(80, size=BATCH, replace=False)
        k = np.where(rng.integers(0, 2, size=BATCH) == 0,
                     rng.integers(1, 33, size=BATCH), rng.integers(33, 81, size=BATCH))
        s = rng.integers(0, 158 - k, size=BATCH)
        cases.append(pick); starts.append(s); strides.append(k)
    return {'cases': np.asarray(cases), 'starts': np.asarray(starts), 'strides': np.asarray(strides)}


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    split = json.loads((ROOT / 'configs/splits/body_120.json').read_text())
    assert len(split['train']) == 80 and len(split['val']) == 8
    protocol = {
        'task': 'direct_state_conditioned_leadtime_prediction_0_to_20_seconds',
        'source': 'existing K8 weights; new time-input lifting columns zero initialized',
        'main_change': 'direct future query vs autoregressive one-step prediction',
        'initial_information': 'same target geometry and four true history frames',
        'time_input': {'raw': 'actual target time minus last support time, divided by 20',
                       'sin_cos_periods_seconds': PERIODS, 'oracle_frequency_used': False},
        'runs': [{'mode':'direct','seed':11},{'mode':'direct','seed':22},{'mode':'control','seed':11}],
        'control': 'same augmented FNO with extra time columns zero; next-frame training; autonomous rollout',
        'budget': {'updates':UPDATES,'batch':BATCH,'lr':1e-4,'AdamW_weight_decay':1e-4,
                   'gradient_clip':1.,'AMP':False,'optimizer_state':'fresh AdamW for all runs'},
        'sampling': 'paired seed11 case/support indices; half strides uniform 1..32, half 33..80; control stride1',
        'training_field_loss': 'train80 fixed per-channel RMS; masked normalized MSE',
        'checks': CHECKS,
        'selection': 'minimum val8 mean_R at 20s; tie max_R then update; short-horizon gates evaluated afterward',
        'continue_gate': 'both direct seeds reduce 20s mean_R >=20% versus stronger of frozen K8 and selected control; '
                         '20s max_R no worse than that same comparator; all 8 pass at 2s and 8s',
        'evaluation': 'same separate-channel per-frame L2 and R formula; support indices0..3; no future inputs',
        'test32_or_confirmation12_used': False,
        'no_further_training_or_CFD_if_gate_fails': True,
        'train_ids': split['train'], 'val_ids':split['val'],
        'base_checkpoint_file_sha256':sha(BASE),'data_sha256':sha(DATA),'script_sha256':sha(__file__),
        'official_F0_and_K8_remain_frozen':True,
        'literature': 'Time-conditioned neural operators and direct evaluations are established; this pilot is an application experiment.'}
    for seed in [11,22]:
        path = OUT / f'tasks_seed{seed}.npz'
        np.savez(path, **tasks(seed))
        protocol[f'tasks_seed{seed}_sha256'] = sha(path)
    dump(OUT/'preregister.json', protocol)
    print('prepared fixed protocol and task schedules', flush=True)


def load_data(ids, device, torch):
    with h5py.File(DATA, 'r') as h:
        geom = np.stack([h['cases'][cid]['geom'][...].astype(np.float32) for cid in ids])
        fields = np.stack([h['cases'][cid]['fields'][...].astype(np.float32) for cid in ids])
        masks = np.stack([h['cases'][cid]['mask'][...].astype(np.float32) for cid in ids])
        clocks = np.stack([h['cases'][cid]['times'][...].astype(np.float32) for cid in ids])
    if not np.isfinite(fields).all():raise ValueError('Non-finite fields')
    return tuple(torch.from_numpy(x).to(device) for x in [geom, fields, masks, clocks])


def run(mode, seed, smoke=False):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    sys.path.insert(0,str(ROOT/'src'))
    from flowproxy.models import build_model
    from flowproxy.paths import load_config
    from flowproxy.round2.metrics import per_frame_rel, TARGET
    from round2_complete40_v1 import frame_summary, summarize, horizon_selection
    protocol=json.loads((OUT/'preregister.json').read_text())
    if sha(BASE)!=protocol['base_checkpoint_file_sha256'] or sha(DATA)!=protocol['data_sha256']:
        raise ValueError('Frozen source changed')
    path=OUT/f'tasks_seed{seed}.npz'
    if sha(path)!=protocol[f'tasks_seed{seed}_sha256']:raise ValueError('Sampling schedule changed')
    scheduled=np.load(path)
    job=OUT/(f'smoke_{mode}_seed{seed}' if smoke else f'{mode}_seed{seed}')
    job.mkdir(exist_ok=False)
    device=torch.device('cuda')
    cfg=load_config(ROOT/'configs/default.yaml')
    saved=torch.load(BASE,map_location='cpu',weights_only=False)
    old_n=int(saved['in_channels'])
    if old_n!=16:raise ValueError('Expected 4 geometry and 12 history channels')
    old=build_model(saved['model_name'],old_n,saved['out_channels'],cfg).to(device)
    old.load_state_dict(saved['model']);old.eval()
    model=build_model(saved['model_name'],old_n+13,saved['out_channels'],cfg).to(device)
    state=model.state_dict()
    for key,value in saved['model'].items():
        if key=='lift.0.weight':
            state[key].zero_();state[key][:,:old_n]=value[:,:old_n]
            state[key][:,-2:]=value[:,-2:]
        else:state[key].copy_(value)
    model.load_state_dict(state)
    train_geom,train_fields,train_mask,train_times=load_data(protocol['train_ids'],device,torch)
    vg,vf,vm,vt=load_data(protocol['val_ids'],device,torch)
    valid=train_mask[:,None,None]>.5
    scales=torch.sqrt((train_fields.square()*valid).sum(dim=(0,1,3,4))/(valid.sum()*train_fields.shape[1])+1e-8)
    periods=torch.tensor(PERIODS,device=device)
    def inputs(geom,history,tau,with_time):
        hist=history.reshape(len(history),12,64,64)
        if with_time:
            phase=2*np.pi*tau[:,None]/periods[None]
            code=torch.cat([tau[:,None]/20.,torch.sin(phase),torch.cos(phase)],dim=1)
        else:code=torch.zeros(len(history),13,device=device)
        return torch.cat([geom,hist,code[:,:,None,None].expand(-1,-1,64,64)],dim=1)
    model.eval()
    with torch.inference_mode():
        a=old(torch.cat([vg[:1],vf[:1,:4].reshape(1,12,64,64)],dim=1))[:,:3]
        b=model(inputs(vg[:1],vf[:1,:4],torch.tensor([8.],device=device),True))[:,:3]
        initial_delta=float((a-b).abs().max())
    if initial_delta>1e-6:raise ValueError('Lift initialization does not preserve old outputs')
    del old
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    history=[]
    @torch.inference_mode()
    def evaluate():
        model.eval();cases=[]
        for i,cid in enumerate(protocol['val_ids']):
            if mode=='direct':
                predictions=[]
                for start in range(4,84,8):
                    end=min(start+8,84);n=end-start
                    tau=vt[i,start:end]-vt[i,3]
                    predictions.append(model(inputs(vg[i:i+1].expand(n,-1,-1,-1),
                        vf[i:i+1,:4].expand(n,-1,-1,-1,-1),tau,True))[:,:3])
                pred=torch.cat(predictions)
            else:
                window=deque(vf[i,:4].unbind(0),maxlen=4);predictions=[]
                for _ in range(80):
                    x=inputs(vg[i:i+1],torch.stack(tuple(window))[None],torch.zeros(1,device=device),False)
                    p=model(x)[0,:3];window.append(p);predictions.append(p)
                pred=torch.stack(predictions)
            err=per_frame_rel(pred,vf[i,4:84],vm[i]).cpu().numpy()
            times=vt[i].cpu().numpy()
            horizons={}
            for seconds in HORIZONS:
                pick,complete=horizon_selection(times,seconds)
                horizons[f'{seconds:g}s']=frame_summary(err[pick],float(times[3]),float(times[4+pick[-1]]),seconds,complete)
            cases.append({'id':cid,'horizons':horizons,'curve_times':times[4:84].tolist(),'curve':err.tolist()})
        result={'cases':cases,'summary':{}}
        for seconds in HORIZONS:
            key=f'{seconds:g}s'
            result['summary'][key]=summarize([c['horizons'][key] for c in cases])
        return result
    best_key=(float('inf'),float('inf'),float('inf'))
    started=time.perf_counter()
    def checkpoint(update,loss):
        nonlocal best_key
        evaluation=evaluate()
        row={'update':update,'loss':loss,'elapsed_seconds':time.perf_counter()-started,'summary':evaluation['summary']}
        history.append(row)
        dump(job/f'eval_{update}.json',evaluation)
        dump(job/'history.json',history)
        s=evaluation['summary']['20s'];key=(s['mean_R'],s['max_R'],update)
        if key<best_key:
            best_key=key
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                        'update':update,'mode':mode,'seed':seed,'in_channels':old_n+13,
                        'time_periods':PERIODS,'model_name':saved['model_name'],'out_channels':saved['out_channels'],
                        'scales':scales.detach().cpu(),'protocol':protocol},job/'best.pt')
            dump(job/'selected_eval.json',{'update':update,**evaluation})
        print(mode,seed,'update',update,'R20',round(s['mean_R'],4),'maxR20',round(s['max_R'],4),
              'pass8',evaluation['summary']['8s']['n_pass'],flush=True)
    if not smoke:checkpoint(0,None)
    total=3 if smoke else UPDATES
    for step in range(1,total+1):
        model.train()
        ids=torch.from_numpy(scheduled['cases'][step-1]).to(device)
        starts=torch.from_numpy(scheduled['starts'][step-1]).to(device)
        stride=torch.from_numpy(scheduled['strides'][step-1]).to(device) if mode=='direct' else torch.ones(BATCH,dtype=torch.long,device=device)
        support=starts[:,None]+torch.arange(4,device=device)[None]
        hist=train_fields[ids[:,None],support]
        future=starts+3+stride
        tau=train_times[ids,future]-train_times[ids,starts+3]
        target=train_fields[ids,future]
        pred=model(inputs(train_geom[ids],hist,tau,mode=='direct'))[:,:3]
        fluid=(train_mask[ids]>.5)[:,None]
        loss=(((pred-target)/scales[None,:,None,None]).square()*fluid).sum()/fluid.sum()/3.
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('Non-finite gradient')
        opt.step()
        if step%50==0 or step==1:
            dump(job/'status.json',{'update':step,'total':total,'loss':float(loss.detach()),
                                    'gradient_norm':float(grad),'elapsed_seconds':time.perf_counter()-started})
        if not smoke and step in CHECKS:checkpoint(step,float(loss.detach()))
    if smoke:
        dump(job/'smoke.json',{'steps':3,'initial_lift_max_delta':initial_delta,'loss':float(loss.detach()),
                              'gradient_norm':float(grad),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                              'scale_ux_uy_p':scales.detach().cpu().tolist()})
        print('SMOKE_OK',mode,seed,flush=True)
        return
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'update':UPDATES,'mode':mode,'seed':seed},job/'last.pt')
    dump(job/'done.json',{'status':'complete','elapsed_seconds':time.perf_counter()-started,
                         'selected_update':best_key[-1],'initial_lift_max_delta':initial_delta,
                         'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'data_sha256':sha(DATA),
                         'base_checkpoint_file_sha256':sha(BASE),'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
    print('DONE',mode,seed,round(time.perf_counter()-started,1),flush=True)


def aggregate():
    frozen=json.loads((ROOT/'outputs/cpu_round1/full40_eval/complete40_val.json').read_text())['models']['K8']['horizons']
    control=json.loads((OUT/'control_seed11/selected_eval.json').read_text())
    candidates={seed:json.loads((OUT/f'direct_seed{seed}/selected_eval.json').read_text()) for seed in [11,22]}
    stronger='K8' if frozen['20s']['summary']['mean_R']<=control['summary']['20s']['mean_R'] else 'control'
    comparison={k:v['summary'] for k,v in frozen.items() if k in ['2s','8s','20s']} if stronger=='K8' else control['summary']
    gates={}
    for seed,c in candidates.items():
        s=c['summary'];baseline=comparison['20s']
        gain=1-s['20s']['mean_R']/baseline['mean_R']
        gates[seed]={'relative_R20_gain':gain,'max_R20_no_worse':s['20s']['max_R']<=baseline['max_R'],
                     'all_pass_at_2_and_8':all(s[k]['n_pass']==8 for k in ['2s','8s'])}
        gates[seed]['passed']=bool(gain>=.20 and gates[seed]['max_R20_no_worse'] and gates[seed]['all_pass_at_2_and_8'])
    report={'task':'direct_leadtime_original_val8_development_pilot','stronger_comparator':stronger,
            'frozen_K8':{k:v['summary'] for k,v in frozen.items() if k in ['2s','8s','20s']},
            'control':{'update':control['update'],'summary':control['summary']},
            'direct':{seed:{'update':c['update'],'summary':c['summary']} for seed,c in candidates.items()},
            'gate_by_seed':gates,'continue_gate_passed':all(g['passed'] for g in gates.values()),
            'test32_or_confirmation12_used':False,'no_new_40s_score_for_this_pilot':True,
            'official_weights_replaced':False}
    dump(OUT/'pilot_summary.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--mode',choices=['direct','control'],default='direct')
    parser.add_argument('--seed',type=int,choices=[11,22],default=11)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.aggregate:aggregate()
    else:run(args.mode,args.seed,args.smoke)
