#!/usr/bin/env python3
"""Geometry / mechanics-scale attack. Original data and formal weights immutable.

Subcommands: prepare, resolution, data, smoke, train, evaluate.
All new scores use the original uniform physical grid and pressure reference.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/uniform128_attack_v1'
PREVIOUS = ROOT / 'outputs/geometry_attack_v1'
DATA = ROOT / 'data/processed/dataset_long40_v1.h5'
WARP = ROOT / 'data/processed/uniform128_native_v1.h5'
BASE = ROOT / 'outputs/checkpoints/fno_k8_base/base.pt'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
MODES = ['control']
STEPS = 6000
CHECKS = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            d.update(b)
    return d.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_suffix(path.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    t.replace(path)


def protocol():
    return json.loads((OUT / 'preregister.json').read_text())


def attrs(g):
    return {k: str(g.attrs[k]) if k == 'shape' else float(g.attrs.get(k, 0.))
            for k in ['shape', 'Re', 'D', 'cx', 'cy']}


def mapping(cfg, row, n=64):
    """Monotone tensor-product map. Physical domain endpoints stay fixed."""
    from flowproxy.geometry import obstacle_sdf, wall_sdf
    d = cfg['domain']; D = row['D']; cx = row['cx']; cy = row['cy']
    xr = np.array([d['x_min'], -2., 0., 10., d['x_max']])
    xp = np.array([d['x_min'], cx - D, cx, cx + 5 * D, d['x_max']])
    yr = np.array([d['y_min'], -3., 0., 3., d['y_max']])
    yp = np.array([d['y_min'], cy - 1.5 * D, cy, cy + 1.5 * D, d['y_max']])
    if min(np.diff(xp)) <= 0 or min(np.diff(yp)) <= 0:
        raise ValueError('Map would fold')
    x0 = np.linspace(d['x_min'], d['x_max'], n)
    y0 = np.linspace(d['y_min'], d['y_max'], n)
    x = np.interp(x0, xr, xp); y = np.interp(y0, yr, yp)
    xx, yy = np.meshgrid(x, y)
    sdf = obstacle_sdf(xx, yy, row['shape'], cx, cy, D).astype(np.float32)
    mask = (sdf > 0).astype(np.float32)
    wall = wall_sdf(yy, d['y_min'], d['y_max'])
    geom = np.stack([sdf, mask, wall, np.full_like(sdf, row['Re'] / 200)])
    forward = np.stack([2 * (xx - d['x_min']) / (d['x_max'] - d['x_min']) - 1,
                        2 * (yy - d['y_min']) / (d['y_max'] - d['y_min']) - 1], -1)
    # Evaluation and training loss are always on the original physical 64x64 grid.
    physical_x = np.linspace(d['x_min'], d['x_max'], 64)
    physical_y = np.linspace(d['y_min'], d['y_max'], 64)
    # grid_sample interpolates neighboring sampled nodes, so invert the actual
    # sampled physical coordinates rather than the unsampled analytic knot map.
    bx = np.interp(physical_x, x, x0); by = np.interp(physical_y, y, y0)
    bxx, byy = np.meshgrid(bx, by)
    backward = np.stack([2 * (bxx - d['x_min']) / (d['x_max'] - d['x_min']) - 1,
                         2 * (byy - d['y_min']) / (d['y_max'] - d['y_min']) - 1], -1)
    coords = np.stack([(xx - cx) / (20 * D), (yy - cy) / (8 * D), wall / (8 * D)])
    return {k: np.asarray(v, np.float32) for k, v in {
        'xx': xx, 'yy': yy, 'mask': mask, 'geom': geom, 'forward': forward,
        'backward': backward, 'coords': coords}.items()}


class NativeSampler:
    """Linear triangle interpolation at fixed query points; geometry cached only."""
    def __init__(self, xx, yy, mask):
        self.q = np.stack([xx.ravel(), yy.ravel()], 1)
        self.shape = xx.shape; self.mask = mask; self.points = None

    def sample(self, path):
        from cpu_data_prepare import read_point_fields
        from scipy.spatial import Delaunay
        points, values = read_point_fields(path)
        if self.points is None:
            self.points = points.copy(); tri = Delaunay(points)
            simp = tri.find_simplex(self.q); self.valid = simp >= 0
            safe = np.maximum(simp, 0)
            tr = tri.transform[safe]
            b = np.einsum('nij,nj->ni', tr[:, :2], self.q - tr[:, 2])
            self.weights = np.column_stack([b, 1 - b.sum(1)])
            self.vertices = tri.simplices[safe]
        elif not np.array_equal(points, self.points):
            raise ValueError('Native coordinates changed')
        a = np.einsum('nk,nkc->nc', self.weights, values[self.vertices])
        a[~self.valid] = 0
        a = a.reshape(*self.shape, 3).transpose(2, 0, 1).astype(np.float32)
        return a * self.mask[None]


def prepare():
    import shutil
    if OUT.exists():raise FileExistsError('Uniform128 already exists')
    OUT.mkdir()
    p=json.loads((ROOT/'outputs/geometry_attack_v2/preregister.json').read_text())
    p['registered_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p['version']='uniform128_attack_v1'
    p['seeds']=[11,22];p['modes']=['control']
    for seed in p['seeds']:
        shutil.copy2(PREVIOUS/f'tasks_seed{seed}.npz',OUT/f'tasks_seed{seed}.npz')
    p['warm_checkpoint_sha256']={'control_seed'+str(seed):sha(BASE) for seed in p['seeds']}
    p['initialization']='formal K8; fresh zero-output query head; AdamW restart'
    p['state_grid']='uniform physical128; 16384 sites/frame, four times uniform64; state/query physical-area losses as v2'
    p['clock_step']='raw FNO output, no D clock rescaling'
    p['readout']='bilinear latent and velocity to uniform64; valid-fluid pressure base; same zero-output query head as v2'
    p['native_field_targets']='80 training cases native128 labels; validation native reads four startup frames only'
    p['script_sha256']=sha(__file__)
    save(OUT/'preregister.json',p);print('UNIFORM128_PREPARED',flush=True)


def prepare_data():
    from cpu_data_prepare import frames
    from flowproxy.geometry import geometry_channels
    p=protocol();cfg=copy.deepcopy(p['config']);cfg['grid']={'nx':128,'ny':128}
    tmp=WARP.with_suffix('.partial.h5')
    if WARP.exists() or tmp.exists():raise FileExistsError('128 cache already exists')
    rows=[];started=time.perf_counter()
    with h5py.File(DATA,'r') as original,h5py.File(tmp,'x') as dest:
        dest.attrs['protocol_sha256']=sha(OUT/'preregister.json');cases=dest.create_group('cases')
        for role in ['train','val']:
            for cid in p[f'{role}_ids']:
                g=original['cases'][cid];q=attrs(g);geo=geometry_channels(cfg,**q)
                items=frames(ROOT/'data/raw/cases_long40'/cid)
                sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask']);count=161 if role=='train' else 4
                group=cases.create_group(cid)
                field=group.create_dataset('fields',(count,3,128,128),dtype='f4',chunks=(1,3,128,128),compression='lzf')
                for i,(_,path) in enumerate(items[:count]):field[i]=sampler.sample(path)
                for k in ['geom','mask']:group.create_dataset(k,data=geo[k])
                for k,v in q.items():group.attrs[k]=v
                group.create_dataset('times',data=g['times'][:count])
                rows.append({'id':cid,'role':role,'frames':count,'finite':bool(np.isfinite(field[...]).all())})
                if not rows[-1]['finite']:raise ValueError('Non-finite native128')
                save(OUT/'data_status.json',{'status':'running','completed':len(rows),'total':88})
                if len(rows)%8==0:print('128_DATA',len(rows),round(time.perf_counter()-started,1),flush=True)
    tmp.replace(WARP)
    save(OUT/'data_report.json',{'status':'complete','rows':rows,'h5_sha256':sha(WARP),'seconds':time.perf_counter()-started})
    save(OUT/'data_status.json',{'status':'complete','completed':88,'total':88})
    print('128_DATA_COMPLETE',flush=True)


def run(mode, seed, smoke=False):
    import torch
    import torch.nn.functional as F
    from flowproxy.models import build_model
    from flowproxy.geometry import geometry_channels
    from flowproxy.round2.metrics import per_frame_rel
    from round2_complete40_v1 import frame_summary,summarize,state_sha
    torch.set_num_threads(1);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    p=protocol();cfg=p['config'];device='cuda'
    if sha(BASE)!=p['base_sha256']:raise ValueError('Base changed')
    tasks_path=OUT/f'tasks_seed{seed}.npz'
    if sha(tasks_path)!=p[f'tasks_seed{seed}_sha256']:raise ValueError('Tasks changed')
    tasks=np.load(tasks_path); job=OUT/(f'smoke_{mode}_seed{seed}' if smoke else f'{mode}_seed{seed}')
    job.mkdir(exist_ok=False)
    def load(ids,role):
        cfg128=copy.deepcopy(cfg);cfg128['grid']={'nx':128,'ny':128}
        values={k:[] for k in ['fields','native','geom','mask','times','D','state_mask','state_weight','query_coords']}
        with h5py.File(DATA,'r') as h,h5py.File(WARP,'r') as native:
            for cid in ids:
                g=h['cases'][cid];q=attrs(g);ng=native['cases'][cid]
                values['fields'].append(g['fields'][:161 if role=='train' else 164])
                values['native'].append(ng['fields'][...]);values['geom'].append(ng['geom'][...])
                values['mask'].append(g['mask'][...]);values['times'].append(g['times'][...]);values['D'].append(q['D'])
                geo=geometry_channels(cfg128,**q);sx=geo['xx'][0];sy=geo['yy'][:,0]
                wx=np.r_[np.diff(sx)[0]/2,(sx[2:]-sx[:-2])/2,np.diff(sx)[-1]/2]
                wy=np.r_[np.diff(sy)[0]/2,(sy[2:]-sy[:-2])/2,np.diff(sy)[-1]/2]
                area=wy[:,None]*wx[None,:];values['state_weight'].append(area/area.mean()*geo['mask']);values['state_mask'].append(geo['mask'])
                gg=geometry_channels(cfg,**q)
                values['query_coords'].append(np.stack([(gg['xx']-q['cx'])/(20*q['D']),
                    (gg['yy']-q['cy'])/(8*q['D']),gg['sdf_wall']/(8*q['D']),gg['sdf_obs']/q['D'],np.full((64,64),q['Re']/200)]))
        return {k:torch.from_numpy(np.stack(v).astype(np.float32)).to(device) for k,v in values.items()}
    train=load(p['train_ids'],'train');val=load(p['val_ids'],'val')
    warm=BASE
    if sha(warm)!=p['warm_checkpoint_sha256'][f'{mode}_seed{seed}']:raise ValueError('Warm weights changed')
    saved=torch.load(warm,map_location='cpu',weights_only=False);inc=16 if mode=='control' else 21
    model=build_model(saved['model_name'],inc,3,cfg).to(device)
    model.load_state_dict(saved['model'])
    capture={}
    hook=model.project[1].register_forward_hook(lambda module,inputs,output:capture.__setitem__('latent',output))
    query_head=torch.nn.Sequential(torch.nn.Conv2d(75,32,1),torch.nn.GELU(),torch.nn.Conv2d(32,3,1)).to(device)
    torch.nn.init.zeros_(query_head[-1].weight);torch.nn.init.zeros_(query_head[-1].bias)
    scales=torch.tensor(p['normalization']['channel_rms'],device=device)
    parameters=list(model.parameters())+list(query_head.parameters())
    opt=torch.optim.AdamW(parameters,lr=p['lr'],weight_decay=p['weight_decay'])
    def sample_forward(data,idx,history):
        if mode=='warp64':
            b,k,c,h,w=history.shape
            grid=data['forward'][idx,None].expand(-1,k,-1,-1,-1).reshape(b*k,64,64,2)
            return F.grid_sample(history.reshape(b*k,c,h,w),grid,align_corners=True,padding_mode='border').reshape(b,k,c,64,64)
        return history
    def startup(data,idx,starts):
        source=data['native']
        hist=torch.stack([source[int(i),int(s):int(s)+4] for i,s in zip(idx,starts)])
        return sample_forward(data,idx,hist)*data['state_mask'][idx,None,None]
    def step(data,idx,hist,dt):
        x=torch.cat([data['geom'][idx],hist.reshape(len(idx),12,128,128)],1)
        if mode!='control':
            constants=torch.stack([data['D'][idx]-1,dt*cfg['physics']['U_inf']/(.25*data['D'][idx])-1],1)
            extra=torch.cat([data['coords'][idx],constants[:,:,None,None].expand(-1,-1,64,64)],1)
            x=torch.cat([x,extra],1)
        raw=model(x)[:,:3];latent=capture.pop('latent')
        if mode=='control':v=raw
        else:
            alpha=(dt*cfg['physics']['U_inf']/(.25*data['D'][idx]))[:,None,None,None]
            v=hist[:,-1]+alpha*(raw-hist[:,-1])
        v=v*data['state_mask'][idx,None]
        return v,latent
    def readout(data,idx,v,latent,dt):
        base=F.interpolate(v,size=(64,64),mode='bilinear',align_corners=True)
        valid=F.interpolate(data['state_mask'][idx,None],size=(64,64),mode='bilinear',align_corners=True)
        base=torch.cat([base[:,:2],base[:,2:3]/valid.clamp_min(1e-6)],1)
        features=F.interpolate(latent,size=(64,64),mode='bilinear',align_corners=True)
        constants=torch.stack([data['D'][idx]-1,dt*cfg['physics']['U_inf']/(.25*data['D'][idx])-1],1)
        query_code=torch.cat([data['query_coords'][idx],constants[:,:,None,None].expand(-1,-1,64,64)],1)
        return (base+query_head(torch.cat([features,base,query_code,valid],1)))*data['mask'][idx,None]
    initial_network_delta=None
    history=[];best=None
    @torch.inference_mode()
    def evaluate(update):
        model.eval();query_head.eval();cases=[]
        for i,cid in enumerate(p['val_ids']):
            idx=torch.tensor([i],device=device);hist=startup(val,idx,[0]);pred=[]
            for n in range(160):
                dt=val['times'][i,4+n]-val['times'][i,3+n]
                v,latent=step(val,idx,hist,dt[None]);hist=torch.cat([hist[:,1:],v[:,None]],1)
                pred.append(readout(val,idx,v,latent,dt[None])[0])
            errors=per_frame_rel(torch.stack(pred),val['fields'][i,4:164],val['mask'][i]).cpu().numpy()
            hs={}
            for seconds,n in [(2,8),(8,32),(20,80),(40,160)]:
                hs[f'{seconds}s']=frame_summary(errors[:n],float(val['times'][i,3]),float(val['times'][i,n+3]),seconds,True)
            cases.append({'id':cid,'horizons':hs})
        report={'step':update,'mode':mode,'seed':seed,'summary':{h:summarize([x['horizons'][h] for x in cases]) for h in ['2s','8s','20s','40s']},'cases':cases}
        save(job/f'eval_{update}.json',report)
        return report
    def checkpoint(update,report):
        nonlocal best
        v=report['summary']['8s'];key=(v['mean_R'],v['max_R'],update)
        history.append({'step':update,'summary':report['summary']})
        if best is None or key<best[0]:
            best=(key,update)
            torch.save({'model':model.state_dict(),'query_head':query_head.state_dict(),'optimizer':opt.state_dict(),'step':update,'mode':mode,'seed':seed,
                        'model_name':saved['model_name'],'in_channels':inc,'out_channels':3,'protocol_sha256':sha(OUT/'preregister.json')},job/'selected.pt')
            save(job/'selected_eval.json',report)
        save(job/'history.json',history)
        save(job/'status.json',{'status':'running','step':update,'selected_step':best[1],'selected_R8':best[0][0],
                               'seconds':time.perf_counter()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
        print('CHECK',mode,seed,update,'R8',round(v['mean_R'],4),'max',round(v['max_R'],4),'pass',v['n_pass'],flush=True)
    started=time.perf_counter();checkpoint(0,evaluate(0))
    max_steps=12 if smoke else STEPS
    update_durations=[];clip_hits=0
    for update in range(1,max_steps+1):
        tick=time.perf_counter();model.train();query_head.train();idx=torch.from_numpy(tasks['cases'][update-1]).to(device);starts=tasks['starts'][update-1]
        hist=startup(train,idx,starts);K=8
        losses=[]
        for j in range(K):
            frame=torch.tensor(starts+4+j,device=device)
            dt=train['times'][idx,frame]-train['times'][idx,frame-1]
            v,latent=step(train,idx,hist,dt);mapped=readout(train,idx,v,latent,dt)
            true_uniform=train['fields'][idx,frame];mask=train['mask'][idx,None]
            native_target=train['native'][idx,frame]
            native_target=native_target*train['state_mask'][idx,None]
            w=train['state_weight'][idx,None]
            state_loss=(((v-native_target)/scales[None,:,None,None]).square()*w).sum((1,2,3))/(3*w.sum((1,2,3)))
            query_loss=(((mapped-true_uniform)/scales[None,:,None,None]).square()*mask).sum((1,2,3))/(3*mask.sum((1,2,3)))
            losses.append((.5*state_loss+.5*query_loss).mean());hist=torch.cat([hist[:,1:],v[:,None]],1)
        loss=.5*torch.stack(losses).mean()+.5*losses[-1]
        if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        clip_hits+=int(float(norm)>1);opt.step();torch.cuda.synchronize();update_durations.append(time.perf_counter()-tick)
        if not smoke and update in CHECKS:
            result=evaluate(update)
            result['training_at_checkpoint']={'last_loss':float(loss),'preclip_grad_norm':float(norm),'K':K}
            checkpoint(update,result)
    if smoke:
        save(job/'smoke.json',{'K':8,'batch':2,'median_update_seconds':float(np.median(update_durations[2:])),
                             'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'last_loss':float(loss),
                             'initial_network_delta':initial_network_delta,'finite_gradients':bool(torch.isfinite(norm))})
    else:
        torch.save({'model':model.state_dict(),'query_head':query_head.state_dict(),'optimizer':opt.state_dict(),'step':max_steps,'mode':mode,'seed':seed,
                    'model_name':saved['model_name'],'in_channels':inc,'out_channels':3},job/'last.pt')
        save(job/'done.json',{'status':'complete','selected_step':best[1],'selected_R8':best[0][0],
            'seconds':time.perf_counter()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'median_update_seconds':float(np.median(update_durations)),'clip_hits':clip_hits,
            'selected_backbone_state_sha256':state_sha(torch.load(job/'selected.pt',map_location='cpu',weights_only=False)['model']),
            'selected_query_state_sha256':state_sha(torch.load(job/'selected.pt',map_location='cpu',weights_only=False)['query_head']),
            'selected_checkpoint_sha256':sha(job/'selected.pt')})
    save(job/'status.json',{'status':'complete','step':max_steps,'selected_step':best[1],'selected_R8':best[0][0]})
    print('DONE',mode,seed,'selected',best[1],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','data','smoke','train'])
    parser.add_argument('--mode',choices=MODES,default='control');parser.add_argument('--seed',type=int,default=11)
    a=parser.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='data':prepare_data()
    else:run(a.mode,a.seed,smoke=a.command=='smoke')
