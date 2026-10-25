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
OUT = ROOT / 'outputs/geometry_attack_v1'
DATA = ROOT / 'data/processed/dataset_long40_v1.h5'
WARP = ROOT / 'data/processed/geometry_warp_v1.h5'
BASE = ROOT / 'outputs/checkpoints/fno_k8_base/base.pt'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
MODES = ['control', 'scale', 'warp64', 'warp_native']
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
    from flowproxy.paths import load_config
    if OUT.exists() or WARP.exists():
        raise FileExistsError('Existing experiment directory')
    OUT.mkdir()
    split = json.loads((ROOT / 'configs/splits/body_120.json').read_text())
    if [len(split[x]) for x in ['train', 'val', 'test']] != [80, 8, 32]:
        raise ValueError('Split changed')
    cfg = load_config(ROOT / 'configs/default.yaml')
    p = {'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'scope': 'Mechanics-scale / geometry-adaptive short autonomous forecasts',
         'train_ids': split['train'], 'val_ids': split['val'], 'old_seen_test_ids': split['test'],
         'old32_and_previous_confirmation12_are_seen_development_evidence': True,
         'no_training_on_test32_or_previous_confirmation12': True,
         'selection': 'minimum val8 mean R at 8s; ties: max R then earlier step; step0 eligible',
         'final_independent_confirmation_requires_new_preregistration': True,
         'modes': MODES, 'seeds': [11, 22], 'max_updates': STEPS, 'checks': CHECKS,
         'batch': 2, 'K_schedule': [[1, 2000, 1], [2001, 3000, 4], [3001, 6000, 8]],
         'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 1e-4, 'grad_clip': 1.,
         'field_loss': 'same train80 channel RMS; uniform original physical mask; .5 mean_t + .5 endpoint',
         'extras': ['(x-cx)/(20D)', '(y-cy)/(8D)', 'wall/(8D)', 'D-1', 'dt*Uinf/(.25D)-1'],
         'clock_step': 'scale/warp: last + (dt*Uinf/D)/.25 * (raw_output-last); control: raw_output',
         'warp': 'piecewise monotone map xref=[xmin,-2,0,10,xmax] to [xmin,cx-D,cx,cx+5D,xmax]; '
                 'yref=[ymin,-3,0,3,ymax] to [ymin,cy-1.5D,cy,cy+1.5D,ymax]',
         'warp_observations': '4096 sites/frame for all arms; warp_native samples new sites from native CFD; '
                              'warp64 interpolates the existing uniform 4096 observations',
         'initialization': 'frozen K8; extra lifting columns zero, original grid columns relocated',
         'dtype': 'float32 without AMP; seeded torch/numpy, grid_sample backward not claimed bit-deterministic',
         'normalization': json.loads((ROOT / 'outputs/round2/normalization.json').read_text()),
         'base_sha256': sha(BASE), 'data_view_sha256': sha(DATA),
         'source_h5_sha256': sha(DATA.with_name('dataset_long.h5')),
         'tail_h5_sha256': sha(DATA.with_name('dataset_long40_tail_v1.h5')),
         'script_sha256': sha(__file__), 'config': cfg,
         'primary_horizon': 8, 'reported_horizons': [2, 8, 20, 40],
         'forecast_support': 'four true fields, no future true fields fed after startup',
         'resolution_diagnostic': {'models': ['F0', 'K8'], 'sets': ['val', 'test_already_seen'],
             'arms': ['native64', 'upsample64_to128', 'native128'],
             'all_scoring': 'original uniform64 mask/raw pressure; 128 prediction interpolated back',
             'native128_uses_four_times_spatial_observations': True,
             'zero_shot_resolution_failure_does_not_disprove_better_representation_after_training': True}}
    for seed in p['seeds']:
        rng = np.random.default_rng(seed)
        tasks = OUT / f'tasks_seed{seed}.npz'
        np.savez(tasks, cases=rng.integers(0, 80, (STEPS, 2)), starts=rng.integers(0, 150, (STEPS, 2)))
        # 161 frames: start=149 gives support149..152, future153..160 (K8).
        p[f'tasks_seed{seed}_sha256'] = sha(tasks)
    save(OUT / 'preregister.json', p)
    print('PREPARED', flush=True)


def prepare_data():
    from cpu_data_prepare import frames
    from flowproxy.geometry import geometry_channels
    p = protocol(); cfg = p['config']; tmp = WARP.with_suffix('.partial.h5')
    if WARP.exists() or tmp.exists():
        raise FileExistsError('Warp cache already exists')
    if sha(DATA.with_name('dataset_long.h5')) != p['source_h5_sha256']:
        raise ValueError('Source data changed')
    started = time.perf_counter(); rows = []
    with h5py.File(DATA, 'r') as original, h5py.File(tmp, 'x') as dest:
        dest.attrs['protocol_sha256'] = sha(OUT / 'preregister.json')
        cases = dest.create_group('cases')
        for role in ['train', 'val']:
            for cid in p[f'{role}_ids']:
                g = original['cases'][cid]; row = attrs(g); m = mapping(cfg, row)
                items = frames(ROOT / 'data/raw/cases_long40' / cid)
                sampler = NativeSampler(m['xx'], m['yy'], m['mask'])
                # Test barycentric implementation against the original interpolator on a training anchor.
                if role == 'train' and not rows:
                    from scipy.interpolate import LinearNDInterpolator
                    from cpu_data_prepare import read_point_fields
                    pts, values = read_point_fields(items[0][1])
                    expected = LinearNDInterpolator(pts, values, fill_value=0)(sampler.q)
                    expected = expected.reshape(64, 64, 3).transpose(2, 0, 1).astype(np.float32) * m['mask'][None]
                    actual = sampler.sample(items[0][1])
                    delta = float(np.abs(expected - actual).max())
                    if delta > 1e-6: raise ValueError('Interpolator differs')
                    save(OUT / 'interpolation_smoke.json', {'max_absolute_delta': delta})
                count = 161 if role == 'train' else 4
                out = cases.create_group(cid)
                field = out.create_dataset('fields', (count, 3, 64, 64), dtype='f4', chunks=(1, 3, 64, 64), compression='lzf')
                for i, (_, native) in enumerate(items[:count]):
                    field[i] = sampler.sample(native)
                for key, value in m.items(): out.create_dataset(key, data=value)
                for key, value in row.items(): out.attrs[key] = value
                out.create_dataset('times', data=g['times'][:count])
                rows.append({'id': cid, 'role': role, 'frames': count, 'finite': bool(np.isfinite(field[...]).all())})
                if not rows[-1]['finite']:raise ValueError('Non-finite native warp')
                save(OUT / 'data_status.json', {'status': 'running', 'completed': len(rows), 'total': 88})
                if len(rows) % 8 == 0:print('WARP_DATA', len(rows), round(time.perf_counter()-started,1), flush=True)
    tmp.replace(WARP)
    save(OUT / 'data_report.json', {'status': 'complete', 'rows': rows, 'warp_h5_sha256': sha(WARP),
                                 'seconds': time.perf_counter()-started, 'training_future_labels': 'original uniform64',
                                 'native_validation_reads': 'four support frames only'})
    save(OUT / 'data_status.json', {'status': 'complete', 'completed': 88, 'total': 88})
    print('WARP_DATA_COMPLETE', flush=True)


def resolution():
    import torch
    import torch.nn.functional as F
    from flowproxy.models import build_model
    from flowproxy.geometry import geometry_channels
    from flowproxy.round2.metrics import per_frame_rel
    from cpu_data_prepare import frames, SampleCache
    from round2_complete40_v1 import MODELS, frame_summary, summarize
    torch.set_num_threads(1); p = protocol(); cfg = p['config']; device = 'cuda'
    out = OUT / 'resolution'; out.mkdir(exist_ok=False)
    rows = {name: {arm: {'val': [], 'test_already_seen': []} for arm in ['native64','upsample64_to128','native128']} for name in MODELS}
    supports = {}
    with h5py.File(DATA, 'r') as h:
        for role, ids in [('val',p['val_ids']), ('test_already_seen',p['old_seen_test_ids'])]:
            for cid in ids:
                g=h['cases'][cid]; q=attrs(g)
                cfg128=copy.deepcopy(cfg);cfg128['grid']={'nx':128,'ny':128}
                geo=geometry_channels(cfg128,**q); sampler=SampleCache(cfg128,geo['mask'])
                items=frames(ROOT/'data/raw/cases_long40'/cid)
                native=np.stack([sampler.sample(path) for _,path in items[:4]])
                supports[cid]=(torch.from_numpy(g['fields'][:4]).to(device),torch.from_numpy(g['geom'][...])[None].to(device),
                    torch.from_numpy(native).to(device),torch.from_numpy(geo['geom'])[None].to(device),g['times'][...],role)
    reference=json.loads((ROOT/'outputs/cpu_round1/full40_eval/complete40_val.json').read_text())
    parity=0.
    for name,path in MODELS.items():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=build_model(saved['model_name'],saved['in_channels'],saved['out_channels'],cfg).to(device)
        model.load_state_dict(saved['model']);model.eval()
        with h5py.File(DATA,'r') as h,torch.inference_mode():
            for cid,(f,g,fn,gn,t,role) in supports.items():
                for arm in rows[name]:
                    if arm=='native64': init=f; geom=g; n=64
                    elif arm=='upsample64_to128':
                        init=F.interpolate(f,size=(128,128),mode='bilinear',align_corners=True)
                        geom=F.interpolate(g,size=(128,128),mode='bilinear',align_corners=True);n=128
                    else:init=fn;geom=gn;n=128
                    window=deque(init.unbind(0),maxlen=4);pred=[]
                    for step in range(32):
                        v=model(torch.cat([geom,torch.stack(tuple(window)).reshape(1,12,n,n)],1))[0,:3]
                        window.append(v);pred.append(v)
                    prediction=torch.stack(pred)
                    if n!=64:prediction=F.interpolate(prediction,size=(64,64),mode='bilinear',align_corners=True)
                    true=torch.from_numpy(h['cases'][cid]['fields'][4:36]).to(device)
                    mask=torch.from_numpy(h['cases'][cid]['mask'][...]).to(device)
                    err=per_frame_rel(prediction,true,mask).cpu().numpy()
                    metrics=frame_summary(err,float(t[3]),float(t[35]),8,True)
                    rows[name][arm][role].append({'id':cid,**metrics})
                    if arm=='native64' and role=='val':
                        old=next(x for x in reference['models'][name]['horizons']['8s']['cases'] if x['id']==cid)
                        for k in ['mean','end','max']:parity=max(parity,float(np.abs(np.array(old[k])-metrics[k]).max()))
        del model;saved=None;torch.cuda.empty_cache()
        print('RESOLUTION_MODEL_DONE',name,flush=True)
    if parity>1e-6:raise ValueError('Resolution control fails frozen parity')
    summary={name:{arm:{role:summarize(cases) for role,cases in sets.items()} for arm,sets in arms.items()} for name,arms in rows.items()}
    save(out/'report.json',{'cases':rows,'summary':summary,'native64_val_max_absolute_delta':parity,
                          'native128_extra_spatial_observations':4,'no_training_performed':True})
    save(out/'summary.json',summary)
    print('RESOLUTION_COMPLETE',flush=True)


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
    def load(ids, role):
        fields=[];native=[];geom=[];mask=[];times=[];coord=[];forward=[];backward=[];Ds=[]
        with h5py.File(DATA,'r') as h:
            wh=h5py.File(WARP,'r') if mode=='warp_native' else None
            try:
                for cid in ids:
                    g=h['cases'][cid];q=attrs(g);m=mapping(cfg,q)
                    fields.append(g['fields'][:161 if role=='train' else 164]);mask.append(g['mask'][...]);times.append(g['times'][...])
                    if mode.startswith('warp'):
                        geom.append(m['geom']);coord.append(m['coords'])
                    else:
                        gg=geometry_channels(cfg,**q);geom.append(g['geom'][...])
                        coord.append(np.stack([(gg['xx']-q['cx'])/(20*q['D']),(gg['yy']-q['cy'])/(8*q['D']),gg['sdf_wall']/(8*q['D'])]))
                    forward.append(m['forward']);backward.append(m['backward']);Ds.append(q['D'])
                    if wh is not None:native.append(wh['cases'][cid]['fields'][...])
            finally:
                if wh is not None:wh.close()
        values={k:torch.from_numpy(np.stack(v).astype(np.float32)).to(device) for k,v in
                {'fields':fields,'geom':geom,'mask':mask,'times':times,'coords':coord,'forward':forward,'backward':backward}.items()}
        values['D']=torch.tensor(Ds,device=device)
        if native:values['native']=torch.from_numpy(np.stack(native)).to(device)
        return values
    train=load(p['train_ids'],'train');val=load(p['val_ids'],'val')
    saved=torch.load(BASE,map_location='cpu',weights_only=False);inc=16 if mode=='control' else 21
    model=build_model(saved['model_name'],inc,3,cfg).to(device)
    state=model.state_dict()
    for key,value in saved['model'].items():
        if key=='lift.0.weight' and inc>16:
            state[key].zero_();state[key][:,:16]=value[:,:16];state[key][:,-2:]=value[:,-2:]
        else:state[key].copy_(value)
    model.load_state_dict(state);del state
    scales=torch.tensor(p['normalization']['channel_rms'],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    def sample_forward(data,idx,history):
        if mode=='warp64':
            b,k,c,h,w=history.shape
            grid=data['forward'][idx,None].expand(-1,k,-1,-1,-1).reshape(b*k,64,64,2)
            return F.grid_sample(history.reshape(b*k,c,h,w),grid,align_corners=True,padding_mode='border').reshape(b,k,c,64,64)
        return history
    def startup(data,idx,starts):
        source=data['native'] if mode=='warp_native' else data['fields']
        hist=torch.stack([source[int(i),int(s):int(s)+4] for i,s in zip(idx,starts)])
        return sample_forward(data,idx,hist)
    def step(data,idx,hist,dt):
        x=torch.cat([data['geom'][idx],hist.reshape(len(idx),12,64,64)],1)
        if mode!='control':
            constants=torch.stack([data['D'][idx]-1,dt*cfg['physics']['U_inf']/(.25*data['D'][idx])-1],1)
            extra=torch.cat([data['coords'][idx],constants[:,:,None,None].expand(-1,-1,64,64)],1)
            x=torch.cat([x,extra],1)
        raw=model(x)[:,:3]
        if mode=='control':return raw
        alpha=(dt*cfg['physics']['U_inf']/(.25*data['D'][idx]))[:,None,None,None]
        return hist[:,-1]+alpha*(raw-hist[:,-1])
    def physical(data,idx,pred):
        if mode.startswith('warp'):
            return F.grid_sample(pred,data['backward'][idx],align_corners=True,padding_mode='border')
        return pred
    # The scale arm's network before the declared clock update preserves original K8.
    model.eval();initial_network_delta=None
    if mode in ['control','scale']:
        old=build_model(saved['model_name'],16,3,cfg).to(device);old.load_state_dict(saved['model']);old.eval()
        with torch.inference_mode():
            idx=torch.tensor([0],device=device);hist=val['fields'][:1,:4]
            x=torch.cat([val['geom'][:1],hist.reshape(1,12,64,64)],1)
            ref=old(x)
            if inc>16:x=torch.cat([x,torch.zeros(1,5,64,64,device=device)],1)
            initial_network_delta=float((ref-model(x)).abs().max())
        if initial_network_delta>1e-6:raise ValueError('Lift conversion fails parity')
        del old
    history=[];best=None
    @torch.inference_mode()
    def evaluate(update):
        model.eval();cases=[]
        for i,cid in enumerate(p['val_ids']):
            idx=torch.tensor([i],device=device);hist=startup(val,idx,[0]);pred=[]
            for n in range(160):
                dt=val['times'][i,4+n]-val['times'][i,3+n]
                v=step(val,idx,hist,dt[None]);hist=torch.cat([hist[:,1:],v[:,None]],1)
                pred.append(physical(val,idx,v)[0])
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
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':update,'mode':mode,'seed':seed,
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
        tick=time.perf_counter();model.train();idx=torch.from_numpy(tasks['cases'][update-1]).to(device);starts=tasks['starts'][update-1]
        hist=startup(train,idx,starts);K=8 if smoke else (1 if update<=2000 else 4 if update<=3000 else 8)
        losses=[]
        for j in range(K):
            frame=torch.tensor(starts+4+j,device=device)
            dt=train['times'][idx,frame]-train['times'][idx,frame-1]
            v=step(train,idx,hist,dt);mapped=physical(train,idx,v)
            target=train['fields'][idx,frame];mask=train['mask'][idx,None]
            loss=(((mapped-target)/scales[None,:,None,None]).square()*mask).sum((1,2,3))/(3*mask.sum((1,2,3)))
            losses.append(loss.mean());hist=torch.cat([hist[:,1:],v[:,None]],1)
        loss=.5*torch.stack(losses).mean()+.5*losses[-1]
        if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
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
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':max_steps,'mode':mode,'seed':seed,
                    'model_name':saved['model_name'],'in_channels':inc,'out_channels':3},job/'last.pt')
        save(job/'done.json',{'status':'complete','selected_step':best[1],'selected_R8':best[0][0],
            'seconds':time.perf_counter()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'median_update_seconds':float(np.median(update_durations)),'clip_hits':clip_hits,
            'selected_state_sha256':state_sha(torch.load(job/'selected.pt',map_location='cpu',weights_only=False)['model'])})
    save(job/'status.json',{'status':'complete','step':max_steps,'selected_step':best[1],'selected_R8':best[0][0]})
    print('DONE',mode,seed,'selected',best[1],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','data','resolution','smoke','train'])
    parser.add_argument('--mode',choices=MODES,default='control');parser.add_argument('--seed',type=int,default=11)
    a=parser.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='data':prepare_data()
    elif a.command=='resolution':resolution()
    else:run(a.mode,a.seed,smoke=a.command=='smoke')
