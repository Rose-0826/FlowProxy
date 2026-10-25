"""Two matching data/schedule arms, with a bounded 32-step rollout difference."""
from pathlib import Path
import json, argparse, traceback, time
import numpy as np
import torch
import optimize as o
from train_core import dataset, build_candidate

def rollout32_backward(model,g,f,m,scale,weight,factor=.25):
    # Forward states remain self-fed across all 32 steps. Backpropagation is
    # truncated every 8 steps so the real deployment drift is exercised without
    # retaining an excessive graph. Only the two newshape TRAIN picks are used.
    batch=len(f);window=[f[:,k] for k in range(4)];total=0.
    channels=torch.tensor([1.,2.,1.],device=f.device)[None,None,:,None,None]
    spatial=weight[:,None,None]
    for block in range(4):
        outputs=[]
        for k in range(8):
            pred=model(torch.cat([g,torch.stack(window[-4:],1).reshape(batch,12,64,64)],1))[:,:3]
            window.append(pred);outputs.append(pred)
        target=f[:,4+block*8:12+block*8];error=(torch.stack(outputs,1)-target)/scale[None,None,:,None,None]
        loss=(error.square()*spatial*channels).sum()/((spatial.sum()*channels.sum()*8).clamp_min(1))
        assert torch.isfinite(loss);(factor*loss/4).backward();total+=float(loss.detach())/4
        window=[v.detach() for v in window[-4:]]
    return total

def step(model,base,opt,g,f,m,scale,periods,picks,weights,rollout):
    model.train();opt.zero_grad(set_to_none=True)
    loss=.5*o.old.train_loss(model,g,f[:,:12],m,scale,1)+.5*o.old.train_loss(model,g,f[:,:12],m,scale,8)+.1*o.old.cycle(model,g,f[:,:12],m,scale,periods,picks)[0]
    assert torch.isfinite(loss);loss.backward()
    startup=rollout32_backward(model,g[2:],f[2:],m[2:],scale,weights[2:]) if rollout else 0.
    norm=float(torch.nn.utils.clip_grad_norm_(base.parameters(),1.));assert np.isfinite(norm);opt.step()
    return dict(base_loss=float(loss.detach()),startup32_loss=startup,gradient_norm=norm)

def tasks():
    path=o.OUT/'training_tasks.npz'
    if not path.exists():
        rng=np.random.default_rng(1602)
        # Two original TRAIN100, one prior newshape TRAIN24, one added TRAIN32.
        picks=np.column_stack([rng.integers(0,100,(1500,2)),rng.integers(100,124,1500),rng.integers(124,156,1500)])
        np.savez(path,picks=picks,starts=rng.integers(0,130,1500))
    return np.load(path)

def arm(name):
    cfg=o.check();arm=next(a for a in cfg['arms'] if a['name']==name);folder=o.OUT/name;folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    torch.manual_seed(1602);torch.set_num_threads(1);recs,data,templates,periods,weights=dataset()
    parent=json.loads((o.OUT/'parent_system.json').read_text());initial=parent['new_shapes']['checkpoint']
    base,saved=o.old.model_from(initial,o.old.protocol()['config'],seed=1602);model=o.old.ProjectedState(o.old.ReflectionOperator(base),True)
    scale=torch.as_tensor(o.old.p.scales(),device='cuda',dtype=torch.float32)
    opt=torch.optim.AdamW(base.parameters(),lr=arm['lr'],weight_decay=1e-4);plan=tasks();history=[];first=1;state=folder/'resume.pt'
    if state.exists():
        s=torch.load(state,map_location='cuda',weights_only=False);base.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);first=s['step']+1;history=s['history']
        for checkpoint in cfg['checkpoints']:
            if checkpoint>=first or any(h['name']==name+'_step'+str(checkpoint) for h in history):continue
            cp=folder/f'checkpoint_{checkpoint}.pt';assert cp.exists()
            history.append(build_candidate(cp,folder/f'development_{checkpoint}',recs,data,templates,name+'_step'+str(checkpoint)))
    begin=time.perf_counter()
    for k in range(first,arm['steps']+1):
        picks=plan['picks'][k-1];start=int(plan['starts'][k-1])
        g=torch.as_tensor(np.stack([data[int(j)]['geom'] for j in picks]),device='cuda')
        f=torch.as_tensor(np.stack([data[int(j)]['fields'][start:start+36] for j in picks]),device='cuda')
        m=torch.as_tensor(np.stack([data[int(j)]['mask'] for j in picks]),device='cuda')
        w=torch.as_tensor(np.stack([weights[int(j)] for j in picks]),device='cuda')
        metrics=step(model,base,opt,g,f,m,scale,periods,picks,w,arm['rollout32'])
        if k%25==0:
            o.save(folder/'status.json',dict(stage='training',step=k,**metrics,wall_seconds=time.perf_counter()-begin,utc=o.now()))
            tmp=folder/'resume.tmp.pt';torch.save(dict(model=base.state_dict(),optimizer=opt.state_dict(),step=k,history=history),tmp);tmp.replace(state)
        if k in cfg['checkpoints']:
            cp=folder/f'checkpoint_{k}.pt';torch.save(dict(model=base.state_dict(),model_name=saved['model_name'],in_channels=saved['in_channels'],out_channels=saved['out_channels'],
                config=o.old.protocol()['config'],seed=1602,additional_update=k,arm=arm,training_case_count=156,training_source_sha256=o.sha(o.OUT/'TRAIN156_manifest.json'),
                initial_checkpoint_sha256=o.sha(initial),requires_research_operator_wrapper='ProjectedState(ReflectionOperator(base),True)'),cp)
            o.save(folder/'status.json',dict(stage='building_TRAIN_libraries',step=k,utc=o.now()))
            history.append(build_candidate(cp,folder/f'development_{k}',recs,data,templates,name+'_step'+str(k)));o.save(folder/'history.json',history)
            torch.save(dict(model=base.state_dict(),optimizer=opt.state_dict(),step=k,history=history),state)
    o.save(folder/'done.json',dict(step=arm['steps'],utc=o.now()));o.save(folder/'status.json',dict(stage='complete',step=arm['steps'],utc=o.now()))

def smoke():
    cfg=o.check();torch.set_num_threads(1);recs,data,templates,periods,weights=dataset(False)
    parent=json.loads((o.OUT/'parent_system.json').read_text());results=[]
    for rollout in [False,True]:
        base,saved=o.old.model_from(parent['new_shapes']['checkpoint'],o.old.protocol()['config'],seed=1602)
        model=o.old.ProjectedState(o.old.ReflectionOperator(base),True);opt=torch.optim.AdamW(base.parameters(),lr=1e-5,weight_decay=1e-4)
        scale=torch.as_tensor(o.old.p.scales(),device='cuda',dtype=torch.float32);picks=np.array([0,50,100,112]);start=20
        g=torch.as_tensor(np.stack([data[j]['geom'] for j in picks]),device='cuda');f=torch.as_tensor(np.stack([data[j]['fields'][start:start+36] for j in picks]),device='cuda')
        m=torch.as_tensor(np.stack([data[j]['mask'] for j in picks]),device='cuda');w=torch.as_tensor(np.stack([weights[j] for j in picks]),device='cuda')
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        for _ in range(2):metrics=step(model,base,opt,g,f,m,scale,periods,picks,w,rollout)
        torch.cuda.synchronize();results.append(dict(rollout32=rollout,seconds_two_steps=time.perf_counter()-begin,peak_GPU_bytes=torch.cuda.max_memory_allocated(),**metrics))
        del opt,model,base;torch.cuda.empty_cache()
    o.save(o.OUT/'training_smoke.json',dict(results=results,TRAIN124_only=True,smoke_weights_discarded=True,production_weights_unchanged=True));print('TRAINING_SMOKE_OK',json.dumps(results),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:
        if args.smoke:smoke()
        else:arm(args.arm)
    except Exception as exc:
        o.save(o.OUT/((args.arm or 'smoke')+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),utc=o.now()));raise
