#!/usr/bin/env python3
"""Learn mean/H1 spatial residuals from TRAIN96 only, with a frozen causal clock."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,scales,score,summaries,protected,now
from research_v2_outer_support_train import dataset
from research_v2_final_confirmation import load,infer
from research_v3_clock_probe import features
from startup_observability_v1 import frequency
from research_v2_template_probe import basis,fit_startup
from research_v4_sparse_observer import constrain
from research_v5_split_observer import DEST,PILOT,context,rotate,neural_template,observe,hybrid,balanced,candidate_guard
from geometry_attack_v2 import save,sha

PREP=DEST/'residual_training_data'
WEIGHTS=OUT/'final_research_confirmation/selected_research_base_weights.pt'
STARTS=[0,24,48,72]
SEEDS=[1011,1022]

def features_for(row,Ccanonical):
    scale=np.asarray(scales(),np.float32)[:,None,None]
    return np.concatenate([row['geom'],(row['fields']/scale).reshape(12,64,64),
        (Ccanonical[:3]/scale).reshape(9,64,64)],0).astype(np.float32)

def prepare():
    protected();PREP.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    torch.manual_seed(1001);begin=time.perf_counter()
    data=dataset();assert len(data)==96
    periods=json.loads((OUT/'outer_support_training/periods96.json').read_text())['rows']
    assert [q['id'] for q in data]==[q['id'] for q in periods]
    save(PREP/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),
        'training':'96 corrected TRAIN cases,4 startup phases each;no development/fresh fields used as labels',
        'starts':STARTS,'samples':384,'input':'geometry4 + startup12 normalized + frozen mean/H1 coefficients9 normalized',
        'target':'TRAIN96 CFD full-window fitted mean/H1 minus frozen neural mean/H1,phase gauge at last startup frame',
        'future_frequency':'training-only label for CFD coefficient targets;not predictor input',
        'base_weights':str(WEIGHTS),'base_sha256':sha(WEIGHTS),
        'frozen_bootstrap':'same adaptive24/48,C2BC wrapper,selfH4 frequency using TRAIN96 prior',
        'reflection_augmentation':'exact y reflection,Uy sign reversal;not counted as new CFD'})
    prior={};prior_data=json.loads((OUT/'next_round/clock_probe96/TRAIN96_frequency_prior.json').read_text())
    for a in prior_data['models']:
        prior[a['shape']]=RBFInterpolator(np.array([q['features'] for q in a['geometry_groups']]),
            np.array([q['mean_field_St'] for q in a['geometry_groups']]),
            kernel='thin_plate_spline',degree=1,smoothing=a['selected']['smoothing'])
    model=load({'path':str(WEIGHTS)});inputs=[];targets=[];masks=[];startup_B=[];startup_res=[];registry=[]
    scale=np.asarray(scales(),np.float32)[:,None,None]
    for r,p in zip(data,periods):
        full_t=r['times']-r['times'][0]
        true_C=np.linalg.lstsq(basis(full_t,p['frequency']),r['fields'].reshape(165,-1),rcond=1e-10)[0].reshape(9,3,64,64)
        for start in STARTS:
            absolute_last=r['times'][start+3]
            row={'id':r['id'],'q':r['q'],'fields':r['fields'][start:start+4].copy(),
                'geom':r['geom'],'mask':r['mask'],'times':r['times'][start]+np.arange(165)*.25}
            own,d=infer(model,row,True);n=d['prefix_steps']
            obs=np.concatenate([row['fields'],own[:n]]);t=row['times'][:n+4]-row['times'][0]
            center=float(prior[r['q']['shape']](features(r['q'])[None])[0])/r['q']['D']
            f,_=frequency(t,obs[:,1][:,row['mask']>.5].astype(np.float64),4,center,.2)
            C=neural_template(row,own,n,f);params,initial=fit_startup(C,row,f,False)
            canonical=rotate(C,params[0])
            target=rotate(true_C,2*np.pi*p['frequency']*(absolute_last-r['times'][0]))
            inputs.append(features_for(row,canonical))
            targets.append(((target[:3]-canonical[:3])/scale).reshape(9,64,64).astype(np.float32))
            masks.append(row['mask'].astype(np.float32))
            B=basis(row['times'][:4]-row['times'][3],f,0.)
            residual=row['fields']-(B@canonical.reshape(9,-1)).reshape(4,3,64,64)
            startup_B.append(B[:,:3].astype(np.float32));startup_res.append((residual/scale).astype(np.float32))
            registry.append({'id':r['id'],'start_index':start,'prefix_steps':n,'self_frequency':f,
                'TRAIN_label_frequency':p['frequency'],'startup_phase':params[0],
                'initial_field_sha256':hashlib.sha256(row['fields'].tobytes()).hexdigest()})
        save(PREP/'status.json',{'status':'preparing_training_only','utc':now(),
            'cases_completed':len(inputs)//4,'total':96,'wall_seconds':time.perf_counter()-begin})
    file=PREP/'train96_fourstarts.npz'
    np.savez_compressed(file,inputs=np.stack(inputs),targets=np.stack(targets),masks=np.stack(masks),
        startup_B=np.stack(startup_B),startup_res=np.stack(startup_res))
    save(PREP/'training_manifest.json',{'utc':now(),'training_cases':96,'samples':len(inputs),
        'file':str(file),'sha256':sha(file),'registry':registry,'target_fields_train_only':True})
    protected();save(PREP/'status.json',{'status':'complete','utc':now(),
        'wall_seconds':time.perf_counter()-begin,'samples':len(inputs)})

def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU(),nn.Conv2d(b,b,3,padding=1),nn.GELU())

class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__();self.e0=block(25,32);self.e1=block(32,64);self.e2=block(64,128)
        self.d1=block(128+64,64);self.d0=block(64+32,32);self.out=nn.Conv2d(32,9,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):
        a=self.e0(x);b=self.e1(F.avg_pool2d(a,2));c=self.e2(F.avg_pool2d(b,2))
        d=self.d1(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        d=self.d0(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(d)

def predicted_coefficients(model,row,C,params,arm):
    canonical=rotate(C,params[0]);x=features_for(row,canonical)
    with torch.inference_mode():delta=model(torch.as_tensor(x[None],device='cuda'))[0].cpu().numpy().reshape(3,3,64,64)
    scale=np.asarray(scales(),np.float32)[:,None,None];delta*=scale
    corrected=canonical.copy()
    if arm=='pressure':corrected[:3,2]+=delta[:,2]
    elif arm=='full':corrected[:3]+=delta
    else:raise ValueError(arm)
    return constrain(corrected,row['mask'])

def evaluate(model,step,dest,inference_ref,short_ref):
    model.eval();forecasts={'full':[],'pressure':[]};registry=[]
    for q in rows():
        row,own,C,params,initial,clean,noisy=context(q)
        for arm in forecasts:
            coeff=predicted_coefficients(model,row,C,params,arm)
            pred,detail=observe(coeff,params[1],row,clean,[0.,params[1]])
            pred=hybrid(own,pred);forecasts[arm].append((q,row,pred))
            registry.append({'id':q['id'],'arm':arm,'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest()})
    save(dest/('forecasts_step'+str(step)+'.json'),{'utc':now(),'all_before_future_scoring':True,'registry':registry})
    results={}
    for arm,part in forecasts.items():
        cases=[score(q,pred,row) for q,row,pred in part];summary=summaries(cases)
        results[arm]={'summary':summary,'cases':cases,'balanced_R40':balanced(summary),
            'eligible':candidate_guard(summary,inference_ref,short_ref)}
    save(dest/('eval_step'+str(step)+'.json'),{'utc':now(),'step':step,'results':results})
    return results

def train(seed):
    assert seed in SEEDS;protected();torch.set_num_threads(1);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    dest=DEST/('residual_seed'+str(seed));dest.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    while not (PREP/'training_manifest.json').exists() or not (PILOT/'report.json').exists():
        if time.perf_counter()-begin>7200:raise RuntimeError('2h preparation dependency limit')
        if (PREP/'failure.json').exists() or (PILOT/'failure.json').exists():raise RuntimeError('Preparation dependency failed')
        save(dest/'status.json',{'status':'waiting_for_training_data_and_inference_pilot','utc':now()});time.sleep(20)
    prep=json.loads((PREP/'training_manifest.json').read_text());assert sha(prep['file'])==prep['sha256']
    with np.load(prep['file']) as a:data={k:a[k] for k in a.files}
    n=len(data['inputs']);assert n==384
    pilot=json.loads((PILOT/'report.json').read_text());ref=pilot['results'][pilot['selected']['method']]['summary']
    short=json.loads((OUT/'final_research_confirmation/outer96_seed622_development.json').read_text())['summary']
    save(dest/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'seed':seed,
        'TRAIN_manifest_sha256':sha(PREP/'training_manifest.json'),'data':prep['sha256'],
        'architecture':'25-input,mean/H1 9-output zero-initialized two-level convolutional residual network,32/64/128 channels',
        'optimizer':'AdamW lr1e-4,weight_decay1e-4,clip1,1000updates,batch8',
        'loss':'masked normalized coefficient MSE,mean weight1,H1 coefficients weight.5,+.1 four-startup reconstruction MSE',
        'checkpoint_candidates':[0,250,500,750,1000],'arms':['full','pressure'],
        'augmentation':'exact reflection50%,Uy signs changed in input,target,residual',
        'evaluation':'old56 development only; old independent8 batches excluded from selection',
        'eligibility':'against frozen selected inference-only correction: preserve8s,40s pass count,and either-group meanR/maxR (1% tolerance)',
        'selection':'eligible min balancedR40;step0 included;no last-weight default',
        'new_independent_confirmation':'only if BOTH seeds >=5% lower balancedR40 than inference reference,at least48/56 pass,and1%noise guard',
        'probe_budget':76,'fallback_unchanged':True})
    model=ResidualNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    weight=torch.tensor([1.,1.,1.,.5,.5,.5,.5,.5,.5],device='cuda')[None,:,None,None]
    entries=[];selected=None;selected_key=None
    def checkpoint(step):
        nonlocal selected,selected_key
        result=evaluate(model,step,dest,ref,short)
        for arm,a in result.items():
            entry={'step':step,'arm':arm,'balanced_R40':a['balanced_R40'],'eligible':a['eligible'],
                'summary':a['summary']};entries.append(entry)
            if a['eligible']:
                key=(a['balanced_R40'],a['summary']['development_all56']['40s']['max_R'],step,arm)
                if selected_key is None or key<selected_key:
                    selected_key=key;selected=entry
                    torch.save({'state_dict':model.state_dict(),'seed':seed,'step':step,'arm':arm,
                        'source_sha256':sha(__file__),'train_manifest_sha256':sha(PREP/'training_manifest.json')},dest/'selected.pt')
        print('RESIDUAL_EVAL',seed,step,[(a['arm'],round(a['balanced_R40'],5),a['eligible']) for a in entries if a['step']==step],flush=True)
    checkpoint(0)
    for step in range(1,1001):
        model.train();idx=rng.integers(0,n,8)
        x=data['inputs'][idx].copy();target=data['targets'][idx].copy();m=data['masks'][idx].copy()
        residual=data['startup_res'][idx].copy();B=data['startup_B'][idx]
        flips=rng.random(8)<.5
        for i in np.flatnonzero(flips):
            x[i]=x[i,:,::-1,:].copy();target[i]=target[i,:,::-1,:].copy()
            m[i]=m[i,::-1,:].copy();residual[i]=residual[i,:,:,::-1,:].copy()
            x[i,[5,8,11,14,17,20,23]]*=-1
            target[i,[1,4,7]]*=-1;residual[i,:,1]*=-1
        xx=torch.as_tensor(x,device='cuda');yy=torch.as_tensor(target,device='cuda')
        mask=torch.as_tensor(m[:,None],device='cuda');rr=torch.as_tensor(residual,device='cuda')
        bb=torch.as_tensor(B,device='cuda');delta=model(xx)*mask
        loss_coef=((delta-yy).square()*weight*mask).sum()/(mask.sum()*weight.sum())
        rebuilt=torch.einsum('btk,bkchw->btchw',bb,delta.reshape(8,3,3,64,64))
        loss_start=((rebuilt-rr).square()*mask[:,None]).sum()/(mask.sum()*12)
        loss=loss_coef+.1*loss_start;assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0:
            save(dest/'status.json',{'status':'training','utc':now(),'seed':seed,'step':step,
                'loss':float(loss.detach()),'coefficient_loss':float(loss_coef.detach()),
                'startup_loss':float(loss_start.detach()),'grad_norm':float(norm),
                'selected':None if selected is None else {k:selected[k] for k in ['step','arm','balanced_R40']},
                'wall_seconds':time.perf_counter()-begin})
        if step%250==0:checkpoint(step)
    protected();save(dest/'report.json',{'utc':now(),'seed':seed,'candidates':entries,'selected':selected,
        'reference_method':pilot['selected']['method'],'reference_balanced_R40':balanced(ref),
        'selected_sha256':None if selected is None else sha(dest/'selected.pt'),
        'selection_data':'old56 only; not prior independent batches','fallback_unchanged':True})
    save(dest/'status.json',{'status':'complete','utc':now(),'seed':seed,'wall_seconds':time.perf_counter()-begin,
        'selected':None if selected is None else {k:selected[k] for k in ['step','arm','balanced_R40']}})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    try:
        if args.prepare:prepare()
        else:train(args.seed)
    except Exception as e:
        dest=PREP if args.prepare else DEST/('residual_seed'+str(args.seed))
        dest.mkdir(parents=True,exist_ok=True);save(dest/'failure.json',{'utc':now(),'error':repr(e)});raise
