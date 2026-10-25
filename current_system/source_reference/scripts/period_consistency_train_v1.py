#!/usr/bin/env python3
"""One-cycle closure regularization; two matched pairs, no additional CFD."""
from pathlib import Path
import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from corrected_benchmark_train_v1 import Store,model_from,train_loss,field_loss_steps,evaluate,protocol
from startup_observability_v1 import frequency
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

OLD=ROOT/'outputs/corrected_benchmark_restart_v1'
OUT=ROOT/'outputs/period_consistency_train_v1'
UPDATES=1000
CHECKS=[0,250,500,750,1000]
SEEDS=[11,22]
LR=1e-5
WEIGHT=.1


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locked():return json.loads((OLD/'selected_model_lock.json').read_text())


def prepare_periods(train):
    rows=[]
    for row in train.rows:
        mask=row['mask']>.5
        t=row['times'].astype(np.float64)-float(row['times'][0])
        fields=row['fields'][:,1][:,mask].astype(np.float64)
        f,diag=frequency(t,fields,4,row['St']/row['q']['D'],.05)
        dt=float(np.median(np.diff(t)));count=1/(f*dt)
        assert np.allclose(np.diff(t),.25) and 12<count<40
        N=int(np.floor(count));alpha=float(count-N)
        start=32
        future=row['fields'][start+N:start+N+4].astype(np.float64)*(1-alpha)+row['fields'][start+N+1:start+N+5]*alpha
        target=row['fields'][start:start+4]
        norm=np.linalg.norm(target*mask[None,None],axis=(-2,-1))
        defect=np.linalg.norm((future-target)*mask[None,None],axis=(-2,-1))/np.maximum(norm,1e-8)
        rows.append({'id':row['id'],'frequency':f,'period_seconds':1/f,'dt':dt,'N':N,'alpha':alpha,
                     'fit':diag,'CFD_interpolated_cycle_defect_mean':defect.mean(0).tolist()})
    assert len(rows)==80
    save(OUT/'train80_periods.json',{'source':'train80 FULL training fields only; four-harmonic frequency fit around its force frequency +-5%',
                                   'rows':rows,'numerical_interpolation':'linear interpolation between adjacent native .25s frames',
                                   'defect_mean':np.mean([r['CFD_interpolated_cycle_defect_mean'] for r in rows],axis=0).tolist()})
    return rows


def make_tasks(periods,seed):
    rng=np.random.default_rng(seed);picks=[];starts=[]
    for i in range(UPDATES):
        pick=rng.choice(80,4,replace=False)
        # Both arms use the same samples and short-field targets; cycle labels need no future fields.
        limit=min(165-4-periods[j]['N']-1 for j in pick)
        picks.append(pick);starts.append(int(rng.integers(0,limit+1)))
    return np.array(picks),np.array(starts)


def cycle_loss(model,geom,fields,mask,scales,periods,pick):
    window=[fields[:,k] for k in range(4)];outputs=[]
    nmax=max(periods[j]['N'] for j in pick)+1
    for k in range(nmax):
        hist=torch.stack(window[-4:],1).reshape(4,12,64,64)
        pred=model(torch.cat([geom,hist],1))[:,:3]
        window.append(pred);outputs.append(pred)
    allpred=torch.stack(outputs,1)
    aligned=[]
    for b,j in enumerate(pick):
        N,alpha=periods[j]['N'],periods[j]['alpha']
        aligned.append((1-alpha)*allpred[b,N-4:N]+alpha*allpred[b,N-3:N+1])
    end=torch.stack(aligned)
    return field_loss_steps(end,fields[:,:4],mask,scales),nmax


def configure():
    OUT.mkdir(exist_ok=False)
    l=locked();assert sha(l['chosen']['path'])==l['chosen']['sha256']
    save(OUT/'preregister.json',{'registered_utc':now(),'network':l['chosen'],'seeds':SEEDS,
        'arms':['control','cycle'],'updates_max':UPDATES,'checks':CHECKS,'include_step0':True,
        'lr':LR,'AdamW_weight_decay':1e-4,'gradient_clip':1.,'lambda_cycle':WEIGHT,'batch_cases':4,
        'short_loss':'same teacher-forced eight targets as corrected K1; .5 mean +.5 last',
        'added_cycle_loss':'autonomous one-period rollout, interpolated last4 phase-matched states vs four true initial states; same channel scales and time weights',
        'period_labels':'train80 field-derived Fourier frequency only, never validation/test/confirmation labels',
        'normalization':'unchanged corrected train80 scales','validation_closure':'same H4, 48 self steps, shape mean +-10%; no new clock or reflection modules',
        'selection':'val8 minimum closure40 meanRi among checkpoints with 8/8 short8s pass; tie maximumRi then step; include step0',
        'stop_each_cycle_arm':'at500 if best meanRi has not improved >=2% vs frozen original and current meanRi>1.25*original, stop; controls complete1000',
        'method_gate':'both seeds: cycle better meanRi than matched control; at least one cycle meanRi<=.8*original and maxRi<=.9*original, keeping short8s 8/8',
        'fair_comparison':'same initialization, seed, samples, target windows, lr, scaling; extra one-cycle forward/backward compute is reported separately',
        'already_seen_retest':'any later 32+12 score is exploratory, not a new blind test',
        'no_new_CFD':True,'original_model_lock_sha256':sha(OLD/'selected_model_lock.json'),
        'original_test_reports_sha256':{role:sha(OLD/role/'report.json') for role in ['already_seen_test32','fresh_confirmation12']},
        'script_sha256':sha(__file__)})
    periods=prepare_periods(Store('train'))
    for seed in SEEDS:
        pick,start=make_tasks(periods,seed);np.savez(OUT/f'tasks_seed{seed}.npz',picks=pick,starts=start)


def worker(arm,seed,smoke=False):
    p=protocol();l=locked();train=Store('train');val=Store('val')
    periods=json.loads((OUT/'train80_periods.json').read_text())['rows']
    assert [r['id'] for r in periods]==[r['id'] for r in train.rows]
    a=np.load(OUT/f'tasks_seed{seed}.npz');picks=a['picks'];starts=a['starts']
    torch.set_num_threads(1)
    model,saved=model_from(l['chosen']['path'],p['config'],seed=seed)
    oldconfig=json.loads((Path(l['chosen']['path']).parent/'configuration.json').read_text())
    scales=torch.as_tensor(oldconfig['normalization_train80'],device='cuda')
    prior={k:(None,v) for k,v in l['shape_mean_train80_St'].items()}
    original=json.loads((Path(l['chosen']['path']).parent/'selected_validation.json').read_text())['summary']['closure']['40s']
    def batch(step):
        pick=picks[step];start=starts[step]
        geom=torch.as_tensor(np.stack([train.rows[j]['geom'] for j in pick]),device='cuda')
        fields=torch.as_tensor(np.stack([train.rows[j]['fields'][start:start+12] for j in pick]),device='cuda')
        mask=torch.as_tensor(np.stack([train.rows[j]['mask'] for j in pick]),device='cuda')
        return pick,geom,fields,mask
    if smoke:
        pick,geom,fields,mask=batch(0);model.train();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();short=train_loss(model,geom,fields,mask,scales,1)
        periodic,nmax=cycle_loss(model,geom,fields,mask,scales,periods,pick)
        loss=short+WEIGHT*periodic;loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
        save(OUT/'smoke.json',{'short_loss':float(short.detach()),'cycle_loss':float(periodic.detach()),'weighted_cycle_loss':float(WEIGHT*periodic.detach()),
            'autonomous_steps':nmax,'seconds_forward_backward':time.perf_counter()-started,'peak_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'finite_gradients':True,'no_optimizer_updates':True})
        print('PERIOD_SMOKE_OK',nmax,float(loss.detach()),flush=True);return
    folder=OUT/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    best=(float('inf'),float('inf'),0);history=[];started=time.perf_counter();peak_grad=0.;clips=0
    save(folder/'configuration.json',{'arm':arm,'seed':seed,'initial_sha256':l['chosen']['sha256'],
         'tasks_sha256':sha(OUT/f'tasks_seed{seed}.npz'),'lambda_cycle':WEIGHT if arm=='cycle' else 0.,'lr':LR})
    last=0
    for step in range(UPDATES+1):
        if step:
            pick,geom,fields,mask=batch(step-1);model.train();opt.zero_grad(set_to_none=True)
            short=train_loss(model,geom,fields,mask,scales,1)
            cyc=torch.zeros((),device='cuda')
            if arm=='cycle':cyc,_=cycle_loss(model,geom,fields,mask,scales,periods,pick)
            loss=short+WEIGHT*cyc;assert torch.isfinite(loss)
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            assert np.isfinite(norm);peak_grad=max(peak_grad,norm);clips+=int(norm>1.)
            opt.step()
            if step%25==0:save(folder/'status.json',{'step':step,'short_loss':float(short.detach()),'cycle_loss':float(cyc.detach()),
                                                    'elapsed_seconds':time.perf_counter()-started})
        if step in CHECKS:
            ev=evaluate(model,val,prior,p);a=ev['summary']['closure'];q=a['40s'];valid=a['8s']['n_pass']==8
            if step==0:assert abs(q['mean_R']-original['mean_R'])<1e-4
            key=(q['mean_R'],q['max_R'],step)
            save(folder/f'validation_{step}.json',ev);history.append({'step':step,'summary':ev['summary'],'eligible_short8s':valid})
            if valid and key<best:
                best=key;payload={'model':model.state_dict(),'model_name':saved['model_name'],'in_channels':saved['in_channels'],
                    'out_channels':saved['out_channels'],'update':step,'seed':seed,'arm':arm,'lambda_cycle':WEIGHT if arm=='cycle' else 0.,
                    'config':p['config'],'data_version':'corrected_benchmark_v1','optimizer':opt.state_dict()}
                torch.save(payload,folder/'selected.pt');save(folder/'selected_validation.json',ev)
                save(folder/'selection.json',{'step':step,'key':list(best),'sha256':sha(folder/'selected.pt')})
            save(folder/'history.json',history);last=step
            print('PERIOD_TRAIN',arm,seed,step,'R',round(q['mean_R'],4),'MAX',round(q['max_R'],4),'PASS',q['n_pass'],flush=True)
            if arm=='cycle' and step==500 and best[0]>.98*original['mean_R'] and q['mean_R']>1.25*original['mean_R']:
                print('PERIOD_EARLY_STOP',arm,seed,flush=True);break
    save(folder/'done.json',{'status':'complete','last_update':last,'best':list(best),'wall_seconds':time.perf_counter()-started,
                            'max_gradient_norm_before_clip':peak_grad,'clip_count':clips,'seed':seed,'arm':arm})


def orchestrate():
    configure()
    with (OUT/'smoke.log').open('x') as log:
        subprocess.run([sys.executable,__file__,'--smoke'],stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke=json.loads((OUT/'smoke.json').read_text())
    assert smoke['finite_gradients'] and smoke['peak_GPU_allocated_GiB']<5.2
    procs=[]
    for seed in SEEDS:
        for arm in ['control','cycle']:
            log=(OUT/f'{arm}_seed{seed}.log').open('x')
            process=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,
                                     env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            procs.append((arm,seed,process,log))
    save(OUT/'status.json',{'status':'training','workers':[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,f in procs],
                          'total':4,'smoke':smoke})
    for arm,seed,process,log in procs:
        rc=process.wait();log.close()
        if rc:raise RuntimeError((arm,seed,rc))
    baseline=json.loads((Path(locked()['chosen']['path']).parent/'selected_validation.json').read_text())['summary']['closure']['40s']
    candidates=[];paired=[]
    for seed in SEEDS:
        scores={}
        for arm in ['control','cycle']:
            folder=OUT/f'{arm}_seed{seed}';sel=json.loads((folder/'selection.json').read_text());ev=json.loads((folder/'selected_validation.json').read_text())
            scores[arm]=ev['summary']['closure']['40s']
            candidates.append({'arm':arm,'seed':seed,'path':str(folder/'selected.pt'),'sha256':sha(folder/'selected.pt'),
                               'selection':sel,'summary':ev['summary']})
        paired.append({'seed':seed,'cycle_better_mean_than_control':scores['cycle']['mean_R']<scores['control']['mean_R'],
                       'control':scores['control'],'cycle':scores['cycle']})
    qualified=[c for c in candidates if c['arm']=='cycle' and c['summary']['closure']['40s']['mean_R']<=.8*baseline['mean_R']
               and c['summary']['closure']['40s']['max_R']<=.9*baseline['max_R']]
    passed=all(v['cycle_better_mean_than_control'] for v in paired) and bool(qualified)
    chosen=min(qualified,key=lambda c:(c['summary']['closure']['40s']['mean_R'],c['summary']['closure']['40s']['max_R'])) if qualified else None
    save(OUT/'completed_report.json',{'pairs':paired,'all_candidates':candidates,'gate_passed':passed,
                                     'selected_cycle_for_followup_if_gate_passed':chosen,'baseline':baseline})
    save(OUT/'status.json',{'status':'complete','gate_passed':passed,'pairs':paired})
    print('PERIOD_COMPLETE_GATE',passed,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['control','cycle']);ap.add_argument('--seed',type=int,default=11);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    try:
        if args.worker or args.smoke:worker(args.worker or 'cycle',args.seed,args.smoke)
        else:orchestrate()
    except Exception as error:
        if not args.worker and not args.smoke and OUT.exists():save(OUT/'status.json',{'status':'failed','error':repr(error),'utc':now()})
        raise
