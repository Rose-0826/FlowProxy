#!/usr/bin/env python3
"""Two paired physical-reflection operator training experiments, broad seen development."""
from pathlib import Path
import argparse, json, os, subprocess, sys, time, hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,score,summaries,scales,protected,now
from corrected_benchmark_train_v1 import Store,model_from,train_loss,protocol,predict
from corrected_benchmark_controls_v1 import dmd_periodic
from period_consistency_train_v1 import cycle_loss
from cycle_landmarks_train_v1 import FORMAL,PERIODS
from research_v2_operator_probe import reflect_input,reflect_output
from geometry_attack_v2 import save,sha

TRAIN=OUT/'reflection_training';SEEDS=[411,422];CHECKS=[0,250,500,750,1000]

class ReflectionOperator(torch.nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,x):
        b=len(x);y=self.base(torch.cat([x,reflect_input(x)],0))[:,:3]
        return .5*(y[:b]+reflect_output(y[b:]))

def evaluate(model,case_list):
    predictions=[];registry=[]
    # Hold all forecasts and freeze their hashes before reading target future fields.
    for q in case_list:
        row=startup(q);direct=predict(model,row)
        closed,diag=dmd_periodic(np.concatenate([row['fields'],direct[:48]]),row['times'][:52],row['times'][4:164],row['mask'],9,scales())
        closed[:48]=direct[:48]
        predictions.append((q,row,direct,closed))
        registry.append({'id':q['id'],'direct_sha256':hashlib.sha256(direct.tobytes()).hexdigest(),
            'DMD9_sha256':hashlib.sha256(closed.tobytes()).hexdigest(),'DMD':diag})
    direct=[];closed=[]
    for q,row,a,b in predictions:
        direct.append(score(q,a,row));closed.append(score(q,b,row))
    return {'summary':{'direct':summaries(direct),'DMD9':summaries(closed)},'cases':{'direct':direct,'DMD9':closed},
        'registry':registry,'all_forecast_hashes_before_target_future_scoring':True}

def balanced(summary,horizon='40s'):
    return .5*(summary['DMD9']['development_inside24'][horizon]['mean_R']+summary['DMD9']['seen_extreme32'][horizon]['mean_R'])

def worst(summary):return summary['DMD9']['development_all56']['40s']['max_R']

def cached_evaluation(arm):
    p=json.loads((OUT/'operator_probe/report.json').read_text())['fallback' if arm=='control' else 'step_reflection']
    return p

def prepare():
    TRAIN.mkdir(exist_ok=False);protected();periods=json.loads(PERIODS.read_text())['rows']
    c={'utc':now(),'seeds':SEEDS,'arms':['control','reflection'],'updates_max':1000,'checks':CHECKS,'include_step0':True,
        'lr':1e-5,'weight_decay':1e-4,'clip':1.,'batch_cases':4,'lambda_cycle':.1,'training_cases':80,
        'loss':'same eight teacher-forced one-step targets +0.1 full-cycle return; only operator reflection changes',
        'reflection':'C2 Reynolds projection: average F(x) and reflected F(reflected x) at EACH step before feedback; all vector signs correct',
        'unchanged':['model parameter count','four true input fields','fixed train80 channel scales','native .25s','DMD9 decoder'],
        'initialization':'same immutable fallback weights; AdamW restarted; no new training data',
        'development':'all56 already-seen; inside24 and extreme32 jointly used for candidate selection, not blind tests',
        'selection':'minimum0.5inside24 meanRi40+0.5extreme32 meanRi40; eligible only if balanced8s meanRi<=1.05 fallback and8s pass loss<=1/56; tie maxRi40 then earliest',
        'method_signal':'both reflection seeds lower balancedRi40 than matched controls; at least one >=10% lower than original fallback; neither inside24 nor extreme32 meanRi40 worse than fallback',
        'early_stop':'at500 if best balancedRi40 not improved2% from own step0 and current score>1.25 own step0',
        'fair_compute':'reflection has two backbone calls per operator step and extra training cost; walltime reported, not claimed equal',
        'fallback_immutable':True,'no_new_fresh_confirmation':True,'source_sha256':sha(__file__),
        'baseline_probe_sha256':sha(OUT/'operator_probe/report.json'),'channel_scales':scales()}
    save(TRAIN/'preregister.json',c)
    for seed in SEEDS:
        rng=np.random.default_rng(seed);picks=[];starts=[]
        for k in range(1000):
            pick=rng.choice(80,4,replace=False);limit=min(165-4-periods[int(j)]['N']-1 for j in pick)
            picks.append(pick);starts.append(int(rng.integers(0,limit+1)))
        np.savez(TRAIN/f'tasks_seed{seed}.npz',picks=np.array(picks),starts=np.array(starts))

def worker(arm,seed,smoke=False):
    torch.set_num_threads(1);protected();c=json.loads((TRAIN/'preregister.json').read_text())
    assert sha(__file__)==c['source_sha256']
    data=Store('train').rows;periods=json.loads(PERIODS.read_text())['rows'];tasks=np.load(TRAIN/f'tasks_seed{seed}.npz')
    assert [r['id'] for r in data]==[r['id'] for r in periods]
    base,saved=model_from(FORMAL,protocol()['config'],seed=seed)
    model=base if arm=='control' else ReflectionOperator(base)
    channel_scales=torch.as_tensor(c['channel_scales'],device='cuda')
    def batch(k):
        pick=tasks['picks'][k];start=int(tasks['starts'][k])
        geom=torch.as_tensor(np.stack([data[int(j)]['geom'] for j in pick]),device='cuda')
        fields=torch.as_tensor(np.stack([data[int(j)]['fields'][start:start+12] for j in pick]),device='cuda')
        mask=torch.as_tensor(np.stack([data[int(j)]['mask'] for j in pick]),device='cuda')
        return pick,geom,fields,mask
    if smoke:
        model.train();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();pick,geom,fields,mask=batch(0)
        short=train_loss(model,geom,fields,mask,channel_scales,1)
        cyc,n=cycle_loss(model,geom,fields,mask,channel_scales,periods,pick)
        loss=short+.1*cyc;loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in base.parameters() if p.grad is not None)
        save(TRAIN/'smoke.json',{'passed':True,'no_optimizer_updates':True,'finite_gradients':True,'unrolled_steps':n,
            'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'wall_seconds':time.perf_counter()-begin,
            'loss':float(loss.detach()),'operator_is_exact_C2_reflection_projection':True})
        return
    folder=TRAIN/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False)
    opt=torch.optim.AdamW(base.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    save(folder/'configuration.json',{'arm':arm,'seed':seed,'operator_reflection':arm=='reflection',
        'tasks_sha256':sha(TRAIN/f'tasks_seed{seed}.npz'),'lr':c['lr'],'initial_weight_sha256':sha(FORMAL)})
    original=cached_evaluation('control')['summary'];initial=cached_evaluation(arm)['summary']
    best=(float('inf'),float('inf'),0);history=[];starttime=time.perf_counter();clips=0;maxnorm=0.;last=0
    for step in range(1001):
        if step:
            model.train();opt.zero_grad(set_to_none=True);pick,geom,fields,mask=batch(step-1)
            short=train_loss(model,geom,fields,mask,channel_scales,1)
            cyc,_=cycle_loss(model,geom,fields,mask,channel_scales,periods,pick)
            loss=short+.1*cyc;assert torch.isfinite(loss);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(base.parameters(),c['clip']));assert np.isfinite(norm)
            maxnorm=max(maxnorm,norm);clips+=int(norm>c['clip']);opt.step()
            if step%25==0:save(folder/'status.json',{'status':'training','step':step,'elapsed_seconds':time.perf_counter()-starttime,
                'loss':float(loss.detach()),'gradient_norm_before_clip':norm,'utc':now()})
        if step in CHECKS:
            ev=cached_evaluation(arm) if step==0 else evaluate(model,rows())
            s=ev['summary'];score40=balanced(s);max40=worst(s)
            eligible=balanced(s,'8s')<=1.05*balanced(original,'8s') and s['direct']['development_all56']['8s']['n_pass']>=original['direct']['development_all56']['8s']['n_pass']-1
            key=(score40,max40,step);save(folder/f'development_{step}.json',ev)
            history.append({'step':step,'balanced_R40':score40,'maximum_R40':max40,'eligible_short8s':eligible,'summary':s})
            if eligible and key<best:
                best=key
                torch.save({'model':base.state_dict(),'model_name':saved['model_name'],'in_channels':saved['in_channels'],
                    'out_channels':saved['out_channels'],'config':protocol()['config'],'seed':seed,'update':step,
                    'operator_reflection':arm=='reflection','requires_research_operator_wrapper':arm=='reflection',
                    'optimizer':opt.state_dict(),'role':'seen-development-selected research candidate; NOT blind-confirmed'},folder/'selected_base_weights.pt')
                save(folder/'selected_development.json',ev)
                save(folder/'selection.json',{'step':step,'key':list(best),'sha256':sha(folder/'selected_base_weights.pt')})
            save(folder/'history.json',history);last=step
            print('REFLECTION_TRAIN',arm,seed,step,'balancedR40',round(score40,6),'eligible',eligible,flush=True)
            if step==500 and best[0]>.98*balanced(initial) and score40>1.25*balanced(initial):break
    protected();save(folder/'done.json',{'status':'complete','utc':now(),'last_update':last,'best':list(best),
        'max_gradient_norm':maxnorm,'clip_count':clips,'wall_seconds':time.perf_counter()-starttime})

def orchestrate():
    prepare();env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    with (TRAIN/'smoke.log').open('x') as f:
        subprocess.run([sys.executable,__file__,'--worker','reflection','--seed',str(SEEDS[0]),'--smoke'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    smoke=json.loads((TRAIN/'smoke.json').read_text());assert smoke['passed'] and smoke['peak_GPU_GiB']<5.2
    jobs=[]
    for seed in SEEDS:
        for arm in ['control','reflection']:
            log=(TRAIN/f'{arm}_seed{seed}.log').open('x')
            p=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((arm,seed,p,log))
    try:
        while True:
            completed=[{'arm':a,'seed':s,'exit_code':p.returncode} for a,s,p,f in jobs if p.poll() is not None]
            save(TRAIN/'status.json',{'status':'training','utc':now(),'workers':[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,f in jobs],
                'completed':completed,'smoke':smoke})
            if any(r['exit_code']!=0 for r in completed):raise RuntimeError(completed)
            if len(completed)==4:break
            time.sleep(15)
    except BaseException as e:
        for a,s,p,f in jobs:
            if p.poll() is None:p.terminate()
        save(TRAIN/'status.json',{'status':'failed','utc':now(),'error':repr(e)})
        raise
    for a,s,p,f in jobs:f.close()
    pairs=[];candidates=[];original=cached_evaluation('control')['summary']
    for seed in SEEDS:
        scores={}
        for arm in ['control','reflection']:
            folder=TRAIN/f'{arm}_seed{seed}';s=json.loads((folder/'selected_development.json').read_text())['summary']
            sel=json.loads((folder/'selection.json').read_text());scores[arm]=balanced(s)
            candidates.append({'arm':arm,'seed':seed,'path':str(folder/'selected_base_weights.pt'),'sha256':sha(folder/'selected_base_weights.pt'),
                'operator_reflection':arm=='reflection','selection':sel,'summary':s,'balanced_R40':balanced(s)})
        pairs.append({'seed':seed,'reflection_better_than_control':scores['reflection']<scores['control'],'scores':scores})
    qualified=[r for r in candidates if r['operator_reflection'] and r['balanced_R40']<=.9*balanced(original)
        and all(r['summary']['DMD9'][g]['40s']['mean_R']<=original['DMD9'][g]['40s']['mean_R'] for g in ['development_inside24','seen_extreme32'])]
    signal=all(p['reflection_better_than_control'] for p in pairs) and len(qualified)>0
    protected();save(TRAIN/'completed_report.json',{'utc':now(),'research_signal_passed':signal,'pairs':pairs,'candidates':candidates,
        'fallback_unchanged':True,'fresh_independent_confirmation_used':False,'role':'all evaluation DEVELOPMENT; not fresh test'})
    save(TRAIN/'status.json',{'status':'complete','utc':now(),'research_signal_passed':signal,'fallback_unchanged':True})
    print('REFLECTION_ROUND_COMPLETE',signal,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['control','reflection']);ap.add_argument('--seed',type=int)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.worker:worker(args.worker,args.seed,args.smoke)
    else:orchestrate()
