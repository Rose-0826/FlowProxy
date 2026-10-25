#!/usr/bin/env python3
"""Two matched seeds: replace TF eight targets by autonomous8, retain period return."""
from pathlib import Path
import argparse, json, os, subprocess, sys, time, hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT, rows, startup, scales, score, summaries, protected, now
from research_v2_outer_support_train import dataset, cycle
from research_v2_final_confirmation import infer
from research_v2_reflection_train import ReflectionOperator
from research_v2_state_probe import ProjectedState
from corrected_benchmark_train_v1 import model_from, protocol, train_loss
from geometry_attack_v2 import save, sha

DEST=OUT/'next_round/paired_rollout8';SEEDS=[711,722];CHECKS=[0,250,500,750,1000]
INITIAL=OUT/'final_research_confirmation/selected_research_base_weights.pt'
INITIAL_REPORT=OUT/'final_research_confirmation/outer96_seed622_development.json'
PERIODS=OUT/'outer_support_training/periods96.json'

def balance(s,h='40s'):
    return .5*(s['development_inside24'][h]['mean_R']+s['seen_extreme32'][h]['mean_R'])

def evaluate(model):
    model.eval();forecasts=[];registry=[]
    for q in rows():
        r=startup(q);p,d=infer(model,r,True);forecasts.append((q,r,p))
        registry.append({'id':q['id'],'diagnostic':d,'prediction_sha256':hashlib.sha256(p.tobytes()).hexdigest()})
    return {'summary':summaries([score(q,p,r) for q,r,p in forecasts]),
        'registry':registry,'all_forecasts_before_target_future_scoring':True}

def prepare():
    protected();DEST.mkdir(parents=True,exist_ok=False)
    c={'utc':now(),'source_sha256':sha(__file__),'arms':['TF','AR8'],'seeds':SEEDS,
        'initial_weights':str(INITIAL),'initial_sha256':sha(INITIAL),'updates':1000,'checks':CHECKS,
        'lr':1e-5,'weight_decay':1e-4,'clip':1.,'batch':4,
        'loss':'TF: eight true-history one-step targets; AR8: same eight targets but entirely own history after first4; both +.1 one-period return',
        'data':'same original80+outer16 TRAINING; 3 old +1 outer per batch; same prewritten windows in each pair',
        'operator':'same C2 reflection +solid mask +known velocityBC before feedback',
        'decoder':'same causal 24/48 spectral guard +unit-circleDMD9',
        'optimizer':'both restart AdamW from same frozen research weights; no optimizer-state imbalance',
        'selection_data':'previously seen56 DEVELOPMENT only; latest independent8 excluded from selection and training',
        'selection':'balancedRi40; both group means and all56 maxRi cannot worsen from frozen research candidate; short8s balanced <=1.05 and pass loss<=1/56; include0',
        'signal':'both AR8 seeds outperform matched TF controls; at least one >=10% balancedRi40 reduction, all predeclared guards',
        'early_stop':'after500 if no2% improvement and current balanced score >1.25 frozen baseline',
        'channel_scales':scales(),'fallback_immutable':True}
    save(DEST/'preregister.json',c)
    periods=json.loads(PERIODS.read_text())['rows']
    for seed in SEEDS:
        rng=np.random.default_rng(seed);picks=[];starts=[]
        for k in range(c['updates']):
            pick=np.r_[rng.choice(80,3,replace=False),rng.integers(80,96)]
            limit=min(165-4-periods[int(j)]['N']-1 for j in pick)
            picks.append(pick);starts.append(int(rng.integers(0,limit+1)))
        np.savez(DEST/f'tasks_seed{seed}.npz',picks=picks,starts=starts)

def worker(arm,seed,smoke=False):
    torch.set_num_threads(1);protected();c=json.loads((DEST/'preregister.json').read_text())
    assert sha(__file__)==c['source_sha256'] and sha(INITIAL)==c['initial_sha256']
    data=dataset();periods=json.loads(PERIODS.read_text())['rows'];task=np.load(DEST/f'tasks_seed{seed}.npz')
    assert [a['id'] for a in data]==[a['id'] for a in periods]
    base,saved=model_from(INITIAL,protocol()['config'],seed=seed)
    model=ProjectedState(ReflectionOperator(base),True)
    scale=torch.as_tensor(c['channel_scales'],device='cuda')
    def batch(k):
        pick=task['picks'][k];s=int(task['starts'][k])
        if smoke:
            pick=np.argsort([p['N'] for p in periods])[-4:];s=0
        g=torch.as_tensor(np.stack([data[int(j)]['geom'] for j in pick]),device='cuda')
        f=torch.as_tensor(np.stack([data[int(j)]['fields'][s:s+12] for j in pick]),device='cuda')
        m=torch.as_tensor(np.stack([data[int(j)]['mask'] for j in pick]),device='cuda')
        return pick,g,f,m
    def loss(k):
        pick,g,f,m=batch(k)
        return train_loss(model,g,f,m,scale,1 if arm=='TF' else 8)+.1*cycle(model,g,f,m,scale,periods,pick)[0]
    if smoke:
        model.train();torch.cuda.reset_peak_memory_stats();a=loss(0);a.backward();torch.cuda.synchronize()
        assert torch.isfinite(a) and all(torch.isfinite(p.grad).all() for p in base.parameters() if p.grad is not None)
        save(DEST/f'smoke_{arm}.json',{'passed':True,'arm':arm,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'loss':float(a.detach()),'no_optimizer_update':True});return
    folder=DEST/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False)
    opt=torch.optim.AdamW(base.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    initial=json.loads(INITIAL_REPORT.read_text());ref=initial['summary'];best=(float('inf'),float('inf'),0)
    history=[];begin=time.perf_counter();clips=0
    for step in range(c['updates']+1):
        if step:
            model.train();opt.zero_grad(set_to_none=True);l=loss(step-1);assert torch.isfinite(l);l.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(base.parameters(),c['clip']));assert np.isfinite(norm)
            clips+=int(norm>c['clip']);opt.step()
            if step%25==0:save(folder/'status.json',{'status':'training','step':step,'loss':float(l.detach()),'elapsed_seconds':time.perf_counter()-begin,'utc':now()})
        if step in CHECKS:
            ev=initial if step==0 else evaluate(model);s=ev['summary'];b=balance(s);worst=s['development_all56']['40s']['max_R']
            eligible=(balance(s,'8s')<=1.05*balance(ref,'8s')
                and s['development_all56']['8s']['n_pass']>=ref['development_all56']['8s']['n_pass']-1
                and worst<=ref['development_all56']['40s']['max_R']+1e-6
                and all(s[g]['40s']['mean_R']<=ref[g]['40s']['mean_R']+1e-6 for g in ['development_inside24','seen_extreme32']))
            key=(b,worst,step);save(folder/f'development_{step}.json',ev)
            history.append({'step':step,'balanced_R40':b,'max_R40':worst,'eligible':eligible,'summary':s})
            if eligible and key<best:
                best=key;torch.save({'model':base.state_dict(),'model_name':saved['model_name'],
                    'in_channels':saved['in_channels'],'out_channels':saved['out_channels'],'config':protocol()['config'],
                    'seed':seed,'update':step,'optimizer':opt.state_dict(),
                    'requires_research_operator_wrapper':'ProjectedState(ReflectionOperator(base),True)',
                    'role':'seen56 development-selected next-round research ONLY'},folder/'selected_base_weights.pt')
                save(folder/'selected_development.json',ev);save(folder/'selection.json',{'step':step,'key':list(best),'sha256':sha(folder/'selected_base_weights.pt')})
            save(folder/'history.json',history);print('ROLLOUT8',arm,seed,step,round(b,6),eligible,flush=True)
            if step>=500 and best[0]>.98*balance(ref) and b>1.25*balance(ref):break
    protected();save(folder/'done.json',{'utc':now(),'last_step':step,'best':list(best),'clips':clips,'wall_seconds':time.perf_counter()-begin})
    save(folder/'status.json',{'status':'complete','utc':now(),'step':step,'best':list(best)})

def main():
    prepare();env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    for arm in ['TF','AR8']:
        with (DEST/f'smoke_{arm}.log').open('x') as log:
            subprocess.run([sys.executable,__file__,'--worker',arm,'--seed',str(SEEDS[0]),'--smoke'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    maxmem=max(json.loads((DEST/f'smoke_{a}.json').read_text())['peak_GPU_GiB'] for a in ['TF','AR8'])
    assert maxmem<5.3,('Too much GPU memory for 4 workers',maxmem)
    jobs=[]
    for seed in SEEDS:
        for arm in ['TF','AR8']:
            log=(DEST/f'{arm}_seed{seed}.log').open('x');p=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,env=env)
            jobs.append((arm,seed,p,log))
    try:
        while True:
            done=[{'arm':a,'seed':s,'exit':p.returncode} for a,s,p,f in jobs if p.poll() is not None]
            save(DEST/'status.json',{'status':'training','utc':now(),'workers':[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,f in jobs],'completed':done})
            if any(d['exit']!=0 for d in done):raise RuntimeError(done)
            if len(done)==4:break
            time.sleep(15)
    except BaseException:
        for a,s,p,f in jobs:
            if p.poll() is None:p.terminate()
        raise
    candidates=[];pairs=[];ref=json.loads(INITIAL_REPORT.read_text())['summary']
    for seed in SEEDS:
        values={}
        for arm in ['TF','AR8']:
            folder=DEST/f'{arm}_seed{seed}';ev=json.loads((folder/'selected_development.json').read_text())
            sel=json.loads((folder/'selection.json').read_text());values[arm]=balance(ev['summary'])
            candidates.append({'arm':arm,'seed':seed,'path':str(folder/'selected_base_weights.pt'),'selection':sel,'summary':ev['summary']})
        pairs.append({'seed':seed,'AR8_better':values['AR8']<values['TF'],'scores':values})
    signal=all(p['AR8_better'] for p in pairs) and any(a['arm']=='AR8' and balance(a['summary'])<=.9*balance(ref) for a in candidates)
    protected();save(DEST/'completed_report.json',{'utc':now(),'pairs':pairs,'candidates':candidates,'research_signal_passed':signal,'independent8_unused_for_selection':True,'fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'research_signal_passed':signal})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['TF','AR8']);ap.add_argument('--seed',type=int);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    try:
        if a.worker:worker(a.worker,a.seed,a.smoke)
        else:main()
    except Exception as e:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
