#!/usr/bin/env python3
"""Finite dependency batch: same96 continuation versus96+8 joint-corner training."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,scales,protected,now
from research_v2_outer_support_train import dataset as original_dataset,cycle
from research_v2_reflection_train import ReflectionOperator
from research_v2_state_probe import ProjectedState
from research_v3_rollout_train import evaluate,balance
from corrected_benchmark_train_v1 import model_from,protocol,train_loss
from startup_observability_v1 import frequency
from geometry_attack_v2 import save,sha

DEST=OUT/'next_round/paired_corner_training'
MANIFEST=OUT/'next_round/corner_training_cfd/training_manifest.json'
INITIAL=OUT/'final_research_confirmation/selected_research_base_weights.pt'
INITIAL_REPORT=OUT/'final_research_confirmation/outer96_seed622_development.json'
SEEDS=[811,822];CHECKS=[0,250,500,750,1000]

def data104():
    data=original_dataset();m=json.loads(MANIFEST.read_text());cache=Path(m['cache'])
    for q in m['cases']:
        meta=json.loads((cache/(q['id']+'.json')).read_text());assert meta['CFD_stable'] and sha(cache/(q['id']+'.npz'))==meta['cache_sha256']
        with np.load(cache/(q['id']+'.npz')) as a:
            data.append({'id':q['id'],'q':{k:q[k] for k in ['shape','Re','cx','cy','D']},
                **{k:a[k] for k in ['fields','geom','mask','times']}})
    assert len(data)==104;return data

def prepare():
    data=data104();periods=json.loads((OUT/'outer_support_training/periods96.json').read_text())['rows']
    m=json.loads(MANIFEST.read_text());cache=Path(m['cache'])
    for row in data[96:]:
        meta=json.loads((cache/(row['id']+'.json')).read_text())
        f,diag=frequency(row['times']-row['times'][0],row['fields'][:,1][:,row['mask']>.5],4,meta['St']/row['q']['D'],.05)
        count=1/f/.25;N=int(np.floor(count));periods.append({'id':row['id'],'frequency':f,'N':N,'alpha':float(count-N),'fit':diag,'role':'corner TRAINING ONLY'})
    assert [r['id'] for r in data]==[r['id'] for r in periods]
    save(DEST/'periods104.json',{'rows':periods,'source':'TRAIN104 only; no validation/fresh clocks'})
    c={'utc':now(),'source_sha256':sha(__file__),'arms':['same96','plus104'],'seeds':SEEDS,'updates':1000,'checks':CHECKS,
        'lr':1e-5,'weight_decay':1e-4,'clip':1.,'loss':'unchanged TF8 one-step +.1 autonomous period return',
        'initial':'both frozen outer96_seed622 research weights; same restarted AdamW',
        'initial_sha256':sha(INITIAL),'batch':'3 identical unique TRAIN96 cases; fourth control TRAIN96 or new TRAIN8 corner; same windows and updates',
        'single_factor':'joint-corner training support; fixed operator,clock guard,decoder,learning rate and loss',
        'operator':'ProjectedState(ReflectionOperator(base),True)','decoder':'same24/48 self-clock harmonic gate and unit-circleDMD9',
        'channel_scales':scales(),'selection':'seen56 DEVELOPMENT only; latest confirmation8 neither training nor selection',
        'guards':'both group Ri40 means and all56 maxRi40 no worse; balanced8s<=1.05 and pass loss<=1/56; include0',
        'signal':'both plus104 seeds better matched same96 controls; one>=10% balancedRi40 improvement from frozen research',
        'early_stop':'at500 if no2% selected improvement and current >1.25 initial balancedR40',
        'no_automatic_material_or_model_replacement':True,'training_manifest_sha256':sha(MANIFEST)}
    save(DEST/'preregister.json',c)
    for seed in SEEDS:
        rng=np.random.default_rng(seed);common=[];old=[];new=[];starts=[]
        for k in range(c['updates']):
            pick=rng.choice(96,3,replace=False);a=int(rng.choice(np.setdiff1d(np.arange(96),pick)));b=int(rng.integers(96,104))
            limit=min(165-4-periods[int(j)]['N']-1 for j in list(pick)+[a,b])
            common.append(pick);old.append(a);new.append(b);starts.append(int(rng.integers(0,limit+1)))
        np.savez(DEST/f'tasks_seed{seed}.npz',common=common,old=old,new=new,starts=starts)

def worker(arm,seed,smoke=False):
    torch.set_num_threads(1);protected();c=json.loads((DEST/'preregister.json').read_text());assert sha(__file__)==c['source_sha256'] and sha(INITIAL)==c['initial_sha256']
    data=data104();periods=json.loads((DEST/'periods104.json').read_text())['rows'];tasks=np.load(DEST/f'tasks_seed{seed}.npz')
    base,saved=model_from(INITIAL,protocol()['config'],seed=seed);model=ProjectedState(ReflectionOperator(base),True)
    scale=torch.as_tensor(c['channel_scales'],device='cuda')
    def batch(k):
        pick=np.r_[tasks['common'][k],tasks['old' if arm=='same96' else 'new'][k]];s=int(tasks['starts'][k])
        if smoke:pick=np.argsort([p['N'] for p in periods])[-4:];s=0
        g=torch.as_tensor(np.stack([data[int(j)]['geom'] for j in pick]),device='cuda')
        f=torch.as_tensor(np.stack([data[int(j)]['fields'][s:s+12] for j in pick]),device='cuda')
        m=torch.as_tensor(np.stack([data[int(j)]['mask'] for j in pick]),device='cuda')
        return pick,g,f,m
    def loss(k):
        pick,g,f,m=batch(k);return train_loss(model,g,f,m,scale,1)+.1*cycle(model,g,f,m,scale,periods,pick)[0]
    if smoke:
        model.train();torch.cuda.reset_peak_memory_stats();l=loss(0);l.backward();assert torch.isfinite(l)
        assert all(torch.isfinite(v.grad).all() for v in base.parameters() if v.grad is not None)
        save(DEST/'smoke.json',{'passed':True,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'no_optimizer_update':True});return
    folder=DEST/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False);opt=torch.optim.AdamW(base.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    initial=json.loads(INITIAL_REPORT.read_text());ref=initial['summary'];best=(float('inf'),float('inf'),0);history=[];begin=time.perf_counter()
    for step in range(c['updates']+1):
        if step:
            model.train();opt.zero_grad(set_to_none=True);l=loss(step-1);assert torch.isfinite(l);l.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(base.parameters(),c['clip']));assert np.isfinite(norm);opt.step()
            if step%25==0:save(folder/'status.json',{'status':'training','step':step,'loss':float(l.detach()),'utc':now(),'elapsed_seconds':time.perf_counter()-begin})
        if step in CHECKS:
            ev=initial if step==0 else evaluate(model);s=ev['summary'];b=balance(s);w=s['development_all56']['40s']['max_R']
            ok=(balance(s,'8s')<=1.05*balance(ref,'8s') and s['development_all56']['8s']['n_pass']>=ref['development_all56']['8s']['n_pass']-1
                and w<=ref['development_all56']['40s']['max_R']+1e-6
                and all(s[g]['40s']['mean_R']<=ref[g]['40s']['mean_R']+1e-6 for g in ['development_inside24','seen_extreme32']))
            key=(b,w,step);save(folder/f'development_{step}.json',ev);history.append({'step':step,'balanced_R40':b,'max_R40':w,'eligible':ok,'summary':s})
            if ok and key<best:
                best=key;torch.save({'model':base.state_dict(),'model_name':saved['model_name'],'in_channels':saved['in_channels'],'out_channels':saved['out_channels'],
                    'config':protocol()['config'],'seed':seed,'update':step,'optimizer':opt.state_dict(),'training_arm':arm,
                    'requires_research_operator_wrapper':'ProjectedState(ReflectionOperator(base),True)','role':'DEVELOPMENT-selected research ONLY'},folder/'selected_base_weights.pt')
                save(folder/'selected_development.json',ev);save(folder/'selection.json',{'step':step,'key':list(best),'sha256':sha(folder/'selected_base_weights.pt')})
            save(folder/'history.json',history);print('CORNER_TRAIN',arm,seed,step,round(b,6),ok,flush=True)
            if step==500 and best[0]>.98*balance(ref) and b>1.25*balance(ref):break
    protected();save(folder/'done.json',{'utc':now(),'last_step':step,'best':list(best),'wall_seconds':time.perf_counter()-begin})
    save(folder/'status.json',{'status':'complete','utc':now(),'step':step,'best':list(best)})

def main():
    protected();DEST.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    previous=OUT/'next_round/paired_rollout8/status.json'
    while not MANIFEST.exists() or not previous.exists() or json.loads(previous.read_text()).get('status')!='complete':
        if time.perf_counter()-begin>18000:raise RuntimeError('5h finite dependency limit')
        for p in [OUT/'next_round/corner_training_cfd/failure.json',OUT/'next_round/paired_rollout8/failure.json']:
            if p.exists():raise RuntimeError('Dependency failed: '+str(p))
        save(DEST/'status.json',{'status':'waiting_for_training_corners_and_GPU','utc':now(),'elapsed_seconds':time.perf_counter()-begin});time.sleep(30)
    prepare();env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    with (DEST/'smoke.log').open('x') as log:subprocess.run([sys.executable,__file__,'--worker','plus104','--seed',str(SEEDS[0]),'--smoke'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads((DEST/'smoke.json').read_text())['peak_GPU_GiB']<5.3
    jobs=[]
    for seed in SEEDS:
        for arm in ['same96','plus104']:
            log=(DEST/f'{arm}_seed{seed}.log').open('x');p=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((arm,seed,p,log))
    try:
        while True:
            done=[{'arm':a,'seed':s,'exit':p.returncode} for a,s,p,f in jobs if p.poll() is not None]
            save(DEST/'status.json',{'status':'training','utc':now(),'completed':done,'workers':[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,f in jobs]})
            if any(x['exit']!=0 for x in done):raise RuntimeError(done)
            if len(done)==4:break
            time.sleep(15)
    except BaseException:
        for a,s,p,f in jobs:
            if p.poll() is None:p.terminate()
        raise
    pairs=[];candidates=[];ref=json.loads(INITIAL_REPORT.read_text())['summary']
    for seed in SEEDS:
        scores={}
        for arm in ['same96','plus104']:
            folder=DEST/f'{arm}_seed{seed}';ev=json.loads((folder/'selected_development.json').read_text());scores[arm]=balance(ev['summary'])
            candidates.append({'arm':arm,'seed':seed,'path':str(folder/'selected_base_weights.pt'),'summary':ev['summary'],'selection':json.loads((folder/'selection.json').read_text())})
        pairs.append({'seed':seed,'plus104_better':scores['plus104']<scores['same96'],'scores':scores})
    signal=all(x['plus104_better'] for x in pairs) and any(a['arm']=='plus104' and balance(a['summary'])<=.9*balance(ref) for a in candidates)
    protected();save(DEST/'completed_report.json',{'utc':now(),'pairs':pairs,'candidates':candidates,'research_signal_passed':signal,'latest_confirmation8_unused_for_selection':True,'fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'research_signal_passed':signal})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['same96','plus104']);ap.add_argument('--seed',type=int);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    try:
        if a.worker:worker(a.worker,a.seed,a.smoke)
        else:main()
    except Exception as e:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});save(DEST/'status.json',{'status':'failed','utc':now(),'error':repr(e)});raise
