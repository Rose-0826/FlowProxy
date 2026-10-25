#!/usr/bin/env python3
"""Finite data-dependent paired training: original80 versus80+16 outer anchors."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,scales,protected,now
from research_v2_reflection_train import ReflectionOperator,evaluate,balanced,worst
from research_v2_state_probe import ProjectedState
from corrected_benchmark_train_v1 import Store,model_from,train_loss,field_loss_steps,protocol
from cycle_landmarks_train_v1 import FORMAL,PERIODS
from startup_observability_v1 import frequency
from geometry_attack_v2 import save,sha

TRAIN=OUT/'outer_support_training';SEEDS=[611,622];CHECKS=[0,250,500,750,1000]
MANIFEST=OUT/'outer_support_training_cfd/training_manifest.json'

def prepare():
    protected();TRAIN.mkdir(exist_ok=False)
    manifest=json.loads(MANIFEST.read_text());assert len(manifest['cases'])==16
    existing=Store('train').rows;periods=json.loads(PERIODS.read_text())['rows'];new=[]
    assert [r['id'] for r in existing]==[r['id'] for r in periods]
    for q in manifest['cases']:
        cache=Path(manifest['cache']);file=cache/(q['id']+'.npz')
        meta=json.loads((cache/(q['id']+'.json')).read_text())
        assert sha(file)==meta['cache_sha256'] and meta['CFD_stable']
        with np.load(file) as a:
            fields=a['fields'];times=a['times'];mask=a['mask']
        f,diag=frequency(times-times[0],fields[:,1][:,mask>.5],4,meta['St']/q['D'],.05)
        count=1/f/.25;N=int(np.floor(count))
        new.append({'id':q['id'],'frequency':f,'N':N,'alpha':float(count-N),
            'source_role':'TRAINING ONLY','cache_sha256':sha(file),'fit':diag})
    save(TRAIN/'periods96.json',{'rows':periods+new,'source':'original80 sealed periods +outer16 full TRAINING fields; no development field-derived clocks'})
    c={'utc':now(),'source_sha256':sha(__file__),'seeds':SEEDS,'arms':['original80','outer96'],
        'updates':1000,'checks':CHECKS,'lr':1e-5,'weight_decay':1e-4,'clip':1.,
        'initialization':'same immutable formal base weights, AdamW restarted',
        'operator':'BOTH arms use same exact reflection projection +solid zero+known velocityBC feedback',
        'decoder':'BOTH fixed4 true+48 neural, unit-circle DMD9',
        'batch':'BOTH4 cases:3 common original cases +1 old control or1 new outer anchor; same windows and total targets/updates',
        'loss':'same eight TF one-step targets +.1 autonomous one-period return; fixed originaltrain80 scales',
        'data_roles':'new outer16 training only; all56 prior observed cases DEVELOPMENT; not independent tests',
        'selection':'balanced meanRi40 over inside24/extreme32; short8s meanRi <=1.05 original fallback and pass loss<=1/56; include step0',
        'method_signal':'BOTH expanded seeds better matched controls; >=20% balancedRi40 improvement over formal fallback; neither group average worse; all maxRi and8s reported',
        'future_confirmation':'only new independent geometries after algorithm and checkpoint freeze',
        'channel_scales':scales(),'training_manifest_sha256':sha(MANIFEST),'fallback_immutable':True}
    save(TRAIN/'preregister.json',c)
    allperiod=periods+new
    for seed in SEEDS:
        rng=np.random.default_rng(seed);common=[];extra_old=[];extra_new=[];starts=[]
        for k in range(c['updates']):
            a=rng.choice(80,3,replace=False);old=int(rng.choice(np.setdiff1d(np.arange(80),a)))
            new_index=int(rng.integers(80,96))
            limit=min(165-4-allperiod[int(j)]['N']-1 for j in list(a)+[old,new_index])
            common.append(a);extra_old.append(old);extra_new.append(new_index);starts.append(int(rng.integers(0,limit+1)))
        np.savez(TRAIN/f'tasks_seed{seed}.npz',common=np.array(common),extra_old=extra_old,extra_new=extra_new,starts=starts)

def cycle(model,geom,fields,mask,scale,periods,pick):
    b=len(fields);window=[fields[:,k] for k in range(4)];outputs=[]
    n=max(periods[int(j)]['N'] for j in pick)+1
    for _ in range(n):
        y=model(torch.cat([geom,torch.stack(window[-4:],1).reshape(b,12,64,64)],1))[:,:3]
        window.append(y);outputs.append(y)
    prediction=torch.stack(outputs,1);aligned=[]
    for i,j in enumerate(pick):
        N,a=periods[int(j)]['N'],periods[int(j)]['alpha']
        aligned.append((1-a)*prediction[i,N-4:N]+a*prediction[i,N-3:N+1])
    return field_loss_steps(torch.stack(aligned),fields[:,:4],mask,scale),n

def dataset():
    data=Store('train').rows;manifest=json.loads(MANIFEST.read_text());cache=Path(manifest['cache'])
    for q in manifest['cases']:
        with np.load(cache/(q['id']+'.npz')) as a:
            data.append({'id':q['id'],'q':{k:q[k] for k in ['shape','Re','cx','cy','D']},
                'fields':a['fields'],'times':a['times'],'geom':a['geom'],'mask':a['mask']})
    assert len(data)==96;return data

def worker(arm,seed,smoke=False):
    torch.set_num_threads(1);protected();c=json.loads((TRAIN/'preregister.json').read_text());assert sha(__file__)==c['source_sha256']
    data=dataset();periods=json.loads((TRAIN/'periods96.json').read_text())['rows']
    assert [q['id'] for q in data]==[q['id'] for q in periods]
    tasks=np.load(TRAIN/f'tasks_seed{seed}.npz');base,saved=model_from(FORMAL,protocol()['config'],seed=seed)
    model=ProjectedState(ReflectionOperator(base),True);scale=torch.as_tensor(c['channel_scales'],device='cuda')
    def batch(k):
        extra=tasks['extra_old' if arm=='original80' else 'extra_new'][k]
        pick=np.concatenate([tasks['common'][k],[extra]]);start=int(tasks['starts'][k])
        geom=torch.as_tensor(np.stack([data[int(j)]['geom'] for j in pick]),device='cuda')
        fields=torch.as_tensor(np.stack([data[int(j)]['fields'][start:start+12] for j in pick]),device='cuda')
        mask=torch.as_tensor(np.stack([data[int(j)]['mask'] for j in pick]),device='cuda')
        return pick,geom,fields,mask
    if smoke:
        model.train();torch.cuda.reset_peak_memory_stats();p,g,f,m=batch(0)
        loss=train_loss(model,g,f,m,scale,1)+.1*cycle(model,g,f,m,scale,periods,p)[0];loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in base.parameters() if v.grad is not None)
        save(TRAIN/'smoke.json',{'passed':True,'no_optimizer_updates':True,
            'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'loss':float(loss.detach())});return
    folder=TRAIN/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False);begin=time.perf_counter();history=[]
    opt=torch.optim.AdamW(base.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    original=json.loads((OUT/'operator_probe/report.json').read_text())['fallback']['summary']
    initial=json.loads((OUT/'state_probe/report.json').read_text())['reflection_velocity_BC']
    best=(float('inf'),float('inf'),0);clips=0
    for step in range(c['updates']+1):
        if step:
            model.train();opt.zero_grad(set_to_none=True);p,g,f,m=batch(step-1)
            loss=train_loss(model,g,f,m,scale,1)+.1*cycle(model,g,f,m,scale,periods,p)[0]
            assert torch.isfinite(loss);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(base.parameters(),c['clip']));assert np.isfinite(norm)
            clips+=int(norm>c['clip']);opt.step()
            if step%25==0:save(folder/'status.json',{'status':'training','step':step,'loss':float(loss.detach()),'norm':norm,'utc':now()})
        if step in CHECKS:
            ev=initial if step==0 else evaluate(model,rows());s=ev['summary']
            eligible=balanced(s,'8s')<=1.05*balanced(original,'8s') and s['DMD9']['development_all56']['8s']['n_pass']>=original['DMD9']['development_all56']['8s']['n_pass']-1
            key=(balanced(s),worst(s),step);save(folder/f'development_{step}.json',ev)
            history.append({'step':step,'balanced_R40':key[0],'max_R40':key[1],'eligible':eligible,'summary':s})
            if eligible and key<best:
                best=key
                torch.save({'model':base.state_dict(),'model_name':saved['model_name'],'in_channels':saved['in_channels'],
                    'out_channels':saved['out_channels'],'config':protocol()['config'],'seed':seed,'update':step,
                    'requires_research_operator_wrapper':'ProjectedState(ReflectionOperator(base),True)',
                    'training_arm':arm,'optimizer':opt.state_dict(),'role':'DEVELOPMENT-selected research candidate'},folder/'selected_base_weights.pt')
                save(folder/'selected_development.json',ev)
                save(folder/'selection.json',{'step':step,'key':list(best),'sha256':sha(folder/'selected_base_weights.pt')})
            save(folder/'history.json',history);print('OUTER_TRAIN',arm,seed,step,'balanced',round(key[0],6),'eligible',eligible,flush=True)
    protected();save(folder/'done.json',{'utc':now(),'status':'complete','wall_seconds':time.perf_counter()-begin,'best':list(best),'clips':clips})
    save(folder/'status.json',{'utc':now(),'status':'complete','step':c['updates'],'best':list(best)})

def orchestrate():
    OUT.mkdir(exist_ok=True);waitfile=OUT/'outer_support_training_wait.json';begin=time.perf_counter()
    # Finite dependency pipeline, not a recurring monitor or automation.
    while not MANIFEST.exists():
        if time.perf_counter()-begin>10800:raise RuntimeError('3h CFD dependency limit reached')
        p=OUT/'outer_support_training_cfd/status.json'
        state=json.loads(p.read_text()) if p.exists() else {}
        if state.get('errors') or state.get('status')=='failed':raise RuntimeError('Outer CFD failed; do not train on partial/selected successes')
        save(waitfile,{'utc':now(),'status':'waiting_for_complete16_training_CFD','completed':len(state.get('completed',[])),
            'total':16,'elapsed_seconds':time.perf_counter()-begin,'fallback_unchanged':True})
        time.sleep(30)
    prepare();env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    with (TRAIN/'smoke.log').open('x') as log:
        subprocess.run([sys.executable,__file__,'--worker','outer96','--seed',str(SEEDS[0]),'--smoke'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke=json.loads((TRAIN/'smoke.json').read_text());assert smoke['passed'] and smoke['peak_GPU_GiB']<5.2
    jobs=[]
    for seed in SEEDS:
        for arm in ['original80','outer96']:
            log=(TRAIN/f'{arm}_seed{seed}.log').open('x')
            proc=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((arm,seed,proc,log))
    try:
        while True:
            completed=[{'arm':a,'seed':s,'return_code':p.returncode} for a,s,p,f in jobs if p.poll() is not None]
            save(TRAIN/'status.json',{'utc':now(),'status':'training','completed':completed,
                'workers':[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,f in jobs]})
            if any(z['return_code']!=0 for z in completed):raise RuntimeError(completed)
            if len(completed)==4:break
            time.sleep(15)
    except BaseException:
        for a,s,p,f in jobs:
            if p.poll() is None:p.terminate()
        raise
    for a,s,p,f in jobs:f.close()
    candidates=[];pairs=[];original=json.loads((OUT/'operator_probe/report.json').read_text())['fallback']['summary']
    for seed in SEEDS:
        scores={}
        for arm in ['original80','outer96']:
            folder=TRAIN/f'{arm}_seed{seed}';ev=json.loads((folder/'selected_development.json').read_text());selection=json.loads((folder/'selection.json').read_text())
            scores[arm]=balanced(ev['summary'])
            candidates.append({'arm':arm,'seed':seed,'selection':selection,'summary':ev['summary'],
                'balanced_R40':scores[arm],'path':str(folder/'selected_base_weights.pt')})
        pairs.append({'seed':seed,'expanded_better':scores['outer96']<scores['original80'],'scores':scores})
    qualified=[a for a in candidates if a['arm']=='outer96' and a['balanced_R40']<=.8*balanced(original)
        and all(a['summary']['DMD9'][g]['40s']['mean_R']<=original['DMD9'][g]['40s']['mean_R'] for g in ['development_inside24','seen_extreme32'])]
    signal=all(p['expanded_better'] for p in pairs) and len(qualified)>0
    protected();save(TRAIN/'completed_report.json',{'utc':now(),'pairs':pairs,'candidates':candidates,
        'research_signal_passed':signal,'fresh_confirmation_used':False,'fallback_unchanged':True})
    save(TRAIN/'status.json',{'utc':now(),'status':'complete','research_signal_passed':signal})
    save(waitfile,{'utc':now(),'status':'complete','CFD_dependency_satisfied':True,'training_complete':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['original80','outer96']);ap.add_argument('--seed',type=int);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    try:
        if args.worker:worker(args.worker,args.seed,args.smoke)
        else:orchestrate()
    except Exception as e:
        save(OUT/'outer_support_training_failure.json',{'utc':now(),'error':repr(e)});raise
