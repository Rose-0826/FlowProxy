#!/usr/bin/env python3
"""Two matched pairs: known convective timestep as an explicit FNO input."""
from pathlib import Path
import argparse, copy, datetime, json, os, subprocess, sys, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from flowproxy.models import build_model
from corrected_benchmark_train_v1 import Store,model_from,field_loss_steps,protocol
from corrected_benchmark_controls_v1 import dmd_periodic
from period_consistency_train_v1 import cycle_loss,make_tasks
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

PREV=ROOT/'outputs/period_consistency_train_v1'
EVAL=ROOT/'outputs/period_consistency_evaluate_v1'
OUT=ROOT/'outputs/physical_timestep_train_v1'
SEEDS=[44,55]
CHECKS=[0,250,500,750,1000]
MAX_UPDATES=1000
LR=1e-5
WEIGHT=.1

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def config():return json.loads((OUT/'preregister.json').read_text())

def expanded_model(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    c=config();saved=torch.load(c['network']['path'],map_location='cpu',weights_only=False)
    model=build_model(saved['model_name'],saved['in_channels']+1,saved['out_channels'],protocol()['config']).cuda()
    state=copy.deepcopy(saved['model']);old=state['lift.0.weight'];new=old.new_zeros((old.shape[0],old.shape[1]+1,*old.shape[2:]))
    new[:,:4]=old[:,:4];new[:,5:]=old[:,4:];state['lift.0.weight']=new
    model.load_state_dict(state)
    return model,saved

def add_feature(row,arm,c):
    dt=float(np.median(np.diff(row['times'])))
    assert np.allclose(np.diff(row['times']),.25)
    value=(dt*c['U_inf']/row['q']['D']-c['feature_mean_train80'])/c['feature_std_train80'] if arm=='physical' else 0.
    return np.concatenate([row['geom'],np.full((1,64,64),value,np.float32)],axis=0)

def predict(model,row,arm,c):
    model.eval();geom=torch.as_tensor(add_feature(row,arm,c)[None],device='cuda')
    window=[torch.as_tensor(f[None],device='cuda') for f in row['fields'][:4]];outputs=[]
    with torch.inference_mode():
        for k in range(160):
            hist=torch.stack(window[-4:],1).reshape(1,12,64,64)
            pred=model(torch.cat([geom,hist],1))[:,:3];outputs.append(pred[0].cpu().numpy());window.append(pred)
    return np.stack(outputs)

def evaluate(model,store,arm,c):
    direct=[];closed=[]
    for row in store.rows:
        base=predict(model,row,arm,c)
        observed=np.concatenate([row['fields'][:4],base[:48]])
        pred,diag=dmd_periodic(observed,row['times'][:52],row['times'][4:164],row['mask'],9,c['channel_scales_train80'])
        pred[:48]=base[:48]
        truth=row['fields'][4:164]
        direct.append({'id':row['id'],'horizons':metrics(base,truth,row['mask'],row['times'])})
        closed.append({'id':row['id'],'horizons':metrics(pred,truth,row['mask'],row['times']),'DMD':diag})
    return {'cases_direct':direct,'cases_DMD9':closed,'summary':{name:{h:summarize([x['horizons'][h] for x in rows]) for h in ['2s','8s','20s','40s']} for name,rows in [('direct',direct),('DMD9',closed)]}}

def short_loss(model,geom,fields,mask,scales):
    b=fields.shape[0]
    hist=torch.stack([fields[:,k:k+4] for k in range(8)],1).reshape(b*8,12,64,64)
    gs=geom[:,None].expand(-1,8,-1,-1,-1).reshape(b*8,5,64,64)
    pred=model(torch.cat([gs,hist],1))[:,:3].reshape(b,8,3,64,64)
    return field_loss_steps(pred,fields[:,4:12],mask,scales)

def configure():
    OUT.mkdir(exist_ok=False)
    previous=json.loads((PREV/'completed_report.json').read_text())
    network=previous['selected_cycle_for_followup_if_gate_passed'];assert network['seed']==11 and sha(network['path'])==network['sha256']
    train=Store('train');p=protocol();U=float(p['config']['physics']['U_inf']);assert U==1.
    values=np.array([float(np.median(np.diff(r['times'])))*U/r['q']['D'] for r in train.rows])
    original=json.loads((EVAL/'validation_report.json').read_text())['summary']['cycle_seed11_self_DMD_rank9']
    oldconfig=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/training/warm_K1_seed33/configuration.json').read_text())
    c={'registered_utc':now(),'network':network,'arms':['control','physical'],'seeds':SEEDS,'max_updates':MAX_UPDATES,'checks':CHECKS,
       'U_inf':U,'feature':'known dt*U_inf/D, z-scored using train80 only; no target force, future frequency or phase',
       'feature_mean_train80':float(values.mean()),'feature_std_train80':float(values.std()),'channel_scales_train80':oldconfig['normalization_train80'],
       'expanded_input':'geom4 + feature1 + history12; coordinate embedding unchanged; feature lifting weight initialized zero',
       'control':'same expanded architecture, feature identically zero; same initial output and parameter count',
       'both_losses':'same eight teacher-forced field targets +0.1 autonomous one-cycle closure; train80 periods unchanged',
       'lr':LR,'AdamW_weight_decay':1e-4,'gradient_clip':1.,'batch_cases':4,
       'selection':'original val8 only; minimum DMD9 40s meanRi among short8s8/8, then maxRi, then earlier checkpoint; include0',
       'gate':'both pairs physical meanRi<control; one physical >=10% lower meanRi AND >=10% lower maxRi vs initial, short8s8/8',
       'early_stop':'at500 if no2% improvement in best meanRi AND current meanRi>1.25*initial; applies both arms',
       'posthoc_cases':'32+12+new4 are now all seen; do not train or select on them; no fresh claim for future rescoring',
       'baseline_validation':original,'new_CFD':False,'script_sha256':sha(__file__),'periods_sha256':sha(PREV/'train80_periods.json')}
    assert c['feature_std_train80']>0
    save(OUT/'preregister.json',c)
    periods=json.loads((PREV/'train80_periods.json').read_text())['rows']
    assert [x['id'] for x in periods]==[x['id'] for x in train.rows]
    for seed in SEEDS:
        picks,starts=make_tasks(periods,seed);np.savez(OUT/f'tasks_seed{seed}.npz',picks=picks,starts=starts)

def worker(arm,seed,smoke=False):
    torch.set_num_threads(1);c=config();train=Store('train');val=Store('val')
    periods=json.loads((PREV/'train80_periods.json').read_text())['rows']
    task=np.load(OUT/f'tasks_seed{seed}.npz');model,saved=expanded_model(seed)
    scales=torch.as_tensor(c['channel_scales_train80'],device='cuda')
    def batch(step):
        pick=task['picks'][step];start=int(task['starts'][step])
        geom=torch.as_tensor(np.stack([add_feature(train.rows[j],arm,c) for j in pick]),device='cuda')
        fields=torch.as_tensor(np.stack([train.rows[j]['fields'][start:start+12] for j in pick]),device='cuda')
        mask=torch.as_tensor(np.stack([train.rows[j]['mask'] for j in pick]),device='cuda')
        return pick,geom,fields,mask
    if smoke:
        pick,geom,fields,mask=batch(0);old,_=model_from(c['network']['path'],protocol()['config'])
        hist=fields[:,:4].reshape(4,12,64,64)
        model.eval();old.eval()
        with torch.no_grad():difference=float((model(torch.cat([geom,hist],1))-old(torch.cat([geom[:,:4],hist],1))).abs().max())
        assert difference<2e-6
        model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        short=short_loss(model,geom,fields,mask,scales);cyc,nmax=cycle_loss(model,geom,fields,mask,scales,periods,pick)
        loss=short+WEIGHT*cyc;loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
        save(OUT/'smoke.json',{'initial_max_abs_difference':difference,'finite_gradients':True,'short_loss':float(short.detach()),'cycle_loss':float(cyc.detach()),
             'new_channel_gradient_norm':float(model.lift[0].weight.grad[:,4].norm()),'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,
             'steps_autonomous':nmax,'seconds_forward_backward':time.perf_counter()-start,'no_optimizer_updates':True})
        print('TIMESTEP_SMOKE_OK',difference,nmax,flush=True);return
    folder=OUT/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False);opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    best=(float('inf'),float('inf'),0);history=[];started=time.perf_counter();maxgrad=0.;clips=0;last=0
    save(folder/'configuration.json',{'arm':arm,'seed':seed,'initial_model_sha256':c['network']['sha256'],'tasks_sha256':sha(OUT/f'tasks_seed{seed}.npz'),
         'in_channels':17,'lr':LR,'lambda_cycle':WEIGHT,'feature_normalization_train80':[c['feature_mean_train80'],c['feature_std_train80']]})
    for step in range(MAX_UPDATES+1):
        if step:
            pick,geom,fields,mask=batch(step-1);model.train();opt.zero_grad(set_to_none=True)
            short=short_loss(model,geom,fields,mask,scales);cyc,_=cycle_loss(model,geom,fields,mask,scales,periods,pick)
            loss=short+WEIGHT*cyc;assert torch.isfinite(loss);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm)
            maxgrad=max(maxgrad,norm);clips+=int(norm>1);opt.step()
            if step%25==0:save(folder/'status.json',{'step':step,'elapsed_seconds':time.perf_counter()-started,'field_loss':float(short.detach()),'cycle_loss':float(cyc.detach())})
        if step in CHECKS:
            ev=evaluate(model,val,arm,c);q=ev['summary']['DMD9']['40s'];valid=ev['summary']['direct']['8s']['n_pass']==8
            if step==0:assert abs(q['mean_R']-c['baseline_validation']['40s']['mean_R'])<2e-4
            key=(q['mean_R'],q['max_R'],step);save(folder/f'validation_{step}.json',ev)
            history.append({'step':step,'summary':ev['summary'],'eligible_short8s':valid})
            if valid and key<best:
                best=key
                torch.save({'model':model.state_dict(),'model_name':saved['model_name'],'in_channels':17,'out_channels':saved['out_channels'],
                   'update':step,'seed':seed,'arm':arm,'config':protocol()['config'],'feature_configuration':c,'optimizer':opt.state_dict()},folder/'selected.pt')
                save(folder/'selected_validation.json',ev);save(folder/'selection.json',{'step':step,'key':list(key),'sha256':sha(folder/'selected.pt')})
            save(folder/'history.json',history);last=step
            print('TIMESTEP_TRAIN',arm,seed,step,'R',round(q['mean_R'],4),'MAX',round(q['max_R'],4),'PASS',q['n_pass'],flush=True)
            initial=c['baseline_validation']['40s']['mean_R']
            if step==500 and best[0]>.98*initial and q['mean_R']>1.25*initial:break
    save(folder/'done.json',{'status':'complete','last_update':last,'best':list(best),'wall_seconds':time.perf_counter()-started,'max_grad':maxgrad,'clip_count':clips})

def orchestrate():
    configure()
    with (OUT/'smoke.log').open('x') as f:subprocess.run([sys.executable,__file__,'--smoke'],stdout=f,stderr=subprocess.STDOUT,check=True)
    smoke=json.loads((OUT/'smoke.json').read_text());assert smoke['peak_GPU_GiB']<5.2
    procs=[]
    for seed in SEEDS:
        for arm in ['control','physical']:
            log=(OUT/f'{arm}_seed{seed}.log').open('x')
            proc=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            procs.append((arm,seed,proc,log))
    save(OUT/'status.json',{'status':'training','workers':[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,f in procs]})
    for arm,seed,proc,log in procs:
        rc=proc.wait();log.close()
        if rc:raise RuntimeError((arm,seed,rc))
    baseline=config()['baseline_validation']['40s'];pairs=[];candidates=[]
    for seed in SEEDS:
        scores={}
        for arm in ['control','physical']:
            f=OUT/f'{arm}_seed{seed}';ev=json.loads((f/'selected_validation.json').read_text());sel=json.loads((f/'selection.json').read_text());q=ev['summary']['DMD9']['40s']
            scores[arm]=q;candidates.append({'arm':arm,'seed':seed,'path':str(f/'selected.pt'),'sha256':sha(f/'selected.pt'),'selection':sel,'summary':ev['summary']})
        pairs.append({'seed':seed,'physical_better_mean_than_control':scores['physical']['mean_R']<scores['control']['mean_R'],'scores':scores})
    qualified=[x for x in candidates if x['arm']=='physical' and x['summary']['DMD9']['40s']['mean_R']<=.9*baseline['mean_R'] and x['summary']['DMD9']['40s']['max_R']<=.9*baseline['max_R']]
    gate=all(x['physical_better_mean_than_control'] for x in pairs) and bool(qualified)
    chosen=min(qualified,key=lambda x:(x['summary']['DMD9']['40s']['mean_R'],x['summary']['DMD9']['40s']['max_R'])) if gate else None
    save(OUT/'completed_report.json',{'gate_passed':gate,'pairs':pairs,'candidates':candidates,'selected_if_gate_passed':chosen,'baseline':baseline})
    save(OUT/'status.json',{'status':'complete','gate_passed':gate,'pairs':pairs});print('TIMESTEP_COMPLETE_GATE',gate,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['control','physical']);ap.add_argument('--seed',type=int,default=44);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    try:
        if a.worker or a.smoke:worker(a.worker or 'physical',a.seed,a.smoke)
        else:orchestrate()
    except Exception as e:
        if not a.worker and not a.smoke and OUT.exists():save(OUT/'status.json',{'status':'failed','error':repr(e),'utc':now()})
        raise
