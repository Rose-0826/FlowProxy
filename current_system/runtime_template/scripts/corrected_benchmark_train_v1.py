#!/usr/bin/env python3
"""Paired K1/K8 retraining on corrected train80; validation only for selection."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys,json,time,copy,subprocess,os,fcntl
import h5py,numpy as np,torch
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from flowproxy.models import build_model
from self_trajectory_closure_v1 import closure,metrics
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/corrected_benchmark_restart_v1'
DATA=ROOT/'data/processed/dataset_corrected_benchmark_v1.h5'
TRAINVAL=ROOT/'data/processed/dataset_corrected_trainval_v1.h5'
CACHE=ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
BASE=ROOT/'outputs/checkpoints/fno_k8_base/base.pt'

def protocol():return json.loads((OUT/'preregister.json').read_text())

class Store:
    def __init__(self,role):
        p=protocol();self.rows=[]
        source=TRAINVAL if role!='test' and TRAINVAL.exists() else DATA
        with h5py.File(source,'r') as h:
            for cid in p['splits'][role]:
                g=h['cases'][cid]
                self.rows.append({'id':cid,'q':{k:str(g.attrs[k]) if k=='shape' else float(g.attrs[k]) for k in ['shape','Re','cx','cy','D']},
                                  'fields':g['fields'][...], 'geom':g['geom'][...],'mask':g['mask'][...],
                                  'times':g['times'][...],'St':float(g.attrs['St'])})
        assert len(self.rows)=={'train':80,'val':8,'test':32}[role]

def model_from(path,cfg,seed=0,scratch=False):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(saved['model_name'],saved['in_channels'],saved['out_channels'],cfg).to('cuda')
    if not scratch:model.load_state_dict(saved['model'])
    return model,saved

def priors_and_scales(train):
    acc=np.zeros(3);count=0
    for r in train.rows:
        fluid=r['mask']>.5;a=r['fields'][:,:,fluid].astype(np.float64)
        acc+=np.sum(a*a,axis=(0,2));count+=len(a)*int(fluid.sum())
    scales=np.sqrt(acc/count+1e-8).astype(np.float32)
    prior={shape:(None,float(np.mean([r['St'] for r in train.rows if r['q']['shape']==shape]))) for shape in ['cylinder','square']}
    return scales,prior

def predict(model,row):
    model.eval();geom=torch.as_tensor(row['geom'][None],device='cuda');window=[torch.as_tensor(a[None],device='cuda') for a in row['fields'][:4]]
    out=[]
    with torch.inference_mode():
        for k in range(160):
            hist=torch.stack(window[-4:],1).reshape(1,12,64,64)
            pred=model(torch.cat([geom,hist],1))[:,:3];out.append(pred[0].cpu().numpy());window.append(pred)
    return np.stack(out)

def evaluate(model,store,prior,p):
    direct=[];closed=[]
    for row in store.rows:
        base=predict(model,row)
        pred,diag=closure(row['q'],row['fields'][:4],base,row['times'],row['mask'],prior,p['closure'])
        assert np.array_equal(base[:48],pred[:48])
        direct.append({'id':row['id'],'horizons':metrics(base,row['fields'][4:164],row['mask'],row['times'])})
        closed.append({'id':row['id'],'horizons':metrics(pred,row['fields'][4:164],row['mask'],row['times']),'closure':diag})
    summaries={mode:{h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']} for mode,rows in [('direct',direct),('closure',closed)]}
    return {'cases_direct':direct,'cases_closure':closed,'summary':summaries}

def field_loss_steps(pred,target,mask,scales):
    # B,K,C,H,W. Identical weighting for K1 and K8; fluid counts include cases.
    fluid=(mask>.5)[:,None,None]
    error=(pred-target)/scales[None,None,:,None,None]
    losses=(error.square()*fluid).sum((0,2,3,4))/(fluid.sum().clamp_min(1)*3)
    return .5*losses.mean()+.5*losses[-1]

def train_loss(model,geom,fields,mask,scales,K):
    b=fields.shape[0]
    if K==1:
        hist=torch.stack([fields[:,k:k+4] for k in range(8)],1).reshape(b*8,12,64,64)
        gs=geom[:,None].expand(-1,8,-1,-1,-1).reshape(b*8,4,64,64)
        pred=model(torch.cat([gs,hist],1))[:,:3].reshape(b,8,3,64,64)
    else:
        window=[fields[:,k] for k in range(4)];outputs=[]
        for k in range(8):
            hist=torch.stack(window[-4:],1).reshape(b,12,64,64)
            a=model(torch.cat([geom,hist],1))[:,:3];outputs.append(a);window.append(a)
        pred=torch.stack(outputs,1)
    return field_loss_steps(pred,fields[:,4:12],mask,scales)

def tasks(seed,total):
    rng=np.random.default_rng(seed)
    picks=[];starts=[]
    for i in range(total):picks.append(rng.choice(80,4,replace=False));starts.append(int(rng.integers(0,154)))
    return np.array(picks),np.array(starts)

def score_key(report,step):
    r=report['summary']['closure']['40s'];return (r['mean_R'],r['max_R'],step)

def worker(arm,seed,K,smoke=False):
    p=protocol();cfg=p['config'];train=Store('train');val=Store('val')
    scales,prior=priors_and_scales(train)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    model,saved=model_from(BASE,cfg,seed,scratch=arm=='scratch')
    if smoke:
        folder=OUT/'smoke_gpu';folder.mkdir(exist_ok=False)
        geom=torch.as_tensor(np.stack([r['geom'] for r in train.rows[:4]]),device='cuda')
        fields=torch.as_tensor(np.stack([r['fields'][:12] for r in train.rows[:4]]),device='cuda')
        mask=torch.as_tensor(np.stack([r['mask'] for r in train.rows[:4]]),device='cuda')
        s=torch.as_tensor(scales,device='cuda');loss=train_loss(model,geom,fields,mask,s,8);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        a=evaluate(model,val,prior,p)
        save(folder/'smoke.json',{'loss':float(loss.detach()),'finite_gradients':True,'validation_cases':8,'scores':a['summary'],'no_checkpoint_promoted':True})
        print('CORRECTED_GPU_SMOKE_OK',flush=True);return
    name=f'{arm}_K{K}_seed{seed}';folder=OUT/'training'/name;folder.mkdir(parents=True,exist_ok=False)
    lr=p['train']['lr_scratch' if arm=='scratch' else 'lr_warm']
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['train']['weight_decay'])
    picks,starts=tasks(seed,p['train']['updates']);np.savez(folder/'training_tasks.npz',picks=picks,starts=starts)
    scale_t=torch.as_tensor(scales,device='cuda');history=[];best=(float('inf'),float('inf'),0);started=time.perf_counter()
    save(folder/'configuration.json',{'arm':arm,'seed':seed,'K':K,'lr':lr,'initial_checkpoint_sha256':sha(BASE),
        'initial_weights_loaded':arm=='warm','normalization_train80':scales.tolist(),'shape_mean_train80_St':{k:v[1] for k,v in prior.items()},
        'data_path':str(TRAINVAL if TRAINVAL.exists() else DATA),'data_sha256':sha(TRAINVAL if TRAINVAL.exists() else DATA),'paired_task_sha256':sha(folder/'training_tasks.npz'),'float32_TF32_disabled':True,'cudnn_deterministic':True,'script_sha256':sha(__file__)})
    for step in range(p['train']['updates']+1):
        if step:
            model.train();pick=picks[step-1];start=int(starts[step-1])
            geom=torch.as_tensor(np.stack([train.rows[i]['geom'] for i in pick]),device='cuda')
            fields=torch.as_tensor(np.stack([train.rows[i]['fields'][start:start+12] for i in pick]),device='cuda')
            mask=torch.as_tensor(np.stack([train.rows[i]['mask'] for i in pick]),device='cuda')
            opt.zero_grad(set_to_none=True);loss=train_loss(model,geom,fields,mask,scale_t,K)
            assert torch.isfinite(loss), (name,step)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%25==0:save(folder/'status.json',{'step':step,'total':p['train']['updates'],'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-started})
        if step in p['train']['checks']:
            ev=evaluate(model,val,prior,p);key=score_key(ev,step)
            save(folder/f'validation_{step}.json',ev)
            row={'step':step,'elapsed_seconds':time.perf_counter()-started,'summary':ev['summary']};history.append(row)
            payload={'model':model.state_dict(),'model_name':saved['model_name'],'in_channels':saved['in_channels'],'out_channels':saved['out_channels'],
                     'update':step,'seed':seed,'training_K':K,'arm':arm,'config':cfg,'scales':scales,'optimizer':opt.state_dict(),'data_version':'corrected_benchmark_v1'}
            torch.save(payload,folder/'last.pt')
            if key<best:
                best=key;torch.save(payload,folder/'selected.pt');save(folder/'selected_validation.json',ev)
                save(folder/'selection.json',{'step':step,'key':list(key),'checkpoint_sha256':sha(folder/'selected.pt')})
            save(folder/'history.json',history)
            print('CORRECTED_TRAIN',name,step,'closure_R',round(key[0],4),'max',round(key[1],4),flush=True)
    save(folder/'done.json',{'name':name,'seed':seed,'K':K,'arm':arm,'best':list(best),'wall_seconds':time.perf_counter()-started,'selected_sha256':sha(folder/'selected.pt')})

def all_training_locked():
    p=protocol();assert (OUT/'dataset_manifest.json').exists() or (OUT/'dataset_trainval_manifest.json').exists()
    subprocess.run([sys.executable,__file__,'--smoke'],check=True)
    arms=[('warm',seed,K) for seed in p['train']['seeds'] for K in [1,8]]+[('scratch',11,K) for K in [1,8]]
    logdir=OUT/'training_logs';logdir.mkdir(exist_ok=False);done=[]
    def one(task):
        arm,seed,K=task;name=f'{arm}_K{K}_seed{seed}'
        with (logdir/(name+'.log')).open('x') as f:
            proc=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed),'--K',str(K)],stdout=f,stderr=subprocess.STDOUT,
                                  env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            rc=proc.wait()
        if rc:raise RuntimeError((name,rc))
        return name
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(one,a) for a in arms]
        for f in as_completed(fs):done.append(f.result());save(OUT/'training_status.json',{'status':'running','completed':done,'total':len(arms)})
    candidates=[]
    for arm,seed,K in arms:
        folder=OUT/'training'/f'{arm}_K{K}_seed{seed}';s=json.loads((folder/'selection.json').read_text());candidates.append({'name':folder.name,'key':s['key'],'path':str(folder/'selected.pt'),'sha256':sha(folder/'selected.pt')})
    chosen=min(candidates,key=lambda c:(*c['key'],c['name']))
    prior=json.loads((OUT/'training'/chosen['name']/'configuration.json').read_text())['shape_mean_train80_St']
    save(OUT/'selected_model_lock.json',{'locked_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
         'chosen':chosen,'all_candidates':candidates,'closure':p['closure'],'shape_mean_train80_St':prior,
         'training_dataset_path':str(TRAINVAL if TRAINVAL.exists() else DATA),'training_dataset_sha256':sha(TRAINVAL if TRAINVAL.exists() else DATA),'no_test_or_confirmation_used_for_selection':True})
    save(OUT/'training_status.json',{'status':'complete','completed':done,'total':len(arms)})
    print('CORRECTED_MODEL_LOCKED',chosen['name'],flush=True)

def all_training():
    # The data producer also invokes --all after all120. Never duplicate training.
    with (OUT/'training_execution.lock').open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        if (OUT/'selected_model_lock.json').exists():
            lock=json.loads((OUT/'selected_model_lock.json').read_text())
            assert sha(lock['training_dataset_path'])==lock['training_dataset_sha256']
            assert sha(lock['chosen']['path'])==lock['chosen']['sha256']
            print('TRAINING_ALREADY_LOCKED_NO_DUPLICATE',flush=True);return
        all_training_locked()

def formal(role):
    from startup_observability_v1 import basis
    from corrected_benchmark_controls_v1 import controls,dmd_periodic
    p=protocol();lock=json.loads((OUT/'selected_model_lock.json').read_text());chosen=lock['chosen']
    assert sha(chosen['path'])==chosen['sha256']
    prior={k:(None,v) for k,v in lock['shape_mean_train80_St'].items()}
    selected_config=json.loads((Path(chosen['path']).parent/'configuration.json').read_text())
    scales=selected_config['normalization_train80']
    baseline_lock=json.loads((OUT/'controls_preregister.json').read_text())
    assert baseline_lock['DMD_ranks']==[3,9] and baseline_lock['bootstrap_steps']==48
    assert sha(ROOT/'scripts/corrected_benchmark_controls_v1.py')==baseline_lock['controls_script_sha256']
    if role=='test':
        rows=[]
        with h5py.File(DATA,'r') as h:
            for cid in p['splits']['test']:
                g=h['cases'][cid]
                rows.append({'id':cid,'q':{k:str(g.attrs[k]) if k=='shape' else float(g.attrs[k]) for k in ['shape','Re','cx','cy','D']},
                             'fields':g['fields'][:4],'geom':g['geom'][...],'mask':g['mask'][...],'times':g['times'][...]})
    else:
        rows=[]
        for q in p['confirmation_cases']:
            a=np.load(CACHE/(q['id']+'_startup.npz'));rows.append({'id':q['id'],'q':{k:q[k] for k in ['shape','Re','cx','cy','D']},'fields':a['fields'],
               'geom':a['geom'],'mask':a['mask'],'times':a['times'],'cache':str(CACHE/(q['id']+'.npz'))});a.close()
    folder=OUT/('already_seen_test32' if role=='test' else 'fresh_confirmation12');folder.mkdir(exist_ok=False)
    saved_models=[('old_F0',ROOT/'outputs/checkpoints/fno_long/best.pt'),('old_K8',BASE),('corrected_selected',Path(chosen['path']))]
    models={name:model_from(path,p['config'])[0] for name,path in saved_models}
    prediction_manifest=[]
    # Freeze all predictions before any scoring. Confirmation loads only four fields above.
    for row in rows:
        for name,model in models.items():
            base=predict(model,row);closed,diag=closure(row['q'],row['fields'][:4],base,row['times'],row['mask'],prior,p['closure'])
            variants=[(name,base),(name+'_closure',closed)]
            alternate=controls(row['fields'][:4],base,row['times'],row['mask'],p['closure'],diag['frequency'],scales)
            variants.extend((name+'_'+label,value[0]) for label,value in alternate.items())
            for label,a in variants:
                f=folder/(label+'__'+row['id']+'.npz');np.savez_compressed(f,prediction=a)
                prediction_manifest.append({'model':label,'id':row['id'],'sha256':sha(f)})
        relative=row['times']-row['times'][3];f=prior[row['q']['shape']][1]/row['q']['D']
        coef=np.linalg.lstsq(basis(relative[:4],f,1),row['fields'][:4].reshape(4,-1),rcond=1e-10)[0]
        harmonic=(basis(relative[4:164],f,1)@coef).reshape(160,3,64,64).astype(np.float32)
        persistent=np.repeat(row['fields'][3:4],160,axis=0)
        startup_dmd,_=dmd_periodic(row['fields'][:4],row['times'][:4],row['times'][4:164],row['mask'],3,scales)
        for label,a in [('fourframe_harmonic',harmonic),('copy_last',persistent),('real4_DMD_rank3',startup_dmd)]:
            target=folder/(label+'__'+row['id']+'.npz');np.savez_compressed(target,prediction=a)
            prediction_manifest.append({'model':label,'id':row['id'],'sha256':sha(target)})
    save(folder/'predictions_frozen.json',{'model_lock_sha256':sha(OUT/'selected_model_lock.json'),'registry':prediction_manifest,'phase':'all prediction files saved before future scoring'})
    reports={entry['model']:[] for entry in prediction_manifest}
    for row in rows:
        if role=='test':
            with h5py.File(DATA,'r') as h:truth=h['cases'][row['id']]['fields'][4:164]
        else:
            with np.load(row['cache']) as a:truth=a['fields'][4:164]
        for label in reports:
            f=folder/(label+'__'+row['id']+'.npz');pred=np.load(f)['prediction']
            reports[label].append({'id':row['id'],'q':row['q'],'horizons':metrics(pred,truth,row['mask'],row['times'])})
    summaries={name:{h:summarize([r['horizons'][h] for r in rs]) for h in ['2s','8s','20s','40s']} for name,rs in reports.items()}
    save(folder/'report.json',{'role':'already-seen development geometries' if role=='test' else 'previously unused confirmation12; frozen before CFD generation',
         'cases':reports,'summary':summaries,'no_model_selection_from_this_report':True,'old_closure_uses_new_train80_prior':True,
         'controls_preregister_sha256':sha(OUT/'controls_preregister.json'),'DMD_scales_from_corrected_train80_only':True})
    print('CORRECTED_FORMAL_DONE',role,[(k,round(v['40s']['mean_R'],4),v['40s']['n_pass']) for k,v in summaries.items()],flush=True)

def interface_smoke():
    from cpu_data_prepare import frames
    from geometry_attack_v2 import NativeSampler
    from flowproxy.geometry import geometry_channels
    p=protocol();cfg=p['config'];protected=sha(BASE);rows=[]
    quality=json.loads((ROOT/'outputs/true2d_recheck_v1/preregister.json').read_text())
    for q in quality['cases']:
        geo=geometry_channels(cfg,**{k:q[k] for k in ['shape','Re','cx','cy','D']})
        sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
        items=frames(ROOT/'data/raw/true2d_recheck_v1/fine_dt0005'/q['id'])
        rows.append({'fields':np.stack([sampler.sample(f) for t,f in items[:12]]),'geom':geo['geom'],'mask':geo['mask']})
    batch=rows*2;fields=torch.as_tensor(np.stack([r['fields'] for r in batch]),device='cuda');geom=torch.as_tensor(np.stack([r['geom'] for r in batch]),device='cuda');mask=torch.as_tensor(np.stack([r['mask'] for r in batch]),device='cuda');scales=torch.ones(3,device='cuda')
    torch.set_num_threads(1);model,saved=model_from(BASE,cfg)
    with torch.no_grad():
        b=len(batch);hist=torch.stack([fields[:,k:k+4] for k in range(8)],1).reshape(b*8,12,64,64)
        gs=geom[:,None].expand(-1,8,-1,-1,-1).reshape(b*8,4,64,64)
        vectorized=model(torch.cat([gs,hist],1))[:,:3].reshape(b,8,3,64,64)
        independent=torch.stack([model(torch.cat([geom,fields[:,k:k+4].reshape(b,12,64,64)],1))[:,:3] for k in range(8)],1)
        max_delta=float((vectorized-independent).abs().max());print('TEACHER_BATCH_PARITY',max_delta,flush=True);assert max_delta<2e-6
        zero=float(field_loss_steps(fields[:,4:12],fields[:,4:12],mask,scales));assert zero==0
    results={}
    for K in [1,8]:
        model.zero_grad(set_to_none=True);loss=train_loss(model,geom,fields,mask,scales,K);loss.backward()
        gs=[q.grad for q in model.parameters() if q.grad is not None]
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gs) and sum(float(g.square().sum()) for g in gs)>0
        results[str(K)]={'loss':float(loss.detach()),'finite_nonzero_gradients':True,'gradient_norm':sum(float(g.square().sum()) for g in gs)**.5}
    assert sha(BASE)==protected
    save(OUT/'interface_smoke.json',{'known_reference_cases':2,'purpose':'GPU and loss interface only, no optimizer update or trained model claim','teacher_forced_batch_vs_individual_max_abs_delta':max_delta,'exact_target_loss':zero,'K1_K8':results,'original_weights_unchanged':True,'peak_GPU_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
    print('RESTART_INTERFACE_SMOKE_OK',max_delta,results,flush=True)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--worker',choices=['warm','scratch']);a.add_argument('--seed',type=int,default=11);a.add_argument('--K',type=int,choices=[1,8],default=8)
    a.add_argument('--all',action='store_true');a.add_argument('--smoke',action='store_true');a.add_argument('--interface-smoke',action='store_true');a.add_argument('--test',action='store_true');a.add_argument('--confirm',action='store_true');args=a.parse_args()
    if args.worker:worker(args.worker,args.seed,args.K)
    elif args.smoke:worker('warm',11,8,True)
    elif args.all:all_training()
    elif args.test:formal('test')
    elif args.confirm:formal('confirm')
    elif args.interface_smoke:interface_smoke()
