#!/usr/bin/env python3
"""Paired cycle-path supervision and a separately paired CFD-data supplement."""
from pathlib import Path
import argparse, datetime, json, os, subprocess, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from geometry_attack_v2 import save, sha
from corrected_benchmark_train_v1 import Store, model_from, train_loss, field_loss_steps, predict, protocol
from corrected_benchmark_controls_v1 import dmd_periodic
from startup_observability_v1 import frequency
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

FORMAL = ROOT/'outputs/period_consistency_train_v1/cycle_seed11/selected.pt'
FORMAL_SHA = '6111b1944227614893269d8a420ab649dbb89775aa185a7e1724a1719d5ef4f5'
PERIODS = ROOT/'outputs/period_consistency_train_v1/train80_periods.json'
CFD = ROOT/'outputs/continuous_training_cfd_v1'
CACHE = ROOT/'data/processed/continuous_training_cfd_v1_cases'
N_UPDATES = 1000
CHECKS = [0, 250, 500, 750, 1000]
FRACTIONS = [.25, .5, .75]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def out_for(suite): return ROOT/'outputs'/('cycle_landmarks_train_v1' if suite == 'landmarks' else 'continuous_data_train_v1')
def configuration(suite): return json.loads((out_for(suite)/'preregister.json').read_text())

def all_rows(suite):
    rows = Store('train').rows
    if suite == 'data':
        p = json.loads((CFD/'preregister.json').read_text())
        manifest = json.loads((CFD/'training_manifest.json').read_text())
        assert len(manifest['cases']) == len(p['cases']) == 16
        for q in p['cases']:
            result = json.loads((CACHE/(q['id']+'.json')).read_text())
            file = CACHE/(q['id']+'.npz')
            assert sha(file) == result['cache_sha256']
            with np.load(file) as a:
                rows.append({'id': q['id'], 'q': {k:q[k] for k in ['shape','Re','cx','cy','D']},
                    'fields':a['fields'], 'times':a['times'], 'geom':a['geom'], 'mask':a['mask'], 'St':result['St']})
    return rows

def derive_period(row):
    mask = row['mask'] > .5
    t = row['times'].astype(np.float64) - float(row['times'][0])
    assert len(t) == 165 and np.allclose(np.diff(t), .25)
    f, diagnostic = frequency(t, row['fields'][:,1][:,mask].astype(np.float64), 4, row['St']/row['q']['D'], .05)
    count = 1/(f*.25)
    assert 12 < count < 40
    return {'id':row['id'], 'frequency':f, 'period_seconds':1/f, 'dt':.25,
        'N':int(np.floor(count)), 'alpha':float(count-np.floor(count)), 'fit':diagnostic}

def interpolate_four(sequence, b, offset):
    lower = int(np.floor(offset)); alpha = float(offset-lower)
    assert lower >= 0 and lower+4 < sequence.shape[1]
    return (1-alpha)*sequence[b,lower:lower+4] + alpha*sequence[b,lower+1:lower+5]

def orbit_losses(model, geom, fields, mask, scales, periods, picks, with_landmarks):
    batch = fields.shape[0]
    nmax = max(periods[int(j)]['N'] for j in picks) + 1
    assert fields.shape[1] >= nmax+4
    window = [fields[:,k] for k in range(4)]
    for k in range(nmax):
        history = torch.stack(window[-4:], 1).reshape(batch,12,64,64)
        window.append(model(torch.cat([geom,history],1))[:,:3])
    trajectory = torch.stack(window,1)
    aligned = torch.stack([interpolate_four(trajectory,b,periods[int(j)]['N']+periods[int(j)]['alpha']) for b,j in enumerate(picks)])
    cycle = field_loss_steps(aligned,fields[:,:4],mask,scales)
    landmarks = torch.zeros((),device=geom.device)
    if with_landmarks:
        terms=[]
        for fraction in FRACTIONS:
            predicted=[]; target=[]
            for b,j in enumerate(picks):
                count=periods[int(j)]['N']+periods[int(j)]['alpha']
                predicted.append(interpolate_four(trajectory,b,fraction*count))
                target.append(interpolate_four(fields,b,fraction*count))
            terms.append(field_loss_steps(torch.stack(predicted),torch.stack(target),mask,scales))
        landmarks=torch.stack(terms).mean()
    return cycle,landmarks,nmax

def evaluate_dmd(model, store, scales):
    direct=[]; dmd=[]
    for row in store.rows:
        base=predict(model,row)
        observed=np.concatenate([row['fields'][:4],base[:48]])
        extended,diagnostic=dmd_periodic(observed,row['times'][:52],row['times'][4:164],row['mask'],9,scales)
        extended[:48]=base[:48]
        truth=row['fields'][4:164]
        direct.append({'id':row['id'],'horizons':metrics(base,truth,row['mask'],row['times'])})
        dmd.append({'id':row['id'],'horizons':metrics(extended,truth,row['mask'],row['times']),'DMD':diagnostic})
    return {'cases_direct':direct,'cases_DMD9':dmd,
        'summary':{label:{h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']}
                   for label,rows in [('direct',direct),('DMD9',dmd)]}}

def task_sequences(suite, seed, rows, periods):
    rng=np.random.default_rng(seed); first=[]; second=[]; starts=[]
    for step in range(N_UPDATES):
        if suite=='landmarks':
            old=rng.choice(80,4,replace=False); new=old.copy()
        else:
            supplement=80+int(rng.integers(0,16)); q=rows[supplement]['q']
            choices=[j for j,r in enumerate(rows[:80]) if r['q']['shape']==q['shape']]
            nearest=min(choices,key=lambda j:((rows[j]['q']['Re']-q['Re'])/120)**2+((rows[j]['q']['cy']-q['cy'])/.8)**2+((rows[j]['q']['D']-q['D'])/.2)**2)
            shared=rng.choice([j for j in range(80) if j!=nearest],3,replace=False)
            old=np.concatenate([shared,[nearest]]); new=np.concatenate([shared,[supplement]])
        limit=min(len(rows[int(j)]['fields'])-periods[int(j)]['N']-5 for j in np.concatenate([old,new]))
        start=int(rng.integers(0,limit+1)); first.append(old); second.append(new); starts.append(start)
    return np.array(first),np.array(second),np.array(starts)

def configure(suite):
    out=out_for(suite); out.mkdir(exist_ok=False)
    assert sha(FORMAL)==FORMAL_SHA
    rows=all_rows(suite)
    periods=json.loads(PERIODS.read_text())['rows']
    assert [r['id'] for r in rows[:80]]==[p['id'] for p in periods]
    if suite=='data': periods+= [derive_period(row) for row in rows[80:]]
    save(out/'training_periods.json',{'source':'original80 sealed labels; any added periods use SUPPLEMENT TRAINING fields only','rows':periods})
    cfg=json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/training/warm_K1_seed33/configuration.json').read_text())
    baseline=json.loads((ROOT/'outputs/period_consistency_evaluate_v1/validation_report.json').read_text())['summary']['cycle_seed11_self_DMD_rank9']
    seeds=[66,77] if suite=='landmarks' else [88,99]
    arms=['control','landmarks'] if suite=='landmarks' else ['original80','augmented96']
    c={'registered_utc':now(),'suite':suite,'network':{'path':str(FORMAL),'sha256':FORMAL_SHA},
        'seeds':seeds,'arms':arms,'updates_max':N_UPDATES,'checks':CHECKS,'include_step0':True,
        'lr':1e-5,'AdamW_weight_decay':1e-4,'gradient_clip':1.,'batch_cases':4,
        'channel_scales_train80':cfg['normalization_train80'],'lambda_cycle':.1,
        'lambda_landmark':.1 if suite=='landmarks' else 0.,'landmark_fractions':FRACTIONS,
        'loss':'same eight teacher-forced field targets +0.1 full-cycle return; landmark arm adds0.1 mean of three four-state phase-position field losses',
        'landmark_target':'true TRAINING states at quarter,half,three-quarter periods; same adjacent native-frame interpolation for prediction and target; never approximate antisymmetry',
        'information':'new-case inference uses FOUR true starting fields only; no target period, force, phase, or future frames',
        'decoder':'unchanged rank9 unit-circle DMD of4 true +48 self-generated states; direct prefix kept',
        'selection':'original val8 only; minimum40s DMD9 meanRi with8s8/8 pass, then maxRi,then earliest; include0',
        'gate':'both experimental seeds better meanRi than their paired controls; at least one >=10% lower meanRi AND >=10% lower maxRi than frozen initial;8s8/8',
        'early_stop':'at500 if best meanRi not improved2% AND current meanRi>1.25 initial',
        'seen48':'32+12+4 are developmental only; not used for training or validation checkpoint selection',
        'sampling':'same four original cases and start order in each pair' if suite=='landmarks' else 'same3 original cases+same start; fourth slot is new TRAINING case vs metadata-nearest original training control;25% supplement exposure',
        'training_cases':len(rows),'additional_data_manifest_sha256':sha(CFD/'training_manifest.json') if suite=='data' else None,
        'baseline_validation':baseline,'script_sha256':sha(__file__),
        'formal_weight_preserved':True,'hardware_plan':'four workers,one CPU thread each; CFD concurrently <=4workers x2MPI'}
    save(out/'preregister.json',c)
    for seed in seeds:
        a,b,starts=task_sequences(suite,seed,rows,periods)
        np.savez(out/f'tasks_seed{seed}.npz',control_picks=a,experiment_picks=b,starts=starts)
    save(out/'status.json',{'status':'configured','suite':suite,'utc':now()})

def interpolation_check():
    sequence=torch.arange(60,dtype=torch.float64).reshape(1,60,1,1,1)
    for offset in [4.75,9.5,14.25,19.]:
        result=interpolate_four(sequence,0,offset).flatten()
        assert torch.allclose(result,torch.arange(4,dtype=torch.float64)+offset,atol=1e-12,rtol=0)

def worker(suite,arm,seed,smoke=False):
    out=out_for(suite); c=configuration(suite); torch.set_num_threads(1)
    rows=all_rows(suite); val=Store('val'); periods=json.loads((out/'training_periods.json').read_text())['rows']
    task=np.load(out/f'tasks_seed{seed}.npz'); experimental=arm==c['arms'][1]
    picks=task['experiment_picks' if experimental else 'control_picks']; starts=task['starts']
    model,saved=model_from(FORMAL,protocol()['config'],seed=seed)
    scales=torch.as_tensor(c['channel_scales_train80'],device='cuda')
    def batch(step):
        pick=picks[step]; start=int(starts[step]); length=max(periods[int(j)]['N'] for j in pick)+5
        geom=torch.as_tensor(np.stack([rows[int(j)]['geom'] for j in pick]),device='cuda')
        fields=torch.as_tensor(np.stack([rows[int(j)]['fields'][start:start+length] for j in pick]),device='cuda')
        mask=torch.as_tensor(np.stack([rows[int(j)]['mask'] for j in pick]),device='cuda')
        return pick,geom,fields,mask
    if smoke:
        interpolation_check(); pick,geom,fields,mask=batch(0); model.train();torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
        short=train_loss(model,geom,fields[:,:12],mask,scales,1)
        cyc,landmarks,nmax=orbit_losses(model,geom,fields,mask,scales,periods,pick,suite=='landmarks')
        loss=short+.1*cyc+c['lambda_landmark']*landmarks;loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        result={'suite':suite,'no_optimizer_updates':True,'interpolation_linear_ramp_check':True,
            'short_loss':float(short.detach()),'cycle_loss':float(cyc.detach()),'landmark_loss':float(landmarks.detach()),
            'autonomous_steps':nmax,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,
            'seconds_forward_backward':time.perf_counter()-begin,'finite_gradients':True}
        if suite=='landmarks':
            from period_consistency_train_v1 import cycle_loss
            old_cycle,_=cycle_loss(model,geom,fields[:,:12],mask,scales,periods,pick)
            assert abs(float(old_cycle.detach())-result['cycle_loss'])<1e-8
            result['cycle_loss_matches_original_implementation']=True
        save(out/'smoke.json',result);print('CYCLE_PATH_SMOKE_OK',suite,nmax,flush=True);return
    folder=out/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['AdamW_weight_decay'])
    save(folder/'configuration.json',{'arm':arm,'seed':seed,'suite':suite,'initial_weight_sha256':FORMAL_SHA,
        'tasks_sha256':sha(out/f'tasks_seed{seed}.npz'),'lambda_cycle':.1,
        'lambda_landmark':c['lambda_landmark'] if experimental else 0.,'input_channels':saved['in_channels'],
        'new_training_cases_used':16 if suite=='data' and experimental else 0})
    best=(float('inf'),float('inf'),0); history=[]; begin=time.perf_counter(); maxgrad=0.; clips=0; last=0
    for step in range(N_UPDATES+1):
        if step:
            pick,geom,fields,mask=batch(step-1);model.train();optimizer.zero_grad(set_to_none=True)
            short=train_loss(model,geom,fields[:,:12],mask,scales,1)
            cyc,landmark,_=orbit_losses(model,geom,fields,mask,scales,periods,pick,suite=='landmarks')
            weight=c['lambda_landmark'] if experimental else 0.
            loss=short+.1*cyc+weight*landmark;assert torch.isfinite(loss)
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip']))
            assert np.isfinite(norm);maxgrad=max(maxgrad,norm);clips+=int(norm>c['gradient_clip']);optimizer.step()
            if step%25==0:save(folder/'status.json',{'step':step,'elapsed_seconds':time.perf_counter()-begin,
                'short_loss':float(short.detach()),'cycle_loss':float(cyc.detach()),'landmark_loss':float(landmark.detach()),
                'lambda_landmark':weight,'gradient_norm_before_clip':norm})
        if step in CHECKS:
            ev=evaluate_dmd(model,val,c['channel_scales_train80']);q=ev['summary']['DMD9']['40s']
            valid=ev['summary']['direct']['8s']['n_pass']==8
            if step==0: assert abs(q['mean_R']-c['baseline_validation']['40s']['mean_R'])<2e-4
            key=(q['mean_R'],q['max_R'],step);save(folder/f'validation_{step}.json',ev)
            history.append({'step':step,'summary':ev['summary'],'eligible_short8s':valid})
            if valid and key<best:
                best=key;torch.save({'model':model.state_dict(),'model_name':saved['model_name'],
                    'in_channels':saved['in_channels'],'out_channels':saved['out_channels'],'update':step,'seed':seed,
                    'arm':arm,'suite':suite,'config':protocol()['config'],'optimizer':optimizer.state_dict(),
                    'data_version':'corrected_benchmark_v1'+('_plus_training16' if suite=='data' and experimental else '')},folder/'selected.pt')
                save(folder/'selected_validation.json',ev);save(folder/'selection.json',{'step':step,'key':list(best),'sha256':sha(folder/'selected.pt')})
            save(folder/'history.json',history);last=step
            print('CYCLE_PATH_TRAIN',suite,arm,seed,step,'R',round(q['mean_R'],4),'MAX',round(q['max_R'],4),'PASS',q['n_pass'],flush=True)
            initial=c['baseline_validation']['40s']['mean_R']
            if step==500 and best[0]>.98*initial and q['mean_R']>1.25*initial:break
    save(folder/'done.json',{'status':'complete','last_update':last,'best':list(best),'max_gradient_norm':maxgrad,
        'gradient_clip_count':clips,'wall_seconds':time.perf_counter()-begin})

def orchestrate(suite):
    out=out_for(suite);configure(suite);c=configuration(suite)
    with (out/'smoke.log').open('x') as f:
        subprocess.run([sys.executable,__file__,'--suite',suite,'--smoke'],stdout=f,stderr=subprocess.STDOUT,check=True)
    smoke=json.loads((out/'smoke.json').read_text());assert smoke['finite_gradients'] and smoke['peak_GPU_GiB']<5.2
    processes=[]
    for seed in c['seeds']:
        for arm in c['arms']:
            log=(out/f'{arm}_seed{seed}.log').open('x')
            process=subprocess.Popen([sys.executable,__file__,'--suite',suite,'--worker',arm,'--seed',str(seed)],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            processes.append((arm,seed,process,log))
    worker_list=[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,f in processes]
    while True:
        completed=[{'arm':a,'seed':s,'exit_code':p.returncode} for a,s,p,f in processes if p.poll() is not None]
        save(out/'status.json',{'status':'training','workers':worker_list,'completed':completed,'utc':now(),'smoke':smoke})
        failures=[x for x in completed if x['exit_code']!=0]
        if failures:raise RuntimeError(failures)
        if len(completed)==len(processes):break
        time.sleep(15)
    for a,s,p,f in processes:f.close()
    baseline=c['baseline_validation']['40s'];pairs=[];candidates=[]
    for seed in c['seeds']:
        scores={}
        for arm in c['arms']:
            folder=out/f'{arm}_seed{seed}';ev=json.loads((folder/'selected_validation.json').read_text());sel=json.loads((folder/'selection.json').read_text())
            scores[arm]=ev['summary']['DMD9']['40s']
            candidates.append({'arm':arm,'seed':seed,'path':str(folder/'selected.pt'),'sha256':sha(folder/'selected.pt'),'selection':sel,'summary':ev['summary']})
        pairs.append({'seed':seed,'experimental_better_mean_than_control':scores[c['arms'][1]]['mean_R']<scores[c['arms'][0]]['mean_R'],'scores':scores})
    qualified=[x for x in candidates if x['arm']==c['arms'][1] and x['summary']['DMD9']['40s']['mean_R']<=.9*baseline['mean_R'] and x['summary']['DMD9']['40s']['max_R']<=.9*baseline['max_R']]
    passed=all(x['experimental_better_mean_than_control'] for x in pairs) and bool(qualified)
    chosen=min(qualified,key=lambda x:(x['summary']['DMD9']['40s']['mean_R'],x['summary']['DMD9']['40s']['max_R'])) if passed else None
    assert sha(FORMAL)==FORMAL_SHA
    save(out/'completed_report.json',{'gate_passed':passed,'pairs':pairs,'candidates':candidates,'selected_if_gate_passed':chosen,'baseline':baseline,'formal_model_unchanged':True})
    save(out/'status.json',{'status':'complete','gate_passed':passed,'pairs':pairs,'utc':now()})
    print('CYCLE_PATH_COMPLETE',suite,passed,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['landmarks','data'],default='landmarks')
    parser.add_argument('--worker',choices=['control','landmarks','original80','augmented96'])
    parser.add_argument('--seed',type=int);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:
        if args.worker or args.smoke:
            c=configuration(args.suite);worker(args.suite,args.worker or c['arms'][1],args.seed or c['seeds'][0],args.smoke)
        else:orchestrate(args.suite)
    except Exception as error:
        out=out_for(args.suite)
        if out.exists() and not args.worker and not args.smoke:save(out/'failure.json',{'status':'failed','error':repr(error),'utc':now()})
        raise
