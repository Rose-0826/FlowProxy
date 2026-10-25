#!/usr/bin/env python3
"""Train the existing 48-step neural prefix through its exact periodic DMD9 decoder."""
from pathlib import Path
import argparse, datetime, json, os, subprocess, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from geometry_attack_v2 import save, sha
from corrected_benchmark_train_v1 import Store, model_from, train_loss, field_loss_steps, protocol
from corrected_benchmark_controls_v1 import dmd_periodic
from period_consistency_train_v1 import cycle_loss
from cycle_landmarks_train_v1 import evaluate_dmd, FORMAL, FORMAL_SHA, PERIODS

OUT = ROOT/'outputs/end_to_end_periodic_train_v1'
SEEDS = [111, 222]
CHECKS = [0, 250, 500, 750, 1000]
UPDATES = 1000

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def configuration_scales():
    return json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/training/warm_K1_seed33/configuration.json').read_text())['normalization_train80']

def neural_prefix(model, geom, fields, n=48):
    window = [fields[:, k] for k in range(4)]
    for k in range(n):
        history = torch.stack(window[-4:], 1).reshape(len(fields), 12, 64, 64)
        window.append(model(torch.cat([geom, history], 1))[:, :3])
    return torch.stack(window, 1)

def differentiable_dmd(observed, times, future, mask, scales, output_float32=True):
    """Same SVD/eigensystem/global amplitude least squares as frozen NumPy decoder.

    The complete state, basis, eigenvalue angle and coefficients retain gradients.
    Only the locally constant rank choice and reporting diagnostics are detached.
    Complex amplitude LS uses CPU gelsd to retain NumPy's 1e-10 rank convention.
    """
    scales = torch.as_tensor(scales, device=observed.device, dtype=torch.float64)
    fields = observed.to(torch.float64)*mask.to(torch.float64)[None, None]
    Y = (fields/scales[None, :, None, None]).reshape(len(fields), -1).T
    kwargs = {'driver':'gesvd'} if Y.is_cuda else {}
    U, s, Vh = torch.linalg.svd(Y[:, :-1], full_matrices=False, **kwargs)
    r = min(9, int((s > s[0]*1e-9).sum().detach().cpu()))
    assert r == 9, f'Rank changed: {r}'
    retained_s = s[:r]
    AV = (Y[:, 1:] @ Vh[:r].T)/retained_s[None]
    eig, W = torch.linalg.eig(U[:, :r].T @ AV)
    Phi = AV.to(torch.complex128) @ W
    t = torch.as_tensor(times, device=Y.device, dtype=torch.float64)
    tf = torch.as_tensor(future, device=Y.device, dtype=torch.float64)
    dt = (t[1:]-t[:-1]).mean()
    assert torch.allclose(t[1:]-t[:-1], dt, atol=1e-8, rtol=0)
    angular = eig.angle()/dt
    E = torch.exp(1j*(t-t[0])[:, None]*angular[None])
    gram = (Phi.conj().T @ Phi)*(E.conj().T @ E)
    rhs = (E.conj().T*(Phi.conj().T @ Y.to(torch.complex128))).sum(1)
    fitted = torch.linalg.lstsq(gram.cpu(), rhs.cpu(), rcond=1e-10, driver='gelsd')
    coefficients = fitted.solution.to(Y.device)
    assert int(fitted.rank) == 9, f'Amplitude fit rank changed: {int(fitted.rank)}'
    Ef = torch.exp(1j*(tf-t[0])[:, None]*angular[None])
    pred = (Ef @ (Phi*coefficients[None]).T).real.reshape(len(tf), 3, 64, 64)
    pred = pred*scales[None, :, None, None]
    gap = (eig[:, None]-eig[None]).abs()
    gap = gap + torch.eye(r, device=Y.device)*1e10
    diagnostic = {'rank':r, 'eigenvalue_min_separation':float(gap.min().detach()),
        'relative_singular_gap_at_cutoff':float(((s[r-1]-s[r])/s[0]).detach()),
        'retained_singular_condition':float((s[0]/s[r-1]).detach()),
        'amplitude_condition':float((fitted.singular_values[0]/fitted.singular_values[-1]).detach()),
        'all_parameters_differentiated':True}
    return (pred.to(torch.float32) if output_float32 else pred), diagnostic

def final_trajectory_loss(model, geom, fields, mask, scales, times):
    observed = neural_prefix(model, geom, fields)
    decoded = []
    for b in range(len(fields)):
        extended, _ = differentiable_dmd(observed[b], times[b][:52], times[b][4:164], mask[b], scales)
        decoded.append(torch.cat([observed[b, 4:52], extended[48:]], 0))
    prediction = torch.stack(decoded)
    return field_loss_steps(prediction, fields[:, 4:164], mask, scales)

def gradient_norm(grads):
    return float(torch.stack([g.abs().square().sum() for g in grads if g is not None]).sum().sqrt())

def smoke():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    assert sha(FORMAL) == FORMAL_SHA
    train = Store('train'); scales_list = configuration_scales()
    scales = torch.as_tensor(scales_list, device='cuda', dtype=torch.float32)
    periods = json.loads(PERIODS.read_text())['rows']
    assert [r['id'] for r in train.rows] == [p['id'] for p in periods]
    model, _ = model_from(FORMAL, protocol()['config'], seed=111)
    tests = []
    for j in [0, 10, 20, 30, 40, 50, 60, 70]:
        row = train.rows[j]
        geom = torch.as_tensor(row['geom'][None], device='cuda')
        fields = torch.as_tensor(row['fields'][None], device='cuda')
        mask = torch.as_tensor(row['mask'], device='cuda')
        model.eval()
        with torch.no_grad(): observed = neural_prefix(model, geom, fields)[0]
        observed = observed.to(torch.float64).detach().requires_grad_(True)
        pred, diag = differentiable_dmd(observed, row['times'][:52], row['times'][4:164], mask, scales_list, False)
        original, _ = dmd_periodic(observed.detach().cpu().numpy(), row['times'][:52], row['times'][4:164], row['mask'], 9, scales_list)
        delta = (pred.detach().cpu().numpy()-original)*row['mask'][None, None]
        relative = np.linalg.norm(delta, axis=(-2,-1))/np.maximum(np.linalg.norm(original*row['mask'][None,None],axis=(-2,-1)),1e-8)
        assert float(relative.max()) < 2e-5, ('Forward mismatch', row['id'], relative.max())
        target = torch.as_tensor(row['fields'][4:164], device='cuda', dtype=torch.float64)
        objective = (((pred-target)/scales[None,:,None,None]).square()*mask[None,None]).mean()
        grad, = torch.autograd.grad(objective, observed)
        assert torch.isfinite(grad).all() and grad.norm() > 0
        item = {'id':row['id'], 'max_forward_channel_relative_difference':float(relative.max()),
            'finite_decoder_gradient':True, 'gradient_norm':float(grad.norm()), 'DMD':diag}
        if j in [0, 40]:
            direction = torch.zeros_like(observed)
            direction[1:-1] = (observed.detach()[2:]-observed.detach()[:-2])/.5
            direction *= torch.linspace(0,1,52,device='cuda')[:,None,None,None]
            direction /= direction.norm()
            analytic = float((grad*direction).sum())
            fd = []
            for eps in [1e-4, 3e-5, 1e-5]:
                values=[]
                with torch.no_grad():
                    for sign in [-1,1]:
                        q,_ = differentiable_dmd(observed.detach()+sign*eps*direction,row['times'][:52],row['times'][4:164],mask,scales_list,False)
                        values.append(float((((q-target)/scales[None,:,None,None]).square()*mask[None,None]).mean()))
                numerical=(values[1]-values[0])/(2*eps)
                fd.append({'epsilon':eps,'numerical':numerical,'relative_difference':abs(numerical-analytic)/max(abs(numerical),abs(analytic),1e-10)})
            assert min(q['relative_difference'] for q in fd)<.02, ('Gradient check failed',analytic,fd)
            item['directional_derivative']={'analytic':analytic,'finite_differences':fd}
        tests.append(item)
        save(OUT/'decoder_checks.json',{'cases':tests,'training_only':True})
        print('DIFFERENTIABLE_DMD_CHECK',row['id'],'forward',float(relative.max()),'grad',float(grad.norm()),flush=True)
    probes=[]
    params=list(model.parameters())
    for seed in SEEDS:
        rng=np.random.default_rng(seed); picks=rng.choice(80,4,replace=False)
        start=int(rng.integers(0,min(165-4-periods[int(j)]['N']-1 for j in picks)+1))
        origins=rng.integers(0,2,size=4)
        geom=torch.as_tensor(np.stack([train.rows[int(j)]['geom'] for j in picks]),device='cuda')
        mask=torch.as_tensor(np.stack([train.rows[int(j)]['mask'] for j in picks]),device='cuda')
        shortfields=torch.as_tensor(np.stack([train.rows[int(j)]['fields'][start:start+12] for j in picks]),device='cuda')
        longfields=torch.as_tensor(np.stack([train.rows[int(j)]['fields'][int(o):int(o)+164] for j,o in zip(picks,origins)]),device='cuda')
        times=[train.rows[int(j)]['times'][int(o):int(o)+164] for j,o in zip(picks,origins)]
        model.train(); model.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
        short=train_loss(model,geom,shortfields,mask,scales,1)
        cyc,_=cycle_loss(model,geom,shortfields,mask,scales,periods,picks)
        base=short+.1*cyc
        basegrad=torch.autograd.grad(base,params,allow_unused=True)
        end=final_trajectory_loss(model,geom,longfields,mask,scales,times)
        endgrad=torch.autograd.grad(end,params,allow_unused=True)
        assert all(torch.isfinite(g).all() for g in basegrad+endgrad if g is not None)
        bn,en=gradient_norm(basegrad),gradient_norm(endgrad)
        assert bn>0 and en>0
        torch.cuda.synchronize()
        probes.append({'seed':seed,'picks':picks.tolist(),'short_start':start,'long_origins':origins.tolist(),
            'short_loss':float(short.detach()),'cycle_loss':float(cyc.detach()),'final40_loss':float(end.detach()),
            'base_gradient_norm':bn,'final40_gradient_norm':en,'suggested_lambda':.5*bn/en,
            'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'seconds_forward_backward':time.perf_counter()-begin})
        save(OUT/'model_probes.json',{'probes':probes,'no_optimizer_updates':True,'training_only':True})
        print('FULL_MODEL_GRADIENT_CHECK',seed,bn,en,probes[-1]['peak_GPU_GiB'],flush=True)
    weight=float(np.clip(np.median([r['suggested_lambda'] for r in probes]),1e-5,.1))
    peak=max(r['peak_GPU_GiB'] for r in probes)
    concurrency=max(1,min(4,int(21/(peak+.8))))
    report={'status':'passed','utc':now(),'source_sha256':sha(__file__),'formal_sha256':sha(FORMAL),
        'decoder_training_cases_checked':8,'finite_difference_cases':2,'finite_decoder_and_network_gradients':True,
        'no_optimizer_updates':True,'training_only':True,'all_DMD_parameters_differentiated':True,
        'lambda_final40_train_probe_only':weight,'weight_rule':'clip(0.5*median(base gradient norm / final40 gradient norm),1e-5,0.1); freeze before validation',
        'peak_GPU_GiB':peak,'worker_concurrency':concurrency,'probes':probes,'decoder_checks':tests}
    save(OUT/'smoke.json',report)
    print('END_TO_END_SMOKE_PASSED','lambda',weight,'concurrency',concurrency,flush=True)

def configure():
    assert sha(FORMAL) == FORMAL_SHA
    assert not (OUT/'preregister.json').exists()
    checked=json.loads((OUT/'smoke.json').read_text())
    assert checked['status']=='passed' and checked['all_DMD_parameters_differentiated']
    rows=Store('train').rows;periods=json.loads(PERIODS.read_text())['rows']
    assert [r['id'] for r in rows]==[p['id'] for p in periods]
    baseline=json.loads((ROOT/'outputs/period_consistency_evaluate_v1/validation_report.json').read_text())['summary']['cycle_seed11_self_DMD_rank9']
    config={'registered_utc':now(),'source_sha256':sha(__file__),'network':{'path':str(FORMAL),'sha256':FORMAL_SHA},
        'seeds':SEEDS,'arms':['control','joint_DMD40'],'updates_max':UPDATES,'checks':CHECKS,'include_step0':True,
        'lr':1e-6,'AdamW_weight_decay':1e-4,'gradient_clip':1.,'batch_cases':4,'lambda_cycle':.1,
        'lambda_final40':checked['lambda_final40_train_probe_only'],'lambda_rule':checked['weight_rule'],
        'channel_scales_train80':configuration_scales(),'training_cases':80,'new_CFD_cases_used':0,
        'base_loss':'same eight teacher-forced field targets + 0.1 one-cycle four-state return',
        'joint_loss':'base + fixed lambda * (.5 mean + .5 endpoint) scaled field MSE of complete 160-state prediction; neural prefix48 kept, DMD tail112',
        'differentiation':'full gradient through all48 autonomous neural steps, rank9 SVD, eigenvalue phase, DMD modes and global complex amplitude least squares; no detached clock or basis',
        'rank_policy':'rank9 and amplitude rank9 required; any change or nonfinite gradient fails worker, no selective sample skipping',
        'long_targets':'true TRAINING fields only, start index0 or1 because165 states allow164-state40s sequence',
        'short_sampling':'matched prewritten four cases and random cycle-safe starts in each pair; same long origins prewritten too',
        'information_at_inference':'four target initial fields+geometry/Re only; no target future, force, period or phase; unchanged decoder',
        'decoder':'frozen NumPy rank9 unit-circle DMD of4 true +48 autonomous generated states; first48 predictions kept',
        'selection':'val8 only; minimum40s DMD meanRi among8s8/8-passing checkpoints; tie maximumRi then earliest; include0',
        'gate':'both joint seeds strictly better meanRi than matched controls; at least one joint meanRi AND maxRi >=10% lower than frozen initial; keep8s8/8',
        'early_stop':'at500 if best meanRi not improved2% and current meanRi >1.25 initial',
        'seen48':'all32+12+4 already seen; later retest developmental only, not training/selection or fresh confirmation',
        'fair_compute':'same optimizer/sample/lr for controls; joint arm uses extra48-step graph+DMD backward, walltime reported separately',
        'baseline_validation':baseline,'formal_weight_preserved':True,
        'smoke_sha256':sha(OUT/'smoke.json'),'probe_no_optimizer_updates':True}
    save(OUT/'preregister.json',config)
    save(OUT/'training_periods.json',{'source_sha256':sha(PERIODS),'rows':periods})
    for seed in SEEDS:
        rng=np.random.default_rng(seed);origin_rng=np.random.default_rng(seed+10000)
        picks=[];starts=[];origins=[]
        for k in range(UPDATES):
            pick=rng.choice(80,4,replace=False)
            limit=min(165-4-periods[int(j)]['N']-1 for j in pick)
            picks.append(pick);starts.append(int(rng.integers(0,limit+1)))
            origins.append(origin_rng.integers(0,2,size=4))
        np.savez(OUT/f'tasks_seed{seed}.npz',picks=np.array(picks),starts=np.array(starts),origins=np.array(origins))
    save(OUT/'status.json',{'status':'configured','utc':now()})

def worker(arm,seed,combined_smoke=False):
    torch.set_num_threads(1)
    c=json.loads((OUT/'preregister.json').read_text())
    assert c['source_sha256']==sha(__file__) and sha(FORMAL)==FORMAL_SHA
    rows=Store('train').rows;val=Store('val')
    periods=json.loads((OUT/'training_periods.json').read_text())['rows']
    tasks=np.load(OUT/f'tasks_seed{seed}.npz')
    model,saved=model_from(FORMAL,protocol()['config'],seed=seed)
    scales=torch.as_tensor(c['channel_scales_train80'],device='cuda',dtype=torch.float32)
    def loss_for(step):
        picks=tasks['picks'][step];start=int(tasks['starts'][step]);origins=tasks['origins'][step]
        geom=torch.as_tensor(np.stack([rows[int(j)]['geom'] for j in picks]),device='cuda')
        mask=torch.as_tensor(np.stack([rows[int(j)]['mask'] for j in picks]),device='cuda')
        fields=torch.as_tensor(np.stack([rows[int(j)]['fields'][start:start+12] for j in picks]),device='cuda')
        short=train_loss(model,geom,fields,mask,scales,1)
        cycle,_=cycle_loss(model,geom,fields,mask,scales,periods,picks)
        final=torch.zeros((),device='cuda')
        if arm=='joint_DMD40':
            longfields=torch.as_tensor(np.stack([rows[int(j)]['fields'][int(o):int(o)+164] for j,o in zip(picks,origins)]),device='cuda')
            times=[rows[int(j)]['times'][int(o):int(o)+164] for j,o in zip(picks,origins)]
            final=final_trajectory_loss(model,geom,longfields,mask,scales,times)
        return short+.1*cycle+c['lambda_final40']*final,short,cycle,final
    if combined_smoke:
        model.train();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        loss,short,cycle,final=loss_for(0);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        save(OUT/'combined_smoke.json',{'passed':True,'no_optimizer_updates':True,'arm':arm,'seed':seed,
            'total_loss':float(loss.detach()),'short_loss':float(short.detach()),'cycle_loss':float(cycle.detach()),
            'final40_loss':float(final.detach()),'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,
            'seconds_forward_backward':time.perf_counter()-begin,'finite_gradients':True})
        print('COMBINED_SMOKE_PASSED',float(loss.detach()),flush=True);return
    folder=OUT/f'{arm}_seed{seed}';folder.mkdir(exist_ok=False)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['AdamW_weight_decay'])
    save(folder/'configuration.json',{'arm':arm,'seed':seed,'initial_weight_sha256':FORMAL_SHA,
        'tasks_sha256':sha(OUT/f'tasks_seed{seed}.npz'),'lambda_final40':c['lambda_final40'] if arm=='joint_DMD40' else 0.,
        'lr':c['lr'],'training_cases':80,'input_channels':saved['in_channels']})
    best=(float('inf'),float('inf'),0);history=[];begin=time.perf_counter();clips=0;maxgrad=0.;last=0
    for step in range(UPDATES+1):
        if step:
            model.train();opt.zero_grad(set_to_none=True)
            loss,short,cycle,final=loss_for(step-1)
            assert torch.isfinite(loss), f'Nonfinite loss at{step}'
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip']))
            assert np.isfinite(norm),f'Nonfinite gradient at{step}'
            maxgrad=max(maxgrad,norm);clips+=int(norm>c['gradient_clip']);opt.step()
            if step%25==0:
                save(folder/'status.json',{'status':'training','step':step,'utc':now(),'elapsed_seconds':time.perf_counter()-begin,
                    'short_loss':float(short.detach()),'cycle_loss':float(cycle.detach()),'final40_loss':float(final.detach()),
                    'gradient_norm_before_clip':norm,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30})
        if step in CHECKS:
            ev=evaluate_dmd(model,val,c['channel_scales_train80']);q=ev['summary']['DMD9']['40s']
            valid=ev['summary']['direct']['8s']['n_pass']==8
            if step==0:assert abs(q['mean_R']-c['baseline_validation']['40s']['mean_R'])<2e-4
            key=(q['mean_R'],q['max_R'],step);save(folder/f'validation_{step}.json',ev)
            history.append({'step':step,'summary':ev['summary'],'eligible_short8s':valid})
            if valid and key<best:
                best=key
                torch.save({'model':model.state_dict(),'model_name':saved['model_name'],'in_channels':saved['in_channels'],
                    'out_channels':saved['out_channels'],'update':step,'seed':seed,'arm':arm,'config':protocol()['config'],
                    'optimizer':opt.state_dict(),'data_version':'corrected_benchmark_v1_train80'},folder/'selected.pt')
                save(folder/'selected_validation.json',ev)
                save(folder/'selection.json',{'step':step,'key':list(best),'sha256':sha(folder/'selected.pt')})
            save(folder/'history.json',history);last=step
            print('JOINT_PERIODIC_TRAIN',arm,seed,step,'R',q['mean_R'],'MAX',q['max_R'],'PASS40',q['n_pass'],'SHORT',valid,flush=True)
            initial=c['baseline_validation']['40s']['mean_R']
            if step==500 and best[0]>.98*initial and q['mean_R']>1.25*initial:break
    save(folder/'done.json',{'status':'complete','utc':now(),'last_update':last,'best':list(best),
        'max_gradient_norm':maxgrad,'gradient_clip_count':clips,'wall_seconds':time.perf_counter()-begin})

def orchestrate():
    configure();c=json.loads((OUT/'preregister.json').read_text())
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    with (OUT/'combined_smoke.log').open('x') as f:
        subprocess.run([sys.executable,__file__,'--worker','joint_DMD40','--seed',str(SEEDS[0]),'--combined-smoke'],
            env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    combined=json.loads((OUT/'combined_smoke.json').read_text());assert combined['passed']
    concurrency=max(1,min(4,int(21/(combined['peak_GPU_GiB']+.8))))
    queue=[(arm,seed) for seed in SEEDS for arm in c['arms']];active=[];completed=[]
    save(OUT/'execution.json',{'utc':now(),'concurrency':concurrency,'combined_smoke':combined,'controller_pid':os.getpid()})
    try:
        while queue or active:
            while queue and len(active)<concurrency:
                arm,seed=queue.pop(0);log=(OUT/f'{arm}_seed{seed}.log').open('x')
                p=subprocess.Popen([sys.executable,__file__,'--worker',arm,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
                active.append((arm,seed,p,log))
            for entry in active[:]:
                arm,seed,p,log=entry;code=p.poll()
                if code is not None:
                    completed.append({'arm':arm,'seed':seed,'exit_code':code});log.close();active.remove(entry)
                    if code!=0:raise RuntimeError(f'Worker failed: {arm} seed{seed} code{code}')
            save(OUT/'status.json',{'status':'training','utc':now(),'workers':[{'arm':a,'seed':s,'pid':p.pid} for a,s,p,l in active],
                'completed':completed,'queued':[{'arm':a,'seed':s} for a,s in queue],'concurrency':concurrency})
            if queue or active:time.sleep(15)
    except BaseException as e:
        for a,s,p,l in active:
            if p.poll() is None:p.terminate()
        save(OUT/'status.json',{'status':'failed','utc':now(),'error':repr(e),'completed':completed})
        raise
    candidates=[];pairs=[];baseline=c['baseline_validation']['40s']
    for seed in SEEDS:
        scores={}
        for arm in c['arms']:
            f=OUT/f'{arm}_seed{seed}';ev=json.loads((f/'selected_validation.json').read_text());sel=json.loads((f/'selection.json').read_text())
            scores[arm]=ev['summary']['DMD9']['40s']
            candidates.append({'arm':arm,'seed':seed,'path':str(f/'selected.pt'),'sha256':sha(f/'selected.pt'),
                'selection':sel,'summary':ev['summary']})
        pairs.append({'seed':seed,'joint_better_mean_than_control':scores['joint_DMD40']['mean_R']<scores['control']['mean_R'],'scores':scores})
    qualified=[r for r in candidates if r['arm']=='joint_DMD40' and r['summary']['DMD9']['40s']['mean_R']<=.9*baseline['mean_R'] and r['summary']['DMD9']['40s']['max_R']<=.9*baseline['max_R']]
    passed=all(r['joint_better_mean_than_control'] for r in pairs) and len(qualified)>0
    assert sha(FORMAL)==FORMAL_SHA
    save(OUT/'training_summary.json',{'utc':now(),'method_gate_passed':passed,'pairs':pairs,'candidates':candidates,
        'qualified_vs_initial':[{'seed':r['seed'],'arm':r['arm']} for r in qualified],'formal_weight_unchanged':True,
        'fresh_confirmation_used':False,'additional_compute_reported_separately':True})
    save(OUT/'status.json',{'status':'complete','utc':now(),'method_gate_passed':passed,'completed':completed,'formal_weight_unchanged':True})
    print('JOINT_PERIODIC_ROUND_COMPLETE','GATE',passed,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--orchestrate',action='store_true');parser.add_argument('--worker',choices=['control','joint_DMD40'])
    parser.add_argument('--seed',type=int);parser.add_argument('--combined-smoke',action='store_true');args=parser.parse_args()
    if args.smoke:smoke()
    elif args.worker:worker(args.worker,args.seed,args.combined_smoke)
    else:
        assert args.orchestrate
        orchestrate()
