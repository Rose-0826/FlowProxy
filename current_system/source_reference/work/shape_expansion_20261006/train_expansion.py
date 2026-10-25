from expansion_core import *
import argparse,subprocess,traceback,statistics,tarfile,shutil

def training():
 cfg=json.loads((OUT/'run_config.json').read_text());recs=old.metadata();data0=old.data_for(recs);data=[data0[z['id']] for z in recs]
 # Original TRAIN100 templates contain only TRAIN truth, not DEV72 labels.
 templates=[]
 for z in recs:
  with np.load(old.template_path(z)) as a:templates.append({**z,**{k:a[k] for k in a.files}})
 recs=list(templates)
 for q in cfg['design']['train']:
  info=json.loads((CACHE/(q['id']+'.json')).read_text());assert info['CFD_stable'] and info['cache_sha256']==old.p.digest(CACHE/(q['id']+'.npz'))
  with np.load(CACHE/(q['id']+'.npz')) as a:raw={k:a[k] for k in ['fields','geom','mask','times']}
  frequency=info['St']/q['D'];B=old.p.basis(raw['times']-raw['times'][3],frequency)
  truthC=old.b.previous.w.constrain(np.linalg.lstsq(B,raw['fields'].reshape(165,-1),rcond=1e-10)[0].reshape(9,3,64,64),raw['mask'])
  rec=dict(id=q['id'],q={k:q[k] for k in ['shape','Re','cx','cy','D']},fold=0,group_key=q['id'],truth_frequency=frequency,truthC=truthC,mask=raw['mask'],fields=raw['fields'][:4],geom=raw['geom'],times=raw['times'])
  recs.append(rec);data.append(dict(id=q['id'],q=rec['q'],**raw));templates.append(rec)
 assert len(recs)==124 and len(data)==124
 return recs,data,templates

def add_priors(bundle,recs):
 for shape in shapes.SPECS:
  values=[z['truth_frequency']*z['q']['D'] for z in recs if z['q']['shape']==shape];assert len(values)==4
  st=float(np.mean(values));bundle['priors'][shape]=lambda x,st=st:np.full(len(x),st)
 return bundle

def banks_and_development(cp,folder,recs,data,templates):
 base,saved=old.model_from(cp,old.protocol()['config'],seed=1601);model=old.ProjectedState(old.ReflectionOperator(base),True);model.eval()
 # Recompute the original-shape libraries for this network, then score old72.
 old_summary,old_banks=old.build_evaluate(model,templates[:100],data[:100],folder/'old72')
 bundle=dict(model=model,libs={},priors=old.p.geometry_priors());add_priors(bundle,recs);updated=[]
 for rec,raw in zip(templates[100:],data[100:]):
  row=dict(id=rec['id'],q=rec['q'],fields=raw['fields'][:4],geom=raw['geom'],mask=raw['mask'],times=raw['times'])
  own,C,pa,di=bootstrap(bundle,row);updated.append({**rec,'C':C.astype(np.float32),'params':pa})
 new_banks={}
 for shape in shapes.SPECS:
  part=[z for z in updated if z['q']['shape']==shape];lib=old.b.build(part,str(folder/'new_banks'/shape));bundle['libs'][shape]=lib
  path=folder/'new_banks'/shape/(shape+'_bank.npz');new_banks[shape]=dict(path=str(path),sha256=old.p.digest(path))
 # Leave one case out of the new residual bank, while FNO and frequency priors
 # see all TRAIN124. This diagnoses the added module; it is NOT full-model CV.
 cases=[]
 qs=json.loads((OUT/'run_config.json').read_text())['design']['train']
 for index,q in enumerate(qs):
  shape=q['shape'];full=bundle['libs'][shape];part=[z for z in updated if z['q']['shape']==shape and z['id']!=q['id']];assert len(part)==3
  bundle['libs'][shape]=old.b.build(part,str(folder/'module_leaveoneout_banks'/q['id']))
  row=startup(q);values=packets(q,row);pred,di=forecast(bundle,row,values)
  old.p.save(folder/'module_leaveoneout_registry'/(q['id']+'.json'),dict(prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),prediction_before_fullfield_score=True,excluded_residual_source=q['id'],diagnostic=di))
  with np.load(CACHE/(q['id']+'.npz')) as a:truth=a['fields'][4:164]
  cases.append(dict(id=q['id'],q=q,group=q['family'],horizons=old.p.metrics(pred,truth,row['mask'],row['times'])));bundle['libs'][shape]=full
 new_summary={g:{h:old.p.summarize([z['horizons'][h] for z in cases if g=='all' or z['group']==g]) for h in ['8s','40s']} for g in ['all','ellipse','rectangle','irregular']}
 report=dict(checkpoint=str(cp),checkpoint_sha256=old.p.digest(cp),old72=old_summary,new_train_module_cv=new_summary,new_train_cases=cases,old_banks=old_banks,new_banks=new_banks,new_priors={s:float(np.mean([z['truth_frequency']*z['q']['D'] for z in recs if z['q']['shape']==s])) for s in shapes.SPECS},role='old72 SEEN development; new24 TRAIN module leave-one-out ONLY; not independent model validation')
 old.p.save(folder/'comparison.json',report);return report

def train_arm(name):
 setup();torch.manual_seed(1601);cfg=json.loads((OUT/'run_config.json').read_text());arm=next(z for z in cfg['training_arms'] if z['name']==name);recs,data,templates=training();periods=[dict(N=int(np.floor(4/z['truth_frequency'])),alpha=float(4/z['truth_frequency']-np.floor(4/z['truth_frequency']))) for z in recs]
 assert min(p['N'] for p in periods)>=4
 base,saved=old.model_from(cfg['parent']['checkpoint'],old.protocol()['config'],seed=1601);model=old.ProjectedState(old.ReflectionOperator(base),True);scale=torch.as_tensor(cfg['channel_scales'],device='cuda');tasks=np.load(OUT/'tasks.npz');folder=OUT/name;folder.mkdir(exist_ok=True)
 if (folder/'done.json').exists():return
 # Exact optimizer/task checkpoint supports explicit continuation after failure.
 opt=torch.optim.AdamW(base.parameters(),lr=arm['lr'],weight_decay=1e-4);history=[];first=1
 state=folder/'resume.pt'
 if state.exists():
  saved_state=torch.load(state,map_location='cuda',weights_only=False);base.load_state_dict(saved_state['model']);opt.load_state_dict(saved_state['optimizer']);first=saved_state['step']+1;history=saved_state['history']
  for completed_step in [500,1000]:
   if completed_step>=first:continue
   cp=folder/f'checkpoint_{completed_step}.pt'
   if any(Path(h['checkpoint'])==cp for h in history):continue
   assert cp.exists(), 'Keep saved optimizer state; recover missing checkpoint explicitly'
   report=banks_and_development(cp,folder/f'development_{completed_step}',recs,data,templates);history.append(report);old.p.save(folder/'history.json',history)
 for step in range(first,1001):
  picks=tasks['picks'][step-1];s=int(tasks['starts'][step-1]);g=torch.as_tensor(np.stack([data[int(j)]['geom'] for j in picks]),device='cuda');f=torch.as_tensor(np.stack([data[int(j)]['fields'][s:s+12] for j in picks]),device='cuda');m=torch.as_tensor(np.stack([data[int(j)]['mask'] for j in picks]),device='cuda')
  model.train();opt.zero_grad(set_to_none=True);loss=.5*old.train_loss(model,g,f,m,scale,1)+.5*old.train_loss(model,g,f,m,scale,8)+.1*old.cycle(model,g,f,m,scale,periods,picks)[0];assert torch.isfinite(loss);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(base.parameters(),1.));assert np.isfinite(norm);opt.step()
  if step%25==0:
   old.p.save(folder/'status.json',dict(stage='training',step=step,loss=float(loss.detach()),gradient_norm=norm,utc=old.p.now()))
   tmp=folder/'resume.tmp.pt';torch.save(dict(model=base.state_dict(),optimizer=opt.state_dict(),step=step,history=history),tmp);tmp.replace(state)
  if step in [500,1000]:
   cp=folder/f'checkpoint_{step}.pt';torch.save(dict(model=base.state_dict(),model_name=saved['model_name'],in_channels=saved['in_channels'],out_channels=saved['out_channels'],config=old.protocol()['config'],seed=1601,additional_update=step,arm=arm,training_case_count=124,training_source_sha256=cfg['training_manifest_sha256'],initial_checkpoint_sha256=cfg['parent']['checkpoint_sha256'],requires_research_operator_wrapper='ProjectedState(ReflectionOperator(base),True)'),cp)
   old.p.save(folder/'status.json',dict(stage='development',step=step,utc=old.p.now()));report=banks_and_development(cp,folder/f'development_{step}',recs,data,templates);history.append(report);old.p.save(folder/'history.json',history)
   # Building development models must not modify the live training parameters.
   saved_live=base.state_dict();torch.save(dict(model=saved_live,optimizer=opt.state_dict(),step=step,history=history),state)
 old.p.save(folder/'done.json',dict(utc=old.p.now(),step=1000));old.p.save(folder/'status.json',dict(stage='complete',step=1000,utc=old.p.now()))

def load_candidate(info):
 bundle=old.load_bundle(info['checkpoint'],info['old_banks']);basepriors=bundle['priors']
 for shape,b in info['new_banks'].items():
  path=Path(b['path']);assert old.p.digest(path)==b['sha256']
  with np.load(path) as z:lib={k:z[k] for k in z.files}
  lib.update(shape=shape,source=json.loads(path.with_suffix('.json').read_text())['source']);lib['gram']=lib['A']@lib['A'].T;bundle['libs'][shape]=lib
  st=info['new_priors'][shape];basepriors[shape]=lambda x,st=st:np.full(len(x),st)
 return bundle

def adapt():
 cfg=json.loads((OUT/'run_config.json').read_text());dataset=json.loads((OUT/'training_dataset.json').read_text());assert len(dataset['results'])==24
 cfg['training_manifest_sha256']=old.p.digest(OUT/'training_dataset.json');old.p.save(OUT/'run_config.json',cfg)
 tasks=OUT/'tasks.npz'
 if not tasks.exists():
  rng=np.random.default_rng(1601);picks=np.concatenate([rng.integers(0,100,(1000,2)),rng.integers(100,124,(1000,2))],axis=1);np.savez(tasks,picks=picks,starts=rng.integers(0,150,1000))
 workers=[]
 for arm in cfg['training_arms']:
  if (OUT/arm['name']/'done.json').exists():continue
  log=(OUT/(arm['name']+'.log')).open('a');p=subprocess.Popen([sys.executable,__file__,'--arm',arm['name']],cwd=Path(__file__).parent,env=os.environ,stdout=log,stderr=subprocess.STDOUT);log.close();workers.append((arm['name'],p))
 old.p.save(OUT/'training_workers.json',dict(workers=[dict(name=n,pid=p.pid) for n,p in workers],utc=old.p.now()))
 for name,p in workers:
  code=p.wait();assert code==0,(name,code)
 candidates=[]
 for arm in cfg['training_arms']:
  for report in json.loads((OUT/arm['name']/'history.json').read_text()):candidates.append(dict(name=arm['name']+'_step'+Path(report['checkpoint']).stem.split('_')[-1],**report))
 assert len(candidates)==4;old.p.save(OUT/'locked_candidates.json',dict(candidates=candidates,utc=old.p.now(),held_evaluation_future_fields_unread=True,selection_role='four candidate comparison; no new percentage gates'))
 old.p.save(OUT/'overall_status.json',dict(stage='candidates_locked',utc=old.p.now()))

def evaluate_held():
 setup();cfg=json.loads((OUT/'run_config.json').read_text());qs=cfg['design']['evaluation'];locked=json.loads((OUT/'locked_candidates.json').read_text());assert len(qs)==12;reports={}
 baseline=transferred_bundle();reports['parent_transfer']=score_queries(baseline,qs,OUT/'held/parent_transfer',library=False,wide=True);del baseline;torch.cuda.empty_cache()
 for c in locked['candidates']:
  assert old.p.digest(c['checkpoint'])==c['checkpoint_sha256'];bundle=load_candidate(c);reports[c['name']]=score_queries(bundle,qs,OUT/'held'/c['name']);del bundle;torch.cuda.empty_cache()
 old.p.save(OUT/'held_comparison.json',dict(reports=reports,locked_candidates_sha256=old.p.digest(OUT/'locked_candidates.json'),role='first fixed-candidate evaluation on 12 previously unscored parameter cases; comparison used to select winner; profiles are trained, not unseen-family proof'))
 names=[c['name'] for c in locked['candidates']];winner=min(names,key=lambda name:(-reports[name]['summary']['all']['40s']['n_pass'],reports[name]['summary']['all']['40s']['mean_R']));selected=next(c for c in locked['candidates'] if c['name']==winner);noise={}
 for name in ['parent_transfer',winner]:
  b=transferred_bundle() if name=='parent_transfer' else load_candidate(selected);noise[name]={}
  for label,startup_noisy in [('future1pct',False),('startup_and_future1pct',True)]:noise[name][label]=score_queries(b,qs,OUT/'noise'/name/label,library=name!='parent_transfer',wide=name=='parent_transfer',noise=.01,startup_noise=startup_noisy)
  del b;torch.cuda.empty_cache()
 old.p.save(OUT/'noise_report.json',noise)
 parent=json.loads((ROOT/'outputs/practical_targeted_20261006/selected_candidate.json').read_text());combined=dict(original_shapes=parent,new_shapes=selected,routing='known geometry family only; never case ID. Original cylinder/square retain original parent network and 24-dimensional libraries exactly; three new shape families use selected shared adapted network with rank capped by TRAIN library size.')
 old.p.save(OUT/'selected_system.json',combined)
 dominated=[]
 for c in locked['candidates']:
  if c['name']==winner:continue
  a=reports[c['name']]['summary']['all']['40s'];aold=c['old72']['all72']['40s']
  for other in locked['candidates']:
   if other['name']==c['name']:continue
   b=reports[other['name']]['summary']['all']['40s'];bold=other['old72']['all72']['40s']
   if b['n_pass']>=a['n_pass'] and b['mean_R']<=a['mean_R'] and b['max_R']<=a['max_R'] and bold['n_pass']>=aold['n_pass'] and bold['mean_R']<=aold['mean_R'] and bold['max_R']<=aold['max_R'] and (b!=a or bold!=aold):dominated.append(c);break
 cleanup=[]
 if dominated:
  archive=OUT/'dominated_branches.tar.gz';paths=[]
  for c in dominated:
   paths.append(Path(c['checkpoint']));paths.extend(Path(b['path']) for b in {**c['old_banks'],**c['new_banks']}.values())
  paths=list(dict.fromkeys(paths));assert all(p.resolve().is_relative_to(OUT.resolve()) and p.exists() for p in paths)
  hashes={str(p.relative_to(OUT)):old.p.digest(p) for p in paths}
  with tarfile.open(archive,'w:gz') as tar:
   for p in paths:tar.add(p,arcname=str(p.relative_to(OUT)))
  with tarfile.open(archive,'r:gz') as tar:
   for name,sha in hashes.items():assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==sha
  for p in paths:cleanup.append(dict(path=str(p),sha256=hashes[str(p.relative_to(OUT))],bytes=p.stat().st_size));p.unlink()
  old.p.save(OUT/'cleanup_receipt.json',dict(archive=str(archive),sha256=old.p.digest(archive),removed=cleanup,dominated=[c['name'] for c in dominated],real_CFD_preserved=True))
 result=dict(status='COMPLETE',utc=old.p.now(),selected_newshape_candidate=winner,held_summary={n:z['summary'] for n,z in reports.items()},old72_adapted_network=selected['old72'],original72_with_family_routing=parent['summary'],noise_summary={n:{mode:z['summary'] for mode,z in v.items()} for n,v in noise.items()},training_cases=124,new_training_cases=24,new_held_parameter_cases=12,geometry_profiles=shapes.SPECS,old_known_shape_branch_unchanged=True,new_family_unseen_shape_claim=False,dominated=[c['name'] for c in dominated],models_downloaded=False,disk_free_bytes=shutil.disk_usage(ROOT).free)
 old.p.save(OUT/'final_result.json',result);old.p.save(OUT/'overall_status.json',dict(stage='complete',utc=old.p.now(),selected=winner));print('EXPANSION_FINAL',winner,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--evaluate-held',action='store_true');args=ap.parse_args()
 try:
  if args.arm:train_arm(args.arm)
  elif args.evaluate_held:evaluate_held()
  else:adapt()
 except Exception as e:old.p.save(OUT/((args.arm or 'adaptation')+'_failure.json'),dict(utc=old.p.now(),error=repr(e),traceback=traceback.format_exc()));raise
