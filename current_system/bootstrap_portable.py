from pathlib import Path
import os,sys,json,hashlib
P=Path(__file__).resolve().parent;device=sys.argv[1] if len(sys.argv)>1 else 'cuda';assert device in ['cpu','cuda']
template=P/'runtime_template';runtime=P/('runtime_'+device);root='/root/autodl-tmp/flowproxy';changes=[]
for p in template.rglob('*'):
 if not p.is_file():continue
 target=runtime/p.relative_to(template);target.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix in ['.py','.json','.yaml','.yml','.txt']:
  text=p.read_text(encoding='utf-8').replace(root,str(runtime).replace('\\','/'))
  if p.suffix=='.py':
   text=text.replace('.read_text()',".read_text(encoding='utf-8')")
   if p.name=='corrected_benchmark_train_v1.py':text=text.replace(',fcntl','')+"\ntry:\n import fcntl\nexcept ImportError:\n fcntl=None  # Linux-only historical training lock; not used by inference.\n"
  if device=='cpu' and p.suffix=='.py':
   text=text.replace('.cuda()',".to('cpu')").replace("'cuda'","'cpu'").replace('"cuda"','"cpu"').replace('torch.cuda.synchronize()',"(torch.cuda.synchronize() if False else None)")
  target.write_text(text,encoding='utf-8')
 else:target.write_bytes(p.read_bytes())
 changes.append(dict(path=str(target.relative_to(P)),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
cfg=json.loads((P/'portable_system.json').read_text(encoding='utf-8'));(runtime/'system.json').write_text(json.dumps(cfg),encoding='utf-8')
(P/('derived_runtime_'+device+'.json')).write_text(json.dumps(dict(device=device,source_selected_system_SHA=cfg['source_selected_system_sha256'],files=changes),indent=2),encoding='utf-8');print(runtime)
