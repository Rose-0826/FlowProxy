from pathlib import Path
import tarfile,json,hashlib,shutil,time
BASE=Path('/root/autodl-tmp')
R=BASE/'flowproxy';A=R/'outputs/mainline_cleanup_20261005'
LEGACY=BASE/'AIC'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
before=shutil.disk_usage(BASE)
assert LEGACY.resolve()==LEGACY and LEGACY.parent==BASE
files=[p for p in LEGACY.rglob('*') if p.is_file() and not p.is_symlink()]
extra=[p for p in BASE.iterdir() if p.is_file() and p.name in ['FlowProxy_结果与过程说明.docx','flowproxy_framework.zip','二维车形流场预测_讨论回复(1).docx','续训新标签与旧效果_讨论整理(1).docx','三角形续训.md','warmstart_execution_checks.template.json','warmstart_round1_config.template.json','warmstart_round1_setup.py','FlowProxy_第二轮实验与参赛方案_2026-09-30.md']]
records=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'status':'LEGACY_SOURCE_OR_REPORT_ONLY' if p.suffix not in ['.pt','.h5','.npz'] else 'UNRELATED_OLD_SMOKE_EXPERIMENT_REMOVED'} for p in files+extra]
(A/'ancillary_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
bundle=A/'legacy_sources_and_reports.tar.gz'
with tarfile.open(bundle,'w:gz') as tf:
    for p in files+extra:
        if p.suffix not in ['.pt','.h5','.npz'] and '__pycache__' not in p.parts:tf.add(p,arcname=str(p.relative_to(BASE)),recursive=False)
with tarfile.open(bundle,'r:gz') as tf:
    members=tf.getmembers();assert len(members)>0
    for m in members:
        assert not m.name.startswith('/') and '..' not in Path(m.name).parts
        if m.isfile():
            p=BASE/m.name;f=tf.extractfile(m);assert f is not None
            assert hashlib.file_digest(f,'sha256').hexdigest()==sha(p)
shutil.rmtree(LEGACY)
for p in extra:
    assert p.parent==BASE and not p.is_symlink();p.unlink()
after=shutil.disk_usage(BASE)
(A/'ancillary_receipt.json').write_text(json.dumps({'freed_bytes':after.free-before.free,'files_removed':len(records),'preserved_archive':str(bundle),'archive_sha256':sha(bundle),'archive_checked':True,'utc':time.time()},indent=2))
print((A/'ancillary_receipt.json').read_text())
