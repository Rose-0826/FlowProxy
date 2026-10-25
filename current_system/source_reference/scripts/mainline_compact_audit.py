from pathlib import Path
import tarfile,hashlib,shutil,json
R=Path('/root/autodl-tmp/flowproxy');A=R/'outputs/mainline_cleanup_20261005'
dirs=[A/'evidence',A/'historical_reports'];archive=A/'history_evidence.tar.gz'
def sha_file(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
before=shutil.disk_usage(R);records={}
for d in dirs:
    assert d.resolve()==d and R in d.parents
    for p in d.rglob('*'):
        if p.is_file() and not p.is_symlink():records[str(p.relative_to(A))]=sha_file(p)
with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
    for d in dirs:tf.add(d,arcname=d.name,recursive=True)
with tarfile.open(archive,'r:gz') as tf:
    seen=set()
    for member in tf:
        if member.isfile():
            f=tf.extractfile(member);assert f is not None
            assert hashlib.file_digest(f,'sha256').hexdigest()==records[member.name]
            seen.add(member.name)
    assert seen==set(records)
for d in dirs:shutil.rmtree(d)
index=json.loads((A/'historical_path_index.json').read_text())
for q in index:
    q['archive_member_prefix']=str(Path(q['archive']).relative_to(A))
    q['compressed_archive']=str(archive)
(A/'historical_path_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
after=shutil.disk_usage(R)
(A/'audit_compression_receipt.json').write_text(json.dumps({'files_preserved':len(records),'all_decompressed_hashes_verified':True,'archive_bytes':archive.stat().st_size,'archive_sha256':sha_file(archive),'net_freed_bytes':after.free-before.free,'free_after':after.free,'used_after':after.used},indent=2))
print((A/'audit_compression_receipt.json').read_text(),flush=True)
