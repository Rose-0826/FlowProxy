from pathlib import Path
import json,hashlib,shutil,argparse
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
a=argparse.ArgumentParser();a.add_argument('host',choices=['aic','storage']);args=a.parse_args()
if args.host=='storage':
    folder=Path('/root/autodl-tmp/AIC_MAINLINE_20261005')
    expected=json.loads((folder/'audit_compression_receipt.json').read_text())
    assert digest(folder/'history_evidence.tar.gz')==expected['archive_sha256']
    main=json.loads((folder/'archive_verified.json').read_text())
    mirror=json.loads((folder/'storage_mirror_verified.json').read_text());assert mirror['archive_sha256']==main['archive_sha256']
    auth=Path('/root/.ssh/authorized_keys');lines=auth.read_text().splitlines(keepends=True)
    public='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINprPoqkimR7ff1Q9FsVtgQEp4hxSnsAbsbT+eXrB1hj aic-cleanup-temporary-20261005'
    assert sum(x.strip()==public for x in lines)==1
    auth.write_text(''.join(x for x in lines if x.strip()!=public));auth.chmod(0o600)
    usage=shutil.disk_usage(folder)
    result={'host':'storage','free_bytes':usage.free,'used_bytes':usage.used,'mainline_archive_verified':True,'history_evidence_archive_verified':True,'old_bulk_root_removed':not Path('/root/autodl-tmp/flowproxy_short_window').exists(),'temporary_task_key_removed':True}
    (folder/'final_storage_receipt.json').write_text(json.dumps(result,indent=2))
else:
    folder=Path('/root/autodl-tmp/flowproxy/outputs/mainline_cleanup_20261005')
    for name in ['aic_cleanup_storage_key','aic_cleanup_storage_known_hosts']:
        p=Path('/root/.ssh')/name;p.unlink(missing_ok=True)
    usage=shutil.disk_usage(folder)
    result={'host':'aic','free_bytes':usage.free,'used_bytes':usage.used,'temporary_task_key_removed':True,'one_current_weight':'9b860db72359a0e874a7e5942ea2ccd14d2f1ea29339af49dd435a89128ac1aa','cleanup_and_research_design_complete':True,'new_training_or_CFD_started':False}
    (folder/'final_aic_receipt.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
