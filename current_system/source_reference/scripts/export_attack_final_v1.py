#!/usr/bin/env python3
"""Preserve original export, add completed checks and render revisions, hash every delivered file."""
from pathlib import Path
import sys,json,datetime,shutil,subprocess,os,tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
OUT=ROOT/'outputs/aic_attack_2026-10-02'

def run():
    physics=ROOT/'outputs/cfd_numerics_restart_v1'
    assert json.loads((physics/'status.json').read_text())['status']=='complete'
    assert (physics/'sensitivity_report.json').exists()
    assert json.loads((ROOT/'outputs/hybrid_start_phase_v1/status.json').read_text())['status']=='complete'
    original=OUT/'export_manifest.json';initial=OUT/'export_manifest_initial.json'
    assert not initial.exists();shutil.copy2(original,initial)
    for folder in ['cfd_numerics_restart_v1','hybrid_start_phase_v1']:
        source=ROOT/'outputs'/folder;destination=OUT/'snapshots'/folder;destination.mkdir(exist_ok=False)
        for p in source.rglob('*'):
            if p.is_file() and 'predictions' not in p.parts and (p.suffix in ['.json','.csv','.log'] or p.name.endswith('_field_statistics.npz')):
                target=destination/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    log_records=[]
    for variant in ['coarse_dt002','coarse_dt001','fine_dt001']:
        for case in (ROOT/'data/raw/cfd_numerics_restart_v1'/variant).iterdir():
            target=OUT/'snapshots/cfd_numerics_restart_v1/log_tails'/variant/case.name;target.mkdir(parents=True,exist_ok=True)
            for p in case.glob('log.*'):
                if not p.is_file():continue
                text=p.read_text(errors='replace');(target/(p.name+'.tail.txt')).write_text(text[-6000:])
                log_records.append({'full_server_file':str(p),'bytes':p.stat().st_size,'sha256':sha(p),
                  'tail_file':str((target/(p.name+'.tail.txt')).relative_to(OUT))})
            for name in ['Allrun_numerics','Allrun_numerics_resume']:
                p=case/name
                if p.exists():shutil.copy2(p,target/name)
    save(OUT/'snapshots/cfd_numerics_restart_v1/full_solver_log_manifest.json',log_records)
    scripts=['cfd_numerics_restart_v1.py','cfd_numerics_resume_v1.py','cfd_numerics_analyze_v1.py',
      'hybrid_start_phase_v1.py','hybrid_field_figure_v1.py','build_attack_delivery_v1.py','export_attack_final_v1.py']
    for name in scripts:shutil.copy2(ROOT/'scripts'/name,OUT/'code'/name)
    nominee=json.loads((ROOT/'outputs/geometry_attack_confirmation_d1_v1/nomination.json').read_text())
    weight_checks={r['name']:sha(r['path'])==r['parameter_file_sha256'] for r in nominee['bank'] if r['name'] in ['F0','K8']}
    assert all(weight_checks.values())
    assert sha(ROOT/'outputs/self_trajectory_closure_v1/selected_val.json')==nominee['primary']['closure_configuration_sha256']
    patterns=['geometry_attack_v1.py','geometry_attack_v2.py','uniform128_attack_v1.py','hybrid_start_phase_v1.py',
      'cfd_numerics_restart_v1.py','cfd_numerics_resume_v1.py','cfd_numerics_analyze_v1.py','hybrid_field_figure_v1.py']
    processes=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
    active=[]
    for line in processes.splitlines()[1:]:
        parts=line.strip().split(None,2)
        if len(parts)==3 and int(parts[0])!=os.getpid() and any(s in parts[2] for s in patterns):active.append(line.strip())
    assert not active,active
    save(OUT/'final_resource_and_freeze_audit.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'formal_checkpoint_hashes_unchanged':weight_checks,'primary_configuration_sha256_unchanged':True,
      'all18_training_complete':True,'all16_primary_CFD_normal_ends':True,'all6_numerics_CFD_normal_ends':True,
      'phase_stress_complete':True,'matching_active_owned_jobs':active,
      'reference_grid_independence_not_established':True,'full_native_data_and_prediction_arrays_retained_on_server':True})
    files=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*'))
      if p.is_file() and p!=original]
    save(original,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':'final-with-numerics-and-phase-stress',
      'initial_export_preserved':'export_manifest_initial.json','files':files,'manifest_self_excluded':True})
    archive=Path('/tmp/aic_attack_final.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
    print('FINAL_EXPORT',len(files),archive.stat().st_size,sha(archive),flush=True)

if __name__=='__main__':run()
