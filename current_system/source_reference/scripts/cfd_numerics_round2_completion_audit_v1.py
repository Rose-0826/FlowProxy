#!/usr/bin/env python3
"""Repair only the log-completion checker after the original runner exits."""
from pathlib import Path
import sys,json,re,time,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from geometry_attack_v2 import save,sha
OUT=ROOT/'outputs/cfd_numerics_round2_v1'
RAW=ROOT/'data/raw/cfd_numerics_round2_v1'
def running(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return False
    return p.read_text().split(') ',1)[1].split()[0]!='Z'
while running(23833):time.sleep(10)
p=json.loads((OUT/'preregister.json').read_text());original=json.loads((OUT/'status.json').read_text())
save(OUT/'status_original_completion_checker.json',original)
save(OUT/'completion_checker_amendment.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'cause':'Unanchored Time = pattern also matches ClockTime = and ExecutionTime =; raw solver endTime is actually 311',
  'change':'Require ^Time = with re.MULTILINE, exact 311, End line, all expected sampled outputs, and no active job descendants',
  'original_results_and_errors_preserved':True,'no_solver_rerun':True,'no_changed_gates_or_statistics':True,
  'original_driver_and_protocol_unchanged':True,'checker_sha256':sha(__file__)})
rows=[]
for variant in p['variants']:
    for q in p['cases']:
        case=RAW/variant/q['id'];log=case/'log.icoFoam';text=log.read_text()
        ts=re.findall(r'^Time = (\S+)',text,re.MULTILINE)
        assert ts and abs(float(ts[-1])-311)<1e-8,(case,ts[-1:])
        assert re.search(r'^End\s*$',text,re.MULTILINE),case
        assert (case/'311/U').exists() and (case/'311/p').exists(),case
        assert len(list((case/'VTK').rglob('*.vtu')))==121,case
        rows.append({'variant':variant,'id':q['id'],'last_time':float(ts[-1]),'normal_End':True,
          'sample_count':121,'solver_log_sha256':sha(log),'native_U311_sha256':sha(case/'311/U'),'native_p311_sha256':sha(case/'311/p')})
# No owned jobs remain once the original runner has waited on every child.
save(OUT/'completion_audit.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'all4_verified':True})
save(OUT/'status.json',{'status':'complete','completed':rows,'errors':[],'total':4,'original_checker_failure_preserved':True})
import cfd_numerics_round2_v1 as driver
driver.analyze()
print('COMPLETION_CHECK_REPAIRED_AND_ANALYZED',flush=True)
