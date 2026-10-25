#!/usr/bin/env python3
"""Apply one validation-frozen closure to already-seen development cases. No new choices here."""
from pathlib import Path
import sys,json
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from self_trajectory_closure_v1 import closure,build_priors,metrics,P,DATA
from geometry_attack_v2 import sha,save,attrs
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/self_trajectory_closure_v1/old32_seen'


def run():
    OUT.mkdir(exist_ok=False);(OUT/'predictions').mkdir()
    config=json.loads((ROOT/'outputs/self_trajectory_closure_v1/selected_val.json').read_text())['candidate']
    prior=build_priors();rows=[];base_rows=[];registry=[]
    with h5py.File(DATA,'r') as h:
        for cid in P['old_seen_test_ids']:
            g=h['cases'][cid];q=attrs(g);times=g['times'][...];mask=g['mask'][...]
            base=np.load(ROOT/'outputs/geometry_attack_evaluation_v1/v2_bank/old32_seen/predictions'/f"{config['base_model']}__{cid}.npz")['prediction']
            pred,diagnostic=closure(q,g['fields'][:4],base,times,mask,prior,config)
            file=OUT/'predictions'/f'{cid}.npz';np.savez_compressed(file,prediction=pred)
            registry.append({'id':cid,'prediction_sha256':sha(file),'8s_prefix_max_abs_delta':float(np.abs(pred[:32]-base[:32]).max())})
            future=g['fields'][4:164]
            rows.append({'id':cid,**q,'horizons':metrics(pred,future,mask,times),'closure':diagnostic})
            base_rows.append({'id':cid,**q,'horizons':metrics(base,future,mask,times)})
    groups={}
    for name,predicate in [('all32',lambda r:True),('fixed_D1_position16',lambda r:abs(r['D']-1)<1e-6),
                           ('thin_D0.8_8',lambda r:abs(r['D']-.8)<1e-6),('thick_D1.2_8',lambda r:abs(r['D']-1.2)<1e-6)]:
        groups[name]={method:{key:summarize([r['horizons'][key] for r in source if predicate(r)]) for key in ['2s','8s','20s','40s']}
                      for method,source in [('plain',base_rows),('closure',rows)]}
    save(OUT/'report.json',{'config':config,'configuration_sha256':sha(ROOT/'outputs/self_trajectory_closure_v1/selected_val.json'),
                           'role':'already-seen old32 development diagnostic; not fresh independent evidence','cases':rows,'base_cases':base_rows,'groups':groups,'prediction_manifest':registry})
    save(OUT/'summary.json',groups);print('CLOSURE_OLD32_COMPLETE',groups['all32']['closure']['40s'],flush=True)


if __name__=='__main__':run()
