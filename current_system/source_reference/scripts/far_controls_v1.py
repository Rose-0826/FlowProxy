#!/usr/bin/env python3
"""Apply the already-fixed post-hoc controls to the frozen far-position extension."""
from pathlib import Path
import sys,json,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from hybrid_controls_v1 import controls
from nearest_cycle_template_v1 import predict
from template_interpolation_v1 import prepare_library,forecasts
from self_trajectory_closure_v1 import build_priors,P,metrics
from geometry_attack_v2 import NativeSampler,save,sha
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/far_controls_v1'


def run():
    OUT.mkdir(exist_ok=False);folder='geometry_attack_confirmation_far_v1';root=ROOT/'outputs'/folder
    save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_new_parameter_search':True,
        'fixed_methods_before_far_scores':'same cycle replay,DMD,and template-interpolation controls as previous12',
        'posthoc_controls':True,'no_primary_replacement':True,'script_sha256':sha(__file__)})
    config=json.loads((ROOT/'outputs/self_trajectory_closure_v1/selected_val.json').read_text())['candidate']
    priors=build_priors();library=prepare_library();report={}
    for row in json.loads((root/'preregister.json').read_text())['cases']:
        q={k:row[k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(P['config'],**q);mask=geo['mask']
        items=frames(ROOT/'data/raw'/folder/row['id']);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],mask)
        support=np.stack([sampler.sample(path) for _,path in items[:4]]);base=np.load(root/'evaluation/predictions'/f"K8__{row['id']}.npz")['prediction']
        predictions=controls(q,support,base,times,mask,priors,config)
        predictions.update({'aligned_'+name:value for name,value in forecasts(q,support,times,mask,library,priors).items()})
        # Canonical phase changes the nearest donor's parameterization, not its reachable cycle.
        predictions.update(predict(q,support,times,mask,library,priors))
        truth=np.stack([sampler.sample(path) for _,path in items[4:164]])
        for name,(pred,diagnostic) in predictions.items():report.setdefault(name,[]).append({'id':row['id'],'horizons':metrics(pred,truth,mask,times),'diagnostic':diagnostic})
    summary={name:{key:summarize([r['horizons'][key] for r in rows]) for key in ['2s','8s','20s','40s']} for name,rows in report.items()}
    save(OUT/'report.json',{'cases':report,'summary':summary});save(OUT/'summary.json',summary);save(OUT/'status.json',{'status':'complete'})
    print('FAR_CONTROLS', {k:(round(v['40s']['mean_R'],4),v['40s']['n_pass']) for k,v in summary.items()},flush=True)


if __name__=='__main__':run()
