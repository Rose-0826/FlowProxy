#!/usr/bin/env python3
"""Same validation-frozen continuation config, switch only neural base from K8 to F0; post-hoc."""
from pathlib import Path
import sys,json,datetime
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from self_trajectory_closure_v1 import closure,build_priors,metrics,P,DATA
from geometry_attack_v2 import save,sha,attrs,NativeSampler
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/fno_training_closure_ablation_v1'


def run():
    OUT.mkdir(exist_ok=False);configuration=json.loads((ROOT/'outputs/self_trajectory_closure_v1/selected_val.json').read_text())['candidate'];priors=build_priors()
    save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'posthoc':True,'single_change':'F0 neural trajectory instead of K8',
        'all_continuation_choices_unchanged':True,'no_primary_replacement':True,'script_sha256':sha(__file__)})
    results={}
    for role,folder,ids in [('val','geometry_attack_evaluation_v1/v1_bank/val',P['val_ids']),('old32_seen','geometry_attack_evaluation_v1/v2_bank/old32_seen',P['old_seen_test_ids']),
                           ('fresh_diameter8','geometry_attack_confirmation_v1',None),('fresh_fixed_D1_4','geometry_attack_confirmation_d1_v1',None),
                           ('fresh_far_D1_4','geometry_attack_confirmation_far_v1',None)]:
        isnew=ids is None;directory=ROOT/'outputs'/folder
        if isnew:
            rs=json.loads((directory/'preregister.json').read_text())['cases'];lookup={r['id']:r for r in rs};ids=list(lookup)
        rows=[]
        with h5py.File(DATA,'r') as h:
            for cid in ids:
                if isnew:
                    q={k:lookup[cid][k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(P['config'],**q);mask=geo['mask']
                    items=frames(ROOT/'data/raw'/folder/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],mask)
                    initial=np.stack([sampler.sample(path) for _,path in items[:4]])
                else:
                    g=h['cases'][cid];q=attrs(g);times=g['times'][...];mask=g['mask'][...];initial=g['fields'][:4]
                pred_dir=directory/'evaluation/predictions' if isnew else directory/'predictions'
                base=np.load(pred_dir/f'F0__{cid}.npz')['prediction'];pred,diagnostic=closure(q,initial,base,times,mask,priors,configuration)
                truth=np.stack([sampler.sample(path) for _,path in items[4:164]]) if isnew else h['cases'][cid]['fields'][4:164]
                rows.append({'id':cid,'horizons':metrics(pred,truth,mask,times),'diagnostic':diagnostic})
        summary={k:summarize([r['horizons'][k] for r in rows]) for k in ['2s','8s','20s','40s']}
        results[role]={'cases':rows,'summary':summary};print('F0_CLOSURE_ABLATION',role,round(summary['40s']['mean_R'],4),summary['40s']['n_pass'],flush=True)
    save(OUT/'report.json',results);save(OUT/'summary.json',{k:v['summary'] for k,v in results.items()});save(OUT/'status.json',{'status':'complete'})


if __name__=='__main__':run()
