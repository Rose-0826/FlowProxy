#!/usr/bin/env python3
"""Encoding-consistent feedback, known velocity BCs, and support-aware centering."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,baseline,scales,score,summaries,protected,now
from corrected_benchmark_train_v1 import model_from,protocol,predict
from corrected_benchmark_controls_v1 import dmd_periodic
from cycle_landmarks_train_v1 import FORMAL
from research_v2_reflection_train import ReflectionOperator
from research_v2_operator_probe import rollout
from geometry_attack_v2 import save,sha

METHODS=['solid_feedback','velocity_BC_feedback','reflection_velocity_BC','outside_position_center_reflection','outside_position_center_only']

class ProjectedState(torch.nn.Module):
    def __init__(self,base,boundaries):super().__init__();self.base=base;self.boundaries=boundaries
    def forward(self,x):
        y=self.base(x)[:,:3]*x[:,1:2]
        if self.boundaries:
            y=y.clone();y[:,0,:,0]=1.;y[:,1,:,0]=0.;y[:,1,0,:]=0.;y[:,1,-1,:]=0.
        return y

def run():
    torch.set_num_threads(1);protected();out=OUT/'state_probe';out.mkdir(exist_ok=False);(out/'predictions').mkdir()
    cfg={'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,'no_training':True,'all_cases_seen_development':True,
        'encoding':'all training80 solid entries are exactly0; feed predicted solid placeholders back as0',
        'velocity_BC':'inletUx1/inletUy0/top_bottomUy0; supported by training data within2.15e-6',
        'pressure':'NO outlet pressure projection; sampled training pressure there is not exactly0',
        'centering_policy':'only abs(cy)>max training abs(cy)=0.4; others keep original or step-reflection prediction; not ID-based',
        'centering_invariance':'approximate due finite walls; preserve Fourier translation once-only policy'}
    save(out/'configuration.json',cfg)
    base,_=model_from(FORMAL,protocol()['config']);base.eval();group=ReflectionOperator(base);cases=rows();registry=[];begin=time.perf_counter()
    old=OUT/'operator_probe/predictions'
    for q in cases:
        row=startup(q)
        for method in METHODS:
            if method=='solid_feedback':direct=predict(ProjectedState(base,False),row)
            elif method=='velocity_BC_feedback':direct=predict(ProjectedState(base,True),row)
            elif method=='reflection_velocity_BC':direct=predict(ProjectedState(group,True),row)
            elif abs(q['cy'])>.40000001:
                direct=rollout(group if method.endswith('reflection') else base,row,'body_center_fft')
            elif method.endswith('reflection'):
                with np.load(old/('step_reflection__'+q['id']+'.npz')) as a:direct=a['direct']
            else:direct=baseline(q,'direct')
            closed,diag=dmd_periodic(np.concatenate([row['fields'],direct[:48]]),row['times'][:52],row['times'][4:164],row['mask'],9,scales())
            closed[:48]=direct[:48]
            file=out/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,direct=direct,DMD9=closed)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'DMD':diag})
        save(out/'status.json',{'status':'predicting','completed_cases':len(registry)//len(METHODS),'total_cases':56})
    save(out/'predictions_frozen.json',{'utc':now(),'all_predictions_before_future_scoring':True,'registry':registry})
    report={m:{'direct':[],'DMD9':[]} for m in METHODS}
    for q in cases:
        row=startup(q)
        for method in METHODS:
            with np.load(out/'predictions'/(method+'__'+q['id']+'.npz')) as a:
                for k in ['direct','DMD9']:report[method][k].append(score(q,a[k],row))
    result={method:{'summary':{k:summaries(v) for k,v in variants.items()},'cases':variants} for method,variants in report.items()}
    protected();save(out/'report.json',result);save(out/'status.json',{'status':'complete','utc':now(),'wall_seconds':time.perf_counter()-begin})
    for method,r in result.items():
        print('STATE_PROBE',method,flush=True)
        for g in ['development_inside24','seen_extreme32']:print(g,round(r['summary']['DMD9'][g]['40s']['mean_R'],6),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(OUT/'state_probe/failure.json',{'utc':now(),'error':repr(e)});raise
