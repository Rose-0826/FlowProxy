"""Reproduce the frozen v5 inference with four startup fields and causal Uy probes.

Run in the server release root. No training or CFD is launched. Future full fields
are read only after predictions have been saved, and only with --score.
"""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse,sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from research_v2_final_confirmation import load,infer
from research_v3_clock_probe import features
from startup_observability_v1 import frequency
from research_v2_template_probe import fit_startup
from research_v5_split_observer import neural_template,correction,observe,hybrid
from self_trajectory_closure_v1 import metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',default='cylinder_Re94.7_cy0.747_D1.165');ap.add_argument('--score',action='store_true');args=ap.parse_args()
 cache=ROOT/'data/processed/research_v5_split_confirmation8_cases'
 source=ROOT/'outputs/research_v2_multidirection/next_stage_split_observer/new_confirmation8_eval'
 info=json.loads((ROOT/'outputs/research_v2_multidirection/next_stage_split_observer/new_confirmation8_cfd/reference_manifest.json').read_text())
 q=next(c for c in info['cases'] if c['id']==args.case)
 with np.load(cache/(args.case+'_startup.npz')) as a:row={k:a[k] for k in ['fields','times','geom','mask']}
 assert row['fields'].shape==(4,3,64,64)
 row['id']=args.case;row['q']={k:q[k] for k in ['shape','Re','cx','cy','D']}
 torch.set_num_threads(1)
 weights=ROOT/'outputs/research_v2_multidirection/final_research_confirmation/selected_research_base_weights.pt'
 model=load({'path':str(weights)});own,diag=infer(model,row,True);n=diag['prefix_steps']
 prior=json.loads((ROOT/'outputs/research_v2_multidirection/next_round/clock_probe96/TRAIN96_frequency_prior.json').read_text())
 train=next(a for a in prior['models'] if a['shape']==q['shape']);groups=train['geometry_groups']
 fn=RBFInterpolator(np.array([g['features'] for g in groups]),np.array([g['mean_field_St'] for g in groups]),kernel='thin_plate_spline',degree=1,smoothing=train['selected']['smoothing'])
 center=float(fn(features(row['q'])[None])[0])/q['D']
 observed=np.concatenate([row['fields'],own[:n]]);t=row['times'][:n+4]-row['times'][0]
 f,_=frequency(t,observed[:,1][:,row['mask']>.5].astype(np.float64),4,center,.2)
 C=neural_template(row,own,n,f);params,_=fit_startup(C,row,f,False)
 # Separate sensor payload: shape160x4, no full-field future access here.
 with np.load(source/'measurements'/(args.case+'.npz')) as a:values=a['clean']
 pred,observer=observe(correction(C,row,params,'hybrid8_pH1'),f,row,values,params)
 pred=hybrid(own,pred)
 assert np.array_equal(pred[:32],own[:32])
 assert observer['scalar_future_readings']==76 and observer['end_gap_seconds']==2
 dest=ROOT/'reproduced';dest.mkdir(exist_ok=True)
 np.savez_compressed(dest/(args.case+'.npz'),prediction=pred)
 result={'case':args.case,'future_full_field_used_for_prediction':False,'scalar_future_readings':76,'last_gap_seconds':2,'short8_exactly_preserved':True,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}
 frozen=source/'predictions'/('v5_inference_split__'+args.case+'.npz')
 if frozen.exists():
  with np.load(frozen) as a:result['max_abs_difference_from_frozen_prediction']=float(np.max(np.abs(pred-a['prediction'])))
 if args.score:
  with np.load(cache/(args.case+'.npz')) as a:truth=a['fields'][4:164]
  result['horizons']=metrics(pred,truth,row['mask'],row['times'])
 (dest/(args.case+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
