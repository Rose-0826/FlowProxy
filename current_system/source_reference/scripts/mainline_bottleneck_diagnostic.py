"""Read-only diagnostic of frozen predictions; never a new prediction score."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,json,numpy as np
R=Path('/root/autodl-tmp/flowproxy');sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from research_v4_sparse_observer import probes
from startup_observability_v1 import frequency
from research_v2_template_probe import basis
from research_v2_common import scales
S=R/'outputs/research_v2_multidirection/next_stage_split_observer/new_confirmation8_eval'
M=json.loads((S.parent/'new_confirmation8_cfd/reference_manifest.json').read_text())
report={'role':'POST_SCORE_DIAGNOSTIC_ONLY_USES_FUTURE_TRUTH_NOT_A_METHOD_RESULT','cases':[]}
for q in M['cases']:
    with np.load(R/'data/processed/research_v5_split_confirmation8_cases'/(q['id']+'.npz')) as a:
        Y=a['fields'][4:164].astype(np.float64);mask=a['mask']>.5;times=a['times'][4:164];geom=a['geom']
    with np.load(S/'predictions'/('v5_inference_split__'+q['id']+'.npz')) as a:P=a['prediction'].astype(np.float64)
    row={'q':q,'mask':mask.astype(np.float32)};ij,_=probes(row)
    ids=np.arange(7,152,8);yt=np.stack([Y[ids,1,i,j] for i,j in ij],1);pt=np.stack([P[ids,1,i,j] for i,j in ij],1)
    # After the short-field branch/blend, and before each current-time reading.
    late=ids>=48
    sensor=float(np.sqrt(np.mean((pt[late]-yt[late])**2))/scales()[1])
    t=times-times[0]
    # Future truth estimates the oracle clock for decomposition only.
    f,diag=frequency(t,Y[:,1][:,mask],4,.16/q['D'],.3)
    B=basis(t,f);Ctrue=np.linalg.lstsq(B,Y[:,1][:,mask],rcond=1e-10)[0]
    Cpred=np.linalg.lstsq(B,P[:,1][:,mask],rcond=1e-10)[0]
    truthfit=B@Ctrue;predfit=B@Cpred;d=Cpred-Ctrue;den=np.linalg.norm(Y[:,1][:,mask])
    parts={name:float(np.linalg.norm(B[:,sl]@d[sl])/den) for name,sl in [('mean',slice(0,1)),('H1',slice(1,3)),('H2_H4',slice(3,9))]}
    shifts=np.linspace(-np.pi,np.pi,361);best=None
    for theta in shifts:
        cp=Cpred.copy()
        for k in range(1,5):
            c,s=np.cos(k*theta),np.sin(k*theta)
            cp[2*k-1]=Cpred[2*k-1]*c+Cpred[2*k]*s
            cp[2*k]=-Cpred[2*k-1]*s+Cpred[2*k]*c
        # Small Gram avoids multiplying all spatial fields for each shift.
        delta=cp-Ctrue;err=float(np.sqrt(np.sum((delta@delta.T)*(B.T@B)))/den)
        if best is None or err<best[0]:best=(err,theta)
    item={'id':q['id'],'D':q['D'],'shape':q['shape'],'post12s_sensor_preupdate_RMSE_div_train_Uy_scale':sensor,'whole_Uy_stacked_L2':float(np.linalg.norm((P[:,1]-Y[:,1])[:,mask])/den),'oracle_H4_fit_truth_residual':float(np.linalg.norm(truthfit-Y[:,1][:,mask])/den),'pred_H4_vs_true_H4':float(np.linalg.norm(predfit-truthfit)/den),'oracle_global_phase_aligned_H4_error':best[0],'oracle_best_phase_degrees':float(np.rad2deg(best[1])),'parts_normalized_by_whole_Uy':parts}
    report['cases'].append(item)
    print(json.dumps(item),flush=True)
report['warning']='Harmonic/stacked error is not the frozen frame-mean/endpoint metric. Parts are not summed as percentages. Oracle alignment uses future truth and is diagnostic only. Four probes cannot identify unrestricted full-field corrections.'
(R/'outputs/mainline_cleanup_20261005/bottleneck_diagnostic.json').write_text(json.dumps(report,indent=2))
