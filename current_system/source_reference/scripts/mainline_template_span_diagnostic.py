from pathlib import Path
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,numpy as np
R=Path('/root/autodl-tmp/flowproxy');S=R/'outputs/research_v2_multidirection/next_stage_split_observer/new_confirmation8_eval'
D=R/'outputs/mainline_cleanup_20261005';report=json.loads((D/'bottleneck_diagnostic.json').read_text())
for q in report['cases']:
    cid=q['id']
    with np.load(R/'data/processed/research_v5_split_confirmation8_cases'/(cid+'.npz')) as f:
        mask=f['mask']>.5;Y=f['fields'][4:164,1][:,mask].astype(np.float64)
    with np.load(S/'predictions'/('neural_H4_open__'+cid+'.npz')) as f:X=f['prediction'][:,1][:,mask].astype(np.float64)
    # Low-rank snapshot Gram is sufficient; full neural_H4_open is a fixed
    # nine-field template, regardless of its clock.
    ev,U=np.linalg.eigh(X@X.T);order=np.argsort(ev)[::-1];ev=ev[order];U=U[:,order]
    sv=np.sqrt(np.maximum(ev,0));rank=min(9,int(np.sum(sv>sv[0]*1e-6)))
    V=(U[:,:rank].T@X)/sv[:rank,None]
    V=np.linalg.qr(V.T,mode='reduced')[0].T
    reconstruction=float(np.linalg.norm((X@V.T)@V-X)/np.linalg.norm(X))
    assert reconstruction<1e-6,reconstruction
    ideal=(Y@V.T)@V
    err=np.linalg.norm(ideal-Y,axis=1)/np.maximum(np.linalg.norm(Y,axis=1),1e-8)
    q['oracle_fixed_template_span']={'rank':rank,'source_pred_matrix_relative_reconstruction_residual':reconstruction,'mean_frame_relative_Uy':float(err.mean()),'last_frame_relative_Uy':float(err[-1]),'max_frame_relative_Uy':float(err.max()),'all_coefficients_per_frame_fit_to_future_truth':True,'constant_mode_amplitude_also_free':True,'not_a_forecast_or_deployable_result':True}
    print(json.dumps({'id':cid,**q['oracle_fixed_template_span']}),flush=True)
(D/'bottleneck_diagnostic.json').write_text(json.dumps(report,indent=2))
