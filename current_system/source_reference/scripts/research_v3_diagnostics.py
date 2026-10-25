#!/usr/bin/env python3
"""Post-confirmation diagnostics only. No oracle output enters model selection."""
from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT, protected, now
from startup_observability_v1 import basis, frequency
from geometry_attack_v2 import save, sha

DEST=OUT/'next_round/trajectory_diagnostics'

def rel(a,b,m):
    a=a[:,:,m];b=b[:,:,m]
    return (np.linalg.norm(a-b,axis=(0,2))/np.maximum(np.linalg.norm(b,axis=(0,2)),1e-8)).tolist()

def main():
    protected();DEST.mkdir(parents=True,exist_ok=False)
    ref=OUT/'independent8_cfd/SEALED_reference_manifest.json'
    manifest=json.loads(ref.read_text());cache=Path(manifest['cache']);result=[]
    save(DEST/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),
        'purpose':'diagnose previously frozen independent8 AFTER confirmation, never claim these oracle variants as predictions',
        'frequency_source':'full CFD future fields; DIAGNOSTIC ONLY',
        'no_model_selection_on_these8':True,'no_training_on_these8':True})
    for q in manifest['cases']:
        with np.load(cache/(q['id']+'.npz')) as a:
            true=a['fields'][4:164].astype(np.float64);t=a['times'][4:164]-a['times'][4];m=a['mask']>.5
        meta=json.loads((cache/(q['id']+'.json')).read_text())
        with np.load(OUT/'final_research_confirmation/predictions'/('research_selected__'+q['id']+'.npz')) as a:pred=a['prediction'].astype(np.float64)
        ftrue,dt=frequency(t,true[:,1][:,m],4,meta['St']/q['D'],.05)
        fpred,dp=frequency(t,pred[:,1][:,m],4,ftrue,.2)
        ct=np.linalg.lstsq(basis(t,ftrue,4),true.reshape(160,-1),rcond=1e-10)[0]
        cp=np.linalg.lstsq(basis(t,fpred,4),pred.reshape(160,-1),rcond=1e-10)[0]
        reconstructed=(basis(t,fpred,4)@cp).reshape(pred.shape)
        corrected=(basis(t,ftrue,4)@cp).reshape(pred.shape)
        # A separate global phase alignment diagnoses residual offset; not inference.
        shifts=np.linspace(-.5/ftrue,.5/ftrue,121)
        best=None
        for shift in shifts:
            x=(basis(t+shift,ftrue,4)@cp[:,64*64:2*64*64]).reshape(160,64,64)
            e=np.linalg.norm((x-true[:,1])[:,m])/np.linalg.norm(true[:,1][:,m])
            if best is None or e<best[0]:best=(float(e),float(shift))
        item={'id':q['id'],'true_frequency':ftrue,'prediction_frequency':fpred,
            'relative_frequency_error':abs(fpred/ftrue-1),'true_H4_residual':dt,'pred_H4_residual':dp,
            'stack_relative_error':rel(pred,true,m),'pred_H4_fit_error':rel(reconstructed,true,m),
            'oracle_frequency_only_error':rel(corrected,true,m),
            'oracle_frequency_and_phase_Uy':best[0],'oracle_phase_shift_seconds':best[1],
            'reference_H4_error':rel((basis(t,ftrue,4)@ct).reshape(true.shape),true,m)}
        result.append(item);print(q['id'],flush=True)
        print('Uy raw/clock/clock+phase',*[round(x*100,2) for x in [item['stack_relative_error'][1],item['oracle_frequency_only_error'][1],best[0]]],flush=True)
        save(DEST/'status.json',{'status':'running','utc':now(),'completed':len(result),'total':8})
    protected();save(DEST/'report.json',{'utc':now(),'role':'POSTHOC ORACLE DIAGNOSTIC ONLY','cases':result,'fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'completed':8})

if __name__=='__main__':
    try:main()
    except Exception as e:
        DEST.mkdir(parents=True,exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
