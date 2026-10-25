#!/usr/bin/env python3
"""Use only self-generated spectra to guard a shorter bootstrap, compare structured harmonics."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,scales,score,summaries,protected,now
from cycle_landmarks_train_v1 import PERIODS
from startup_observability_v1 import basis,frequency
from geometry_attack_v2 import save,sha

METHODS=['harmonic_coherence_gate','H4_prefix24','H4_prefix32','H4_prefix48']

def coherence(decoder):
    f=np.sort(np.array(decoder['angular_frequencies'])/(2*np.pi))
    f=f[f>1e-4]
    if len(f)!=4:return False,{'positive_pairs':len(f),'harmonic_relative_defect':None}
    defect=float(np.max(abs(f/(f[0]*np.arange(1,5))-1)))
    return defect<=.05,{'positive_pairs':4,'harmonic_relative_defect':defect,
        'positive_frequencies':f.tolist(),'threshold':.05}

def run():
    protected();out=OUT/'spectrum_guard_probe';out.mkdir(exist_ok=False);(out/'predictions').mkdir();begin=time.perf_counter()
    save(out/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'target_observations':4,'operator':'frozen reflection+velocityBC; no parameter training',
        'coherence_rule':'positive4 DMD frequencies must agree with k*f1 to5%; if yes prefix24, otherwise original prefix48',
        'why':'stable periodic flow harmonics have one fundamental clock; competing close oscillatory modes signal unreliable early identification',
        'rule_not_target_ID_based':True,'no_target_future_for_rule':True,
        'H4_clock_search':'minimize self-bootstrap H4 reconstruction residual within+-20% TRAIN80 shape mean field St/D',
        'preserve_prefix':'retain generated prefix, harmonic extrapolation only after it',
        'all56_role':'already-seen DEVELOPMENT; no independent confirmation or automatic fallback promotion'})
    meta={q['id']:q for q in rows(True) if q['group']=='train80'}
    labels=json.loads(PERIODS.read_text())['rows']
    prior={shape:float(np.mean([a['frequency']*meta[a['id']]['D'] for a in labels
        if meta[a['id']]['shape']==shape])) for shape in ['cylinder','square']}
    save(out/'training_clock_prior.json',{'shape_mean_field_St':prior,'label_sha256':sha(PERIODS),'role':'train80 ONLY'})
    registry=[];prefix=OUT/'prefix_probe';lookup={(a['id'],a['prefix_steps']):a['decoder']
        for a in json.loads((prefix/'forecasts_frozen.json').read_text())['registry']}
    for q in rows():
        row=startup(q);ok,diag=coherence(lookup[q['id'],24]);n=24 if ok else 48
        with np.load(prefix/'predictions'/(f'prefix{n}__'+q['id']+'.npz')) as a:pred=a['prediction']
        options={'harmonic_coherence_gate':(pred,{'selected_prefix':n,**diag})}
        with np.load(OUT/'state_probe/predictions'/('reflection_velocity_BC__'+q['id']+'.npz')) as a:direct=a['direct']
        for n in [24,32,48]:
            observed=np.concatenate([row['fields'],direct[:n]])*row['mask'][None,None]
            scaled=observed/np.array(scales())[None,:,None,None]
            Y=scaled.reshape(n+4,-1);t=row['times'][:n+4]-row['times'][0]
            center=prior[q['shape']]/q['D'];f,fd=frequency(t,Y,4,center,.2)
            C=np.linalg.lstsq(basis(t,f,4),Y,rcond=1e-10)[0]
            B=basis(row['times'][4:164]-row['times'][0],f,4)
            pred=(B@C).reshape(160,3,64,64)*np.array(scales())[None,:,None,None]
            pred=pred.astype(np.float32);pred[:n]=direct[:n];assert np.isfinite(pred).all()
            options['H4_prefix'+str(n)]=(pred,{'frequency':f,'clock_search':fd,'generated_prefix':n})
        for method,(pred,diagnostic) in options.items():
            file=out/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'diagnostic':diagnostic})
    save(out/'forecasts_frozen.json',{'utc':now(),'all_predictions_before_target_future_scoring':True,'registry':registry})
    report={}
    for method in METHODS:
        part=[]
        for q in rows():
            with np.load(out/'predictions'/(method+'__'+q['id']+'.npz')) as a:part.append(score(q,a['prediction'],startup(q)))
        report[method]={'summary':summaries(part),'cases':part}
    protected();save(out/'report.json',report)
    save(out/'status.json',{'status':'complete','utc':now(),'wall_seconds':time.perf_counter()-begin})
    for m,a in report.items():
        print('SPECTRUM_GUARD',m,flush=True)
        for g in ['development_inside24','seen_extreme32']:print(g,round(a['summary'][g]['40s']['mean_R'],6),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(OUT/'spectrum_guard_probe/failure.json',{'utc':now(),'error':repr(e)});raise
