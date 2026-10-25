#!/usr/bin/env python3
"""Observation-trust DMD and train-only clock estimation; all targets already seen."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,baseline,scales,score,summaries,protected,now,truth
from cycle_landmarks_train_v1 import PERIODS
from corrected_benchmark_controls_v1 import dmd_periodic
from startup_observability_v1 import frequency
from geometry_attack_v2 import save,sha

RADII=[0.,.25,.5,1.];METHODS=['anchor_real4_only','anchor_balanced13','anchor50',
    'anchor13_RBF_Re_D','anchor13_RBF_even_cy','anchor13_physical_clock_law','anchor13_AR4_filtered']

def feature(q,even_cy=False):
    return [q['Re']/200,q['D']] + ([q['cy']**2/.16] if even_cy else [])

def physical_feature(q):
    r=100/q['Re'];d=(q['D']-1)/.1;c=q['cy']**2/.16
    return np.array([1,r,d,d*d,r*d,c],np.float64)

def ar_clock(fields,q,mask,radius):
    y=fields[:4,1].astype(np.float64)
    if radius:
        sigma=radius*q['D'];y=gaussian_filter(y,sigma=(0,sigma/(16/63),sigma/(28/63)),mode='nearest')
    xx,yy=np.meshgrid(np.linspace(-8,20,64),np.linspace(-8,8,64),indexing='xy')
    region=(mask>.5)&(xx>q['cx']+2*q['D'])&(xx<q['cx']+18*q['D'])&(abs(yy-q['cy'])<4*q['D'])
    d=np.diff(y[:,region],axis=0);den=2*np.sum(d[1]**2)
    c=float(np.sum((d[2]+d[0])*d[1])/max(den,1e-30))
    if not -1<c<1:return None,{'cosine':c,'valid':False}
    return float(np.arccos(c)/(2*np.pi*.25)),{'cosine':c,'valid':True,'effective_nodes':int(region.sum())}

def build_clocks():
    metadata={q['id']:q for q in rows(True) if q['group']=='train80'}
    periods=json.loads(PERIODS.read_text())['rows'];train=[{**metadata[p['id']],'frequency':p['frequency'],'St_field':p['frequency']*metadata[p['id']]['D']} for p in periods]
    ar={r:[] for r in RADII}
    # Multiple true training phases, never a target future, choose the filter on training only.
    for q in train:
        fields=truth(q);m=startup(q)['mask']
        for radius in RADII:
            for start in [0,20,40,80,120]:
                f,diag=ar_clock(fields[start:start+4],q,m,radius)
                error=abs(f/q['frequency']-1) if f else 10.
                ar[radius].append(error)
    chosen=min(RADII,key=lambda r:(np.median(ar[r]),np.quantile(ar[r],.9),r))
    priors={};checks={}
    for shape in ['cylinder','square']:
        part=[q for q in train if q['shape']==shape];y=np.array([q['St_field'] for q in part])
        X=np.array([feature(q) for q in part]);Xe=np.array([feature(q,True) for q in part]);Xp=np.stack([physical_feature(q) for q in part])
        coef=np.linalg.solve(Xp.T@Xp+1e-8*np.eye(Xp.shape[1]),Xp.T@y)
        priors[shape]={'two':RBFInterpolator(X,y,kernel='thin_plate_spline',smoothing=1e-6),
            'even':RBFInterpolator(Xe,y,kernel='thin_plate_spline',smoothing=1e-6),'physical':coef,'St_min':float(y.min()),'St_max':float(y.max())}
        checks[shape]={'St_min':float(y.min()),'St_max':float(y.max()),'physical_law_coefficients':coef.tolist()}
    save(OUT/'clock_anchor_probe/training_clocks.json',{'train80_period_labels_sha256':sha(PERIODS),'radius_selected_training_only':chosen,
        'radius_errors':{str(r):{'median':float(np.median(ar[r])),'p90':float(np.quantile(ar[r],.9))} for r in RADII},'shape_laws':checks})
    return priors,chosen

def decompose(observed,mask):
    sc=np.array(scales(),np.float64);Y=(observed.astype(np.float64)*mask[None,None]/sc[None,:,None,None]).reshape(len(observed),-1).T
    U,s,Vh=np.linalg.svd(Y[:,:-1],full_matrices=False);r=min(9,int((s>s[0]*1e-9).sum()))
    AV=(Y[:,1:]@Vh[:r].T)/s[:r][None];eig,W=np.linalg.eig(U[:,:r].T@AV)
    return Y,AV@W,np.angle(eig)/.25

def anchored(Y,Phi,angular,times,weight,frequency_override=None,fundamental_center=None):
    omega=angular.copy();fitted=None
    if frequency_override is not None:
        positives=omega[omega>1e-4]/(2*np.pi)
        center=frequency_override if fundamental_center is None else fundamental_center
        fitted=float(positives[np.argmin(abs(positives-center))])
        omega*=frequency_override/fitted
    t=np.asarray(times[:52])-times[0];future=np.asarray(times[4:164])-times[0]
    E=np.exp(1j*t[:,None]*omega[None]);w=np.ones(52)
    if weight=='only4':w[4:]=0
    else:w[:4]=weight
    gram=(Phi.conj().T@Phi)*(E.conj().T@(w[:,None]*E))
    rhs=(E.conj().T*(Phi.conj().T@Y)*w[None]).sum(1)
    coef=np.linalg.lstsq(gram,rhs,rcond=1e-10)[0]
    Ef=np.exp(1j*future[:,None]*omega[None]);pred=(Ef@(Phi*coef[None]).T).real.reshape(160,3,64,64)
    pred=(pred*np.array(scales())[None,:,None,None]).astype(np.float32)
    assert np.isfinite(pred).all()
    return pred,{'observed_weight':weight,'frequency_override':frequency_override,'original_matched_frequency':fitted,
        'amplitude_condition':float(np.linalg.cond(gram))}

def run():
    protected();out=OUT/'clock_anchor_probe';out.mkdir(exist_ok=False);(out/'predictions').mkdir();begin=time.perf_counter()
    save(out/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,'no_network_training':True,
        'observations':4,'DMD_modes':'unchanged basis from4 true+48 neural, unchanged rank9',
        'real_data_trust':['only4 real states for amplitudes','real states weight13','real states weight50'],
        'frequency_labels':'full fields of TRAIN80 only, same sealed one-cycle frequency file',
        'AR4':'three temporal differences: d3+d1=2cos(omega dt)d2, spatial smoothing radius selected from TRAIN80 phases only',
        'clock_guard':'if estimate outside train shape St range with25% margin, fallback to positive bounded train-only Re,D RBF; every case still scored',
        'all56_role':'already-seen DEVELOPMENT; no fresh confirmation','first48_outputs_kept':True})
    priors,radius=build_clocks();cases=rows();registry=[];clock_records=[]
    for q in cases:
        row=startup(q);direct=baseline(q,'direct');observed=np.concatenate([row['fields'],direct[:48]])
        Y,Phi,omega=decompose(observed,row['mask']);p=priors[q['shape']]
        f2=float(p['two'](np.array([feature(q)]))[0])/q['D']
        fe=float(p['even'](np.array([feature(q,True)]))[0])/q['D']
        fp=float(physical_feature(q)@p['physical'])/q['D']
        fa,ad=ar_clock(row['fields'],q,row['mask'],radius)
        lo=.75*p['St_min']/q['D'];hi=1.25*p['St_max']/q['D']
        fallback=float(np.clip(f2,lo,hi));clocks={'two':f2,'even':fe,'physical':fp,'AR4':fa}
        used={};rejected={}
        for key,f in clocks.items():
            valid=f is not None and np.isfinite(f) and lo<=f<=hi
            used[key]=float(f) if valid else fallback;rejected[key]=not valid
        clock_records.append({'id':q['id'],'q':row['q'],'AR_radius_training_selected':radius,'raw':clocks,'used':used,'rejected':rejected,'AR':ad})
        verify,_=anchored(Y,Phi,omega,row['times'],1)
        reference=dmd_periodic(observed,row['times'][:52],row['times'][4:164],row['mask'],9,scales())[0]
        relative=np.linalg.norm((verify-reference)*row['mask'][None,None])/max(np.linalg.norm(reference*row['mask'][None,None]),1e-8)
        assert relative<2e-5
        options={'anchor_real4_only':('only4',None),'anchor_balanced13':(13,None),'anchor50':(50,None),
            'anchor13_RBF_Re_D':(13,used['two']),'anchor13_RBF_even_cy':(13,used['even']),
            'anchor13_physical_clock_law':(13,used['physical']),'anchor13_AR4_filtered':(13,used['AR4'])}
        for name,(weight,f) in options.items():
            pred,diag=anchored(Y,Phi,omega,row['times'],weight,f,fallback);pred[:48]=direct[:48]
            file=out/'predictions'/(name+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=pred)
            registry.append({'id':q['id'],'method':name,'sha256':sha(file),'decoder':diag})
        save(out/'status.json',{'status':'predicting','completed_cases':len(clock_records),'total_cases':56})
    save(out/'clock_predictions_frozen.json',{'utc':now(),'training_only_filter_selection':True,'cases':clock_records})
    save(out/'predictions_frozen.json',{'utc':now(),'all_predictions_before_target_future_scoring':True,'registry':registry})
    scored={m:[] for m in METHODS};diagnostic=[]
    lookup={q['id']:q for q in cases}
    for q in cases:
        row=startup(q)
        for m in METHODS:
            with np.load(out/'predictions'/(m+'__'+q['id']+'.npz')) as a:scored[m].append(score(q,a['prediction'],row))
        full=truth(q);meta=json.loads((Path(q['cache'])/(q['id']+'.json')).read_text())
        ft,fd=frequency(row['times']-row['times'][0],full[:,1][:,row['mask']>.5].astype(np.float64),4,meta['St']/q['D'],.05)
        record=next(r for r in clock_records if r['id']==q['id'])
        diagnostic.append({'id':q['id'],'full_future_frequency_DIAGNOSTIC_ONLY':ft,
            'relative_errors_DIAGNOSTIC_ONLY':{k:abs(f/ft-1) for k,f in record['used'].items()},'fit':fd})
    result={m:{'summary':summaries(v),'cases':v} for m,v in scored.items()}
    protected();save(out/'report.json',result);save(out/'future_clock_diagnostic.json',diagnostic)
    save(out/'status.json',{'status':'complete','utc':now(),'wall_seconds':time.perf_counter()-begin})
    for m,v in result.items():
        print('CLOCK_ANCHOR_PROBE',m,flush=True)
        for g in ['development_inside24','seen_extreme32']:print(g,round(v['summary'][g]['40s']['mean_R'],6),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(OUT/'clock_anchor_probe/failure.json',{'utc':now(),'error':repr(e)});raise
