#!/usr/bin/env python3
"""Causal sparse virtual-probe observer. A separate task from autonomous prediction."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import numpy as np
from scipy.linalg import block_diag
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,truth,scales,score,summaries,protected,now
from research_v2_outer_support_train import dataset
from research_v2_template_probe import mapped,fit_startup,gauge_align
from startup_observability_v1 import basis
from geometry_attack_v2 import save,sha

DEST=OUT/'sparse_observer_pilot'
SOURCE=OUT/'next_round/clock_probe96'
METHODS=['neural_H4_open','neural_EKF_2s','neural_EKF_4s','neural_linearKF_2s',
         'nearest96_EKF_2s','aligned4_96_EKF_2s','neural_EKF_2s_noise1pct','nearest96_EKF_2s_noise1pct']

def probes(row):
    q=row['q'];xx,yy=np.meshgrid(np.linspace(-8,20,64),np.linspace(-8,8,64))
    chosen=[];record=[]
    for offset in [2.,4.,6.,8.]:
        x=q['cx']+offset*q['D'];y=q['cy'];dist=(xx-x)**2+(yy-y)**2
        dist[row['mask']<=.5]=np.inf
        i,j=np.unravel_index(np.argmin(dist),dist.shape);assert np.isfinite(dist[i,j])
        chosen.append((int(i),int(j)));record.append({'requested':[x,y],'grid':[int(i),int(j)],'actual':[float(xx[i,j]),float(yy[i,j])]})
    assert len(set(chosen))==4
    return chosen,record

def collect():
    """Measurement-generation process reads CFD; predictor receives only four scalars."""
    protected();(DEST/'measurements').mkdir()
    records=[]
    for q in rows():
        row=startup(q);ij,record=probes(row);target=truth(q)[4:164]
        values=np.stack([target[:,1,i,j] for i,j in ij],1).astype(np.float64)
        seed=int(hashlib.sha256(q['id'].encode()).hexdigest()[:8],16)
        noise=np.random.default_rng(seed).normal(0,.01*scales()[1],values.shape)
        p=DEST/'measurements'/(q['id']+'.npz');np.savez(p,clean=values,noisy=values+noise)
        records.append({'id':q['id'],'probes':record,'measurement_sha256':sha(p)})
    save(DEST/'measurement_registry.json',{'utc':now(),'role':'virtual CFD probes only; no actual hardware claim',
        'locations':'four geometry-only downstream positions, Uy channel; no target-field sensor selection',
        'noise_std':.01*scales()[1],'cases':records})

def library():
    data=dataset();periods=json.loads((OUT/'outer_support_training/periods96.json').read_text())['rows'];result=[]
    assert [r['id'] for r in data]==[p['id'] for p in periods]
    for r,p in zip(data,periods):
        t=r['times']-r['times'][0];C=np.linalg.lstsq(basis(t,p['frequency'],4),r['fields'].reshape(len(t),-1),rcond=1e-10)[0].reshape(9,3,64,64)
        result.append({'q':{**r['q'],'id':r['id']},'coef':C,'mask':r['mask'],'f':p['frequency']})
    return result

def constrain(C,mask):
    C=C.copy()*mask[None,None];C[:,0,:,0]=0.;C[0,0,:,0]=1.
    C[:,1,:,0]=0.;C[:,1,0,:]=0.;C[:,1,-1,:]=0.
    return C*mask[None,None]

class CausalMeasurements:
    def __init__(self,values,interval):self.values=values;self.interval=interval;self.access=[]
    def read(self,step):
        assert step%self.interval==0 and 1<=step<160
        assert not self.access or step>self.access[-1]
        self.access.append(step);return self.values[step-1].copy()

def weights(theta):
    k=np.arange(1,5);b=np.empty(9);b[0]=1.;b[1::2]=np.cos(k*theta);b[2::2]=np.sin(k*theta)
    d=np.empty(9);d[0]=0.;d[1::2]=-k*np.sin(k*theta);d[2::2]=k*np.cos(k*theta)
    return b,d

def forecast(C,f,row,values,interval,kind):
    """Emit each field BEFORE reading that time's probes, then update for later fields."""
    params,initial=fit_startup(C,row,f,False);theta=float(params[0]);omega=2*np.pi*f;omega0=omega
    ij,_=probes(row);H=np.stack([C[:,1,i,j] for i,j in ij],0)
    reader=CausalMeasurements(values,interval) if interval else None
    R=np.eye(4)*(.05*scales()[1])**2
    z=np.array([theta,omega]);P=np.diag([np.deg2rad(5.)**2,(.02*omega)**2])
    A=np.array([[1.,.25],[0.,1.]])
    Q=np.diag([np.deg2rad(.5)**2,(.0005*omega)**2])*.25
    state=weights(theta)[0][1:];PK=np.eye(8)*.05**2
    rotation=block_diag(*[np.array([[np.cos(k*omega*.25),-np.sin(k*omega*.25)],
        [np.sin(k*omega*.25),np.cos(k*omega*.25)]]) for k in range(1,5)])
    predictions=[];updates=[]
    for step in range(1,161):
        if kind=='linear':
            state=rotation@state;PK=rotation@PK@rotation.T+np.eye(8)*.0001*.25
            b=np.r_[1.,state]
        else:
            z=A@z;P=A@P@A.T+Q;b,_=weights(z[0])
        predictions.append((b@C.reshape(9,-1)).reshape(3,64,64))
        if reader is not None and step%interval==0 and step<160:
            y=reader.read(step)
            if kind=='linear':
                h=H[:,1:];gain=np.linalg.solve(h@PK@h.T+R,h@PK).T
                residual=y-H[:,0]-h@state;state=state+gain@residual
                J=np.eye(8)-gain@h;PK=J@PK@J.T+gain@R@gain.T
                updates.append({'step':step,'sensor_residual_norm':float(np.linalg.norm(residual))})
            else:
                b,d=weights(z[0]);h=np.c_[H@d,np.zeros(4)]
                gain=np.linalg.solve(h@P@h.T+R,h@P).T;residual=y-H@b
                delta=gain@residual;delta[0]=np.clip(delta[0],-np.pi/2,np.pi/2);z=z+delta
                z[1]=np.clip(z[1],.8*omega0,1.2*omega0)
                J=np.eye(2)-gain@h;P=J@P@J.T+gain@R@gain.T
                updates.append({'step':step,'phase_delta_degrees':float(np.rad2deg(delta[0])),
                    'frequency':float(z[1]/(2*np.pi)),'sensor_residual_norm':float(np.linalg.norm(residual))})
            if not np.isfinite(P).all():raise RuntimeError('Nonfinite observer covariance')
    pred=np.asarray(predictions,np.float32)*row['mask'][None,None];assert np.isfinite(pred).all()
    accessed=[] if reader is None else reader.access
    return pred,{'initial':initial,'updates':updates,'observed_future_steps':accessed,
        'scalar_future_readings':4*len(accessed),'last_future_reading_seconds':None if not accessed else accessed[-1]*.25,
        'end_forecast_gap_seconds':40. if not accessed else 40.-accessed[-1]*.25,
        'every_field_emitted_before_current_time_measurement':True}

def main():
    protected();DEST.mkdir(exist_ok=False);(DEST/'predictions').mkdir();begin=time.perf_counter()
    save(DEST/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'new_task':'causal full-field reconstruction/forecast with sparse virtual Uy observations; NOT autonomous40s or solver-free geometry-to-flow',
        'startup':'same four real full fields; no later full field supplied to predictor',
        'probes':'4 nearest-fluid grid nodes to x=cx+[2,4,6,8]*D,y=cy, geometry only',
        'measurement_schedule':'2s updates at2,...,38s=76 scalars;4s updates at4,...,36s=36 scalars; no t40 endpoint assimilation',
        'causality':'predict field before current timestamp sensor read; no backward smoothing; final2/4s without measurements',
        'template':'same frozen TRAIN96 research self-bootstrap, H4 fitted only4 true+own24/48; phase initialized only4 startup fields',
        'simple_baselines':'same-data TRAIN96 nearest and phase-aligned4-donor CFD H4 transported by geometry +same EKF; same neural basis +ordinary linear coefficient KF',
        'EKF':'phase and frequency state; fixed P diag(5deg^2,(2%omega)^2),Q diag(.5deg^2,(.05%omega)^2)*dt,R=(5%TRAIN80 Uy scale)^2',
        'linearKF':'8 harmonic coordinates, fixed self frequency; P=.05^2I,Q=.01^2I*dt,sameR',
        'noise':'clean and1%TRAIN80 Uy scale deterministic Gaussian noise; same noisy probes for neural/nearest pair',
        'inference_no_future':True,'observations_generated_separate_process':True,
        'evaluation':'old seen56 DEVELOPMENT only; last independent8 excluded; new confirmation needed before generalization claims',
        'signal':'2s neural EKF >=40/56 pass,balancedRi40<=.75 frozen autonomous and<=.8 BOTH same-probe nearest/aligned4 EKF; maximumRi40 no worse than frozen research; noisy pass loss<=2',
        'limits':'virtual sensors; hardware/measurement cost not demonstrated; extra observations cannot be omitted from accuracy claims',
        'fallback_and_confirmed_research_immutable':True})
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    with (DEST/'measurement_collection.log').open('x') as log:
        subprocess.run([sys.executable,__file__,'--collect'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    templates=library();lookup={a['id']:a['diagnostic'] for a in json.loads((SOURCE/'forecasts_frozen.json').read_text())['registry'] if a['method']=='self_H4'}
    registry=[]
    for q in rows():
        row=startup(q);d=lookup[q['id']];n=d['prefix_steps'];f=d['self_frequency']
        with np.load(SOURCE/'predictions'/('frozen_research__'+q['id']+'.npz')) as a:own=a['prediction'][:n]
        obs=np.concatenate([row['fields'],own]);t=row['times'][:n+4]-row['times'][0]
        neural=constrain(np.linalg.lstsq(basis(t,f,4),obs.reshape(len(t),-1),rcond=1e-10)[0].reshape(9,3,64,64),row['mask'])
        def distance(s):return sum(((q[k]-s['q'][k])/v)**2 for k,v in [('Re',120.),('cy',1.),('D',.5)])
        donors=sorted([s for s in templates if s['q']['shape']==q['shape']],key=lambda s:(distance(s),s['q']['id']))[:4];donor=donors[0]
        nearest=constrain(mapped(donor,q,row['mask']),row['mask']);fn=donor['f']*donor['q']['D']/q['D']
        mixweights=1/(np.array([distance(s) for s in donors])+1e-6);mixweights/=mixweights.sum()
        transported=[mapped(s,q,row['mask']) for s in donors]
        aligned=[transported[0]]+[gauge_align(c,transported[0],row['mask'])[0] for c in transported[1:]]
        mixed=constrain(np.einsum('s,skchw->kchw',mixweights,np.stack(aligned)),row['mask'])
        fm=sum(w*s['f']*s['q']['D'] for w,s in zip(mixweights,donors))/q['D']
        with np.load(DEST/'measurements'/(q['id']+'.npz')) as a:clean=a['clean'];noisy=a['noisy']
        for method in METHODS:
            coef,freq=(nearest,fn) if method.startswith('nearest') else ((mixed,fm) if method.startswith('aligned4') else (neural,f))
            interval=0 if method.endswith('open') else (16 if '4s' in method else 8)
            kind='linear' if 'linearKF' in method else 'phase'
            pred,diag=forecast(coef,freq,row,noisy if 'noise' in method else clean,interval,kind)
            p=DEST/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(p,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(p),'diagnostic':diag,
                'nearest_donor':donor['q']['id'] if method.startswith('nearest') else None,
                'aligned4_donors':[s['q']['id'] for s in donors] if method.startswith('aligned4') else None})
        save(DEST/'status.json',{'status':'predicting_from_startup_and_causal_probes','utc':now(),'completed_cases':len(registry)//len(METHODS),'total':56,'wall_seconds':time.perf_counter()-begin})
    save(DEST/'forecasts_frozen.json',{'utc':now(),'all_forecasts_before_full_future_scoring':True,'registry':registry})
    results={}
    for method in METHODS:
        cases=[]
        for q in rows():
            with np.load(DEST/'predictions'/(method+'__'+q['id']+'.npz')) as a:cases.append(score(q,a['prediction'],startup(q)))
        results[method]={'summary':summaries(cases),'cases':cases}
    def bal(s):return .5*(s['development_inside24']['40s']['mean_R']+s['seen_extreme32']['40s']['mean_R'])
    ref=json.loads((OUT/'final_research_confirmation/outer96_seed622_development.json').read_text())['summary']
    a=results['neural_EKF_2s']['summary'];b=results['nearest96_EKF_2s']['summary'];b4=results['aligned4_96_EKF_2s']['summary'];noisy=results['neural_EKF_2s_noise1pct']['summary']
    signal=(a['development_all56']['40s']['n_pass']>=40 and bal(a)<=.75*bal(ref) and bal(a)<=.8*min(bal(b),bal(b4))
        and a['development_all56']['40s']['max_R']<=ref['development_all56']['40s']['max_R']
        and noisy['development_all56']['40s']['n_pass']>=a['development_all56']['40s']['n_pass']-2)
    protected();save(DEST/'report.json',{'utc':now(),'new_task':'sparse virtual-observation assisted reconstruction and forecast',
        'results':results,'research_signal_passed':signal,'autonomous_result_unchanged':True,'fallback_unchanged':True})
    save(DEST/'status.json',{'status':'complete','utc':now(),'research_signal_passed':signal,'wall_seconds':time.perf_counter()-begin})
    for m,a in results.items():
        z=a['summary']['development_all56']['40s'];print('SPARSE',m,round(z['mean_R'],5),round(z['max_R'],5),z['n_pass'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--collect',action='store_true');a=ap.parse_args()
    try:
        if a.collect:collect()
        else:main()
    except Exception as e:
        DEST.mkdir(exist_ok=True);save(DEST/'failure.json',{'utc':now(),'error':repr(e)});raise
