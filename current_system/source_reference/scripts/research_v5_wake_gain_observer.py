#!/usr/bin/env python3
"""Use the same four probes to estimate phase/frequency and slow wake H1 gains."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,score,summaries,scales,protected,now
from research_v5_split_observer import DEST,PILOT,context,correction,hybrid,balanced,candidate_guard
from research_v4_sparse_observer import weights,probes,CausalMeasurements,library,constrain
from research_v2_template_probe import fit_startup,mapped,gauge_align
from geometry_attack_v2 import save,sha

GAIN=DEST/'wake_gain_pilot'
METHODS=['gain1_neural','gain4_neural','gain4_nearest','gain4_aligned4',
    'gain1_neural_noise1pct','gain4_neural_noise1pct','gain4_nearest_noise1pct']

def spatial_weights(row,count):
    assert count in [1,4]
    x=np.linspace(-8,20,64);q=row['q'];ij,_=probes(row)
    nodes=np.array([x[j] for i,j in ij]);taper=np.clip((x-q['cx'])/(nodes[0]-q['cx']),0.,1.)
    if count==1:W=taper[None]
    else:W=np.stack([np.interp(x,nodes,np.eye(4)[k]) for k in range(4)])*taper
    W=np.broadcast_to(W[:,None,:],(count,64,64)).copy()
    S=np.stack([W[:,i,j] for i,j in ij],0)
    return W,S

def gain_forecast(C,f,row,values,params,count):
    """Same causality/budget as v4; real H1 gains scale coupled template channels."""
    W,S=spatial_weights(row,count);ij,_=probes(row)
    H=np.stack([C[:,1,i,j] for i,j in ij],0)
    omega=2*np.pi*f;z=np.r_[params[0],omega,np.ones(count)]
    P=np.diag(np.r_[np.deg2rad(5.)**2,(.02*omega)**2,np.full(count,.1**2)])
    A=np.eye(2+count);A[0,1]=.25
    Q=np.diag(np.r_[np.deg2rad(.5)**2,(.0005*omega)**2,np.full(count,.005**2)])*.25
    R=np.eye(4)*(.05*scales()[1])**2;reader=CausalMeasurements(values,8)
    predictions=[];updates=[];phases=[]
    for step in range(1,161):
        z=A@z;P=A@P@A.T+Q;b,d=weights(z[0]);phases.append(float(z[0]))
        field=(b@C.reshape(9,-1)).reshape(3,64,64)
        H1=b[1]*C[1]+b[2]*C[2]
        gain=1+np.einsum('k,kij->ij',z[2:]-1,W)
        field+=(gain-1)[None]*H1
        predictions.append(field)
        if step%8==0 and step<160:
            y=reader.read(step);first=b[1]*H[:,1]+b[2]*H[:,2]
            first_d=d[1]*H[:,1]+d[2]*H[:,2]
            g=1+S@(z[2:]-1);expected=H@b+(g-1)*first
            jac=np.c_[H@d+(g-1)*first_d,np.zeros(4),first[:,None]*S]
            K=np.linalg.solve(jac@P@jac.T+R,jac@P).T;residual=y-expected
            delta=K@residual;delta[0]=np.clip(delta[0],-np.pi/2,np.pi/2)
            z+=delta;z[1]=np.clip(z[1],.8*omega,1.2*omega);z[2:]=np.clip(z[2:],.5,1.5)
            J=np.eye(2+count)-K@jac;P=J@P@J.T+K@R@K.T
            updates.append({'step':step,'frequency':float(z[1]/(2*np.pi)),
                'gains':z[2:].tolist(),'sensor_residual_norm':float(np.linalg.norm(residual))})
        assert np.isfinite(P).all()
    pred=np.asarray(predictions,np.float32)*row['mask'][None,None]
    return pred,{'initial_phase':float(params[0]),'frequency':f,'gain_count':count,
        'updates':updates,'phases':phases,'scalar_future_readings':4*len(reader.access),
        'observed_future_steps':reader.access,'last_reading_seconds':38.,'end_gap_seconds':2.,
        'every_field_emitted_before_current_time_measurement':True}

def templates_for(row,templates):
    q=row['q']
    def distance(s):return sum(((q[k]-s['q'][k])/v)**2 for k,v in [('Re',120.),('cy',1.),('D',.5)])
    donors=sorted([s for s in templates if s['q']['shape']==q['shape']],key=lambda s:(distance(s),s['q']['id']))[:4]
    transported=[mapped(s,q,row['mask']) for s in donors]
    nearest=constrain(transported[0],row['mask']);fn=donors[0]['f']*donors[0]['q']['D']/q['D']
    aligned=[transported[0]]+[gauge_align(c,transported[0],row['mask'])[0] for c in transported[1:]]
    w=1/(np.array([distance(s) for s in donors])+1e-6);w/=w.sum()
    mixed=constrain(np.einsum('s,skchw->kchw',w,np.stack(aligned)),row['mask'])
    fm=sum(a*s['f']*s['q']['D'] for a,s in zip(w,donors))/q['D']
    return nearest,fn,mixed,fm

def main():
    protected();GAIN.mkdir(exist_ok=False);begin=time.perf_counter()
    save(GAIN/'preregister.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'data':'old56 DEVELOPMENT;previous independent batches excluded',
        'new_state':'phase,frequency,1 global or4 streamwise real H1 gains;slow random walk',
        'gain_prior_std':.1,'gain_Q_std_per_sqrt_second':.005,'gain_bounds':[.5,1.5],
        'spatial_weights':'linear hat interpolation at geometry-only probe x nodes,taper0 at obstacle and1 at first probe;constant outside last node',
        'field':'gain applies H1 ofall3 coupled template channels;mean andhigher harmonics unchanged',
        'pressure':'same startup-only p mean/H1 ridge as inference winner;clock startup params never refitted',
        'short':'same neural autonomous first8s forALL methods,including retrieval baselines;transition8..12s',
        'fairness':'nearest96 andaligned4 templates getthe same local-gain observer,pressure calibration,andshort branch',
        'budget':'same76 Uy scalar readings,emit before read,final2s without readings',
        'selection':'neural gain1/4 only,existing inference guard,min balancedR40',
        'signal':'>=48/56 pass,>=10% lower balancedR40 than inference winner,<=.8 BOTH strengthened retrieval baselines;noise pass loss<=2',
        'limits':'virtual observations,approximate amplitude modulation;no exact divergence-free claim'})
    templates=library();forecasts={m:[] for m in METHODS};registry=[]
    for q in rows():
        row,own,C,params,initial,clean,noisy=context(q)
        nearest,fn,mixed,fm=templates_for(row,templates)
        for method in METHODS:
            raw,fr=(nearest,fn) if 'nearest' in method else ((mixed,fm) if 'aligned4' in method else (C,params[1]))
            pp=fit_startup(raw,row,fr,False)[0] if raw is not C else params
            coeff=correction(raw,row,pp,'hybrid8_pH1')
            count=1 if method.startswith('gain1') else 4
            pred,detail=gain_forecast(coeff,fr,row,noisy if 'noise' in method else clean,pp,count)
            pred=hybrid(own,pred);forecasts[method].append((q,row,pred))
            registry.append({'id':q['id'],'method':method,
                'prediction_bytes_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'diagnostic':detail})
        save(GAIN/'status.json',{'utc':now(),'status':'predicting','completed_cases':len(registry)//len(METHODS),'total':56})
    save(GAIN/'forecasts_frozen.json',{'utc':now(),'all_before_future_scoring':True,'registry':registry})
    results={}
    for method,part in forecasts.items():
        cases=[score(q,pred,row) for q,row,pred in part]
        results[method]={'summary':summaries(cases),'cases':cases}
    pilot=json.loads((PILOT/'report.json').read_text());ref=pilot['results'][pilot['selected']['method']]['summary']
    short=json.loads((OUT/'final_research_confirmation/outer96_seed622_development.json').read_text())['summary']
    candidates=[{'method':m,'balanced_R40':balanced(results[m]['summary']),
        'eligible':candidate_guard(results[m]['summary'],ref,short)} for m in ['gain1_neural','gain4_neural']]
    eligible=[a for a in candidates if a['eligible']]
    chosen=min(eligible,key=lambda a:(a['balanced_R40'],a['method'])) if eligible else None
    signal=False
    if chosen:
        selected=results[chosen['method']]['summary'];s=selected['development_all56']['40s']
        noise=results[chosen['method']+'_noise1pct']['summary']['development_all56']['40s']
        baseline=min(balanced(results[m]['summary']) for m in ['gain4_nearest','gain4_aligned4'])
        signal=(s['n_pass']>=48 and chosen['balanced_R40']<=.9*balanced(ref)
            and chosen['balanced_R40']<=.8*baseline and noise['n_pass']>=s['n_pass']-2)
    save(GAIN/'report.json',{'utc':now(),'results':results,'candidates':candidates,
        'selected':chosen,'research_signal_passed':signal,'fallback_unchanged':True})
    protected();save(GAIN/'status.json',{'utc':now(),'status':'complete','signal':signal,
        'wall_seconds':time.perf_counter()-begin,'selected':chosen})
    for m,a in results.items():
        s=a['summary']['development_all56']['40s']
        print('GAIN',m,s['n_pass'],round(s['mean_R'],5),round(s['max_R'],5),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        GAIN.mkdir(exist_ok=True);save(GAIN/'failure.json',{'utc':now(),'error':repr(e)});raise
