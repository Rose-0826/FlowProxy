#!/usr/bin/env python3
"""Known validation only: explicit observation budget, Fourier order, and train-only frequency priors."""
from pathlib import Path
import sys,json,datetime,time
import numpy as np
import h5py
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import sha,save
from round2_complete40_v1 import frame_summary,summarize
OUT=ROOT/'outputs/startup_observability_v1'
DATA=ROOT/'data/processed/dataset_long40_v1.h5'


def basis(t,f,H):
    return np.column_stack([np.ones(len(t))]+[a for k in range(1,H+1) for a in [np.cos(2*np.pi*k*f*t),np.sin(2*np.pi*k*f*t)]])


def frequency(t,Y,H,center,radius):
    Y=np.asarray(Y,np.float64);G=Y@Y.T;total=np.trace(G)
    def loss(f):
        Q=np.linalg.qr(basis(t,f,H),mode='reduced')[0]
        return float(max(0,total-np.sum(Q*(G@Q)))/max(total,1e-20))
    grid=np.linspace(center*(1-radius),center*(1+radius),61)
    values=np.array([loss(f) for f in grid]);best=int(values.argmin())
    lo=grid[max(0,best-1)];hi=grid[min(len(grid)-1,best+1)]
    opt=minimize_scalar(loss,bounds=(lo,hi),method='bounded',options={'xatol':1e-9})
    return float(opt.x),{'residual':float(opt.fun),'at_search_boundary':bool(best in [0,60])}


def vector(g):
    return [float(g.attrs['Re'])/200,float(g.attrs['cy'])/.8,float(g.attrs['D'])]


def run():
    P=json.loads((ROOT/'outputs/geometry_attack_v2/preregister.json').read_text())
    OUT.mkdir(exist_ok=False)
    save(OUT/'preregister.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'role':'diagnostic on original val8, never fresh8','observed_frames':[4,8,16,24,32],
        'orders':[1,2,4],'frequency':'train80 full-window Fourier fit H4; shape mean or per-shape RBF prior; startup-only frequency identification within +-10% prior',
        'train_prior_smoothing':1e-6,'frequency_scan':61,'horizons':[8,20],
        'history_cost':'all M frames are true observations; future never read by fit; FNO4 comparison requires matching start',
        'no_cross_case_modal_learning':True,'not_yet_AI_advantage':True,'script_sha256':sha(__file__)})
    priors=[];models={};rows=[];started=time.perf_counter()
    with h5py.File(DATA,'r') as h:
        for cid in P['train_ids']:
            g=h['cases'][cid];force=g['forces'][...];dt=float(np.median(np.diff(force[:,0])))
            spectrum=abs(np.fft.rfft(force[:,2]-force[:,2].mean()));spectrum[0]=0
            center=float(np.fft.rfftfreq(len(force),dt)[spectrum.argmax()])
            t=g['times'][:161].astype(np.float64);t-=t[0]
            Y=g['fields'][:161,1][:,g['mask'][...]>0.5].astype(np.float64)
            f,diagnostic=frequency(t,Y,4,center,.15)
            D=float(g.attrs['D']);priors.append({'id':cid,'shape':str(g.attrs['shape']),'X':vector(g),'St':f*D/P['config']['physics']['U_inf'],'fit':diagnostic})
        save(OUT/'training_priors.json',priors)
        for shape in ['cylinder','square']:
            selected=[p for p in priors if p['shape']==shape]
            X=np.array([p['X'] for p in selected]);y=np.array([p['St'] for p in selected])
            models[shape]=(RBFInterpolator(X,y,kernel='thin_plate_spline',smoothing=1e-6),float(y.mean()))
        for cid in P['val_ids']:
            g=h['cases'][cid];times=g['times'][...].astype(np.float64);mask=g['mask'][...]>0.5;allfields=g['fields'][...].astype(np.float64)
            shape=str(g.attrs['shape']);D=float(g.attrs['D']);rbf,mean=models[shape]
            st=float(rbf(np.array([vector(g)]))[0]);fprior=st*P['config']['physics']['U_inf']/D
            fmean=mean*P['config']['physics']['U_inf']/D
            for M in [4,8,16,24,32]:
                t=times[:M]-times[M-1];Y=allfields[:M].reshape(M,-1)
                Uy=allfields[:M,1][:,mask]
                for H in [1,2,4]:
                    if M<2*H+2:continue
                    fits=[('RBF_prior',fprior,{}),('shape_mean_prior',fmean,{})]
                    f,diag=frequency(t,Uy,H,fprior,.10);fits.append(('startup_identified',f,diag))
                    for method,f,diag in fits:
                        B=basis(t,f,H);coef=np.linalg.lstsq(B,Y,rcond=1e-10)[0]
                        for sec,N in [(8,32),(20,80)]:
                            future_t=times[M:M+N]-times[M-1]
                            pred=(basis(future_t,f,H)@coef).reshape(N,3,64,64)
                            truth=allfields[M:M+N];diff=(pred-truth)*mask[None,None]
                            norm=np.linalg.norm(truth*mask[None,None],axis=(-2,-1))
                            err=np.linalg.norm(diff,axis=(-2,-1))/np.maximum(norm,1e-8)
                            metric=frame_summary(err,float(times[M-1]),float(times[M+N-1]),sec,True)
                            rows.append({'id':cid,'M':M,'H':H,'method':method,'frequency':f,'frequency_diagnostic':diag,
                                         'condition_number':float(np.linalg.cond(B)),'horizon':sec,**metric})
            print('OBSERVABILITY',cid,round(time.perf_counter()-started,1),flush=True)
    keys=sorted(set((r['M'],r['H'],r['method'],r['horizon']) for r in rows));summary={}
    for M,H,method,sec in keys:
        group=[r for r in rows if (r['M'],r['H'],r['method'],r['horizon'])==(M,H,method,sec)]
        summary[f'M{M}_H{H}_{method}_{sec}s']=summarize(group)
    save(OUT/'report.json',{'cases':rows,'summary':summary,'seconds':time.perf_counter()-started})
    save(OUT/'summary.json',summary);save(OUT/'status.json',{'status':'complete'})
    print('OBSERVABILITY_COMPLETE',flush=True)


if __name__=='__main__':run()
