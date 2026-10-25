#!/usr/bin/env python3
"""Post-hoc strong library baseline. Train80 templates; only four target startup fields fit phase/frequency."""
from pathlib import Path
import sys,json,datetime,time
import numpy as np,h5py
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares,minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import sha,save,attrs,NativeSampler
from self_trajectory_closure_v1 import P,DATA,build_priors,metrics,feature
from startup_observability_v1 import basis
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/nearest_cycle_template_v1'


def phase_basis(t,f,phase,H=4):
    z=2*np.pi*f*np.asarray(t)+phase
    B=np.column_stack([np.ones(len(t))]+[a for k in range(1,H+1) for a in [np.cos(k*z),np.sin(k*z)]])
    derivative=np.column_stack([np.zeros(len(t))]+[a for k in range(1,H+1) for a in [-k*np.sin(k*z),k*np.cos(k*z)]])
    return B,derivative


def mapped_coefficients(source,q,cfg):
    geo=geometry_channels(cfg,**q);d=cfg['domain'];s=source['q']
    xx=s['cx']+(geo['xx']-q['cx'])*s['D']/q['D'];yy=s['cy']+(geo['yy']-q['cy'])*s['D']/q['D']
    index=np.array([(yy-d['y_min'])/(d['y_max']-d['y_min'])*63,
                    (xx-d['x_min'])/(d['x_max']-d['x_min'])*63])
    coef=np.stack([map_coordinates(a,index,order=1,mode='nearest',prefilter=False) for a in source['coef'].reshape(27,64,64)]).reshape(9,3,64,64)
    fraction=map_coordinates(source['mask'],index,order=1,mode='nearest',prefilter=False)
    coef[:,2]/=np.maximum(fraction[None],1e-6)
    return coef*geo['mask'][None,None]


def predict(q,support4,times,mask,templates,priors):
    eligible=[s for s in templates if s['q']['shape']==q['shape']]
    def distance(s):return sum(((q[k]-s['q'][k])/scale)**2 for k,scale in [('Re',120),('cy',.8),('D',.2)])
    source=min(eligible,key=lambda s:(distance(s),s['id']));coef=mapped_coefficients(source,q,P['config'])
    active=mask>0.5;scales=np.array(P['normalization']['channel_rms'])
    C=(coef/scales[None,:,None,None])[:,:,active].reshape(9,-1)
    target=(support4/scales[None,:,None,None])[:,:,active].reshape(4,-1)
    t=times[:4]-times[3];center=float(priors[q['shape']][0](feature(q))[0])*P['config']['physics']['U_inf']/q['D']
    def residual(params):return (phase_basis(t,params[1],params[0])[0]@C-target).ravel()
    def jacobian(params):
        der=phase_basis(t,params[1],params[0])[1]@C
        return np.column_stack([der.ravel(),(der*(2*np.pi*t[:,None])).ravel()])
    grid=np.linspace(-np.pi,np.pi,25,endpoint=False);loss=np.array([np.linalg.norm(residual([a,center]))**2 for a in grid]);a=float(grid[loss.argmin()])
    fixed=minimize_scalar(lambda phi:np.linalg.norm(residual([phi,center]))**2,bounds=(a-2*np.pi/25,a+2*np.pi/25),method='bounded',options={'xatol':1e-9})
    optimized=least_squares(residual,[fixed.x,center],jac=jacobian,bounds=([-np.inf,.9*center],[np.inf,1.1*center]),
                            xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=200)
    result={}
    for method,params in [('NN_train_frequency',[fixed.x,center]),('NN_startup_frequency',optimized.x)]:
        B,_=phase_basis(times[4:164]-times[3],params[1],params[0]);pred=(B@coef.reshape(9,-1)).reshape(160,3,64,64).astype(np.float32)
        if not np.isfinite(pred).all():raise ValueError('Nonfinite NN template')
        result[method]=(pred,{'donor':source['id'],'geometry_distance_squared':distance(source),'phase':float(params[0]),
                            'frequency':float(params[1]),'optimizer_success':bool(optimized.success),'at_frequency_bound':bool(abs(params[1]/center-1)>.0999)})
    return result


def run():
    OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'posthoc_after_primary_fresh_results':True,
        'train80_only_library':True,'donor_distance':'same shape, Re/120,cy/.8,D/.2; deterministic ID ties',
        'spatial_map':'translate and D-scale donor to query grid; border-clamped, fluid-only pressure normalization',
        'H':4,'phase_initialization':25,'frequency_bound':'+-10% train-only RBF prior','observed_target_frames':4,
        'no_retrospective_primary_replacement':True,'script_sha256':sha(__file__)})
    priors=build_priors();freqs={r['id']:r['St'] for r in json.loads((ROOT/'outputs/startup_observability_v1/training_priors.json').read_text())};templates=[];results={};start=time.perf_counter()
    with h5py.File(DATA,'r') as h:
        for cid in P['train_ids']:
            g=h['cases'][cid];q=attrs(g);t=g['times'][:161].astype(np.float64);t-=t[0]
            f=freqs[cid]*P['config']['physics']['U_inf']/q['D'];fields=g['fields'][:161].astype(np.float64)
            coef=np.linalg.lstsq(basis(t,f,4),fields.reshape(161,-1),rcond=1e-10)[0].reshape(9,3,64,64)
            templates.append({'id':cid,'q':q,'coef':coef,'mask':g['mask'][...]})
        for role,folder,ids in [('val','',P['val_ids']),('old32_seen','',P['old_seen_test_ids']),
                              ('fresh_diameter8','geometry_attack_confirmation_v1',None),('fresh_fixed_D1_4','geometry_attack_confirmation_d1_v1',None)]:
            isnew=ids is None
            if isnew:
                rows=json.loads((ROOT/'outputs'/folder/'preregister.json').read_text())['cases'];lookup={r['id']:r for r in rows};ids=list(lookup)
            report={name:[] for name in ['NN_train_frequency','NN_startup_frequency']}
            for cid in ids:
                if isnew:
                    q={k:lookup[cid][k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(P['config'],**q);mask=geo['mask']
                    items=frames(ROOT/'data/raw'/folder/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],mask)
                    support4=np.stack([sampler.sample(path) for _,path in items[:4]])
                else:
                    g=h['cases'][cid];q=attrs(g);times=g['times'][...];mask=g['mask'][...];support4=g['fields'][:4]
                predictions=predict(q,support4,times,mask,templates,priors)
                truth=np.stack([sampler.sample(path) for _,path in items[4:164]]) if isnew else h['cases'][cid]['fields'][4:164]
                for method,(pred,diagnostic) in predictions.items():report[method].append({'id':cid,'horizons':metrics(pred,truth,mask,times),'diagnostic':diagnostic})
            summary={name:{key:summarize([r['horizons'][key] for r in rows]) for key in ['2s','8s','20s','40s']} for name,rows in report.items()}
            results[role]={'cases':report,'summary':summary};print('NN_TEMPLATE',role,{name:(round(d['40s']['mean_R'],4),d['40s']['n_pass']) for name,d in summary.items()},flush=True)
    save(OUT/'report.json',results);save(OUT/'summary.json',{r:d['summary'] for r,d in results.items()});save(OUT/'status.json',{'status':'complete','seconds':time.perf_counter()-start})


if __name__=='__main__':run()
