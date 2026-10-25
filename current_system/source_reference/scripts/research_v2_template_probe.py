#!/usr/bin/env python3
"""Corrected train80 periodic library, gauge-aligned transport and four-frame calibration."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar,minimize
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from research_v2_common import OUT,rows,startup,truth,scales,score,summaries,protected,now
from geometry_attack_v2 import save,sha
from cycle_landmarks_train_v1 import PERIODS

METHODS=['NN_donor_clock','NN_startup_clock','aligned4_donor_clock',
    'aligned4_startup_clock','aligned4_startup_coefficients']

def basis(t,f,phase=0):
    angle=2*np.pi*f*np.asarray(t)+phase
    return np.column_stack([np.ones(len(t))]+[v for k in range(1,5) for v in [np.cos(k*angle),np.sin(k*angle)]])

def train_library():
    freq={q['id']:q['frequency'] for q in json.loads(PERIODS.read_text())['rows']}
    templates=[]
    for q in [q for q in rows(True) if q['group']=='train80']:
        r=startup(q);fields=truth(q);t=r['times']-r['times'][0];f=freq[q['id']]
        coef=np.linalg.lstsq(basis(t,f),fields.reshape(len(t),-1),rcond=1e-10)[0].reshape(9,3,64,64)
        templates.append({'q':q,'coef':coef,'mask':r['mask'],'f':f})
    assert len(templates)==80
    return templates

def mapped(s,q,mask):
    xx,yy=np.meshgrid(np.linspace(-8,20,64),np.linspace(-8,8,64))
    a=s['q'];x=a['cx']+(xx-q['cx'])*a['D']/q['D'];y=a['cy']+(yy-q['cy'])*a['D']/q['D']
    index=np.array([(y+8)*63/16,(x+8)*63/28])
    coef=np.stack([map_coordinates(v,index,order=1,mode='nearest',prefilter=False)
        for v in s['coef'].reshape(27,64,64)]).reshape(9,3,64,64)
    fraction=map_coordinates(s['mask'],index,order=1,mode='nearest',prefilter=False)
    # Avoid pressure pollution by obstacle placeholder zeros near mapped boundary.
    coef[:,2]/=np.maximum(fraction[None],1e-6)
    return coef*mask[None,None]

def gauge_align(coef,reference,mask):
    scale=np.asarray(scales())[:,None,None];active=mask>.5
    z=(coef[1]-1j*coef[2])/scale
    zr=(reference[1]-1j*reference[2])/scale
    angle=float(np.angle(np.vdot(z[:,active],zr[:,active])))
    result=coef.copy()
    for k in range(1,5):
        z=coef[2*k-1]-1j*coef[2*k]
        rotated=z*np.exp(1j*k*angle)
        result[2*k-1]=rotated.real;result[2*k]=-rotated.imag
    return result,angle

def fit_startup(coef,row,f,free_clock):
    mask=row['mask']>.5;scale=np.asarray(scales())[None,:,None,None]
    C=(coef/scale)[:,:,mask].reshape(9,-1)
    Y=(row['fields']/scale)[:,:,mask].reshape(4,-1)
    gram=C@C.T;cross=Y@C.T;norm=np.sum(Y*Y);t=row['times'][:4]-row['times'][3]
    def objective(p):
        B=basis(t,p[1],p[0]);return float(np.sum((B@gram)*B)-2*np.sum(B*cross)+norm)
    grid=np.linspace(-np.pi,np.pi,25,endpoint=False)
    angle=float(min(grid,key=lambda v:objective([v,f])))
    fitted=minimize_scalar(lambda phi:objective([phi,f]),bounds=(angle-2*np.pi/25,angle+2*np.pi/25),method='bounded',options={'xatol':1e-8})
    params=np.array([fitted.x,f]);success=bool(fitted.success)
    if free_clock:
        opt=minimize(objective,params,method='L-BFGS-B',bounds=[(params[0]-np.pi,params[0]+np.pi),(.9*f,1.1*f)],
            options={'maxiter':100,'ftol':1e-13,'gtol':1e-8})
        if np.isfinite(opt.fun) and opt.fun<=objective(params):params=opt.x
        success=bool(opt.success)
    return params,{'phase':float(params[0]),'frequency':float(params[1]),
        'startup_scaled_MSE':float(objective(params)/Y.size),'optimizer_success':success,
        'frequency_at_bound':bool(abs(params[1]/f-1)>.0999)}

def calibrate(coef,row,params):
    t=row['times'][:4]-row['times'][3];B=basis(t,params[1],params[0]);A=B[:,:3]
    residual=row['fields'].reshape(4,-1)-B@coef.reshape(9,-1)
    G=A.T@A;lam=.03*np.trace(G)/3
    delta=np.linalg.solve(G+lam*np.diag([.5,1,1]),A.T@residual)
    result=coef.copy();result[:3]+=delta.reshape(3,3,64,64)
    return result*row['mask'][None,None]

def forecast(coef,row,params):
    t=row['times'][4:164]-row['times'][3]
    return (basis(t,params[1],params[0])@coef.reshape(9,-1)).reshape(160,3,64,64).astype(np.float32)

def run():
    protected();out=OUT/'template_probe';out.mkdir(exist_ok=False);(out/'predictions').mkdir();begin=time.perf_counter()
    save(out/'configuration.json',{'utc':now(),'source_sha256':sha(__file__),'methods':METHODS,
        'library':'corrected train80 ONLY; H4 with train-only field-fitted frequency',
        'distance':'same shape; Euclidean Re/120,cy/.8,D/.2; deterministic ID tie',
        'spatial_map':'body translation and D scale, bilinear, border-clamped; finite-wall transport approximate',
        'four_donors':'inverse squared-distance weights; align complex H1 phase, rotate harmonic k by k*phase',
        'target_data':'ONLY4 startup fields, metadata and mask; no target future clock or mode',
        'clock':'donor St divided by query D; startup refinement bounded+-10%',
        'calibration':'ridge-adjust mean and H1 coefficients using4 true fields; fixed lambda .03*trace/3',
        'role':'all56 ALREADY-SEEN DEVELOPMENT; comparison baseline, no novelty claim for parametric template interpolation'})
    templates=train_library();registry=[]
    for q in rows():
        row=startup(q)
        def distance(s):return sum(((q[k]-s['q'][k])/v)**2 for k,v in [('Re',120),('cy',.8),('D',.2)])
        chosen=sorted([s for s in templates if s['q']['shape']==q['shape']],key=lambda s:(distance(s),s['q']['id']))[:4]
        coefficients=[mapped(s,q,row['mask']) for s in chosen]
        weights=1/(np.array([distance(s) for s in chosen])+1e-6);weights/=weights.sum()
        aligned=[coefficients[0]];angles=[0.]
        for c in coefficients[1:]:
            z,a=gauge_align(c,coefficients[0],row['mask']);aligned.append(z);angles.append(a)
        blend=np.einsum('s,skchw->kchw',weights,np.stack(aligned))
        f_nn=chosen[0]['f']*chosen[0]['q']['D']/q['D']
        f_blend=sum(w*s['f']*s['q']['D'] for w,s in zip(weights,chosen))/q['D']
        for method in METHODS:
            coef=coefficients[0] if method.startswith('NN') else blend
            center=f_nn if method.startswith('NN') else f_blend
            free='startup_clock' in method or method.endswith('coefficients')
            params,diag=fit_startup(coef,row,center,free)
            if method.endswith('coefficients'):coef=calibrate(coef,row,params)
            pred=forecast(coef,row,params);assert np.isfinite(pred).all()
            file=out/'predictions'/(method+'__'+q['id']+'.npz');np.savez_compressed(file,prediction=pred)
            registry.append({'id':q['id'],'method':method,'sha256':sha(file),'donors':[s['q']['id'] for s in chosen],
                'weights':weights.tolist(),'gauge_angles':angles,'fit':diag})
        save(out/'status.json',{'status':'predicting','completed_cases':len(registry)//len(METHODS),'total_cases':56,'utc':now()})
    save(out/'predictions_frozen.json',{'utc':now(),'all_predictions_before_target_future_scoring':True,'registry':registry})
    report={}
    for method in METHODS:
        cases=[]
        for q in rows():
            with np.load(out/'predictions'/(method+'__'+q['id']+'.npz')) as a:cases.append(score(q,a['prediction'],startup(q)))
        report[method]={'summary':summaries(cases),'cases':cases}
    protected();save(out/'report.json',report)
    save(out/'status.json',{'status':'complete','utc':now(),'wall_seconds':time.perf_counter()-begin})
    for m,a in report.items():
        print('TEMPLATE',m,flush=True)
        for g in ['development_inside24','seen_extreme32']:print(g,round(a['summary'][g]['40s']['mean_R'],5),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:
        save(OUT/'template_probe/failure.json',{'utc':now(),'error':repr(e)});raise
