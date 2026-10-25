#!/usr/bin/env python3
"""Fixed, post-hoc controls for the nominated hybrid. They cannot replace its frozen result."""
from pathlib import Path
import sys,json,datetime
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from self_trajectory_closure_v1 import closure,build_priors,metrics,P,DATA
from geometry_attack_v2 import sha,save,attrs,NativeSampler
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/hybrid_controls_v1'


def dmd_periodic(observed,times,future,mask,rank):
    scales=np.array(P['normalization']['channel_rms'])
    fields=observed.astype(np.float64)*mask[None,None]
    Y=(fields/scales[None,:,None,None]).reshape(len(fields),-1).T
    U,s,Vh=np.linalg.svd(Y[:,:-1],full_matrices=False)
    r=min(rank,int(np.count_nonzero(s>s[0]*1e-9)));U=U[:,:r];s=s[:r];V=Vh[:r].T
    AV=(Y[:,1:]@V)/s[None];eig,W=np.linalg.eig(U.T@AV);Phi=AV@W
    dt=float(np.mean(np.diff(times)));angular=np.angle(eig)/dt
    relative=np.asarray(times)-times[0];E=np.exp(1j*relative[:,None]*angular[None])
    gram=(Phi.conj().T@Phi)*(E.conj().T@E)
    rhs=np.sum(E.conj().T*(Phi.conj().T@Y),axis=1)
    coefficients=np.linalg.lstsq(gram,rhs,rcond=1e-10)[0]
    Ef=np.exp(1j*(np.asarray(future)-times[0])[:,None]*angular[None])
    pred=(Ef@(Phi*coefficients[None]).T).real.reshape(len(future),3,64,64)*scales[None,:,None,None]
    return pred.astype(np.float32),{'rank':r,'angular_frequencies':angular.tolist(),'raw_eigen_moduli':abs(eig).tolist(),
                                   'amplitudes':'fixed; eigenvalues projected to unit circle; coefficients fitted to supplied startup only'}


def controls(q,support4,base,times,mask,prior,configuration):
    N=configuration['bootstrap_steps'];observed=np.concatenate([support4,base[:N]])*mask[None,None]
    _,diagnostic=closure(q,support4,base,times,mask,prior,configuration)
    period=1/diagnostic['frequency'];observed_times=times[:4+N]
    phase_times=observed_times[-1]-np.mod(observed_times[-1]-times[4+N:164],period)
    hi=np.searchsorted(observed_times,phase_times).clip(1,len(observed_times)-1);lo=hi-1
    alpha=(phase_times-observed_times[lo])/(observed_times[hi]-observed_times[lo])
    replay=base.copy();replay[N:]=(1-alpha[:,None,None,None])*observed[lo]+alpha[:,None,None,None]*observed[hi]
    dmd,dd=dmd_periodic(observed,observed_times,times[4:164],mask,9)
    dmd[:N]=base[:N]
    startup,ds=dmd_periodic(support4,times[:4],times[4:164],mask,3)
    return {'cycle_replay':(replay,{'frequency':diagnostic['frequency'],'cycle':'last generated period, linear interpolation'}),
            'self_DMD_rank9':(dmd,dd),'real4_DMD_rank3':(startup,ds)}


def run():
    OUT.mkdir(exist_ok=False)
    configuration=json.loads((ROOT/'outputs/self_trajectory_closure_v1/selected_val.json').read_text())['candidate']
    save(OUT/'protocol.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'posthoc_after_fresh8_scores':True,
        'no_parameter_search':True,'four_true_fields_only':True,'bootstrap_steps':configuration['bootstrap_steps'],
        'ranks':[3,9],'primary_configuration_unchanged':True,'no_retrospective_primary_replacement':True,'script_sha256':sha(__file__)})
    prior=build_priors();results={}
    for role,folder,ids in [('val','geometry_attack_evaluation_v1/v1_bank/val',P['val_ids']),
                            ('old32_seen','geometry_attack_evaluation_v1/v2_bank/old32_seen',P['old_seen_test_ids']),
                            ('fresh_diameter8','geometry_attack_confirmation_v1',None),
                            ('fresh_fixed_D1_4','geometry_attack_confirmation_d1_v1',None)]:
        directory=ROOT/'outputs'/folder;isnew=ids is None
        if isnew:
            ids=[r['id'] for r in json.loads((directory/'preregister.json').read_text())['cases']]
            rows=json.loads((directory/'preregister.json').read_text())['cases'];lookup={r['id']:r for r in rows}
            prediction_directory=directory/'evaluation/predictions'
        else:prediction_directory=directory/'predictions'
        report={name:[] for name in ['cycle_replay','self_DMD_rank9','real4_DMD_rank3']}
        with h5py.File(DATA,'r') as h:
            for cid in ids:
                if isnew:
                    q={k:lookup[cid][k] for k in ['shape','Re','cx','cy','D']};geo=geometry_channels(P['config'],**q);mask=geo['mask']
                    items=frames(ROOT/'data/raw'/folder/cid);times=np.array([t for t,_ in items]);sampler=NativeSampler(geo['xx'],geo['yy'],mask)
                    support4=np.stack([sampler.sample(path) for _,path in items[:4]])
                else:
                    g=h['cases'][cid];q=attrs(g);times=g['times'][...];mask=g['mask'][...];support4=g['fields'][:4]
                base=np.load(prediction_directory/f'K8__{cid}.npz')['prediction']
                predictions=controls(q,support4,base,times,mask,prior,configuration)
                if isnew:truth=np.stack([sampler.sample(path) for _,path in items[4:164]])
                else:truth=h['cases'][cid]['fields'][4:164]
                for name,(pred,diagnostic) in predictions.items():
                    report[name].append({'id':cid,'horizons':metrics(pred,truth,mask,times),'diagnostic':diagnostic})
        summary={name:{key:summarize([r['horizons'][key] for r in rows]) for key in ['2s','8s','20s','40s']} for name,rows in report.items()}
        results[role]={'cases':report,'summary':summary}
        print('CONTROLS',role,{name:(round(s['40s']['mean_R'],4),s['40s']['n_pass']) for name,s in summary.items()},flush=True)
    save(OUT/'report.json',results);save(OUT/'summary.json',{k:v['summary'] for k,v in results.items()});save(OUT/'status.json',{'status':'complete'})


if __name__=='__main__':run()
