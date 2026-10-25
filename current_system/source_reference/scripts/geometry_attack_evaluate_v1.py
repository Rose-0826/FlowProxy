#!/usr/bin/env python3
"""Frozen-candidate inference. Fresh-reference future truth is read after all predictions are saved."""
from pathlib import Path
import argparse,copy,json,sys,time,datetime
import numpy as np
import h5py
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import mapping,NativeSampler,sha,save,attrs
from flowproxy.geometry import geometry_channels
from flowproxy.models import build_model
from flowproxy.round2.metrics import per_frame_rel
from round2_complete40_v1 import frame_summary,summarize
from cpu_data_prepare import frames
DATA=ROOT/'data/processed/dataset_long40_v1.h5'
P=json.loads((ROOT/'outputs/geometry_attack_v2/preregister.json').read_text())
CFG=P['config']


def specs():
    result=[{'name':'F0','version':'frozen','mode':'control','path':str(ROOT/'outputs/checkpoints/fno_long/best.pt')},
            {'name':'K8','version':'frozen','mode':'control','path':str(ROOT/'outputs/checkpoints/fno_k8_base/base.pt')},
            {'name':'K8_solid_padding','version':'frozen_mask','mode':'control','path':str(ROOT/'outputs/checkpoints/fno_k8_base/base.pt')}]
    for version,folder in [('v1','geometry_attack_v1'),('v2','geometry_attack_v2')]:
        for mode in ['control','scale','warp64','warp_native']:
            for seed in [11,22]:
                directory=ROOT/'outputs'/folder/f'{mode}_seed{seed}'
                if (directory/'done.json').exists():
                    result.append({'name':f'{version}_{mode}_{seed}','version':version,'mode':mode,
                                   'path':str(directory/'selected.pt'),'validation_report':str(directory/'selected_eval.json')})
    for seed in [11,22]:
        directory=ROOT/'outputs/uniform128_attack_v1'/f'control_seed{seed}'
        if (directory/'done.json').exists():
            result.append({'name':f'uniform128_{seed}','version':'uniform128','mode':'control',
                           'path':str(directory/'selected.pt'),'validation_report':str(directory/'selected_eval.json')})
    closure_path=ROOT/'outputs/self_trajectory_closure_v1/selected_val.json'
    if closure_path.exists():
        candidate=json.loads(closure_path.read_text())['candidate']
        base='outputs/checkpoints/fno_k8_base/base.pt' if candidate['base_model']=='K8' else 'outputs/checkpoints/fno_long/best.pt'
        result.append({'name':'neural_periodic_closure','version':'closure','mode':'control','path':str(ROOT/base),
                       'closure_config':candidate,'closure_config_sha256':sha(closure_path),'validation_report':str(closure_path)})
        result.append({'name':'fourframe_harmonic','version':'harmonic','mode':'control',
                       'path':str(ROOT/'outputs/startup_observability_v1/training_priors.json'),
                       'prior':'shape mean St from train80; H1 least squares, exactly four true fields'})
    return result


class Candidate:
    def __init__(self,spec):
        self.spec=spec;self.mode=spec['mode'];self.version=spec['version'];self.device='cuda'
        if self.version=='harmonic':self.model=None;return
        saved=torch.load(spec['path'],map_location='cpu',weights_only=False)
        self.model=build_model(saved['model_name'],saved['in_channels'],saved['out_channels'],CFG).to(self.device)
        self.model.load_state_dict(saved['model']);self.model.eval();self.capture={}
        self.head=None
        if 'query_head' in saved:
            self.head=torch.nn.Sequential(torch.nn.Conv2d(75,32,1),torch.nn.GELU(),torch.nn.Conv2d(32,3,1)).to(self.device)
            self.head.load_state_dict(saved['query_head']);self.head.eval()
            self.hook=self.model.project[1].register_forward_hook(lambda module,inputs,output:self.capture.__setitem__('latent',output))

    def t(self,a):return torch.from_numpy(np.asarray(a,np.float32)).to(self.device)

    @torch.inference_mode()
    def predict(self,q,uniform4,times,raw):
        physical_times=np.asarray(times,np.float64)
        if self.version=='harmonic':
            from self_trajectory_closure_v1 import build_priors
            from startup_observability_v1 import basis
            begin=time.perf_counter();geo=geometry_channels(CFG,**q)
            f=build_priors()[q['shape']][1]*CFG['physics']['U_inf']/q['D']
            relative=physical_times-physical_times[3]
            observations=np.asarray(uniform4,np.float64)*geo['mask'][None,None]
            coefficients=np.linalg.lstsq(basis(relative[:4],f,1),observations.reshape(4,-1),rcond=1e-10)[0]
            result=(basis(relative[4:164],f,1)@coefficients).reshape(160,3,64,64).astype(np.float32)
            return result,time.perf_counter()-begin
        uniform=geometry_channels(CFG,**q);m=mapping(CFG,q)
        if self.version=='uniform128':
            cfg128=copy.deepcopy(CFG);cfg128['grid']={'nx':128,'ny':128};spatial=geometry_channels(cfg128,**q)
        elif self.mode.startswith('warp'):spatial=m
        else:spatial=uniform
        n=spatial['mask'].shape[0];geom=self.t(spatial['geom'])[None]
        state_mask=self.t(spatial['mask'])[None,None];mask=self.t(uniform['mask'])[None,None]
        backward=self.t(m['backward'])[None];forward=self.t(m['forward']).expand(4,-1,-1,-1)
        if self.mode=='warp_native' or self.version=='uniform128':
            sampler=NativeSampler(spatial['xx'],spatial['yy'],spatial['mask'])
            items=frames(raw);hist=self.t(np.stack([sampler.sample(path) for _,path in items[:4]]))[None]
        else:
            hist=self.t(uniform4)[None]
            if self.mode=='warp64':hist=F.grid_sample(hist[0],forward,align_corners=True,padding_mode='border')[None]
        if self.version in ['v2','uniform128','frozen_mask']:hist=hist*state_mask[:,None]
        coord=self.t(np.stack([(spatial['xx']-q['cx'])/(20*q['D']),
                    (spatial['yy']-q['cy'])/(8*q['D']),
                    spatial.get('sdf_wall',(CFG['domain']['y_max']-CFG['domain']['y_min'])/2-np.abs(spatial['yy']))/(8*q['D'])]))[None]
        if self.mode.startswith('warp'):coord=self.t(m['coords'])[None]
        query_coords=self.t(np.stack([(uniform['xx']-q['cx'])/(20*q['D']),
            (uniform['yy']-q['cy'])/(8*q['D']),uniform['sdf_wall']/(8*q['D']),uniform['sdf_obs']/q['D'],
            np.full((64,64),q['Re']/200,dtype=np.float32)]))[None]
        times=self.t(times);diameter=self.t(np.array([q['D']],np.float32));pred=[];start=time.perf_counter()
        steps=self.spec['closure_config']['bootstrap_steps']+self.spec['closure_config']['blend_steps'] if self.version=='closure' else 160
        for j in range(steps):
            dt=times[j+4]-times[j+3]
            x=torch.cat([geom,hist.reshape(1,12,n,n)],1)
            constants=torch.stack([diameter-1,dt*CFG['physics']['U_inf']/(.25*diameter)-1],1)
            if self.mode!='control':x=torch.cat([x,coord,constants[:,:,None,None].expand(-1,-1,n,n)],1)
            raw_pred=self.model(x)[:,:3]
            v=raw_pred if self.mode=='control' else hist[:,-1]+(dt*CFG['physics']['U_inf']/(.25*diameter))[:,None,None,None]*(raw_pred-hist[:,-1])
            latent=self.capture.pop('latent') if self.head is not None else None
            if self.version in ['v2','uniform128','frozen_mask']:v=v*state_mask
            hist=torch.cat([hist[:,1:],v[:,None]],1)
            if self.version=='uniform128':
                base=F.interpolate(v,size=(64,64),mode='bilinear',align_corners=True)
                valid=F.interpolate(state_mask,size=(64,64),mode='bilinear',align_corners=True)
                features=F.interpolate(latent,size=(64,64),mode='bilinear',align_corners=True)
            elif self.mode.startswith('warp'):
                base=F.grid_sample(v,backward,align_corners=True,padding_mode='border')
                valid=F.grid_sample(state_mask,backward,align_corners=True,padding_mode='border')
                features=F.grid_sample(latent,backward,align_corners=True,padding_mode='border') if latent is not None else None
            else:base=v;features=latent;valid=mask
            if self.head is not None:
                if self.mode.startswith('warp') or self.version=='uniform128':base=torch.cat([base[:,:2],base[:,2:3]/valid.clamp_min(1e-6)],1)
                query_code=torch.cat([query_coords,constants[:,:,None,None].expand(-1,-1,64,64)],1)
                base=(base+self.head(torch.cat([features,base,query_code,valid],1)))*mask
            pred.append(base[0])
        torch.cuda.synchronize()
        result=torch.stack(pred).cpu().numpy()
        if self.version=='closure':
            from self_trajectory_closure_v1 import closure,build_priors
            full=np.zeros((160,3,64,64),np.float32);full[:len(result)]=result
            result,diagnostic=closure(q,uniform4,full,physical_times,uniform['mask'],build_priors(),self.spec['closure_config'])
        if not np.isfinite(result).all():raise ValueError('Nonfinite prediction')
        return result,time.perf_counter()-start


def run(role,names,tag='final',fresh_folder='geometry_attack_confirmation_v1'):
    torch.set_num_threads(1)
    candidates=specs()
    if names:candidates=[s for s in candidates if s['name'] in names]
    if not candidates:raise ValueError('No candidates')
    fresh=role=='fresh8'
    if fresh:
        directory=ROOT/'outputs'/fresh_folder
        nomination=json.loads((directory/'nomination.json').read_text())
        available={s['name']:s for s in candidates}
        primary=available[nomination['primary']['name']]
        if sha(primary['path'])!=nomination['primary']['checkpoint_sha256']:raise ValueError('Primary hash changed')
        rows=json.loads((directory/'preregister.json').read_text())['cases']
        ids=[x['id'] for x in rows];out=directory/'evaluation';rawroot=ROOT/'data/raw'/fresh_folder
    else:
        ids=P['val_ids'] if role=='val' else P['old_seen_test_ids']
        out=ROOT/'outputs/geometry_attack_evaluation_v1'/tag/role;rawroot=ROOT/'data/raw/cases_long40'
        rows=[]
    out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    supports={}
    if fresh:
        for row in rows:
            q={k:row[k] for k in ['shape','Re','D','cx','cy']};geo=geometry_channels(CFG,**q)
            items=frames(rawroot/row['id']);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
            supports[row['id']]={'q':q,'fields':np.stack([sampler.sample(path) for _,path in items[:4]]),
                'times':np.array([t for t,_ in items]),'mask':geo['mask']}
    else:
        with h5py.File(DATA,'r') as h:
            for cid in ids:
                g=h['cases'][cid];supports[cid]={'q':attrs(g),'fields':g['fields'][:4],
                    'times':g['times'][...],'mask':g['mask'][...]}
    registry=[]
    for spec in candidates:
        candidate=Candidate(spec)
        for cid in ids:
            d=supports[cid];prediction,seconds=candidate.predict(d['q'],d['fields'],d['times'],rawroot/cid)
            path=out/'predictions'/f"{spec['name']}__{cid}.npz"
            np.savez_compressed(path,prediction=prediction)
            registry.append({'model':spec['name'],'id':cid,'file':str(path),'sha256':sha(path),'forecast_and_D2H_seconds':seconds,
                             'checkpoint_sha256':sha(spec['path'])})
        del candidate;torch.cuda.empty_cache();print('PREDICTED',role,spec['name'],flush=True)
    save(out/'predictions_frozen.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fresh_future_truth_not_read':fresh,'rows':registry})
    reports={s['name']:{'spec':s,'cases':[],'max_validation_reproduction_delta':0.0} for s in candidates}
    truth={}
    if fresh:
        for cid in ids:
            d=supports[cid];geo=geometry_channels(CFG,**d['q']);sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask'])
            truth[cid]=np.stack([sampler.sample(path) for _,path in frames(rawroot/cid)[4:164]])
    else:
        with h5py.File(DATA,'r') as h:
            truth={cid:h['cases'][cid]['fields'][4:164] for cid in ids}
    for spec in candidates:
        report=reports[spec['name']]
        original=json.loads(Path(spec['validation_report']).read_text()) if role=='val' and 'validation_report' in spec else None
        for cid in ids:
            prediction=np.load(out/'predictions'/f"{spec['name']}__{cid}.npz")['prediction'];d=supports[cid]
            err=per_frame_rel(torch.from_numpy(prediction),torch.from_numpy(truth[cid]),torch.from_numpy(d['mask'])).numpy()
            hs={f'{sec}s':frame_summary(err[:n],float(d['times'][3]),float(d['times'][n+3]),sec,True) for sec,n in [(2,8),(8,32),(20,80),(40,160)]}
            report['cases'].append({'id':cid,'horizons':hs})
            if original:
                expected=next(x for x in original['cases'] if x['id']==cid)
                delta=max(float(np.abs(np.array(hs[h][k])-expected['horizons'][h][k]).max()) for h in hs for k in ['mean','end','max'])
                report['max_validation_reproduction_delta']=max(report['max_validation_reproduction_delta'],delta)
        report['summary']={h:summarize([x['horizons'][h] for x in report['cases']]) for h in ['2s','8s','20s','40s']}
        if original and report['max_validation_reproduction_delta']>2e-6:raise ValueError('Validation parity failed '+spec['name']+' '+str(report['max_validation_reproduction_delta']))
        print('SCORED',role,spec['name'],round(report['summary']['8s']['mean_R'],4),report['summary']['8s']['n_pass'],flush=True)
    save(out/'report.json',{'role':role,'candidate_count':len(candidates),'all_cases_reported':True,'reports':reports})
    save(out/'summary.json',{k:v['summary'] for k,v in reports.items()})
    save(out/'status.json',{'status':'complete','role':role,'n_cases':len(ids),'n_models':len(candidates)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['val','old32_seen','fresh8'],required=True)
    p.add_argument('--names',nargs='*');p.add_argument('--tag',default='final');p.add_argument('--fresh-folder',default='geometry_attack_confirmation_v1');a=p.parse_args();run(a.role,a.names,a.tag,a.fresh_folder)
