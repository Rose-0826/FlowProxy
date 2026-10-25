#!/usr/bin/env python3
"""Local convective phase hypothesis; diagnostic-oracle separate from causal trials."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.optimize import least_squares
ROOT=Path('/root/autodl-tmp/flowproxy');sys.path[:0]=[str(ROOT/'work/spatial_observer_20261005'),str(ROOT/'scripts'),str(ROOT/'src')]
import pilot as p
from research_v4_sparse_observer import weights,constrain
from research_v5_split_observer import rotate
DEST=ROOT/'outputs/phase_transport_20261005'

def spatial_basis(q,mask,rank):
 xi=(p.X-q['cx'])/q['D'];on=np.clip((xi-.5)/1.5,0.,1.)
 raw=np.stack([on*np.clip(xi/8.,0.,3.),on*np.clip(xi/8.,0.,3.)**2])[:rank]
 # Unit a represents 10deg local slope or5deg curvature; amplitude prior fixed.
 raw*=np.asarray([np.deg2rad(10.),np.deg2rad(5.)])[:rank,None,None]
 raw[:,mask<=.5]=0.
 return raw

def warp(C,G,a,offset=0.):
 ph=offset+np.einsum('r,rhw->hw',np.asarray(a),G);out=np.asarray(C,float).copy()
 for k in range(1,5):
  cc=np.cos(k*ph);ss=np.sin(k*ph)
  out[2*k-1]=C[2*k-1]*cc[None]+C[2*k]*ss[None]
  out[2*k]=-C[2*k-1]*ss[None]+C[2*k]*cc[None]
 return out

def oracle_fit(C,T,q,mask,rank):
 G=spatial_basis(q,mask,rank);active=mask>.5;sw=p.HWEIGHT[:,0,0,0,None];den=np.linalg.norm(T[:,1,active]*sw)
 # Uy is the identified original velocity bottleneck. Pressure changes reported independently.
 def residual(v):return ((warp(C,G,v[1:],v[0])-T)[:,1,active]*sw).ravel()
 sol=least_squares(residual,np.zeros(1+rank),bounds=(np.r_[-np.pi,np.full(rank,-2.)],np.r_[np.pi,np.full(rank,2.)]),max_nfev=50,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 fit=warp(C,G,sol.x[1:],sol.x[0]);ratios=[];base=[]
 for channel in range(3):
  d=np.linalg.norm(T[:,channel,active]*sw)
  ratios.append(float(np.linalg.norm((fit-T)[:,channel,active]*sw)/max(d,1e-8)))
  base.append(float(np.linalg.norm((C-T)[:,channel,active]*sw)/max(d,1e-8)))
 return {'rank':rank,'base_template_errors':base,'oracle_template_errors':ratios,'oracle_Uy_ratio':ratios[1]/max(base[1],1e-8),
   'oracle_parameters':sol.x.tolist(),'success':bool(sol.success),'nfev':sol.nfev,'not_deployable':True},fit

def information(C,G,params,q,mask):
 ij,_=p.probes({'q':q,'mask':mask});H=np.stack([C[:,1,i,j] for i,j in ij]);gp=np.stack([G[:,i,j] for i,j in ij])
 J=[];N=[]
 for t in np.arange(2.,18.,2.):
  b,d=weights(params[0]+2*np.pi*params[1]*t);dd=H@d
  J.append(dd[:,None]*gp);N.append(np.c_[dd,t*dd])
 J=np.vstack(J);N=np.vstack(N);J-=N@np.linalg.lstsq(N,J,rcond=1e-10)[0]
 sv=np.linalg.svd(J,compute_uv=False);noise=.05*p.SCALE[0,1,0,0]
 return {'rank_after_clock_nuisance':int(np.sum(sv>max(sv[0]*1e-3,noise))),'singular_values':sv.tolist(),
  'minimum_singular_over_noise':float(sv[-1]/noise),'condition':float(sv[0]/max(sv[-1],1e-20)),'sensitivity_from_own_C_geometry_only':True}

def diagnostic():
 p.protect();begin=time.perf_counter();DEST.mkdir(exist_ok=True)
 p.save(DEST/'preregister.json',{'utc':p.now(),'hypothesis':'local streamwise wake phase/wavelength mismatch remains after uniform phase adjustment; not already established cause',
  'priority':'return to frozen4fullstartup FNO task; sparse-only task held after failed initialization',
  'models':['local_phase_slope1','local_phase_slope_curvature2'],'basis':'bodyD streamwise onset0.5..2D,xi/8 andsquare,10deg/5deg scales; a clipped+-2; applies samelocal phase toall Ux/Uy/p harmonics with orderk, mean fixed',
  'diagnostic':'TRAIN96 ownC vsTRAIN trueC; truth phase andwarp fitted only oracle, no oracle deployable claim; FNO previouslytrainedallTRAIN96',
  'continuation':'median oracleUy relative reduction>=20%,90thpercentile oracle ratio<=1.01, both shapes and thinD<.9 benefit; observable supported states rank/full orminimumsvnoise>=3; then fixed finite causal seen64 trial with originalinput76reads',
  'deployment_guards':'samebudget originalv5; meanR>=15%lower,maxnotworse, first8s preserved, thinbothshapes benefit; thengeometryheld deterministic/noise check; independent8 only after freeze and robust clear gain',
  'storage':'server-only; genuineprogress todownload; failedcompactarchive thenremovebinaries','subagents':'gpt-6-luna only; narrow tasks for quota'})
 cases=[]
 for index,row in enumerate(p.read_train()):
  C=np.asarray(row['C'],float);T,_=p.gauge_align(np.asarray(row['truthC'],float),C,row['mask'])
  for rank in [1,2]:
   result,_=oracle_fit(C,T,row['q'],row['mask'],rank);result.update(id=row['id'],q=row['q'],fold=row['fold'],information=information(C,spatial_basis(row['q'],row['mask'],rank),row['params'],row['q'],row['mask']))
   cases.append(result)
  p.save(DEST/'diagnostic_partial.json',{'cases':cases});p.save(DEST/'status.json',{'stage':'oracle_and_observability','completed':index+1,'total':96,'utc':p.now()})
  print('PHASE_DIAG',index+1,row['id'],[round(cases[-2+i]['oracle_Uy_ratio'],4) for i in range(2)],flush=True)
 summaries=[]
 for rank in [1,2]:
  for group in ['all','cylinder','square','thin_cylinder','thin_square']:
   part=[s for s in cases if s['rank']==rank and(group=='all' or(group==s['q']['shape'])or(group=='thin_'+s['q']['shape'] and s['q']['D']<.9))]
   ratios=np.asarray([s['oracle_Uy_ratio'] for s in part]);obs=np.asarray([s['information']['minimum_singular_over_noise'] for s in part])
   summaries.append({'rank':rank,'group':group,'n':len(part),'median_oracle_ratio':float(np.median(ratios)),'p90_oracle_ratio':float(np.quantile(ratios,.9)),
    'mean_baseUy':float(np.mean([s['base_template_errors'][1] for s in part])),'mean_oracleUy':float(np.mean([s['oracle_template_errors'][1] for s in part])),
    'minimum_singular_over_noise_median':float(np.median(obs)),'observable_ge3noise_count':int(np.sum(obs>=3)),
    'mean_baseP':float(np.mean([s['base_template_errors'][2] for s in part])),'mean_oracleP':float(np.mean([s['oracle_template_errors'][2] for s in part]))})
 entries=[]
 for rank in [1,2]:
  groups={s['group']:s for s in summaries if s['rank']==rank};allg=groups['all']
  guards={'median_Uy_reduction20pct':allg['median_oracle_ratio']<=.8,'p90_not_worse':allg['p90_oracle_ratio']<=1.01,
   'both_thin_shapes_benefit':all(groups[g]['median_oracle_ratio']<.95 for g in ['thin_cylinder','thin_square']),
   'observability_median_ge3noise':allg['minimum_singular_over_noise_median']>=3}
  entries.append({'rank':rank,'guards':guards,'passed':all(guards.values())})
 p.save(DEST/'diagnostic_report.json',{'role':'oracle+structural information only, not deployable; neither wholeFNOheldout nor independenttest','cases':cases,'summary':summaries,'continuation':entries,'wall_seconds':time.perf_counter()-begin})
 p.protect();p.save(DEST/'status.json',{'status':'complete_diagnostic','utc':p.now()});print('DIAGNOSTIC_VERDICT',entries,flush=True)

if __name__=='__main__':
 try:diagnostic()
 except Exception as e:p.save(DEST/'failure.json',{'error':repr(e),'utc':p.now()});raise
