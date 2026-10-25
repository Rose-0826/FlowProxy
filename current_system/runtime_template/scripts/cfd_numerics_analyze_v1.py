#!/usr/bin/env python3
"""Phase invariant mesh/time-step sensitivity; independent of AI prediction scoring."""
from pathlib import Path
import sys,json,datetime,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from cpu_data_prepare import frames,SampleCache
from flowproxy.geometry import geometry_channels
from geometry_attack_v2 import sha,save
OUT=ROOT/'outputs/cfd_numerics_restart_v1';RAW=ROOT/'data/raw/cfd_numerics_restart_v1'

def forces(case):
    rows={}
    for p in sorted(case.glob('postProcessing/forceCoeffs*/**/coefficient.dat')):
        for line in p.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):continue
            cols=line.split()
            if len(cols)>=5:rows[float(cols[0])]=[float(cols[0]),float(cols[1]),float(cols[4])]
    if not rows:raise ValueError('Missing forces')
    return np.array([rows[t] for t in sorted(rows)],dtype=np.float64)

def interval_values(t,q,start,end):
    assert t[0]<=start<end<=t[-1]
    middle=(t>start)&(t<end)
    def at(stamp):
        index=np.searchsorted(t,stamp,side='right')-1
        if t[index]==stamp or index==len(t)-1:return q[index]
        a=(stamp-t[index])/(t[index+1]-t[index]);return (1-a)*q[index]+a*q[index+1]
    return np.r_[start,t[middle],end],np.concatenate([at(start)[None],q[middle],at(end)[None]],axis=0)

def moment(t,q,start,end):
    ts,qs=interval_values(t,q,start,end)
    mean=np.trapezoid(qs,ts,axis=0)/(end-start)
    second=np.trapezoid(qs*qs,ts,axis=0)/(end-start)
    return mean,np.sqrt(np.maximum(second-mean*mean,0)),np.sqrt(np.maximum(second,0))

def case_stats(case,row,cfg,window):
    data=forces(case);sel=(data[:,0]>=window[0])&(data[:,0]<=window[1]);late=data[sel]
    lift=late[:,2]-np.trapezoid(late[:,2],late[:,0])/(late[-1,0]-late[0,0])
    crossings=[]
    for k in np.flatnonzero((lift[:-1]<=0)&(lift[1:]>0)):
        crossings.append(float(late[k,0]-lift[k]*(late[k+1,0]-late[k,0])/(lift[k+1]-lift[k])))
    if len(crossings)<4:raise ValueError('Fewer than three complete late periods')
    begin,end=crossings[-4],crossings[-1]
    force_mean,force_rms,_=moment(data[:,0],data[:,1:],begin,end)
    cycles=[]
    for a,b in zip(crossings[-4:-1],crossings[-3:]):
        mu,rms,_=moment(data[:,0],data[:,1:],a,b)
        cycles.append({'begin':a,'end':b,'Cd_mean':float(mu[0]),'Cl_rms':float(rms[1]),
          'St':row['D']/cfg['physics']['U_inf']/(b-a)})
    geo=geometry_channels(cfg,shape=row['shape'],cx=row['cx'],cy=row['cy'],D=row['D'],Re=row['Re'])
    cache=SampleCache(cfg,geo['mask']);sources=frames(case)
    # Include bracketing snapshots for exact cycle-window endpoints.
    times=np.array([x[0] for x in sources]);left=max(0,np.searchsorted(times,begin)-1);right=min(len(times),np.searchsorted(times,end)+1)
    sources=sources[left:right];times=np.array([x[0] for x in sources])
    fields=np.stack([cache.sample(path) for _,path in sources]).astype(np.float64)
    mu,rms,full=moment(times,fields,begin,end)
    label=case.parent.name+'__'+case.name
    np.savez_compressed(OUT/(label+'_field_statistics.npz'),mean=mu,rms=rms,full_rms=full,mask=geo['mask'])
    log=case/'log.icoFoam';text=log.read_text(errors='replace')
    courts=[float(v) for v in re.findall(r'Courant Number mean:\s+\S+\s+max:\s+(\S+)',text)]
    latest,prior=cycles[-1],cycles[-2]
    cycle_change={key:abs(latest[key]-prior[key])/max(abs(prior[key]),1e-8) for key in ['Cd_mean','Cl_rms','St']}
    summary={'case':row['id'],'directory':str(case),'source_window':window,'cycles':cycles,
      'statistics_window':[begin,end],'field_snapshot_count':len(sources),'field_snapshot_times':times.tolist(),
      'Cd_mean':float(force_mean[0]),'Cl_rms':float(force_rms[1]),
      'St':3*row['D']/cfg['physics']['U_inf']/(end-begin),'last_cycle_relative_change':cycle_change,
      'max_Courant_over_entire_restart':max(courts) if courts else None,
      'force_columns':{'time':0,'Cd':1,'Cl':4},'pressure_gauge':'raw p; no future mean subtraction for mean field',
      'field_statistics_file':label+'_field_statistics.npz','solver_log_sha256':sha(log)}
    save(OUT/(label+'_statistics.json'),summary)
    return summary,{'mean':mu,'rms':rms,'full':full,'mask':geo['mask']>.5}

def difference(reference,target):
    a,fa=reference;b,fb=target;mask=fa['mask'];assert np.array_equal(mask,fb['mask'])
    result={'reference':a['directory'],'target':b['directory'],
      'loads_relative':{key:abs(b[key]-a[key])/max(abs(a[key]),1e-8) for key in ['Cd_mean','Cl_rms','St']}}
    for name in ['mean','rms']:
        result[name+'_field_relative']=[float(np.linalg.norm((fb[name][c]-fa[name][c])[mask])/max(np.linalg.norm(fa[name][c][mask]),1e-8)) for c in range(3)]
        result[name+'_field_difference_over_full_reference_rms']=[float(np.linalg.norm((fb[name][c]-fa[name][c])[mask])/max(np.linalg.norm(fa['full'][c][mask]),1e-8)) for c in range(3)]
    return result

def run():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete' and len(status['completed'])==6
    p=json.loads((OUT/'preregister.json').read_text());rows=[];allstats={}
    for row in p['cases']:
        bank={}
        for variant in p['variants']:
            bank[variant]=case_stats(RAW/variant/row['id'],row,p['config'],p['scoring_window'])
            allstats[variant+'__'+row['id']]=bank[variant][0]
        source=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/row['id']
        bank['original_reference']=case_stats(source,row,p['config'],[171,201])
        allstats['original_reference__'+row['id']]=bank['original_reference'][0]
        rows.append({'case':row['id'],
          'time_step_half':difference(bank['coarse_dt002'],bank['coarse_dt001']),
          'mesh_refinement_at_dt001':difference(bank['coarse_dt001'],bank['fine_dt001']),
          'combined_refinement':difference(bank['coarse_dt002'],bank['fine_dt001']),
          'same_config_restart_vs_original':difference(bank['original_reference'],bank['coarse_dt002'])})
    report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':rows,'statistics':allstats,
      'not_AI_prediction_performance':True,'not_formal_grid_convergence':True,
      'not_phase_aligned_instantaneous_field_error':True,'only_two_anchor_cases':True,
      'scoring':'last three complete positive lift cycles within late window; exact endpoint interpolation and trapezoidal time integration',
      'fine_mesh_mapping_method':'cellPointInterpolate; original unsupported cellVolumeWeight failure retained',
      'script_sha256':sha(__file__)}
    save(OUT/'sensitivity_report.json',report)
    print('NUMERICS_ANALYZED',json.dumps(rows),flush=True)

if __name__=='__main__':run()
