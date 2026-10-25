#!/usr/bin/env python3
"""Export authentic CFD/model comparison data and figures, without changing scores."""
from pathlib import Path
import sys,json,hashlib,shutil,tarfile,datetime,re,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha,NativeSampler
from flowproxy.geometry import geometry_channels
from cpu_data_prepare import frames
O=ROOT/'outputs/true2d_recheck_v1';D=ROOT/'outputs/aic_true2d_recheck_2026-10-03'

def run():
    assert not D.exists();D.mkdir()
    p=json.loads((O/'preregister.json').read_text());final=json.loads((O/'report.json').read_text())
    assert not final['errors'] and len(final['completed'])==4
    figdir=D/'figures';figdir.mkdir();srcdir=D/'snapshots';srcdir.mkdir()
    for f in O.rglob('*'):
        if not f.is_file() or 'predictions' in f.parts or f.suffix not in ['.json','.csv','.npz']:continue
        target=srcdir/f.relative_to(O);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    code=D/'code';code.mkdir()
    for name in ['true2d_recheck_v1.py','true2d_recheck_cpu_resume_v1.py',
                 'true2d_recheck_delivery_v1.py','build_true2d_case_v1.py','cfd_numerics_analyze_v1.py']:
        shutil.copy2(ROOT/'scripts'/name,code/name)
    data=final['evaluations'];models=p['models'];variants=['old_reference','coarse_dt002','fine_dt0005']
    colors=['#61758B','#087F8C','#D88722'];manifest=[]
    fig,axs=plt.subplots(1,2,figsize=(12,5.1),layout='constrained')
    records=[]
    for ax,row in zip(axs,p['cases']):
        cid=row['id'];x=np.arange(len(models));maxval=0
        for j,v in enumerate(variants):
            hs=[next(r for r in data[v]['cases'] if r['id']==cid and r['model']==n)['horizons']['40s'] for n in models]
            y=[r['R'] for r in hs];maxval=max(maxval,max(y))
            bars=ax.bar(x+(j-1)*.24,y,.22,color=colors[j],label=['Original mesh','Repaired coarse','Repaired fine'][j])
            ax.bar_label(bars,labels=[f'{a:.2f}' for a in y],padding=3,fontsize=10)
            for n,h in zip(models,hs):records.append({'case':cid,'variant':v,'model':n,**h})
        ax.axhline(1,color='#872C2B',ls='--',lw=1.4,label='R = 1 threshold')
        ax.set_xticks(x,['F0','K=8','Short trajectory\n+ periodic closure'])
        ax.set_ylabel('Case score R_i (below 1 passes)');ax.set_title(cid,fontsize=11)
        ax.set_ylim(0,maxval*1.18);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=10)
    fig.suptitle('Frozen models: complete 40-second forecasts on two known CFD anchors',fontsize=15)
    for ext in ['png','pdf','svg']:fig.savefig(figdir/f'true2d_case_scores.{ext}',dpi=180)
    plt.close(fig)
    save(D/'plot_score_records.json',{'records':records,'no_selection_among_anchors':True,
        'not_fresh_independent_geometries':True,'coarse_and_fine_dt':{'coarse':.02,'fine':.005}})
    plt.rcParams['pdf.fonttype']=42
    for variant in ['coarse_dt002','fine_dt0005']:
        fig,axs=plt.subplots(2,3,figsize=(13,6.2),layout='constrained')
        arrays={}
        for i,row in enumerate(p['cases']):
            cid=row['id'];q={k:row[k] for k in ['shape','Re','D','cx','cy']};g=geometry_channels(p['config'],**q);mask=g['mask']>.5
            case=ROOT/'data/raw/true2d_recheck_v1'/variant/cid;lo=p['variants'][variant]['input_start']
            items=[(t,f) for t,f in frames(case) if lo-1e-7<=t<=lo+41+1e-7][:164];t,f=items[163]
            truth=NativeSampler(g['xx'],g['yy'],g['mask']).sample(f)
            initial=json.loads((O/('evaluation_'+variant)/'predictions_frozen.json').read_text())['rows']
            predictions=[];sources=[]
            for name in ['K8','neural_periodic_closure']:
                entry=next(r for r in initial if r['id']==cid and r['model']==name)
                assert sha(entry['file'])==entry['sha256'];predictions.append(np.load(entry['file'])['prediction'][-1]);sources.append(entry)
            values=[truth,*predictions];lim=max(float(np.max(np.abs(a[1][mask]))) for a in values)
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#CED6DE')
            errors=[]
            for j,(name,a) in enumerate(zip(['Corrected CFD reference','K=8 forecast','Short trajectory + closure'],values)):
                ax=axs[i,j];image=ax.imshow(np.ma.masked_where(~mask,a[1]),origin='lower',extent=[-8,20,-8,8],
                    interpolation='nearest',cmap=cmap,vmin=-lim,vmax=lim,aspect='equal')
                e=float(np.linalg.norm((a[1]-truth[1])[mask])/np.linalg.norm(truth[1][mask]))
                if j:
                    expected=next(r for r in data[variant]['cases'] if r['id']==cid and r['model']==['K8','neural_periodic_closure'][j-1])['horizons']['40s']['end'][1]
                    assert abs(e-expected)<2e-6
                errors.append(e);ax.set_title(name+f'\nUy error {100*e:.2f}%',fontsize=11)
                ax.set_xlabel('x');ax.set_ylabel('y')
            cb=fig.colorbar(image,ax=axs[i,:].tolist(),fraction=.023,pad=.02,format='%.2f');cb.set_label('Uy')
            arrays.update({f'row{i}_{n}':a for n,a in zip(['CFD','K8','closure'],values)});arrays[f'row{i}_mask']=mask
            manifest.append({'variant':variant,'case':cid,'time':t,'reference_file':str(f),'reference_sha256':sha(f),
                'prediction_files_sha256':{s['file']:s['sha256'] for s in sources},'endpoint_Uy_errors':errors,
                'common_color_limits':[-lim,lim],'whole_native_64x64_sampling_no_crop_or_smoothing':True})
        fig.suptitle(f'40-second endpoint Uy: {variant}; top cylinder Re110, bottom square Re170',fontsize=14)
        for ext in ['png','pdf','svg']:fig.savefig(figdir/f'{variant}_endpoint_fields.{ext}',dpi=160)
        np.savez_compressed(figdir/f'{variant}_endpoint_arrays.npz',**arrays);plt.close(fig)
    save(figdir/'manifest.json',{'rows':manifest,'all_cases_in_preregister_shown':True,
                                'color_limits_shared_within_each_row':True,'geometry_and_time_versions_kept_separate':True})
    invariant=[]
    for rel,digest in p['frozen_sha256'].items():
        observed=sha(ROOT/rel);assert observed==digest;invariant.append({'file':rel,'sha256':observed,'matched':True})
    builds=json.loads((O/'builds.json').read_text())['cases']
    for b in builds:
        if 'source_mesh_hashes' in b:
            for n,h in b['source_mesh_hashes'].items():assert sha(Path(b['source'])/'constant/polyMesh'/n)==h
    quality=json.loads((O/'reference_statistics.json').read_text())
    save(D/'final_audit.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frozen_files':invariant,'original_coarse_sources_unchanged':True,'four_completed_CFD_runs':True,
        'MPI_execution_amendment_preserved':True,'reference_stationarity':quality['late_cycle_quality_gate'],
        'claim_scope':'Two previously examined anchors only. Fine refinement changes both mesh and dt, so not separate spatial/time convergence.',
        'training_performed':False,'new_independent_geometry_confirmation':False})
    files=[{'file':str(f.relative_to(D)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(D.rglob('*')) if f.is_file()]
    save(D/'export_manifest.json',{'files':files,'n':len(files)})
    archive=Path('/tmp/aic_true2d_recheck_2026-10-03.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(D,arcname=str(D.relative_to(ROOT)))
    print('TRUE2D_DELIVERY_READY',len(files),archive.stat().st_size,flush=True)
    print('ARCHIVE_SHA256_FIRST',sha(archive)[:32],flush=True);print('ARCHIVE_SHA256_LAST',sha(archive)[32:],flush=True)

if __name__=='__main__':run()
