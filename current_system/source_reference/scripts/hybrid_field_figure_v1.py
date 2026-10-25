#!/usr/bin/env python3
"""Three rule-selected full-domain field panels, including failures, no rescaling per model."""
from pathlib import Path
import sys,json,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha,NativeSampler
from cpu_data_prepare import frames
from flowproxy.geometry import geometry_channels
from geometry_attack_evaluate_v1 import CFG
OUT=ROOT/'outputs/aic_attack_2026-10-02/figures'

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    d=json.loads((OUT.parent/'combined_evidence.json').read_text());lookup={};fixed=[];diameter=[]
    for group,rows in d['groups'].items():
        folder=Path(rows['source_directory'])
        registry=json.loads((folder/'preregister.json').read_text());geom={r['id']:r for r in registry['cases']}
        for case in rows['models']['neural_periodic_closure']['cases']:
            cid=case['id'];lookup[cid]={'folder':folder,'case':case,'row':geom[cid]}
            (diameter if group=='new_diameter8' else fixed).append(case)
    ordered=sorted(fixed,key=lambda r:(r['horizons']['40s']['R'],r['id']))
    selected=[('Lower median R among new D=1 eight',ordered[(len(ordered)-1)//2]),
              ('Worst R among new D=1 eight',ordered[-1]),
              ('Worst R among new diameter eight',max(diameter,key=lambda r:(r['horizons']['40s']['R'],r['id'])))]
    fig,axs=plt.subplots(3,3,figsize=(11.6,9.3));fig.subplots_adjust(left=.065,right=.87,bottom=.12,top=.855,wspace=.13,hspace=.66)
    records=[]
    for rowindex,(rule,item) in enumerate(selected):
        cid=item['id'];info=lookup[cid];q={k:info['row'][k] for k in ['shape','Re','D','cx','cy']}
        geo=geometry_channels(CFG,**q);mask=geo['mask']>.5;items=frames(ROOT/'data/raw'/info['folder'].name/cid)
        stamp,native=items[163];assert abs(stamp-200.74)<1e-4
        sampler=NativeSampler(geo['xx'],geo['yy'],geo['mask']);truth=sampler.sample(native)
        base=np.load(info['folder']/'evaluation/predictions'/f'K8__{cid}.npz')['prediction'][-1]
        hybrid=np.load(info['folder']/'evaluation/predictions'/f'neural_periodic_closure__{cid}.npz')['prediction'][-1]
        arrays=[truth[1],base[1],hybrid[1]];vmax=max(float(np.abs(a[mask]).max()) for a in arrays)
        err=[float(np.linalg.norm((a-truth[1])[mask])/max(np.linalg.norm(truth[1][mask]),1e-8)) for a in arrays]
        assert abs(err[2]-item['horizons']['40s']['end'][1])<2e-6
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#888888')
        for col,(ax,a,label,e) in enumerate(zip(axs[rowindex],arrays,['CFD reference','K8 rollout','AI trajectory + periodic'],err)):
            shown=np.where(mask,a,np.nan)
            im=ax.imshow(shown,origin='lower',extent=[-8,20,-8,8],interpolation='nearest',cmap=cmap,vmin=-vmax,vmax=vmax,aspect='equal')
            ax.set_title(label+('' if col==0 else f'\nEndpoint Uy error: {100*e:.2f}%'),fontsize=9)
            ax.set_xticks([-8,0,10,20]);ax.set_yticks([-8,0,8]);ax.set_xlabel('x (length unit)')
            if col==0:ax.set_ylabel('y (length unit)')
        top=axs[rowindex,0].get_position().y1
        fig.text(.065,top+.054,f'{rule}: {cid}',fontsize=10,ha='left')
        pos=axs[rowindex,2].get_position();cax=fig.add_axes([.895,pos.y0,.013,pos.height])
        bar=fig.colorbar(im,cax=cax);bar.set_label('Uy (velocity unit)',fontsize=9)
        records.append({'case':cid,'selection_rule':rule,'time':stamp,'same_symmetric_color_limits_all_three_models':[-vmax,vmax],
                        'endpoint_relative_Uy_errors':err,'reference_native_path':str(native),
                        'reference_native_sha256':sha(native),'mask':'fluid >0.5; solids grey; full domain',
                        'rendering':'64x64 original sampled fields, nearest pixels; no smoothing or crop'})
    fig.suptitle('Complete 40 s endpoint: reference, direct rollout, periodic continuation',fontsize=12,y=.99)
    fig.text(.5,.025,'Four true startup fields; t=200.74. Each row shares its color scale. Case selection and raw arrays are supplied.',ha='center',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'hybrid_endpoint_fields.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    arrays_file=OUT/'hybrid_endpoint_fields_arrays.npz'
    # Save numeric plotted fields in addition to export provenance.
    packed={}
    for i,(_,item) in enumerate(selected):
        info=lookup[item['id']];q={k:info['row'][k] for k in ['shape','Re','D','cx','cy']};geo=geometry_channels(CFG,**q)
        stamp,native=frames(ROOT/'data/raw'/info['folder'].name/item['id'])[163]
        packed[f'row{i}_truth']=NativeSampler(geo['xx'],geo['yy'],geo['mask']).sample(native)[1]
        for name,slug in [('K8','K8'),('neural_periodic_closure','closure')]:
            packed[f'row{i}_{slug}']=np.load(info['folder']/'evaluation/predictions'/f"{name}__{item['id']}.npz")['prediction'][-1,1]
        packed[f'row{i}_mask']=geo['mask']
    if arrays_file.exists():
        stored=np.load(arrays_file)
        assert set(stored.files)==set(packed) and all(np.array_equal(stored[k],v) for k,v in packed.items())
    else:np.savez_compressed(arrays_file,**packed)
    save(OUT/'hybrid_endpoint_fields_manifest.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,
      'no_case_exclusion_from_underlying_all16_evidence':True,'figure_inches':[11.6,9.3],'dpi':300,'opaque':True,
      'plot_arrays_file':arrays_file.name,'plot_arrays_sha256':sha(arrays_file),
      'software_method_citation':'https://doi.org/10.48550/arXiv.2609.00065',
      'files':{p.name:sha(p) for p in OUT.glob('hybrid_endpoint_fields.*')}})
    print('FIELD_FIGURE_DONE',[(r['case'],r['endpoint_relative_Uy_errors']) for r in records],flush=True)

if __name__=='__main__':run()
