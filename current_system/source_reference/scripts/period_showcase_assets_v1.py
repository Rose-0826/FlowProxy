#!/usr/bin/env python3
"""Scientific assets from frozen fresh4 predictions; no new model inference."""
from pathlib import Path
import json,hashlib,tarfile,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/period_consistency_confirmation4_v1'
CACHE=ROOT/'data/processed/period_consistency_confirmation4_v1_cases'
OUT=ROOT/'outputs/period_showcase_assets_v1'

def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()

def run():
    OUT.mkdir(exist_ok=False)
    report=json.loads((SRC/'report.json').read_text());rows=report['cases']['cycle_seed11_DMD9']
    through=[r for r in rows if r['horizons']['40s']['R']<1];assert len(through)==1
    selected={'passing':through[0]['id'],'worst':max(rows,key=lambda r:r['horizons']['40s']['R'])['id']}
    extent=(-8,20,-8,8)
    cmap=matplotlib.colormaps['RdBu_r'].with_extremes(bad='#737B83')
    ecmap=matplotlib.colormaps['cividis'].with_extremes(bad='#737B83')
    curves=[];provenance=[]
    for row in rows:
        cid=row['id']
        with np.load(CACHE/(cid+'.npz')) as a:
            truth=a['fields'][4:164];mask=a['mask']>.5;times=a['times'][4:164]-a['times'][3]
        pred={}
        for method in ('baseline_DMD9','cycle_seed11_DMD9'):
            f=SRC/'predictions'/(method+'__'+cid+'.npz')
            with np.load(f) as a:pred[method]=a['prediction']
            provenance.append({'path':str(f.relative_to(ROOT)),'sha256':sha(f)})
        norm=np.maximum(np.linalg.norm(truth[:,1,mask],axis=1),1e-8)
        errors={k:(100*np.linalg.norm((p[:,1]-truth[:,1])[:,mask],axis=1)/norm).tolist() for k,p in pred.items()}
        curves.append({'id':cid,'times_seconds':times.tolist(),'Uy_relativeL2_percent':errors})
        if cid not in selected.values():continue
        role=next(k for k,v in selected.items() if v==cid)
        data=[truth[:,1],pred['baseline_DMD9'][:,1],pred['cycle_seed11_DMD9'][:,1]]
        velocity_limit=max(float(np.max(np.abs(v[:,mask]))) for v in data)
        delta=np.abs(data[2]-data[0]);error_limit=float(np.max(delta[:,mask]))
        fig,axes=plt.subplots(1,4,figsize=(15.0,3.3),layout='constrained')
        ims=[]
        titles=['CFD reference','Original FNO + DMD9','Cycle-trained FNO + DMD9','Absolute error (new)']
        for k,ax in enumerate(axes):
            ax.set(title=titles[k],xlabel='x (m)',ylabel='y (m)')
            a=np.ma.masked_where(~mask,data[k][0] if k<3 else delta[0])
            im=ax.imshow(a,origin='lower',extent=extent,cmap=cmap if k<3 else ecmap,
                         vmin=-velocity_limit if k<3 else 0,vmax=velocity_limit if k<3 else error_limit,interpolation='nearest')
            ims.append(im);ax.set_xticks([-8,0,10,20]);ax.set_yticks([-8,0,8])
        fig.colorbar(ims[0],ax=list(axes[:3]),shrink=.75,label='Uy (m/s)')
        fig.colorbar(ims[3],ax=axes[3],shrink=.75,label='|Uy pred - true| (m/s)')
        def draw(k):
            for j,im in enumerate(ims):im.set_data(np.ma.masked_where(~mask,data[j][k] if j<3 else delta[k]))
            eu=errors['cycle_seed11_DMD9'][k]
            fig.suptitle(f'{cid}     Future t = {times[k]:.2f} s     New Uy error = {eu:.2f}%',fontsize=13)
            fig.canvas.draw();return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
        draw(159);fig.savefig(OUT/f'{role}_endpoint.png',dpi=130,facecolor='white')
        draw(31);fig.savefig(OUT/f'{role}_8s.png',dpi=130,facecolor='white')
        frame_ids=list(range(0,160,2))+[159]
        frames=[draw(k).resize((1500,330),Image.Resampling.NEAREST) for k in frame_ids]
        frames[0].save(OUT/f'{role}_comparison.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
        plt.close(fig)
        # A different physical moment supplies the cover image; full domain preserved.
        if role=='passing':
            fig,ax=plt.subplots(figsize=(9.8,5.6),layout='constrained')
            im=ax.imshow(np.ma.masked_where(~mask,truth[47,1]),origin='lower',extent=extent,cmap=cmap,
                         vmin=-velocity_limit,vmax=velocity_limit,interpolation='nearest')
            ax.set(xlabel='x (m)',ylabel='y (m)',title='Stable cylinder wake, CFD Uy at future t = 12 s')
            fig.colorbar(im,ax=ax,label='Uy (m/s)',shrink=.75);fig.savefig(OUT/'cover_cfd.png',dpi=150,facecolor='white');plt.close(fig)
        provenance.append({'case':cid,'selected_role':role,'selection_rule':'unique 40s passing case and maximum-Ri case in fresh4',
            'velocity_color_limits':[-velocity_limit,velocity_limit],'absolute_error_limits':[0,error_limit],
            'native_resolution':[64,64],'whole_domain_m':extent,'interpolation':'nearest','GIF_source_indices':frame_ids,
            'GIF_display_frame_ms':100,'GIF_note':'subsample every2 native frames, no temporal interpolation; elapsed physical time printed per frame'})
    (OUT/'error_curves.json').write_text(json.dumps({'source_report_sha256':sha(SRC/'report.json'),'cases':curves},indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,row in zip(axes.ravel(),curves):
        for label,color,style in [('baseline_DMD9','#285E8E','--'),('cycle_seed11_DMD9','#087F8C','-')]:
            ax.plot(row['times_seconds'],row['Uy_relativeL2_percent'][label],color=color,linestyle=style,label=label)
        ax.axhline(5,color='#A65C00',ls=':',label='Team target 5%')
        ax.set(xlabel='Future time (s)',ylabel='Uy relative L2 (%)',title=row['id'],ylim=(0,None));ax.legend(fontsize=8)
    fig.savefig(OUT/'all4_error_curves.png',dpi=160,facecolor='white');plt.close(fig)
    (OUT/'provenance.json').write_text(json.dumps({'source_report':str(SRC/'report.json'),'source_report_sha256':sha(SRC/'report.json'),
        'purpose':'competition draft assets on corrected benchmark, fixed primary cycle seed11','selection':selected,'records':provenance,
        'scientific_disclosure':'all4 aggregate is primary evidence; two selected examples illustrate passing and worst; color scales fixed across methods/time within case; no smoothing, phase correction, gauge changes or clipped fields'},indent=2))
    files=sorted(f for f in OUT.iterdir() if f.is_file())
    manifest={'files':[{'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]}
    mp=OUT/'manifest.json';mp.write_text(json.dumps(manifest,indent=2));files.append(mp)
    archive=ROOT/'outputs/period_showcase_assets_20261003.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for f in files:t.add(f,arcname=str(f.relative_to(OUT)))
    print('SHOWCASE_ASSETS_COMPLETE',archive.stat().st_size,sha(archive),flush=True)

if __name__=='__main__':run()
