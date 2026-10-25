#!/usr/bin/env python3
"""Audit, index, and render frozen results without altering raw reports."""
from pathlib import Path
import sys,json,datetime,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from round2_complete40_v1 import summarize
OUT=ROOT/'outputs/aic_attack_2026-10-02'


def run():
    OUT.mkdir(exist_ok=False);groups={};csvrows=[];checks=[]
    settings=[('new_diameter8','geometry_attack_confirmation_v1',8),('new_fixed_D1_near4','geometry_attack_confirmation_d1_v1',4),('new_fixed_D1_far4','geometry_attack_confirmation_far_v1',4)]
    for group,folder,n in settings:
        directory=ROOT/'outputs'/folder;r=json.loads((directory/'evaluation/report.json').read_text());registry=json.loads((directory/'evaluation/predictions_frozen.json').read_text())
        for item in registry['rows']:
            if sha(item['file'])!=item['sha256']:raise ValueError('Prediction SHA changed')
        groups[group]={'source_report_sha256':sha(directory/'evaluation/report.json'),'source_directory':str(directory),'models':r['reports']}
        for name,report in r['reports'].items():
            assert len(report['cases'])==n
            for case in report['cases']:
                for key,m in case['horizons'].items():
                    recomputed=float(max(np.array(m['mean'])/np.array([.05,.05,.08])));recomputed=max(recomputed,float(max(np.array(m['end'])/np.array([.05,.05,.08]))))
                    checks.append(abs(recomputed-m['R']))
                    row={'group':group,'model':name,'case':case['id'],'horizon':key,'R':m['R'],'passed':m['R']<1}
                    for label,values in [('mean',m['mean']),('end',m['end'])]:
                        for ch,value in zip(['Ux','Uy','p'],values):row[label+'_'+ch+'_percent']=100*value
                    csvrows.append(row)
            for h in ['2s','8s','20s','40s']:
                m=summarize([c['horizons'][h] for c in report['cases']]);assert abs(m['mean_R']-report['summary'][h]['mean_R'])<1e-7
                assert m['n_pass']==report['summary'][h]['n_pass']
        prefix=[]
        for case in r['reports']['K8']['cases']:
            cid=case['id'];pred=directory/'evaluation/predictions'
            a=np.load(pred/f'K8__{cid}.npz')['prediction'];b=np.load(pred/f'neural_periodic_closure__{cid}.npz')['prediction']
            prefix.append(float(np.max(np.abs(a[:48]-b[:48]))))
        groups[group]['first48_predictions_max_absolute_delta']=max(prefix)
        if max(prefix)>1e-6:raise ValueError('Primary bootstrap not preserved')
    combined={}
    for label,selected in [('new_fixed_D1_all8',['new_fixed_D1_near4','new_fixed_D1_far4']),('all_new16',list(groups))]:
        combined[label]={}
        for name in groups[selected[0]]['models']:
            rows=sum([groups[g]['models'][name]['cases'] for g in selected],[])
            combined[label][name]={h:summarize([r['horizons'][h] for r in rows]) for h in ['2s','8s','20s','40s']}
    with (OUT/'new_confirmation_cases.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    save(OUT/'combined_evidence.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'groups':groups,'combined':combined,
        'cohort_timing':'initial diameter8 and near4 preregistered and nominated before initial new target reads; far4 extension registered after initial12 scores with unchanged primary',
        'observations':'four true initial fields only, N=48 AI-generated fields for nominated primary','original32_and_prior12_are_seen':True})
    save(OUT/'numerical_audit.json',{'rows':len(csvrows),'max_R_formula_recalculation_delta':max(checks),
        'all_saved_prediction_sha256_verified':True,'all_groups_count_and_mean_R_verified':True,
        'all16_first48_primary_equals_K8':True,'all_horizons_2_8_20_40_present':True})
    plot(groups)
    print('ATTACK_DELIVERY_READY',len(csvrows),combined['new_fixed_D1_all8']['neural_periodic_closure']['40s'],flush=True)


def plot(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(11.6,4.8),sharey=True)
    specs=[('new_fixed_D1_near4','New fixed D=1, |cy|=0.2\nRe=110/170; n=4'),
           ('new_fixed_D1_far4','New fixed D=1, |cy|=0.7\nOutside training positions; n=4'),
           ('new_diameter8','New D=0.8/1.2, |cy|=0.2\nOutside training diameters; n=8')]
    source=[]
    for ax,(group,title) in zip(axs,specs):
        models=groups[group]['models'];base={r['id']:r['horizons']['40s']['R'] for r in models['K8']['cases']}
        hybrid={r['id']:r['horizons']['40s']['R'] for r in models['neural_periodic_closure']['cases']}
        for j,cid in enumerate(sorted(base)):
            jitter=(j-(len(base)-1)/2)*.014
            color='#0072B2' if cid.startswith('cylinder') else '#D55E00'
            marker='o' if cid.startswith('cylinder') else 's'
            ax.plot([jitter,1+jitter],[base[cid],hybrid[cid]],color=color,alpha=.55,lw=1)
            ax.scatter([jitter,1+jitter],[base[cid],hybrid[cid]],s=29,color=color,marker=marker,zorder=3)
            source.append({'group':group,'case':cid,'K8_R40':base[cid],'closure_R40':hybrid[cid]})
        ax.axhline(1,color='#333333',ls='--',lw=1);ax.set_yscale('log');ax.set_ylim(.08,100)
        ax.set_yticks([.1,1,10,100]);ax.yaxis.set_major_formatter(ScalarFormatter());ax.grid(axis='y',alpha=.16)
        ax.set_xlim(-.27,1.27);ax.set_xticks([0,1],['K8 rollout','AI trajectory\n+ periodic\ncontinuation'])
        ax.set_title(title,fontsize=10,pad=10)
        p0=sum(v<1 for v in base.values());p1=sum(v<1 for v in hybrid.values())
        ax.text(.5,-.36,f'Passed: {p0}/{len(base)} → {p1}/{len(base)}',transform=ax.transAxes,ha='center',fontsize=10)
    axs[0].set_ylabel('Case R at complete 40 s (log scale)')
    fig.suptitle('Four true startup fields only; paired case results at 40 s',fontsize=13,y=.98)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],marker='o',color='#0072B2',label='Cylinder',lw=1),
                        Line2D([0],[0],marker='s',color='#D55E00',label='Square',lw=1)],
               loc='upper center',bbox_to_anchor=(.5,.927),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.025,'R = max(channel-normalized time mean and endpoint); R<1 is the team target. All 16 new cases are shown.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.78,bottom=.32,wspace=.16)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'hybrid_independent_evidence.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    save(folder/'figure_source_data.json',source)
    save(folder/'figure_manifest.json',{'case_count':16,'all_new_cases_included':True,'no_confidence_intervals_claimed':True,
        'common_log_y_scale':[.08,100],'jitter':'fixed visual separation on categorical x only; y values unchanged',
        'colors':{'cylinder':'#0072B2','square':'#D55E00'},'opaque_background':True,'dpi':300,'figure_inches':[11.6,4.8],
        'software_method_citation':'Kassis et al. (2026), Scientific Agent Skills, https://doi.org/10.48550/arXiv.2609.00065',
        'files':{str(p.name):sha(p) for p in folder.iterdir() if p.is_file() and p.name!='figure_manifest.json'}})


if __name__=='__main__':run()
