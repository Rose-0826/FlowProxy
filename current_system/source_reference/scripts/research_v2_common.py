#!/usr/bin/env python3
"""Independent research branch; protect the competition fallback, reuse seen data honestly."""
from pathlib import Path
import datetime, json, sys, shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from geometry_attack_v2 import save,sha
from corrected_benchmark_train_v1 import protocol
from cycle_landmarks_train_v1 import FORMAL,FORMAL_SHA
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

OUT=ROOT/'outputs/research_v2_multidirection'
CACHE=ROOT/'data/processed/corrected_benchmark_restart_v1_cases'
FRESH4=ROOT/'data/processed/period_consistency_confirmation4_v1_cases'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def rows(include_train=False):
    p=protocol();result=[]
    previous=ROOT/'outputs/period_consistency_evaluate_v1'
    for q in p['cases']:
        if q['role']=='train' and not include_train:continue
        role={'train':'train80','val':'seen_val8','test':'seen_extreme32'}[q['role']]
        folder='validation' if q['role']=='val' else 'already_seen_test32'
        result.append({**q,'group':role,'cache':str(CACHE),
            'baseline_dir':str(previous/folder),'baseline_label':'cycle_seed11'})
    for q in p['confirmation_cases']:
        result.append({**q,'group':'seen_continuous12','cache':str(CACHE),
            'baseline_dir':str(previous/'fresh_confirmation12'),'baseline_label':'cycle_seed11'})
    four=json.loads((ROOT/'outputs/period_consistency_confirmation4_v1/preregister.json').read_text())['cases']
    for q in four:
        result.append({**q,'group':'seen_continuous4','cache':str(FRESH4),
            'baseline_dir':str(ROOT/'outputs/period_consistency_confirmation4_v1/predictions'),'baseline_label':'cycle_seed11'})
    assert len(result)==(136 if include_train else 56)
    assert len({q['id'] for q in result})==len(result)
    return result

def startup(q):
    with np.load(Path(q['cache'])/(q['id']+'_startup.npz')) as a:
        row={name:a[name] for name in ['fields','times','geom','mask']}
    assert row['fields'].shape==(4,3,64,64)
    row.update({'id':q['id'],'q':{k:q[k] for k in ['shape','Re','cx','cy','D']}})
    return row

def truth(q):
    with np.load(Path(q['cache'])/(q['id']+'.npz')) as a:return a['fields']

def baseline(q,mode):
    suffix='DMD9' if q['group']=='seen_continuous4' and mode=='DMD9' else ('self_DMD_rank9' if mode=='DMD9' else 'direct')
    p=Path(q['baseline_dir'])/(q['baseline_label']+'_'+suffix+'__'+q['id']+'.npz')
    with np.load(p) as a:return a['prediction']

def scales():
    return json.loads((ROOT/'outputs/corrected_benchmark_restart_v1/training/warm_K1_seed33/configuration.json').read_text())['normalization_train80']

def score(q,pred,row):
    return {'id':q['id'],'q':row['q'],'group':q['group'],'horizons':metrics(pred,truth(q)[4:164],row['mask'],row['times'])}

def summaries(scored):
    groups=['seen_val8','seen_extreme32','seen_continuous12','seen_continuous4']
    out={}
    for group in groups+['development_inside24','development_all56']:
        if group=='development_inside24':part=[q for q in scored if q['group']!='seen_extreme32']
        elif group=='development_all56':part=scored
        else:part=[q for q in scored if q['group']==group]
        out[group]={h:summarize([q['horizons'][h] for q in part]) for h in ['2s','8s','20s','40s']}
    return out

def protected():
    lock=json.loads((OUT/'fallback_lock.json').read_text())
    for p,d in lock['protected_server_files'].items():assert sha(p)==d,p
    assert sha(OUT/'fallback_weights.pt')==FORMAL_SHA

def configure():
    OUT.mkdir(exist_ok=False)
    protected_files=[FORMAL,ROOT/'outputs/period_consistency_evaluate_v1/selected_pipeline_lock.json',
        ROOT/'outputs/period_consistency_confirmation4_v1/report.json',ROOT/'outputs/period_consistency_evaluate_v1/validation_report.json']
    assert sha(FORMAL)==FORMAL_SHA
    shutil.copyfile(FORMAL,OUT/'fallback_weights.pt')
    save(OUT/'fallback_lock.json',{'utc':now(),'user_policy':'previous competition material and model remain fallback; research is separate',
        'protected_server_files':{str(p):sha(p) for p in protected_files},'backup_sha256':sha(OUT/'fallback_weights.pt')})
    manifest=rows();save(OUT/'development_manifest.json',{'cases':manifest,'role':'all56 ALREADY-SEEN DEVELOPMENT; none is a fresh test'})
    save(OUT/'preregister.json',{'utc':now(),'input':'four physical true starting fields, geometry/Re; no future forces, phase or clock at inference',
        'fallback_immutable':True,'channel_scales':scales(),'decoder':'same unit-circle DMD9,4 true+48 generated states; original generated prefix retained',
        'research_dimensions':['geometry extrapolation and resolution','operator reflection symmetry','geometry-centred representation',
            'training-library transport and gauge alignment','four-frame frequency identifiability','neural jumps plus CFD recovery'],
        'development_roles':'inside24=old val8+seen12+seen4; extreme32=old seen test32; report both, never call a fresh test',
        'candidate_ranking':'balanced meanRi40=0.5 inside24 +0.5 extreme32; short8s and maxRi reported as guards',
        'promotion_signal':'>=10% lower balanced meanRi40, no worsening either group meanRi40; balanced meanRi8 <=1.05 baseline and8s pass loss<=1/56',
        'no_automatic_competition_model_replacement':True,'new_independent_confirmation':'only after a research method is frozen',
        'symmetry':'up/down reflection is exact for equal symmetryPlane boundaries; Ux,p even and Uy odd',
        'translation':'finite top/bottom boundaries mean centering is approximate, not an exact invariance; remapping floor reported',
        'citations':{'G_FNO':'https://arxiv.org/abs/2306.05697','solver_in_the_loop':'https://arxiv.org/abs/2007.00016',
            'parametric_DMD':'https://arxiv.org/abs/2204.12006'},'common_source_sha256':sha(__file__)})
    protected()

if __name__=='__main__':configure()
