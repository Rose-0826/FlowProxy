"""Training data and TRAIN-only residual libraries for the finite campaign."""
import copy, json, sys
from pathlib import Path
import numpy as np
import optimize as o
sys.path.insert(0,str(o.ROOT/'work/shape_expansion_20261006'))
import train_expansion as t

def dataset(include_additional=True):
    o.check();o.e.setup();recs,data,templates=t.training()
    cfg=o.config();added=[]
    if include_additional:
        audit=json.loads((o.OUT/'CFD_train_manifest.json').read_text());assert len(audit['quality'])==32
        for q in cfg['additional_train']:
            path=o.CACHE/(q['id']+'.npz');meta=json.loads(path.with_suffix('.json').read_text())
            assert meta['CFD_stable'] and meta['cache_sha256']==o.sha(path)
            with np.load(path) as z:raw={k:z[k] for k in ['fields','times','geom','mask']}
            frequency=meta['St']/q['D'];basis=o.old.p.basis(raw['times']-raw['times'][3],frequency)
            truthC=o.old.b.previous.w.constrain(np.linalg.lstsq(basis,raw['fields'].reshape(165,-1),rcond=1e-10)[0].reshape(9,3,64,64),raw['mask'])
            rec=dict(id=q['id'],q={k:q[k] for k in ['shape','Re','cx','cy','D']},fold=0,group_key=q['id'],truth_frequency=frequency,truthC=truthC,
                     mask=raw['mask'],fields=raw['fields'][:4],geom=raw['geom'],times=raw['times'])
            recs.append(rec);templates.append(rec);data.append(dict(id=q['id'],q=rec['q'],**raw));added.append(dict(id=q['id'],cache_sha256=o.sha(path)))
    assert len(recs)==(156 if include_additional else 124) and len(data)==len(recs) and len(templates)==len(recs)
    ids={r['id'] for r in recs};assert len(ids)==len(recs)
    assert not ids.intersection({q['id'] for q in cfg['development']+cfg['fresh_confirmation']})
    periods=[dict(N=int(np.floor(4/r['truth_frequency'])),alpha=float(4/r['truth_frequency']-np.floor(4/r['truth_frequency']))) for r in recs]
    weights=[]
    for raw in data:
        q=raw['q'];geo=o.e.shapes.geometry_channels(o.old.protocol()['config'],**q)
        wake=(geo['xx']>=q['cx'])&(geo['xx']<=q['cx']+8*q['D'])&(np.abs(geo['yy']-q['cy'])<=2*q['D'])
        weights.append((raw['mask']*(1+wake.astype(np.float32))).astype(np.float32))
    if include_additional:
        manifest=dict(ids=[r['id'] for r in recs],counts=dict(original=100,prior_newshape=24,additional=32),additional=added,
                      no_development_or_fresh_ID_in_training=True,role='TRAIN only; original DEV reflection-equivalent cases remain seen evidence')
        path=o.OUT/'TRAIN156_manifest.json'
        if path.exists():assert json.loads(path.read_text())==manifest
        else:o.save(path,manifest)
    return recs,data,templates,periods,weights

def build_candidate(cp,folder,recs,data,templates,name):
    o.check();base,saved=o.old.model_from(cp,o.old.protocol()['config'],seed=1602)
    model=o.old.ProjectedState(o.old.ReflectionOperator(base),True);model.eval()
    parent=json.loads((o.OUT/'parent_system.json').read_text());bundle=dict(model=model,libs={},priors=o.old.p.geometry_priors())
    priors={}
    for shape in o.e.shapes.SPECS:
        st=float(np.mean([r['truth_frequency']*r['q']['D'] for r in recs[100:] if r['q']['shape']==shape]));priors[shape]=st
        bundle['priors'][shape]=lambda x,st=st:np.full(len(x),st)
    updated=[]
    for rec,raw in zip(templates[100:],data[100:]):
        row=dict(id=rec['id'],q=rec['q'],fields=raw['fields'][:4],geom=raw['geom'],mask=raw['mask'],times=raw['times'])
        own,C,pa,di=o.e.bootstrap(bundle,row);updated.append({**rec,'C':C.astype(np.float32),'params':pa})
    banks={};counts={}
    for shape in o.e.shapes.SPECS:
        rows=[r for r in updated if r['q']['shape']==shape];counts[shape]=len(rows)
        assert len(rows)==(12 if shape in o.TARGETS else 4)
        lib=o.old.b.build(rows,str(folder/'new_banks'/shape));path=folder/'new_banks'/shape/(shape+'_bank.npz')
        banks[shape]=dict(path=str(path),sha256=o.sha(path));bundle['libs'][shape]=lib
    # Original shapes route to the unchanged original network. These adapter old
    # banks are loader compatibility only and never used by a supported route.
    info=copy.deepcopy(parent['new_shapes']);info.update(name=name,checkpoint=str(cp),checkpoint_sha256=o.sha(cp),
        new_banks=banks,new_priors=priors,training_case_count=156,residual_source_counts=counts,
        original_adapted_banks_unused_and_not_rebuilt=True)
    system=copy.deepcopy(parent);system['new_shapes']=info
    system['routing']='Original circle/square unchanged; six new profiles share this adapted network; geometry-only routing.'
    system_path=folder/'system.json';o.save(system_path,system)
    record=dict(name=name,checkpoint=str(cp),checkpoint_sha256=o.sha(cp),bank_folder=str(folder),system_path=str(system_path),system_sha256=o.sha(system_path),source_counts=counts,
                training_manifest_sha256=o.sha(o.OUT/'TRAIN156_manifest.json'),no_development_truth_in_banks=True)
    o.save(folder/'candidate.json',record)
    del bundle,model,base
    import torch;torch.cuda.empty_cache()
    return record

def candidate_index():
    cfg=o.check();records=[]
    for arm in cfg['arms']:
        history=json.loads((o.OUT/arm['name']/'history.json').read_text())
        assert len(history)==2
        for record in history:
            assert o.sha(record['checkpoint'])==record['checkpoint_sha256']
            assert o.sha(record['system_path'])==record['system_sha256'];records.append(record)
    assert len(records)==4
    o.save(o.OUT/'candidate_index.json',dict(candidates=records,utc=o.now(),all_four_before_new_confirmation=True))
