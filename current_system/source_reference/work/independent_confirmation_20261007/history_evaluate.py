"""Restore the omitted historical16 with current fixed-system regression results."""
from pathlib import Path
import sys,json,hashlib,re,time,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import campaign as c

def run():
    cfg=c.check();c.e.setup();model=c.Predictor(cfg['system_path'])
    history=json.loads((c.OUT/'historical_identity_input.json').read_text())
    records=[];reports={m:[] for m in cfg['methods']}
    for batch in history['batches'][:2]:
        for cid in batch['ids']:
            match=re.fullmatch(r'(cylinder|square)_Re([^_]+)_cy([^_]+)_D([^_]+)',cid);assert match
            shape,Re,cy,D=match.groups();q=dict(id=cid,shape=shape,Re=float(Re),cx=0.,cy=float(cy),D=float(D),family=shape,role='historical_seen_regression',historical_batch=batch['batch'])
            folder='research_v2_independent8_SEALED_cases' if batch['batch']==1 else 'research_v4_sparse_confirmation8_cases'
            cache=c.ROOT/'data/processed'/folder;meta=json.loads((cache/(cid+'.json')).read_text())
            assert meta['CFD_stable'] and meta['uniform_frames']==165 and meta['max_Courant_dense']<1 and meta['cache_sha256']==c.sha(cache/(cid+'.npz'))
            with np.load(cache/(cid+'_startup.npz')) as z:row={k:z[k] for k in ['fields','times','geom','mask']}
            row.update(id=cid,q={k:q[k] for k in ['shape','Re','cx','cy','D']});assert row['fields'].shape==(4,3,64,64)
            ij,_=c.e.old.p.probes(row);values=np.full((160,4),np.nan)
            with np.load(cache/(cid+'.npz')) as z:
                for step in range(8,160,8):values[step-1]=[z['fields'][step+3,1,i,j] for i,j in ij]
            assert np.isfinite(values).sum()==76
            predictions=[]
            for method in cfg['methods']:
                pred,di=c.predict(method,model,q,row,values,cfg);assert pred.shape==(160,3,64,64) and np.isfinite(pred).all()
                target=c.OUT/'historical16/predictions'/method/(cid+'.npz');target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,prediction=pred)
                c.save(c.OUT/'historical16/registry'/method/(cid+'.json'),dict(id=cid,prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),file_sha256=c.sha(target),registered_before_fullfield_score=True,diagnostic=di,startup_fields=4,future_scalar_readings=76))
                predictions.append((method,pred))
            with np.load(cache/(cid+'.npz')) as z:truth=z['fields'][4:164]
            for method,pred in predictions:reports[method].append(dict(id=cid,q=q,horizons=c.e.old.p.metrics(pred,truth,row['mask'],row['times'])))
            records.append(dict(q=q,cache_file=str(cache/(cid+'.npz')),cache_sha256=meta['cache_sha256'],CFD_stable=True,max_Courant_dense=meta['max_Courant_dense'],historical_version=batch['method'],historical_summary=batch['historical_summary']))
            c.save(c.OUT/'historical16/status.json',dict(stage='evaluating',completed=len(records),total=16,utc=c.now()))
    result={}
    for method,rows in reports.items():
        summary={g:{h:c.e.old.p.summarize([z['horizons'][h] for z in rows if g=='all' or z['q']['historical_batch']==g]) for h in ['8s','40s']} for g in ['all',1,2]}
        result[method]=dict(cases=rows,summary=summary)
    c.check();c.save(c.OUT/'historical16/report.json',dict(reports=result,primary='final_system',role='Restored historical16 SEEN regression, never fresh independent confirmation',no_model_reselection=True,source=records))
    c.save(c.OUT/'historical16/status.json',dict(stage='complete',completed=16,total=16,utc=c.now(),primary=result['final_system']['summary']))
    print('HISTORICAL16_COMPLETE',json.dumps(result['final_system']['summary']),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:c.save(c.OUT/'historical16/failure.json',dict(utc=c.now(),error=repr(e),traceback=__import__('traceback').format_exc()));raise
