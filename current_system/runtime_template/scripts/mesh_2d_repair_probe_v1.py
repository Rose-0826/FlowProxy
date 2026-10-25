#!/usr/bin/env python3
"""Extract a planar face patch and extrude exactly one layer; two-case interface probe."""
from pathlib import Path
import sys,json,re,shutil,subprocess,shlex,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from geometry_attack_v2 import save,sha
OUT=ROOT/'outputs/mesh_2d_repair_probe_v1';RAW=ROOT/'data/raw/mesh_2d_repair_probe_v1'
HEADER='FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n'
def points(p):
    rows=re.findall(r'^\(([^()\n]+)\)$',p.read_text(),re.MULTILINE)
    return np.array([[float(v) for v in x.split()] for x in rows if len(x.split())==3])
def command(case,cmd):
    log=case/('log.'+cmd.split()[0]);line='source '+shlex.quote(str(ROOT/'cfd/env.sh'))+'; cd '+shlex.quote(str(case))+'; '+cmd
    if log.exists():
        i=1
        while (case/(log.name+f'.retry{i}')).exists():i+=1
        log=case/(log.name+f'.retry{i}')
    with log.open('x') as f:
        result=subprocess.run(['bash','-c',line],stdout=f,stderr=subprocess.STDOUT,timeout=180,env={**__import__('os').environ,'OMP_NUM_THREADS':'1'})
    if result.returncode:raise RuntimeError(str(log)+' '+log.read_text()[-1800:])
    return log
def run():
    OUT.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    ids=['cylinder_Re110_cy-0.2_D1','square_Re170_cy0.2_D1']
    protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':ids,
      'source':'original coarse snappy meshes; copied without mutating existing cases',
      'method':'extract frontAndBack faces at min z, createPatch, extrudeMesh linearNormal nLayers1',
      'purpose':'mesh structure and 2-second solver interface only; no new AI accuracy score',
      'required':'exactly two z planes, Mesh OK, successful icoFoam 0->2 seconds',
      'official_sources':['https://api.openfoam.com/2506/extrudeMesh_8C_source.html','https://api.openfoam.com/2312/classFoam_1_1extrudePatchMesh.html'],
      'script_sha256':sha(__file__)}
    if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',protocol)
    else:save(OUT/'compatibility_amendment.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'initial_failure':'OpenFOAM extrudeMesh requires exposedPatchName to exist in source mesh',
      'correction':'Use existing frontAndBack as exposed patch; keep original failure log',
      'no_changes_to_original_meshes_or_accuracy_claims':True,'script_sha256':sha(__file__)})
    results=[]
    for cid in ids:
        original=ROOT/'data/raw/geometry_attack_confirmation_d1_v1'/cid
        original_mesh_hashes={name:sha(original/'constant/polyMesh'/name) for name in ['points','faces','boundary']}
        source=RAW/(cid+'_surface_source');target=RAW/cid
        for case in [source,target]:
            if case.exists():continue
            case.mkdir()
            shutil.copytree(original/'system',case/'system')
            shutil.copytree(original/'constant',case/'constant',ignore=shutil.ignore_patterns('polyMesh') if case==target else None)
            shutil.copytree(original/'0.orig',case/'0')
            control=case/'system/controlDict';t=control.read_text();t=re.sub(r'\bstartTime\s+[^;]+;', 'startTime 0;',t);t=re.sub(r'\bstartFrom\s+[^;]+;', 'startFrom startTime;',t);t=re.sub(r'\bendTime\s+[^;]+;', 'endTime 2;',t);t=re.sub(r'\bdeltaT\s+[^;]+;', 'deltaT 0.005;',t);control.write_text(t)
        pp=original/'constant/polyMesh';a=points(pp/'points');faces=[np.array([int(v) for v in x.split()],int) for x in re.findall(r'^\d+\(([^\n]+)\)$',(pp/'faces').read_text(),re.MULTILINE)]
        boundary=(pp/'boundary').read_text();m=re.search(r'\bfrontAndBack\s*\{([^}]+)\}',boundary);assert m
        start=int(re.search(r'\bstartFace\s+(\d+)',m[1])[1]);n=int(re.search(r'\bnFaces\s+(\d+)',m[1])[1]);zmin=a[:,2].min()
        selected=[i for i in range(start,start+n) if np.max(np.abs(a[faces[i],2]-zmin))<1e-8];assert len(selected)*2==n
        d=source/'constant/polyMesh/sets';d.mkdir(exist_ok=True);(d/'frontFaces').write_text('FoamFile {version 2.0;format ascii;class faceSet;object frontFaces;}\n'+str(len(selected))+'\n(\n'+'\n'.join(map(str,selected))+'\n)\n')
        (source/'system/createPatchDict').write_text(HEADER%'createPatchDict'+'pointSync false;\npatches ({name front2d;patchInfo {type patch;} constructFrom set;set frontFaces;});\n')
        if not (source/'log.createPatch').exists():command(source,'createPatch -overwrite')
        (target/'system/extrudeMeshDict').write_text(HEADER%'extrudeMeshDict'+f'constructFrom patch;\nsourceCase "{source}";\nsourcePatches (front2d);\nexposedPatchName frontAndBack;\nflipNormals true;\nextrudeModel linearNormal;\nnLayers 1;\nexpansionRatio 1;\nlinearNormalCoeffs {{thickness 0.1;}}\nmergeFaces false;\n')
        command(target,'extrudeMesh')
        b=target/'constant/polyMesh/boundary';text=b.read_text()
        for name in ['front2d','back2d','frontAndBack']:
            text=re.sub(r'(\b'+name+r'\s*\{[^}]*?\btype\s+)\w+;',r'\g<1>empty;',text)
        b.write_text(text)
        for name in ['U','p']:
            f=target/'0'/name;t=f.read_text();t=re.sub(r'\bfrontAndBack\s*\{[^}]+\}', '"frontAndBack|front2d|back2d" { type empty; }',t);f.write_text(t)
        check=command(target,'checkMesh');assert 'Mesh OK.' in check.read_text()
        zz=np.unique(np.round(points(target/'constant/polyMesh/points')[:,2],8));assert len(zz)==2
        solve=command(target,'icoFoam');t=solve.read_text();ts=re.findall(r'^Time = (\S+)',t,re.MULTILINE);assert float(ts[-1])==2 and re.search(r'^End\s*$',t,re.MULTILINE)
        cells=int(re.search(r'^\s*cells:\s*(\d+)',check.read_text(),re.MULTILINE)[1])
        assert all(sha(original/'constant/polyMesh'/name)==h for name,h in original_mesh_hashes.items())
        results.append({'case':cid,'source_points_sha256':original_mesh_hashes['points'],
          'source_points_still_unchanged':True,'original_mesh_hashes':original_mesh_hashes,
          'old_z_planes':np.unique(np.round(a[:,2],8)).tolist(),'new_z_planes':zz.tolist(),'new_cells':cells,
          'source_front_face_count':len(selected),'exact_one_layer':cells==len(selected),'Mesh_OK':True,'icoFoam_2s_normal_end':True,
          'checkMesh_sha256':sha(check),'solver_log_sha256':sha(solve),'not_AI_score':True})
        save(OUT/'status.json',{'status':'running','completed':results});print('MESH_REPAIR_OK',results[-1],flush=True)
    save(OUT/'report.json',{'cases':results,'mesh_interface_fixed_for_two_probes':True,'no_new_long_CFD_or_AI_scores':True})
    save(OUT/'status.json',{'status':'complete','n':2});print('MESH_REPAIR_PROBE_COMPLETE',flush=True)

if __name__=='__main__':run()
