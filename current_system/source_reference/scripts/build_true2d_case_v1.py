#!/usr/bin/env python3
"""Create a new cold-start, true single-layer case from an existing snappy mesh."""
from pathlib import Path
import sys,re,shutil,argparse,json,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from geometry_attack_v2 import save,sha
from mesh_2d_repair_probe_v1 import command,points,HEADER

def build_case(original,target,delta_t=.005,end_time=201,write_interval=.25):
    original=Path(original).resolve();target=Path(target).resolve();base=(ROOT/'data/raw').resolve()
    assert original.is_relative_to(base) and target.is_relative_to(base), 'Use explicit data/raw case paths'
    assert delta_t>0 and end_time>0 and write_interval>0
    surface=target.with_name(target.name+'_surface_source');assert not target.exists() and not surface.exists()
    before={n:sha(original/'constant/polyMesh'/n) for n in ['points','faces','boundary']}
    target.parent.mkdir(parents=True,exist_ok=True)
    for case in [surface,target]:
        case.mkdir();shutil.copytree(original/'system',case/'system')
        shutil.copytree(original/'constant',case/'constant',ignore=shutil.ignore_patterns('polyMesh') if case==target else None)
        initial=original/'0.orig' if (original/'0.orig').exists() else original/'0'
        shutil.copytree(initial,case/'0')
        f=case/'system/controlDict';t=f.read_text()
        for key,value in [('startFrom','startTime'),('startTime',0),('endTime',end_time),('deltaT',delta_t),('writeInterval',write_interval)]:t=re.sub(r'\b'+key+r'\s+[^;]+;',f'{key} {value};',t)
        f.write_text(t)
    p=original/'constant/polyMesh';xyz=points(p/'points')
    faces=[np.array([int(i) for i in x.split()],int) for x in re.findall(r'^\d+\(([^\n]+)\)$',(p/'faces').read_text(),re.MULTILINE)]
    b=(p/'boundary').read_text();block=re.search(r'\bfrontAndBack\s*\{([^}]+)\}',b)[1]
    start=int(re.search(r'\bstartFace\s+(\d+)',block)[1]);n=int(re.search(r'\bnFaces\s+(\d+)',block)[1]);zmin=xyz[:,2].min()
    selected=[i for i in range(start,start+n) if np.max(np.abs(xyz[faces[i],2]-zmin))<1e-8];assert len(selected)*2==n
    d=surface/'constant/polyMesh/sets';d.mkdir(exist_ok=True)
    (d/'frontFaces').write_text('FoamFile {version 2.0;format ascii;class faceSet;object frontFaces;}\n'+str(len(selected))+'\n(\n'+'\n'.join(map(str,selected))+'\n)\n')
    (surface/'system/createPatchDict').write_text(HEADER%'createPatchDict'+'pointSync false;patches ({name front2d;patchInfo {type patch;} constructFrom set;set frontFaces;});\n')
    command(surface,'createPatch -overwrite')
    thickness=xyz[:,2].max()-zmin;assert thickness>0
    (target/'system/extrudeMeshDict').write_text(HEADER%'extrudeMeshDict'+f'constructFrom patch;sourceCase "{surface}";sourcePatches (front2d);exposedPatchName frontAndBack;flipNormals true;extrudeModel linearNormal;nLayers 1;expansionRatio 1;linearNormalCoeffs {{thickness {thickness};}} mergeFaces false;\n')
    command(target,'extrudeMesh')
    p=target/'constant/polyMesh/boundary';t=p.read_text()
    for name in ['front2d','frontAndBack']:t=re.sub(r'(\b'+name+r'\s*\{[^}]*?\btype\s+)\w+;',r'\g<1>empty;',t)
    p.write_text(t)
    for name in ['U','p']:
        p=target/'0'/name;t=p.read_text();t=re.sub(r'\bfrontAndBack\s*\{[^}]+\}', '"frontAndBack|front2d" {type empty;}',t);p.write_text(t)
    check=command(target,'checkMesh');assert 'Mesh OK.' in check.read_text()
    zz=np.unique(np.round(points(target/'constant/polyMesh/points')[:,2],8));assert len(zz)==2
    cells=int(re.search(r'^\s*cells:\s*(\d+)',check.read_text(),re.MULTILINE)[1]);assert cells==len(selected)
    # This runner deliberately does not invoke blockMesh/snappyHexMesh again.
    (target/'Allrun_true2d').write_text(f'#!/bin/bash\nset -euo pipefail\ncd "{target}"\nsource "{ROOT}/cfd/env.sh"\ncheckMesh > log.checkMesh_runtime\nicoFoam > log.icoFoam_runtime\nfoamToVTK -time "160:{end_time}" > log.foamToVTK_runtime\n')
    assert all(sha(original/'constant/polyMesh'/name)==h for name,h in before.items())
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(original),'target':str(target),
      'source_mesh_hashes':before,'source_unchanged':True,'cells':cells,'source_planar_faces':len(selected),
      'z_planes':zz.tolist(),'exact_single_layer':True,'mesh_OK':True,'checkMesh_sha256':sha(check),
      'delta_t':delta_t,'end_time':end_time,'write_interval':write_interval,'cold_initial_fields':True,
      'solver_not_run_by_builder':True,'run_script':'Allrun_true2d','original_Allrun_not_copied_or_executed':True,
      'builder_sha256':sha(__file__)}
    save(target/'true2d_build.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--delta-t',type=float,default=.005);p.add_argument('--end-time',type=float,default=201);p.add_argument('--write-interval',type=float,default=.25);a=p.parse_args()
    print(json.dumps(build_case(a.source,a.output,a.delta_t,a.end_time,a.write_interval),ensure_ascii=True))
