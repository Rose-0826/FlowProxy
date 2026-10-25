#!/usr/bin/env python3
"""Preserve exact generated VTK bytes, deduplicating static grid blocks."""
from pathlib import Path
import base64,gzip,hashlib,json,re,os

def digest(data):return hashlib.sha256(data).hexdigest()

def encode_vtk(path,case):
    path=Path(path).resolve();case=Path(case).resolve();vtk=case/'VTK'
    assert path.is_relative_to(vtk) and path.suffix in ['.vtu','.vtp']
    original=path.read_bytes();blocks=vtk/'.blocks';blocks.mkdir(exist_ok=True)
    parts=[];cursor=0
    for match in re.finditer(rb'<DataArray\b[^>]*>.*?</DataArray>',original,re.DOTALL):
        prefix=original[cursor:match.start()]
        if prefix:parts.append({'literal':base64.b64encode(prefix).decode('ascii')})
        data=match.group();key=digest(data);block=blocks/(key+'.gz')
        if not block.exists():
            temporary=block.with_suffix('.partial')
            with gzip.open(temporary,'wb',compresslevel=6) as f:f.write(data)
            assert digest(gzip.decompress(temporary.read_bytes()))==key
            temporary.replace(block)
        parts.append({'block':key});cursor=match.end()
    suffix=original[cursor:]
    if suffix:parts.append({'literal':base64.b64encode(suffix).decode('ascii')})
    assert any('block' in p for p in parts),'No inline VTK data arrays'
    pointer=path.with_name(path.name+'.dedup.json')
    assert not pointer.exists()
    record={'version':1,'original_file':path.name,'original_bytes':len(original),'original_sha256':digest(original),
            'block_directory':os.path.relpath(blocks,path.parent),'parts':parts}
    pointer.write_text(json.dumps(record,separators=(',',':')),encoding='utf-8')
    assert reconstruct(pointer)==original
    path.unlink()
    return record

def reconstruct(pointer):
    pointer=Path(pointer).resolve();record=json.loads(pointer.read_text(encoding='utf-8'))
    blocks=(pointer.parent/record['block_directory']).resolve()
    assert blocks.name=='.blocks' and blocks.parent.name=='VTK' and pointer.is_relative_to(blocks.parent)
    result=[]
    for part in record['parts']:
        if 'literal' in part:result.append(base64.b64decode(part['literal'],validate=True))
        else:
            key=part['block'];assert re.fullmatch('[0-9a-f]{64}',key)
            b=gzip.decompress((blocks/(key+'.gz')).read_bytes());assert digest(b)==key;result.append(b)
    data=b''.join(result)
    assert len(data)==record['original_bytes'] and digest(data)==record['original_sha256']
    return data

def restore(pointer):
    pointer=Path(pointer).resolve();record=json.loads(pointer.read_text(encoding='utf-8'))
    destination=pointer.parent/record['original_file'];assert destination.parent==pointer.parent and not destination.exists()
    destination.write_bytes(reconstruct(pointer));return destination

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('pointer');a=p.parse_args();print(restore(a.pointer))
