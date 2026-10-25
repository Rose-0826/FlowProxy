"""Verify every packaged file against the evidence manifest, using only Python."""
from pathlib import Path
import sys, json, hashlib

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
manifest = root / 'manifest.json'
items = json.loads(manifest.read_text(encoding='utf-8'))['files']
expected = set()
total = 0
for item in items:
    name = item['file']
    path = (root / name).resolve()
    assert path.is_relative_to(root) and path != manifest, name
    assert name not in expected, 'Duplicate manifest item: ' + name
    expected.add(name)
    assert path.is_file() and path.stat().st_size == item['bytes'], name
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == item['sha256'], 'SHA mismatch: ' + name
    total += item['bytes']
actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != manifest}
assert actual == expected, {'missing': sorted(expected - actual), 'unexpected': sorted(actual - expected)}
print(json.dumps(dict(status='ALL_MANIFEST_FILES_VERIFIED',files=len(items),bytes=total,
                      manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())), flush=True)
