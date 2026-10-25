#!/usr/bin/env python3
"""CPU-only audit and versioned HDF5 view of the reruns to t=201.

The old HDF5 stays immutable. Full overlap is checked at native VTK byte level;
five regridded anchors per case tie that source to the cached HDF5 fields.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import resource
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flowproxy.geometry import geometry_channels
from flowproxy.interpolate import interpolate_points_to_grid
from flowproxy.paths import load_config
from flowproxy.geometry import make_grid
from scipy.interpolate import LinearNDInterpolator

OLD = ROOT / 'data/processed/dataset_long.h5'
OLD_RAW = ROOT / 'data/raw/cases_long'
NEW_RAW = ROOT / 'data/raw/cases_long40'
OUT = ROOT / 'outputs/cpu_round1'
TAIL = ROOT / 'data/processed/dataset_long40_tail_v1.h5'
VIEW = ROOT / 'data/processed/dataset_long40_v1.h5'


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frames(case: Path) -> list[tuple[float, Path]]:
    series = json.loads((case / 'VTK' / (case.name + '.vtm.series')).read_text())
    rows = []
    for row in series['files']:
        vtm = case / 'VTK' / row['name']
        dataset = ET.parse(vtm).find('.//DataSet[@name="internal"]')
        if dataset is None:
            raise ValueError(f'No internal dataset: {vtm}')
        rows.append((float(row['time']), vtm.parent / dataset.attrib['file']))
    rows.sort(key=lambda pair: pair[0])
    if len(set(t for t, _ in rows)) != len(rows):
        raise ValueError('Duplicate VTK time')
    return rows


def binary_array(node: ET.Element, root: ET.Element) -> np.ndarray:
    if node.attrib['format'] != 'binary' or root.get('compressor'):
        raise ValueError('Reader supports the observed uncompressed inline binary format only')
    endian = '<' if root.attrib['byte_order'] == 'LittleEndian' else '>'
    header = np.dtype(endian + ('u8' if root.attrib['header_type'] == 'UInt64' else 'u4'))
    types = {'Float32': 'f4', 'Float64': 'f8', 'Int32': 'i4', 'Int64': 'i8',
             'UInt32': 'u4', 'UInt64': 'u8', 'UInt8': 'u1'}
    raw = base64.b64decode(''.join((node.text or '').split()), validate=True)
    count = int(np.frombuffer(raw, dtype=header, count=1)[0])
    if count != len(raw) - header.itemsize:
        raise ValueError('VTK binary byte length mismatch')
    result = np.frombuffer(raw, dtype=np.dtype(endian + types[node.attrib['type']]),
                           offset=header.itemsize)
    components = int(node.get('NumberOfComponents', '1'))
    return result.reshape(-1, components)


def read_point_fields(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    piece = root.find('./UnstructuredGrid/Piece')
    if piece is None:
        raise ValueError('VTK Piece missing')
    pts_node = piece.find('./Points/DataArray')
    arrays = {node.attrib['Name']: node for node in piece.findall('./PointData/DataArray')}
    if pts_node is None or not {'U', 'p'} <= arrays.keys():
        raise ValueError('Expected point U and p arrays')
    points = binary_array(pts_node, root)
    velocity = binary_array(arrays['U'], root)
    pressure = binary_array(arrays['p'], root)
    if len(points) != len(velocity) or len(points) != len(pressure):
        raise ValueError('Point array lengths differ')
    return points[:, :2], np.concatenate([velocity[:, :2], pressure], axis=1)


def sample(path: Path, mask: np.ndarray, cfg: dict) -> np.ndarray:
    points, values = read_point_fields(path)
    result = interpolate_points_to_grid(points, values, cfg, fill=0.0)
    return (result * mask[None]).astype(np.float32)


class SampleCache:
    """Reuse only static interpolation geometry, checking coordinates each frame."""
    def __init__(self, cfg: dict, mask: np.ndarray):
        _, _, xx, yy = make_grid(cfg)
        self.query = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self.shape = xx.shape
        self.mask = mask
        self.points = None
        self.interp = None

    def sample(self, path: Path) -> np.ndarray:
        points, values = read_point_fields(path)
        if self.interp is None:
            self.points = points.copy()
            self.interp = LinearNDInterpolator(points, values, fill_value=0.0)
        else:
            if not np.array_equal(points, self.points):
                raise ValueError('Native grid moved; cached triangulation cannot be reused')
            self.interp.values = np.ascontiguousarray(values, dtype=np.float64)
        out = np.nan_to_num(self.interp(self.query), nan=0.0)
        result = out.reshape(*self.shape, 3).transpose(2, 0, 1).astype(np.float32)
        return (result * self.mask[None]).astype(np.float32)


def read_forces(case: Path) -> np.ndarray:
    rows = {}
    for path in sorted(case.glob('postProcessing/forceCoeffs*/**/coefficient.dat')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.startswith('#'):
                    continue
                cols = line.split()
                if len(cols) >= 5:
                    stamp = float(cols[0])
                    rows[stamp] = [stamp, float(cols[1]), float(cols[4])]
    if not rows:
        raise ValueError(f'No forces for {case}')
    return np.asarray([rows[t] for t in sorted(rows)], dtype=np.float32)


def relative(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> list[float]:
    fluid = mask > 0.5
    return [float(np.linalg.norm((a[c] - b[c])[fluid]) /
                  max(float(np.linalg.norm(b[c][fluid])), 1e-8)) for c in range(3)]


def geometry(group, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    attrs = group.attrs
    geo = geometry_channels(cfg, shape=str(attrs['shape']), cx=float(attrs['cx']),
                            cy=float(attrs['cy']), D=float(attrs['D']), Re=float(attrs['Re']))
    if not np.array_equal(geo['mask'], group['mask'][...]):
        raise ValueError('Recomputed mask differs from cached HDF5')
    if not np.array_equal(geo['geom'], group['geom'][...]):
        raise ValueError('Recomputed geometry differs from cached HDF5')
    return geo['mask'], geo['geom']


def probe(cid: str) -> None:
    cfg = load_config(ROOT / 'configs/default.yaml')
    with h5py.File(OLD, 'r') as handle:
        group = handle['cases'][cid]
        mask, _ = geometry(group, cfg)
        new = frames(NEW_RAW / cid)
        sampler = SampleCache(cfg, mask)
        print('PROBE', cid, flush=True)
        for index in [0, 3, 160]:
            result = sampler.sample(new[index][1])
            truth = group['fields'][index]
            print('GRID_MATCH', index, bool(np.array_equal(result, truth)),
                  relative(result, truth, mask), flush=True)
            if not np.array_equal(result, truth):
                raise ValueError('Probe regridding is not byte-exact to cached fields')
            if not np.array_equal(result, sample(new[index][1], mask, cfg)):
                raise ValueError('Cached sampler differs from the original interpolation')
        print('CLOCK', new[3][0], new[163][0], new[163][0] - new[3][0], flush=True)
    print('MAX_RSS_KIB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'protocol_id': 'cpu_data_integrity_v1',
        'overlap_proof': 'every native internal.vtu byte-exact old/new at 160..200',
        'hdf_grid_anchors_per_case': [0, 1, 2, 3, 160],
        'grid_match_atol': 0.0, 'grid_match_rtol': 0.0,
        'mask_geometry_match': 'exact', 'force_rows_match': 'exact float32 at all cached timestamps',
        'old_h5_immutable': True, 'resampling_time': False,
        'output_storage': 'versioned VDS reuses byte-equivalent cached overlap plus separately sampled new tail',
        'future_model_evaluation': 'not performed by this data audit',
        'threads': 1,
    }
    prereg = OUT / 'preregister_data.json'
    if prereg.exists() and json.loads(prereg.read_text()) != protocol:
        raise ValueError('Protocol already exists with different contents')
    save_json(prereg, protocol)
    if VIEW.exists() or TAIL.exists() or TAIL.with_suffix('.partial.h5').exists():
        raise ValueError('Refusing to overwrite a previous prepared data version')
    started = time.perf_counter()
    cfg = load_config(ROOT / 'configs/default.yaml')
    old_sha_before = sha_file(OLD)
    rows = []
    tail_tmp = TAIL.with_suffix('.partial.h5')
    audit = {'status': 'running', 'protocol': str(prereg.relative_to(ROOT)),
             'source_h5_sha256': old_sha_before, 'compared': rows}
    save_json(OUT / 'data_audit.json', audit)
    with h5py.File(OLD, 'r') as old, h5py.File(tail_tmp, 'x') as tail:
        split = {role: [item.decode() for item in old['splits'][role][...]]
                 for role in ['train', 'val', 'test']}
        roles = {cid: role for role, ids in split.items() for cid in ids}
        if len(roles) != 120:
            raise ValueError('Expected the original 120 cases')
        tail.attrs['derived_from_verified_reruns'] = True
        tail.create_group('cases')
        for cid in sorted(roles):
            case_started = time.perf_counter()
            group = old['cases'][cid]
            native_old = frames(OLD_RAW / cid)
            native_new = frames(NEW_RAW / cid)
            if len(native_old) != 161 or len(native_new) != 165:
                raise ValueError(f'{cid}: expected 161 old and 165 new frames')
            mask, _ = geometry(group, cfg)
            sampler = SampleCache(cfg, mask)
            new_meta = json.loads((NEW_RAW / cid / 'meta.json').read_text())
            for key in ['shape', 'Re', 'cx', 'cy', 'D']:
                if new_meta.get(key, 0.0 if key == 'cx' else None) != group.attrs[key]:
                    raise ValueError(f'{cid}: metadata differs for {key}')
            old_times = group['times'][...]
            new_times = np.asarray([t for t, _ in native_new], dtype=np.float32)
            if not np.array_equal(old_times, new_times[:161]):
                raise ValueError(f'{cid}: HDF5 time clock differs')
            native_manifest = hashlib.sha256()
            for (ot, op), (nt, npth) in zip(native_old, native_new[:161]):
                if ot != nt:
                    raise ValueError(f'{cid}: native timestamps differ')
                oh, nh = sha_file(op), sha_file(npth)
                if oh != nh:
                    raise ValueError(f'{cid}: native VTK differs at t={ot}')
                native_manifest.update(f'{ot:.8f}:{nh}\n'.encode())
            cached = group['fields'][...]
            if not np.isfinite(cached).all():
                raise ValueError(f'{cid}: cached fields not finite')
            anchors = []
            for index in protocol['hdf_grid_anchors_per_case']:
                grid = sampler.sample(native_new[index][1])
                if not np.array_equal(grid, cached[index]):
                    raise ValueError(f'{cid}: cached grid differs at anchor {index}')
                anchors.append({'index': index, 'time': native_new[index][0],
                                'relative_ux_uy_p': relative(grid, cached[index], mask)})
            forces = read_forces(NEW_RAW / cid)
            previous_forces = group['forces'][...]
            pick = np.searchsorted(forces[:, 0], previous_forces[:, 0])
            if np.any(pick >= len(forces)) or not np.array_equal(forces[pick], previous_forces):
                raise ValueError(f'{cid}: forces differ at cached timestamps')
            tail_fields = np.stack([sampler.sample(path) for _, path in native_new[161:]])
            if not np.isfinite(tail_fields).all():
                raise ValueError(f'{cid}: new tail fields not finite')
            dest = tail['cases'].create_group(cid)
            dest.create_dataset('fields', data=tail_fields, compression='gzip', compression_opts=1)
            dest.create_dataset('forces', data=forces, compression='gzip', compression_opts=1)
            dest.create_dataset('times', data=new_times[161:])
            for key, value in group.attrs.items():
                dest.attrs[key] = value
            if abs(native_new[163][0] - native_new[3][0] - 40.0) > 1e-10:
                raise ValueError(f'{cid}: full-40s endpoint missing')
            rows.append({'id': cid, 'split': roles[cid], 'native_overlap_frames': 161,
                         'native_overlap_byte_exact': True,
                         'native_overlap_manifest_sha256': native_manifest.hexdigest(),
                         'cached_fields_sha256': hashlib.sha256(cached.tobytes()).hexdigest(),
                         'grid_anchors': anchors, 'mask_geometry_exact': True,
                         'matched_force_rows': len(previous_forces), 'forces_exact': True,
                         'new_times': [t for t, _ in native_new[161:]],
                         'tail_fields_sha256': hashlib.sha256(tail_fields.tobytes()).hexdigest(),
                         'seconds': time.perf_counter() - case_started})
            tail.flush()
            audit['elapsed_seconds'] = time.perf_counter() - started
            save_json(OUT / 'data_audit.json', audit)
            print('AUDITED', len(rows), cid, 'native=161 grid=5 forces=exact',
                  f'{rows[-1]["seconds"]:.1f}s', flush=True)
    old_sha_after = sha_file(OLD)
    if old_sha_after != old_sha_before:
        raise ValueError('Original HDF5 changed during preparation')
    tail_tmp.replace(TAIL)
    with h5py.File(OLD, 'r') as old, h5py.File(VIEW, 'x', libver='latest') as view:
        view.attrs['protocol_id'] = 'dataset_long40_v1_verified_cached_overlap'
        view.attrs['old_h5_sha256'] = old_sha_before
        view.attrs['old_overlap_reused_after_native_equivalence_check'] = True
        root = view.create_group('cases')
        for cid in sorted(roles):
            group = old['cases'][cid]
            dest = root.create_group(cid)
            for key, value in group.attrs.items():
                dest.attrs[key] = value
            for key in ['geom', 'mask']:
                dest[key] = h5py.ExternalLink(OLD.name, f'/cases/{cid}/{key}')
            dest['forces'] = h5py.ExternalLink(TAIL.name, f'/cases/{cid}/forces')
            native = frames(NEW_RAW / cid)
            dest.create_dataset('times', data=np.asarray([t for t, _ in native], dtype=np.float32))
            layout = h5py.VirtualLayout((165, 3, 64, 64), dtype=np.float32)
            layout[:161] = h5py.VirtualSource(OLD.name, f'/cases/{cid}/fields', (161, 3, 64, 64))
            layout[161:] = h5py.VirtualSource(TAIL.name, f'/cases/{cid}/fields', (4, 3, 64, 64))
            dest.create_virtual_dataset('fields', layout, fillvalue=np.nan)
        sr = view.create_group('splits')
        for role, ids in split.items():
            sr.create_dataset(role, data=np.asarray(ids, dtype='S'))
    with h5py.File(OLD, 'r') as old, h5py.File(TAIL, 'r') as tail, h5py.File(VIEW, 'r') as view:
        for cid in sorted(roles):
            fields = view['cases'][cid]['fields']
            if not np.array_equal(fields[:161], old['cases'][cid]['fields'][...]):
                raise ValueError('VDS overlap validation failed')
            if not np.array_equal(fields[161:], tail['cases'][cid]['fields'][...]):
                raise ValueError('VDS tail validation failed')
    timing = {'protocol_id': 'complete_40_seconds_data_v1', 'support_indices': [0, 1, 2, 3],
              'origin_time': 160.74, 'future_indices_inclusive': [4, 163],
              'future_steps': 160, 'endpoint_time': 200.74, 'future_seconds': 40.0,
              'extra_last_index': 164, 'extra_last_time': 201.0,
              'storage_times_dtype': 'float32; authoritative physical times are from VTK series',
              'nonuniform_dt': [0.24, 0.26], 'temporal_resampling': False,
              'model_scores': 'not evaluated',
              'required_files_together': [OLD.name, TAIL.name, VIEW.name],
              'data_h5': str(VIEW.relative_to(ROOT)),
              'old_157_step_results_unchanged': True}
    save_json(OUT / 'complete40_data_protocol.json', timing)
    audit.update({'status': 'complete', 'cases': len(rows), 'all_native_overlap_byte_exact': True,
                  'all_cached_grid_anchors_exact': True, 'all_forces_exact': True,
                  'original_h5_unchanged': True, 'tail_h5_sha256': sha_file(TAIL),
                  'view_h5_sha256': sha_file(VIEW), 'elapsed_seconds': time.perf_counter() - started,
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    save_json(OUT / 'data_audit.json', audit)
    print('COMPLETE', len(rows), 'full40_steps=160', f'{audit["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', default=None)
    args = parser.parse_args()
    if args.probe:
        probe(args.probe)
    else:
        build()
