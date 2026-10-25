#!/usr/bin/env python3
"""Static competition figure; all evaluated cases retained, no smoothing or selection.

Usage: python plot_frozen_evidence.py --root /root/autodl-tmp/flowproxy
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    old_path = root / 'outputs/cpu_round1/full40_eval/complete40_test.json'
    new_path = root / 'outputs/frozen_confirmation_v1/report.json'
    old, new = load(old_path), load(new_path)
    dest = root / 'outputs/frozen_evidence_figure_v2'
    dest.mkdir(exist_ok=True)
    if (dest / 'frozen_evidence.png').exists():
        raise FileExistsError('Refusing to overwrite delivered figure')
    primary = set(new['protocol']['primary_ids'])
    models = ['F0', 'K8']
    colors = {'F0': '#0072B2', 'K8': '#B34E00'}
    markers = {'F0': 'o', 'K8': 's'}
    source_data = {'old32': {}, 'confirmation12': {}, 'R_formula':
        'max(mean_t(eUx)/.05, mean_t(eUy)/.05, mean_t(ep)/.08, '
        'end(eUx)/.05, end(eUy)/.05, end(ep)/.08); pass iff R<1'}
    style = {'font.family': 'DejaVu Sans', 'font.size': 8,
             'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7,
             'axes.spines.top': False, 'axes.spines.right': False,
             'pdf.fonttype': 42, 'svg.fonttype': 'path',
             'figure.facecolor': 'white', 'axes.facecolor': 'white'}
    with plt.rc_context(style):
        fig, ax = plt.subplots(2, 2, figsize=(10.2, 7.3), layout='constrained')
        fig.suptitle('Frozen autonomous forecasts: supported scope and remaining failures', fontsize=12)
        groups = [('Position |cy|=0.8, D=1', lambda s: '_D1' == s[s.rfind('_'):]),
                  ('Diameter D=0.8', lambda s: s.endswith('_D0.8')),
                  ('Diameter D=1.2', lambda s: s.endswith('_D1.2'))]
        for model in models:
            rows = old['models'][model]['horizons']['8s']['cases']
            source_data['old32'][model] = rows
            for i, (label, predicate) in enumerate(groups):
                selected = [r for r in rows if predicate(r['id'])]
                vals = [r['R'] for r in selected]
                offset = -.13 if model == 'F0' else .13
                # Deterministic horizontal positions reveal each case; no value jitter.
                spread = np.linspace(-.085, .085, len(vals))
                ax[0, 0].scatter(i + offset + spread, vals, s=23,
                                color=colors[model], marker=markers[model],
                                label=model if i == 0 else None,
                                facecolors='none' if model == 'F0' else colors[model])
        ax[0, 0].axhline(1, color='#555555', linestyle='--', linewidth=.9)
        ax[0, 0].set(xticks=range(3),
                     xticklabels=['Position (n=16)\npass 15 to 16', 'D=0.8 (n=8)\npass 0 to 0', 'D=1.2 (n=8)\npass 3 to 3'],
                     ylabel='Case R at 8 s (lower is better)', ylim=(0, 8.7), xlim=(-.55, 2.55),
                     xlabel='Pass counts shown as F0 to K8',
                     title='A  Original held-out 32: all three predefined groups')
        ax[0, 0].legend(loc='center right', frameon=False)

        rows_by_model = {m: new['models'][m]['groups']['all12']['8s']['cases'] for m in models}
        ordered = [r for r in rows_by_model['K8'] if r['id'] in primary]
        ordered += [r for r in rows_by_model['K8'] if r['id'] not in primary]
        for model in models:
            by_id = {r['id']: r for r in rows_by_model[model]}
            source_data['confirmation12'][model] = rows_by_model[model]
            y = np.arange(12) + (-.12 if model == 'F0' else .12)
            ax[0, 1].scatter([by_id[r['id']]['R'] for r in ordered], y, s=25,
                            color=colors[model], marker=markers[model],
                            facecolors='none' if model == 'F0' else colors[model], label=model)
        labels = []
        for r in ordered:
            name = r['id'].replace('cylinder_', 'C ').replace('square_', 'S ')
            name = name.replace('_', ' ').replace('Re', 'Re=')
            labels.append(('* ' if r['id'] in primary else '') + name)
        ax[0, 1].axvline(1, color='#555555', linestyle='--', linewidth=.9)
        ax[0, 1].axhline(3.5, color='#999999', linewidth=.7)
        ax[0, 1].set(yticks=range(12), yticklabels=labels, ylim=(11.6, -.6), xlim=(0, 1.85),
                     xlabel='Case R at 8 s; * = four preregistered primary cases',
                     title='B  New confirmation: all 12 cases retained')
        ax[0, 1].tick_params(axis='y', labelsize=7)
        ax[0, 1].legend(loc='lower right', frameon=False)

        for panel, source, scope, title in [
            (ax[1, 0], old, 'old32', 'C  Original 32: endpoint Uy error'),
            (ax[1, 1], new, 'new12', 'D  New 12: endpoint Uy error')]:
            for model in models:
                if scope == 'old32':
                    blocks = source['models'][model]['horizons']
                else:
                    blocks = source['models'][model]['groups']['all12']
                x = [2, 8, 20, 40]
                vals = [blocks[f'{s}s']['summary']['end'][1] * 100 for s in x]
                panel.plot(x, vals, color=colors[model], marker=markers[model],
                           linestyle='--' if model == 'F0' else '-', linewidth=1.2,
                           markersize=4, label=model)
                panel.annotate(f'{model}: {vals[-1]:.1f}%', (x[-1], vals[-1]),
                               xytext=(-3, -16 if model == 'F0' else 8), ha='right',
                               textcoords='offset points', color=colors[model], fontsize=7)
            panel.axhline(5, color='#555555', linestyle=':', linewidth=.9, label='Team Uy target: 5%')
            panel.set(xlabel='Forecast horizon (s)', ylabel='Case-mean endpoint relative L2 (%)',
                      xticks=[2, 8, 20, 40], xlim=(0, 42), ylim=(0, 82), title=title)
            panel.legend(loc='upper left', frameon=False)
        for panel in ax.flat:
            panel.grid(axis='y', color='#DDDDDD', linewidth=.4)
            panel.set_axisbelow(True)
        for fmt in ['png', 'pdf', 'svg']:
            fig.savefig(dest / f'frozen_evidence.{fmt}', dpi=300, facecolor='white', bbox_inches=None)
        plt.close(fig)
    (dest / 'figure_source_data.json').write_text(json.dumps(source_data, indent=2), encoding='utf-8')
    with Image.open(dest / 'frozen_evidence.png') as im:
        image_metadata = {'pixels': im.size, 'mode': im.mode, 'dpi': im.info.get('dpi')}
    manifest = {
        'destination': 'Provisional competition report / slides; no journal specification claimed',
        'source_sha256': {str(old_path): digest(old_path), str(new_path): digest(new_path)},
        'script_sha256': digest(Path(__file__)), 'package_version': {'matplotlib': matplotlib.__version__},
        'physical_size_inches': [10.2, 7.3], 'raster': image_metadata,
        'transformations': ['R scatter retains all cases', 'deterministic horizontal separation without value jitter',
                            'endpoint Uy case arithmetic mean converted to percent',
                            'lines connect four evaluated horizons; no smoothing or inferred observations'],
        'replication': 'Independent CFD cases; one checkpoint per model; no uncertainty interval or seed replication claimed',
        'exclusions': [], 'missing_data': 'None for the 2/8/20/40 s fields used here',
        'posthoc_scope': 'D=1 main scope chosen after original test32; fixed before all new confirmation predictions and scoring',
        'target_note': '5/5/8% are team thresholds, not official competition criteria; R uses time mean and endpoint, not max_t',
        'alt_text': 'Original position cases mostly pass at eight seconds, while all thin-diameter cases fail. '
                    'Both frozen models pass four new primary cases; K8 passes ten of all twelve new cases. '
                    'Endpoint Uy errors rise strongly by forty seconds in both old and new sets.',
        'software_reference': 'Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). '
                              'Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. '
                              'https://doi.org/10.48550/arXiv.2609.00065 (record checked 2026-10-02)',
        'output_sha256': {f.name: digest(f) for f in dest.glob('frozen_evidence.*')},
    }
    (dest / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(dest), 'raster': image_metadata}))


if __name__ == '__main__':
    main()
