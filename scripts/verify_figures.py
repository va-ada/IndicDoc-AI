"""
Figures of the IndicDoc Verify benchmark (from outputs/metrics/verify_benchmark_*.json, written by
scripts/evaluate_verify.py) for the report, slides and paper.

    docs/figures/fig_verify_by_type.png   share of injected errors found at the right question, per error type,
                                          clean renders vs scan-degraded pages
    docs/figures/fig_verify_ablation.png  found / false alarms with and without the number pass and the
                                          geometric option segmentation, and with a noisy layout
Usage: python scripts/verify_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'outputs' / 'metrics'
FIG = ROOT / 'docs' / 'figures'
INDIGO, GOLD, GREY = '#14284B', '#D9A441', '#94A3B8'


def load(name):
    p = M / f'verify_benchmark_{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', alpha=0.3, lw=0.6)


def by_type(main):
    cases = main['cases']
    types = list(main['labels'])
    rates = {}
    for deg in (False, True):
        rates[deg] = []
        for t in types:
            rows = [c for c in cases if c['type'] == t and c['degraded'] == deg]
            rates[deg].append(sum(c['found_at_place'] for c in rows) / len(rows) if rows else 0)
    order = sorted(range(len(types)), key=lambda i: rates[False][i] + rates[True][i])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    y = range(len(types))
    ax.barh([k + 0.2 for k in y], [rates[False][i] for i in order], 0.4, color=INDIGO, label='clean render (born-digital PDF)')
    ax.barh([k - 0.2 for k in y], [rates[True][i] for i in order], 0.4, color=GOLD, label='scan-degraded')
    ax.set_yticks(list(y), [main['labels'][types[i]] for i in order])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('share of injected errors found at the right question')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_verify_by_type.png', dpi=200)
    plt.close(fig)


def ablation():
    runs = [('Full system', 'oracle'), ('No number pass', 'oracle_no_number_pass'),
            ('No layout-based options', 'oracle_no_geometry'), ('Noisy layout', 'noisy')]
    found, alarms, names = [], [], []
    for label, name in runs:
        r = load(name)
        if r:
            names.append(label)
            found.append(r['overall']['found_at_place'])
            alarms.append(r['overall']['false_alarms_per_clean_pair'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6))
    a1.barh(names[::-1], found[::-1], color=INDIGO)
    a1.set_xlim(0, 1.05)
    a1.set_xlabel('errors found at the right question')
    a2.barh(names[::-1], alarms[::-1], color=GOLD)
    a2.set_xlabel('false alarms per clean paper pair')
    a2.set_yticks([])
    for ax, vals, fmt in ((a1, found[::-1], '{:.0%}'), (a2, alarms[::-1], '{:.1f}')):
        for k, v in enumerate(vals):
            ax.text(v, k, ' ' + fmt.format(v), va='center', fontsize=8.5)
        style(ax)
    a2.set_xlim(0, max(alarms) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_verify_ablation.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main = load('oracle')
    by_type(main)
    ablation()
    print('figures written to', FIG)
