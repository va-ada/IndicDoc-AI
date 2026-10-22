"""
IndicDoc AI - Figures and summary numbers for the research study (paper and Phase-2 slides).

Reads only measured results: outputs/metrics/*_metrics.json, study_robustness.json, study_script_transfer.json,
ocr_evaluation.json and dataset/class_counts.csv. Writes figures to docs/figures/ and the summary numbers to
outputs/metrics/study_summary.json.
Usage: python scripts/study_figures.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS = PROJECT_ROOT / 'outputs' / 'metrics'
FIG = PROJECT_ROOT / 'docs' / 'figures'

NAVY, GOLD, GREEN, RED, GREY = '#14284B', '#D9A441', '#2E7D32', '#C62828', '#94A3B8'
MODELS = [('doclayout', 'authors', 'DocLayout-YOLO (authors)', NAVY),
          ('yolov8', 'proposed', 'YOLOv8s (ours)', GOLD),
          ('fasterrcnn', 'baseline', 'Faster R-CNN (ours)', GREY)]
CORR_LABELS = {'blur': 'Blur', 'noise': 'Noise', 'jpeg': 'JPEG q15', 'low_res': 'Half resolution', 'faded': 'Faded print'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})


def load(name):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


def language_figure(metrics, summary):
    langs = sorted(metrics['authors']['per_group'])
    x = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.27
    for i, (_, key, label, color) in enumerate(MODELS):
        g = metrics[key]['per_group']
        vals = [g[l]['map50'] for l in langs]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
        summary.setdefault('language_ci', {})[key] = min(vals) / max(vals)
        summary.setdefault('language_spread', {})[key] = max(vals) - min(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in langs], rotation=25)
    ax.set_ylabel('mAP@50')
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False)
    save(fig, 'fig_per_language.png')


def domain_figure(rob, summary):
    if not rob:
        return
    fig, ax = plt.subplots(figsize=(11, 4.2))
    doms = sorted(rob['doclayout']['clean']['per_domain'], key=lambda d: -rob['doclayout']['clean']['per_domain'][d]['map50'])
    x = np.arange(len(doms))
    w = 0.27
    for i, (mt, key, label, color) in enumerate(MODELS):
        if mt not in rob:
            continue
        pd = rob[mt]['clean']['per_domain']
        ax.bar(x + (i - 1) * w, [pd[d]['map50'] for d in doms], w, label=label, color=color)
        summary.setdefault('domain_ci', {})[key] = rob[mt]['clean']['domain_consistency']
    pd = rob['doclayout']['clean']['per_domain']
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={pd[d]['images']})" for d in doms], fontsize=8.5)
    ax.set_ylabel('mAP@50')
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False)
    save(fig, 'fig_per_domain.png')
    summary['per_domain_authors'] = {d: pd[d]['map50'] for d in doms}
    summary['domain_pages'] = {d: pd[d]['images'] for d in doms}


def robustness_figure(rob, summary):
    if not rob:
        return
    corrs = list(CORR_LABELS)
    x = np.arange(len(corrs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [3, 2]})
    w = 0.27
    for i, (mt, key, label, color) in enumerate(MODELS):
        if mt not in rob or 'robustness_index' not in rob[mt]:
            continue
        ri = rob[mt]['robustness_index']
        a1.bar(x + (i - 1) * w, [ri[c] for c in corrs], w, label=label, color=color)
        summary.setdefault('robustness_index', {})[key] = ri
        summary.setdefault('mean_ri', {})[key] = rob[mt]['mean_robustness_index']
        summary.setdefault('robust_map50', {})[key] = {c: rob[mt][c]['map50'] for c in ['clean'] + corrs}
    a1.axhline(1, color='black', lw=0.8, ls='--')
    a1.set_xticks(x)
    a1.set_xticklabels([CORR_LABELS[c] for c in corrs])
    a1.set_ylabel('Robustness Index (damaged / clean mAP@50)')
    a1.set_ylim(0, 1.1)
    a1.legend(fontsize=8.5, frameon=False, loc='lower left')
    a1.set_title('How much accuracy survives each kind of damage')
    # absolute mAP for the best model
    d = rob['doclayout']
    vals = [d['clean']['map50']] + [d[c]['map50'] for c in corrs]
    a2.barh(['Clean'] + [CORR_LABELS[c] for c in corrs][::-1][::-1], vals, color=[GREEN] + [NAVY] * len(corrs))
    for i, v in enumerate(vals):
        a2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.set_xlabel('mAP@50')
    a2.set_title('DocLayout-YOLO, absolute')
    save(fig, 'fig_robustness.png')


def class_figure(metrics, summary):
    counts = {}
    with open(PROJECT_ROOT / 'dataset' / 'class_counts.csv') as f:
        for row in csv.DictReader(f):
            counts[row['class_name']] = int(row['train'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (_, key, label, color) in zip(axes, [MODELS[0], MODELS[1]]):
        ap = {c: v for c, v in metrics[key]['per_class_ap50'].items() if v is not None}
        names = [c for c in ap if counts.get(c, 0) > 0]
        xs = np.array([counts[c] for c in names])
        ys = np.array([ap[c] for c in names])
        rho = spearman(xs, ys)
        summary.setdefault('class_freq_spearman', {})[key] = rho
        ax.scatter(xs, ys, color=color, s=28)
        ax.set_xscale('log')
        for c, xv, yv in zip(names, xs, ys):
            if (key == 'authors' and yv < 0.25) or xv > 3000:
                ax.annotate(c, (xv, yv), fontsize=8, xytext=(4, 3), textcoords='offset points')
        n_zero = int((ys < 0.05).sum())
        if n_zero:
            ax.text(0.03, 0.93, f'{n_zero} of {len(ys)} classes have AP@50 < 0.05', transform=ax.transAxes, fontsize=9)
        ax.set_xlabel('Training boxes of this class (our 2,293-page subset, log scale)')
        ax.set_ylabel('AP@50 on test pages')
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f'{label}  (Spearman rho = {rho:.2f})', fontsize=11)
    save(fig, 'fig_class_frequency.png')
    # hardest / easiest classes for the authors' model
    ap = {c: v for c, v in metrics['authors']['per_class_ap50'].items() if v is not None}
    order = sorted(ap, key=ap.get)
    summary['hardest_classes'] = {c: ap[c] for c in order[:6]}
    summary['easiest_classes'] = {c: ap[c] for c in order[-6:][::-1]}


def transfer_figure(tr, summary):
    if not tr:
        return
    groups = ['other scripts', 'dravidian (held out)']
    x = np.arange(2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (name, label, color) in enumerate([('seen', 'Trained on all 12 languages', NAVY),
                                              ('held_out', 'Trained WITHOUT Dravidian scripts', GOLD)]):
        vals = [tr[name][g]['map50'] for g in groups]
        a1.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=label, color=color)
        for xi, v in zip(x + (i - 0.5) * 0.38, vals):
            a1.text(xi, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    a1.set_xticks(x)
    a1.set_xticklabels(['Other 8 languages\n(seen by both)', 'Kannada, Malayalam,\nTamil, Telugu'])
    a1.set_ylabel('mAP@50 (test pages)')
    a1.legend(fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    langs = sorted(tr['seen']['per_language'])
    ratios = [tr['held_out']['per_language'][l] / tr['seen']['per_language'][l] for l in langs]
    colors = [RED if l in tr['held_out_languages'] else NAVY for l in langs]
    a2.bar([l.capitalize() for l in langs], ratios, color=colors)
    a2.axhline(1, color='black', lw=0.8, ls='--')
    a2.set_ylabel('held-out model / all-language model')
    a2.set_title('Per language (red = never seen in training)', fontsize=11)
    a2.tick_params(axis='x', rotation=40)
    save(fig, 'fig_script_transfer.png')
    summary['cstr'] = tr['cstr']
    summary['cstr_other'] = tr['cstr_other_scripts']
    summary['transfer'] = {n: {g: tr[n][g]['map50'] for g in groups} for n in ('seen', 'held_out')}
    summary['transfer_epochs'] = {n: tr[n].get('epochs') for n in ('seen', 'held_out')}
    summary['transfer_pages'] = tr['splits']['seen']['train_pages']


def ocr_figure(ocr, summary):
    if not ocr:
        return
    langs = sorted(ocr['per_language'])
    vals = [ocr['per_language'][l]['correct'] / ocr['per_language'][l]['pages'] for l in langs]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar([l.capitalize() for l in langs], vals, color=NAVY)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Pages with the right script')
    ax.set_title(f"Script detection by region probe: {ocr['script_detection_correct']}/{ocr['pages']} pages "
                 f"(Tesseract OSD: 70/120)", fontsize=11)
    ax.tick_params(axis='x', rotation=35)
    save(fig, 'fig_script_detection.png')


def main():
    metrics = {key: load(f'{key}_metrics.json') for _, key, _, _ in MODELS}
    rob = load('study_robustness.json')
    tr = load('study_script_transfer.json')
    ocr = load('ocr_evaluation.json')
    summary = {}
    language_figure(metrics, summary)
    domain_figure(rob, summary)
    robustness_figure(rob, summary)
    class_figure(metrics, summary)
    transfer_figure(tr, summary)
    ocr_figure(ocr, summary)
    (METRICS / 'study_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
