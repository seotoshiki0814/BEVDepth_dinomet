"""TensorBoard ログから loss 推移を描き、減少を統計的に示す PNG を出力する.

各指標 1 パネル (small multiples, y 軸は 1 本):
  - 薄い線: step ごとの生値 (batch=1 なので揺れが大きい)
  - 濃い線: 移動平均 (--window step)
  - 注記: 最初/最後の N step の平均と変化率、Spearman ρ (step vs loss)、
          Mann-Whitney U の片側 p 値 (最後の N step < 最初の N step)

usage:
  python scripts/plot_loss.py --logdir outputs/xxx/lightning_logs/version_0 \
      --out outputs/xxx/loss_curve.png [--window 25] [--edge 50]
"""
import argparse
import glob
import os.path as osp

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator  # noqa: E402

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT2 = '#52514e'
GRID = '#e4e3df'
SERIES = '#2a78d6'

PANELS = [
    ('train/loss_heatmap', 'Heatmap loss (GaussianFocal)', True),
    ('train/loss_bbox', 'BBox loss (L1 × 0.25)', False),
    ('detection_loss', 'Detection loss (= training loss)', True),
    ('depth_loss_monitor', 'Depth loss monitor (not trained, log only)',
     False),
    ('depth_loss', 'Depth loss (BCE × 3)', False),
]


def moving_average(v, w):
    w = max(1, min(w, len(v)))
    c = np.cumsum(np.insert(v, 0, 0.0))
    ma = (c[w:] - c[:-w]) / w
    return np.arange(w - 1, len(v)), ma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=int, default=25)
    ap.add_argument('--edge', type=int, default=50)
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    ev = sorted(glob.glob(osp.join(args.logdir, 'events.out.tfevents.*')))
    ea = EventAccumulator(ev[-1], size_guidance={'scalars': 0})
    ea.Reload()
    tags = set(ea.Tags()['scalars'])
    panels = [p for p in PANELS if p[0] in tags]

    ncol = 2
    nrow = (len(panels) + 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.2 * nrow),
                             facecolor=SURFACE, squeeze=False)
    report = []
    for ax, (tag, label, logy) in zip(axes.flat, panels):
        sc = ea.Scalars(tag)
        steps = np.array([s.step for s in sc])
        vals = np.array([s.value for s in sc], dtype=float)
        n = min(args.edge, len(vals) // 3)
        first, last = vals[:n], vals[-n:]
        change = (last.mean() - first.mean()) / first.mean() * 100
        rho, p_rho = stats.spearmanr(steps, vals)
        p_mw = stats.mannwhitneyu(last, first, alternative='less').pvalue

        ax.set_facecolor(SURFACE)
        ax.plot(steps, vals, color=SERIES, alpha=0.22, lw=1.0)
        i, ma = moving_average(vals, args.window)
        ax.plot(steps[i], ma, color=SERIES, lw=2.0)
        for seg, xs in ((first, steps[:n]), (last, steps[-n:])):
            ax.hlines(seg.mean(), xs[0], xs[-1], colors=TEXT, lw=1.2,
                      linestyles='--')
        if logy:
            ax.set_yscale('log')
        ax.set_title(label, loc='left', fontsize=12, color=TEXT)
        ax.set_xlabel('step', color=TEXT2)
        ax.set_ylabel('loss' + (' (log scale)' if logy else ''), color=TEXT2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=TEXT2)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        verdict = ('significant decrease' if (p_mw < 0.05 and rho < 0)
                   else 'no significant decrease')
        txt = (f'mean first {n} steps {first.mean():.2f} -> last {n} steps '
               f'{last.mean():.2f} ({change:+.1f}%)\n'
               f'Spearman rho = {rho:+.2f} (p = {p_rho:.1e})   '
               f'Mann-Whitney one-sided p = {p_mw:.1e}\n=> {verdict}')
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha='right',
                va='top', fontsize=9, color=TEXT,
                bbox=dict(facecolor=SURFACE, edgecolor=GRID, pad=4))
        report.append(f'{tag:22s} first{n}={first.mean():9.3f} '
                      f'last{n}={last.mean():9.3f} change={change:+6.1f}% '
                      f'rho={rho:+.3f} (p={p_rho:.1e}) MW p={p_mw:.1e} '
                      f'-> {verdict}')
    for ax in list(axes.flat)[len(panels):]:
        ax.axis('off')

    title = args.title or osp.normpath(args.logdir)
    fig.suptitle(f'{title}\nthin: per-step value / thick: {args.window}-step '
                 f'moving average / dashed: mean of first & last '
                 f'{args.edge} steps',
                 x=0.01, ha='left', fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=110, facecolor=SURFACE)
    print('\n'.join(report))
    print('saved', args.out)


if __name__ == '__main__':
    main()
