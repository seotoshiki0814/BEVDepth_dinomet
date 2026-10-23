"""同じサンプル集合 (val, augmentation なし) で学習前後の検出 loss を比較する.

学習曲線は batch ごとにサンプルが違うので、loss の低下に「学習の効果」と
「サンプル(シーン)の違い」が混ざる。ここでは全モデルを同じサンプルで評価し、
サンプル単位の対応ありで比較する (Wilcoxon 符号順位検定、片側: 学習後 < 学習前).

usage:
  python scripts/fixed_set_loss.py --exp dinov3_metric3d --depth_prior_mode lift \
      --ckpts init outputs/xxx/.../epoch=0-step=323.ckpt \
      --out outputs/xxx/fixed_set_loss.png
  ("init" = 学習前: run_cli と同じ seed 0 で初期化したモデル)
"""
import argparse
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT2 = '#52514e'
GRID = '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']


def build(exp, mode, ckpt):
    if exp == 'dinov3_metric3d':
        from bevdepth.exps.nuscenes.mv.bev_depth_dinov3_metric3d import \
            BEVDepthLightningModel
        kw = dict(depth_prior_mode=mode) if mode else {}
    else:
        from bevdepth.exps.nuscenes.mv.bev_depth_dinov3 import \
            BEVDepthLightningModel
        kw = {}
    pl.seed_everything(0)
    m = BEVDepthLightningModel(batch_size_per_device=1, **kw)
    if ckpt != 'init':
        sd = torch.load(ckpt, map_location='cpu')['state_dict']
        m.load_state_dict(sd)
    return m.cuda().eval()


@torch.no_grad()
def per_sample_loss(m, loader):
    # The val dataset returns empty GT on purpose ("Temporary solution for
    # test"), so rebuild it with get_gt(). No bda/ida augmentation on val,
    # so these boxes are in the same frame as the network input.
    ds = loader.dataset
    cams = m.ida_aug_conf['cams']
    rows = []
    for i, batch in enumerate(loader):
        sweep_imgs, mats, _, img_metas = batch[:4]
        assert img_metas[0]['token'] == ds.infos[i]['sample_token']
        gt_boxes, gt_labels = ds.get_gt(ds.infos[i], cams)
        mats = {k: v.cuda() for k, v in mats.items()}
        with torch.cuda.amp.autocast():
            preds = m.model(sweep_imgs.cuda(), mats)
        targets = m.model.get_targets([gt_boxes.cuda()], [gt_labels.cuda()])
        det = m.model.loss(targets, preds)
        c = m.model.head.last_loss_components
        rows.append(dict(token=img_metas[0]['token'],
                         detection=float(det),
                         heatmap=float(c['loss_heatmap']),
                         bbox=float(c['loss_bbox'])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='dinov3_metric3d')
    ap.add_argument('--depth_prior_mode', default=None)
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    labels = args.labels or [
        'init (untrained)' if c == 'init' else c.split('/')[-1]
        for c in args.ckpts
    ]

    results = {}
    for lab, ck in zip(labels, args.ckpts):
        m = build(args.exp, args.depth_prior_mode, ck)
        results[lab] = per_sample_loss(m, m.val_dataloader())
        print(f'[{lab}] mean detection={np.mean([r["detection"] for r in results[lab]]):.3f}',
              flush=True)
        del m
        torch.cuda.empty_cache()

    keys = ['detection', 'heatmap', 'bbox']
    base = labels[0]
    lines = []
    for k in keys:
        b = np.array([r[k] for r in results[base]])
        for lab in labels[1:]:
            v = np.array([r[k] for r in results[lab]])
            p = stats.wilcoxon(v, b, alternative='less').pvalue
            win = (v < b).mean()
            lines.append(
                f'{k:10s} {base} -> {lab}: mean {b.mean():.3f} -> '
                f'{v.mean():.3f} ({(v.mean() - b.mean()) / b.mean() * 100:+.1f}%)'
                f', median {np.median(b):.3f} -> {np.median(v):.3f}, '
                f'lower on {win * 100:.0f}% of samples, Wilcoxon p={p:.1e}')
    print('\n'.join(lines))

    with open(args.out.rsplit('.', 1)[0] + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['token'] + [f'{lab}:{k}' for lab in labels for k in keys])
        for i in range(len(results[base])):
            w.writerow([results[base][i]['token']] +
                       [results[lab][i][k] for lab in labels for k in keys])

    # Paired plot: one panel per loss, x = model, one gray line per sample.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURFACE)
    for ax, k in zip(axes, keys):
        data = np.array([[r[k] for r in results[lab]] for lab in labels])
        x = np.arange(len(labels))
        for j in range(data.shape[1]):
            ax.plot(x, data[:, j], color='#c3c2b7', lw=0.8, alpha=0.7)
        for i in x:
            ax.scatter(np.full(data.shape[1], i), data[i], s=14,
                       color=SERIES[i % len(SERIES)], zorder=3,
                       edgecolor=SURFACE, linewidth=0.5)
            med = np.median(data[i])
            ax.hlines(med, i - 0.25, i + 0.25, colors=TEXT, lw=2.2, zorder=4)
            ax.annotate(f'median {med:.2f}', (i + 0.28, med), va='center',
                        fontsize=9, color=TEXT)
        if k != 'bbox':
            ax.set_yscale('log')
        ax.set_xticks(x, labels, color=TEXT2)
        ax.set_xlim(-0.4, len(labels) - 0.2)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis='y', color=GRID, lw=0.8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT2)
        b = data[0]
        sub = []
        for i in range(1, len(labels)):
            p = stats.wilcoxon(data[i], b, alternative='less').pvalue
            sub.append(f'{labels[i]}: lower on {(data[i] < b).mean() * 100:.0f}%'
                       f' of samples, Wilcoxon p={p:.1e}')
        ax.set_title(f'{k} loss' + (' (log scale)' if k != 'bbox' else '')
                     + '\n' + '\n'.join(sub), loc='left', fontsize=10,
                     color=TEXT)
    fig.suptitle(f'Same {len(results[base])} val samples (no augmentation) '
                 'evaluated by each model - one gray line per sample, '
                 'black bar = median', x=0.01, ha='left', fontsize=11,
                 color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=110, facecolor=SURFACE)
    print('saved', args.out)


if __name__ == '__main__':
    main()
