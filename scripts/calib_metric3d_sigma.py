"""学習データの LiDAR で Metric3Dv2 深度の誤差特性を測り、
ソフト深度分布の μ スケール s と σ(d) = sigma_min + k * d を決める.

val は評価に使うので較正には使わない (train split, augmentation なし).

usage: python scripts/calib_metric3d_sigma.py [num_train_samples]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from bevdepth.exps.nuscenes.mv.bev_depth_dinov3_metric3d import \
    BEVDepthLightningModel

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

m = BEVDepthLightningModel(batch_size_per_device=1).cuda().eval()
bb = m.model.backbone
ds = NuscDetDataset(ida_aug_conf=m.ida_aug_conf,
                    bda_aug_conf=m.bda_aug_conf,
                    classes=m.class_names,
                    data_root=m.data_root,
                    info_paths=m.train_info_paths,
                    is_train=False,
                    img_conf=m.img_conf,
                    num_sweeps=1,
                    return_depth=True)

preds, gts = [], []
idxs = np.linspace(0, len(ds) - 1, min(N, len(ds))).astype(int)
for i in idxs:
    imgs, mats, _, _, _, _, depth_labels = collate_fn([ds[i]],
                                                      is_return_depth=True)
    mats = {k: v.cuda() for k, v in mats.items()}
    with torch.no_grad(), torch.cuda.amp.autocast():
        feats = bb.get_cam_feats(imgs.cuda())
        bb._depth_prior(mats, feats.shape[-2:])
    if depth_labels.dim() == 5:
        depth_labels = depth_labels[:, 0]
    gt = depth_labels[0].float().unsqueeze(1).cuda()
    pred = F.interpolate(bb.last_metric_depth, gt.shape[-2:],
                         mode='bilinear', align_corners=False)
    v = (gt > 1.0) & (gt < 60.0)
    preds.append(pred[v].cpu())
    gts.append(gt[v].cpu())

p = torch.cat(preds).numpy()
g = torch.cat(gts).numpy()
print(f'samples={len(idxs)} lidar points={len(g)}')

# 1) scale: median(gt / pred)
s = float(np.median(g / p))
ps = p * s
print(f'scale s = median(gt/pred) = {s:.4f}')

# 2) error spread vs predicted depth (robust std = 1.4826 * MAD)
edges = np.array([1, 5, 10, 15, 20, 25, 30, 40, 50, 60])
centers, sig = [], []
print(f'{"pred depth":>12s} {"n":>8s} {"bias":>7s} {"robust_std":>10s} '
      f'{"std/d":>6s}')
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (ps >= lo) & (ps < hi)
    if sel.sum() < 200:
        continue
    e = g[sel] - ps[sel]
    rstd = 1.4826 * np.median(np.abs(e - np.median(e)))
    c = float(np.median(ps[sel]))
    centers.append(c)
    sig.append(rstd)
    print(f'{lo:5.0f}-{hi:<5.0f}m {sel.sum():8d} {np.median(e):+7.2f} '
          f'{rstd:10.2f} {rstd / c:6.3f}')

# 3) fit sigma(d) = a + k d (least squares on bin robust stds)
A = np.stack([np.ones(len(centers)), centers], 1)
a, k = np.linalg.lstsq(A, np.array(sig), rcond=None)[0]
k_only = float(np.sum(np.array(sig) * centers) / np.sum(np.square(centers)))
print(f'fit sigma(d) = {a:.3f} + {k:.4f} * d')
print(f'fit sigma(d) = {k_only:.4f} * d  (through origin)')
