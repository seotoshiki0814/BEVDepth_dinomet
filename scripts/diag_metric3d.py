"""Metric3Dv2 深度 prior を LiDAR 深度 GT と比較し、焦点距離変換の正しさを確認する.

usage: python scripts/diag_metric3d.py [concat|replace] [num_batches]
"""
import sys
import time

import torch
import torch.nn.functional as F

from bevdepth.exps.nuscenes.mv.bev_depth_dinov3_metric3d import \
    BEVDepthLightningModel

mode = sys.argv[1] if len(sys.argv) > 1 else 'concat'
num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 3

m = BEVDepthLightningModel(batch_size_per_device=1,
                           depth_prior_mode=mode).cuda()
m.train()
bb = m.model.backbone
it = iter(m.train_dataloader())
for b in range(num_batches):
    sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels = next(it)
    mats = {k: v.cuda() for k, v in mats.items()}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.cuda.amp.autocast():
        preds, depth_preds = m(sweep_imgs.cuda(), mats)
    torch.cuda.synchronize()
    dt = time.time() - t0

    # LiDAR GT: (B, N, H, W) sparse depth at input resolution.
    if depth_labels.dim() == 5:
        depth_labels = depth_labels[:, 0]
    gt = depth_labels.cuda().float()
    gt = gt.reshape(-1, 1, *gt.shape[-2:])
    pred = F.interpolate(bb.last_metric_depth, size=gt.shape[-2:],
                         mode='bilinear', align_corners=False)
    valid = (gt > 1.0) & (gt < 60.0)
    g, p = gt[valid], pred[valid]
    absrel = ((p - g).abs() / g).mean().item()
    scale = (g.median() / p.median()).item()
    d1 = (torch.max(p / g, g / p) < 1.25).float().mean().item()

    targets = m.model.get_targets([x.cuda() for x in gt_boxes],
                                  [x.cuda() for x in gt_labels])
    det_loss = m.model.loss(targets, preds)
    depth_loss = m.get_depth_loss(depth_labels.cuda(), depth_preds)
    print(f'[batch {b}] n_lidar={valid.sum().item()} AbsRel={absrel:.3f} '
          f'delta<1.25={d1:.3f} median(gt)/median(pred)={scale:.3f} '
          f'prior_coverage={bb.last_depth_prior.sum(1).mean():.3f} '
          f'det_loss={det_loss.item():.1f} depth_loss={depth_loss.item():.2f} '
          f'fwd={dt:.2f}s peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB',
          flush=True)
