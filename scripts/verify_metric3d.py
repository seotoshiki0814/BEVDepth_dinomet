"""Metric3Dv2 の (1) 公式重みの完全ロード (2) 凍結 (3) 深度品質 を検証する.

usage: python scripts/verify_metric3d.py [num_val_samples]
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from bevdepth.exps.nuscenes.mv.bev_depth_dinov3_metric3d import \
    BEVDepthLightningModel

CKPT = '/root/.cache/torch/hub/checkpoints/metric_depth_vit_small_800k.pth'
NUM_VAL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

torch.manual_seed(0)
m = BEVDepthLightningModel(batch_size_per_device=1).cuda()
bb = m.model.backbone
m3d = bb.metric3d

# ---------------------------------------------------------------- [1] weights
print('=' * 70, '\n[1] Official weights fully loaded? (hubconf uses strict=False)')
ckpt = torch.load(CKPT, map_location='cpu')['model_state_dict']
sd = m3d.state_dict()
missing = [k for k in sd if k not in ckpt]
unexpected = [k for k in ckpt if k not in sd]
shape_mm = [k for k in sd if k in ckpt and sd[k].shape != ckpt[k].shape]
value_mm = [k for k in sd if k in ckpt and sd[k].shape == ckpt[k].shape
            and not torch.equal(sd[k].cpu(), ckpt[k].to(sd[k].dtype))]
print(f'  model tensors={len(sd)} ckpt tensors={len(ckpt)}')
print(f'  missing (left random)={len(missing)} {missing[:5]}')
print(f'  unexpected (ignored)={len(unexpected)} {unexpected[:5]}')
print(f'  shape mismatch={len(shape_mm)}  value mismatch={len(value_mm)} '
      f'{value_mm[:5]}')

# ----------------------------------------------------------------- [2] frozen
print('=' * 70, '\n[2] Frozen during training?')
m.train()
n_grad = sum(p.requires_grad for p in m3d.parameters())
n_train_mods = sum(mod.training for mod in m3d.modules())
print(f'  params with requires_grad=True: {n_grad}')
print(f'  submodules in training mode after model.train(): {n_train_mods}')
before = {k: v.detach().clone() for k, v in m3d.state_dict().items()}
before_dino = {k: v.detach().clone() for k, v in
               bb.dinov3.state_dict().items()}
# Deliberately large lr so any leak would visibly change weights.
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                        lr=1e-3)
scaler = torch.cuda.amp.GradScaler()
it = iter(m.train_dataloader())
for step in range(3):
    imgs, mats, _, _, gt_boxes, gt_labels, depth_labels = next(it)
    mats = {k: v.cuda() for k, v in mats.items()}
    if depth_labels.dim() == 5:
        depth_labels = depth_labels[:, 0]
    with torch.cuda.amp.autocast():
        preds, depth_preds = m(imgs.cuda(), mats)
        targets = m.model.get_targets([g.cuda() for g in gt_boxes],
                                      [g.cuda() for g in gt_labels])
        loss = m.model.loss(targets, preds) + m.get_depth_loss(
            depth_labels.cuda(), depth_preds)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
n_with_grad = sum(p.grad is not None for p in m3d.parameters())
changed = [k for k, v in m3d.state_dict().items()
           if not torch.equal(v, before[k])]
changed_dino = [k for k, v in bb.dinov3.state_dict().items()
                if not torch.equal(v, before_dino[k])]
adapter_moved = any(p.grad is not None for p in bb.adapter.parameters())
print(f'  after 3 optimizer steps (lr=1e-3): metric3d tensors changed='
      f'{len(changed)} (incl. BN buffers), params with .grad={n_with_grad}')
print(f'  dinov3 tensors changed={len(changed_dino)}; '
      f'adapter received grad={adapter_moved} (sanity: trainable part trains)')

# ---------------------------------------------------------------- [3] quality
print('=' * 70, '\n[3] Depth quality vs LiDAR (val, no augmentation)')
m.eval()
ds = NuscDetDataset(ida_aug_conf=m.ida_aug_conf,
                    bda_aug_conf=m.bda_aug_conf,
                    classes=m.class_names,
                    data_root=m.data_root,
                    info_paths=m.val_info_paths,
                    is_train=False,
                    img_conf=m.img_conf,
                    num_sweeps=1,
                    return_depth=True)


def metrics(pred, gt):
    v = (gt > 1.0) & (gt < 60.0)
    p, g = pred[v].clamp(min=1e-3), gt[v]
    ratio = torch.max(p / g, g / p)
    return dict(absrel=((p - g).abs() / g).mean().item(),
                rmse=((p - g)**2).mean().sqrt().item(),
                d1=(ratio < 1.25).float().mean().item(),
                scale=(g / p).median().item(),
                n=v.sum().item())


@torch.no_grad()
def official_full_res(img, fx):
    """Official demo pipeline on the raw 1600x900 image."""
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float()
    H, W = x.shape[-2:]
    s = min(616 / H, 1064 / W)
    h, w = round(H * s), round(W * s)
    x = F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
    ph, pw = 616 - h, 1064 - w
    x = (x - MEAN) / STD
    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
    d, conf, _ = m3d.inference({'input': x.cuda()})
    d = d[:, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w].float()
    d = F.interpolate(d, (H, W), mode='bilinear', align_corners=False)
    return d * (fx * s / 1000.0)


rows = {'official@1600x900': [], 'ours@256x704 (in-model)': []}
prior_top1, prior_loss, learned_hint = [], [], []
cams = m.ida_aug_conf['cams']
for idx in range(min(NUM_VAL, len(ds))):
    info = ds.infos[idx]
    lidar_info = info['lidar_infos']
    pts = np.fromfile(os.path.join(m.data_root,
                                   lidar_info['LIDAR_TOP']['filename']),
                      dtype=np.float32).reshape(-1, 5)[..., :4]
    for cam in cams:
        ci = info['cam_infos'][cam]
        img = Image.open(os.path.join(m.data_root, ci['filename']))
        uvd = ds.get_lidar_depth(pts, img, lidar_info, ci)
        gt = torch.zeros(1, 1, img.height, img.width)
        u, v = uvd[:, 0].astype(int), uvd[:, 1].astype(int)
        ok = (u >= 0) & (u < img.width) & (v >= 0) & (v < img.height)
        gt[0, 0, v[ok], u[ok]] = torch.from_numpy(uvd[ok, 2])
        fx = ci['calibrated_sensor']['camera_intrinsic'][0][0]
        rows['official@1600x900'].append(
            metrics(official_full_res(img, fx).cpu(), gt))

    batch = collate_fn([ds[idx]], is_return_depth=True)
    imgs, mats, _, _, _, _, depth_labels = batch
    mats = {k: v.cuda() for k, v in mats.items()}
    with torch.no_grad(), torch.cuda.amp.autocast():
        bb(imgs.cuda(), mats)
    if depth_labels.dim() == 5:
        depth_labels = depth_labels[:, 0]
    gt = depth_labels[0].float().unsqueeze(1)  # (N,1,256,704)
    pred = F.interpolate(bb.last_metric_depth, gt.shape[-2:],
                         mode='bilinear', align_corners=False).cpu()
    rows['ours@256x704 (in-model)'].extend(
        metrics(pred[i:i + 1], gt[i:i + 1]) for i in range(gt.shape[0]))

    # Prior vs BEVDepth's own depth target on the 16x44 grid.
    gt_bins = m.get_downsampled_gt_depth(depth_labels.cuda())  # (M,112)
    prior = bb.last_depth_prior.permute(0, 2, 3, 1).reshape(-1, 112)
    fg = gt_bins.max(1).values > 0
    prior_top1.append((prior[fg].argmax(1) - gt_bins[fg].argmax(1)).abs()
                      .float().cpu())
    p = (prior[fg] + 1e-3)
    p = p / p.sum(1, keepdim=True)
    prior_loss.append(3.0 * F.binary_cross_entropy(
        p.float(), gt_bins[fg].float(), reduction='none').sum(1).mean()
                      .item())

for name, rs in rows.items():
    agg = {k: np.mean([r[k] for r in rs]) for k in rs[0] if k != 'n'}
    print(f'  {name:26s} AbsRel={agg["absrel"]:.3f} RMSE={agg["rmse"]:.2f}m '
          f'd<1.25={agg["d1"]:.3f} median(gt/pred)={agg["scale"]:.3f} '
          f'({len(rs)} images)')
err = torch.cat(prior_top1)
print(f'  prior argmax bin error on 16x44 grid (0.5m bins): '
      f'|err|<=1 bin {(err <= 1).float().mean():.3f}, '
      f'<=4 bins(2m) {(err <= 4).float().mean():.3f}')
print(f'  depth_loss if prior used as-is (replace mode) = '
      f'{np.mean(prior_loss):.2f}   (untrained DepthNet ~18.6, '
      f'DINOv3 after 300it ~17.5)')
