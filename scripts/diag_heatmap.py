"""1バッチで各段の統計を出し、heatmap loss 異常の発生箇所を切り分ける診断スクリプト.

usage: python scripts/diag_heatmap.py [resnet|dinov3]
"""
import sys

import torch

from bevdepth.layers.heads.bev_depth_head import BEVDepthHead

which = sys.argv[1] if len(sys.argv) > 1 else 'dinov3'
if which == 'resnet':
    from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel
else:
    from bevdepth.exps.nuscenes.mv.bev_depth_dinov3 import \
        BEVDepthLightningModel

torch.manual_seed(0)
m = BEVDepthLightningModel(batch_size_per_device=1).cuda()
m.train()
loader = m.train_dataloader()
batch = next(iter(loader))
sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels = batch


def stat(name, t):
    t = t.float()
    print(f'{name:28s} shape={tuple(t.shape)} mean={t.mean():.4f} '
          f'std={t.std():.4f} min={t.min():.4f} max={t.max():.4f}')


stat('img', sweep_imgs)
print('img per-channel mean', sweep_imgs.float().mean(dim=(0, 1, 2, 4, 5)))
print('num gt boxes', [g.shape for g in gt_boxes], 'labels', gt_labels[0][:20])
stat('gt_boxes', gt_boxes[0])

for k, v in mats.items():
    mats[k] = v.cuda()
imgs = sweep_imgs.cuda()
gt_boxes = [g.cuda() for g in gt_boxes]
gt_labels = [g.cuda() for g in gt_labels]

bb = m.model.backbone
hooks = []
for name in ['depth_net']:
    mod = getattr(bb, name)
    hooks.append(mod.register_forward_hook(
        lambda mod, i, o, n=name: stat(n + '_out', o)))
if hasattr(bb, 'adapter'):
    hooks.append(bb.adapter.register_forward_hook(
        lambda mod, i, o: [stat('dinov3_raw', i[0]),
                           stat('adapter_out', o)] and None))
else:
    hooks.append(bb.img_neck.register_forward_hook(
        lambda mod, i, o: stat('img_neck_out', o[0])))

head = m.model.head
hooks.append(head.trunk.register_forward_hook(lambda *a: None))
orig_forward = BEVDepthHead.forward


def wrapped(self, x):
    stat('bev_feat(head in)', x)
    return orig_forward(self, x)


head.forward = wrapped.__get__(head)

with torch.cuda.amp.autocast():
    preds, depth_preds = m(imgs, mats)
stat('depth_preds', depth_preds)
for t, p in enumerate(preds):
    stat(f'task{t} heatmap logits', p[0]['heatmap'])
targets = m.model.get_targets(gt_boxes, gt_labels)
for t, hm in enumerate(targets[0]):
    print(f'task{t} gt_hm num_pos={hm.eq(1).sum().item()} max={hm.max():.3f}'
          f' mask_sum={targets[3][t].sum().item()}')
loss = m.model.loss(targets, preds)
print('loss components', {k: float(v) for k, v in
                          head.last_loss_components.items()})
