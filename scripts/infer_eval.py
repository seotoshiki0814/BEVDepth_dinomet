"""学習済み ckpt で val 推論 → nuScenes 評価 (mAP/NDS) → 可視化 PNG を出力する.

Lightning 2.x では *_epoch_end が呼ばれず -e で評価できないため、独立スクリプトにした.

usage:
  python scripts/infer_eval.py --exp dinov3_metric3d \
      --ckpt outputs/xxx/lightning_logs/version_0/checkpoints/epoch=23.ckpt \
      --out outputs/xxx/eval --num_vis 10 --score_thr 0.3
"""
import argparse
import json
import os
import os.path as osp

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from nuscenes.utils.data_classes import Box  # noqa: E402
from nuscenes.utils.geometry_utils import view_points  # noqa: E402
from PIL import Image  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from bevdepth.evaluators.det_evaluators import DetNuscEvaluator  # noqa: E402

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT']
COLORS = plt.cm.tab10(np.arange(10))[:, :3]


def build_model(exp, ckpt, **kw):
    if exp == 'dinov3_metric3d':
        from bevdepth.exps.nuscenes.mv.bev_depth_dinov3_metric3d import \
            BEVDepthLightningModel
    elif exp == 'dinov3':
        from bevdepth.exps.nuscenes.mv.bev_depth_dinov3 import \
            BEVDepthLightningModel
    else:
        from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel
    m = BEVDepthLightningModel(batch_size_per_device=1, **kw)
    sd = torch.load(ckpt, map_location='cpu')['state_dict']
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f'[ckpt] {ckpt}\n  missing={len(missing)} unexpected='
          f'{len(unexpected)} {missing[:3]} {unexpected[:3]}')
    return m.cuda().eval()


def ego_box_from_global(ann, ego_t, ego_r):
    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
    box.translate(-np.array(ego_t))
    box.rotate(Quaternion(ego_r).inverse)
    return box


def pred_box(b):
    return Box(b[:3], b[[4, 3, 5]], Quaternion(axis=[0, 0, 1], radians=b[6]))


def box_to_cam(box, cam_info):
    """ego-frame Box -> camera-frame Box (copy)."""
    box = Box(box.center.copy(), box.wlh.copy(), box.orientation)
    cs = cam_info['calibrated_sensor']
    box.translate(-np.array(cs['translation']))
    box.rotate(Quaternion(cs['rotation']).inverse)
    return box


def draw_box_img(ax, box_cam, K, color, lw=1.2):
    if box_cam.center[2] < 1.0:
        return
    corners = view_points(box_cam.corners(), K, normalize=True)[:2]
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(corners[0, [i, j]], corners[1, [i, j]], color=color, lw=lw)


def draw_box_bev(ax, box, color, lw=1.2, ls='-'):
    c = box.bottom_corners()[:2]
    ax.plot(np.append(c[0], c[0, 0]), np.append(c[1], c[1, 0]), color=color,
            lw=lw, ls=ls)
    front = c[:, :2].mean(1)
    ax.plot([box.center[0], front[0]], [box.center[1], front[1]],
            color=color, lw=lw, ls=ls)


def visualize(idx, info, det, class_names, data_root, extra, out_path,
              score_thr):
    boxes, scores, labels = det
    keep = scores >= score_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    ego_t = info['cam_infos']['CAM_FRONT']['ego_pose']['translation']
    ego_r = info['cam_infos']['CAM_FRONT']['ego_pose']['rotation']
    gts = [ego_box_from_global(a, ego_t, ego_r) for a in info['ann_infos']]
    preds = [pred_box(b) for b in boxes]

    fig = plt.figure(figsize=(26, 13))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.15])
    for k, cam in enumerate(CAMS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ci = info['cam_infos'][cam]
        ax.imshow(Image.open(osp.join(data_root, ci['filename'])))
        K = np.array(ci['calibrated_sensor']['camera_intrinsic'])
        for g in gts:
            draw_box_img(ax, box_to_cam(g, ci), K, (0.1, 0.9, 0.1), lw=1.0)
        for p, lab in zip(preds, labels):
            draw_box_img(ax, box_to_cam(p, ci), K, COLORS[lab])
        ax.set_xlim(0, 1600)
        ax.set_ylim(900, 0)
        ax.set_title(cam, fontsize=10)
        ax.axis('off')

    # BEV
    ax = fig.add_subplot(gs[0:2, 3])
    lidar = np.fromfile(osp.join(
        data_root, info['lidar_infos']['LIDAR_TOP']['filename']),
                        dtype=np.float32).reshape(-1, 5)[:, :3]
    cs = info['lidar_infos']['LIDAR_TOP']['calibrated_sensor']
    lidar = lidar @ Quaternion(cs['rotation']).rotation_matrix.T + \
        np.array(cs['translation'])
    ax.scatter(lidar[:, 0], lidar[:, 1], s=0.2, c='0.6')
    for g in gts:
        draw_box_bev(ax, g, (0.1, 0.8, 0.1), lw=1.5)
    for p, lab in zip(preds, labels):
        draw_box_bev(ax, p, COLORS[lab], lw=1.2, ls='--')
    ax.set_xlim(-51.2, 51.2)
    ax.set_ylim(-51.2, 51.2)
    ax.set_aspect('equal')
    ax.set_title(f'BEV (ego)  GT=green solid / pred=dashed (score>='
                 f'{score_thr})  #GT={len(gts)} #pred={len(preds)}')
    handles = [plt.Line2D([], [], color=COLORS[i], label=n)
               for i, n in enumerate(class_names)]
    ax.legend(handles=handles, loc='upper right', fontsize=7)

    # Depth row (CAM_FRONT): Metric3D / LSS expectation / LiDAR GT
    titles = ['CAM_FRONT Metric3Dv2 depth [m]',
              'CAM_FRONT LSS depth E[d] (fed to Lift) [m]',
              'CAM_FRONT LiDAR depth GT [m]']
    for j, key in enumerate(['metric3d', 'lss', 'lidar']):
        ax = fig.add_subplot(gs[2, j])
        if extra.get(key) is not None:
            im = ax.imshow(extra[key], cmap='magma_r', vmin=0, vmax=60)
            plt.colorbar(im, ax=ax, fraction=0.025)
        ax.set_title(titles[j], fontsize=10)
        ax.axis('off')
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    ax.text(0, 0.9, f'sample idx={idx}\ntoken={info["sample_token"]}',
            fontsize=10, va='top')
    fig.tight_layout()
    fig.savefig(out_path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='dinov3_metric3d',
                    choices=['dinov3_metric3d', 'dinov3', 'resnet'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_vis', type=int, default=10)
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--depth_prior_mode', default=None,
                    choices=['concat', 'replace', 'lift'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    kw = {}
    if args.depth_prior_mode:
        kw['depth_prior_mode'] = args.depth_prior_mode
    m = build_model(args.exp, args.ckpt, **kw)
    bb = m.model.backbone
    cam_front = CAMS.index('CAM_FRONT')

    # Capture the depth logits actually fed to LSS (DepthNet output, or the
    # Metric3D prior in replace / lift mode, where there may be no DepthNet).
    depth_out = {}
    forward_depth_net = bb._forward_depth_net

    def capture_depth(feat, mats_dict):
        out = forward_depth_net(feat, mats_dict)
        depth_out['x'] = out.detach()
        return out

    bb._forward_depth_net = capture_depth

    loader = m.val_dataloader()
    ds = loader.dataset
    results, metas, extras = [], [], []
    for b, batch in enumerate(loader):
        with torch.no_grad(), torch.cuda.amp.autocast():
            res = m.eval_step(batch, b, 'val')
        for r in res:
            results.append(r[:3])
            metas.append(r[3])
        if b < args.num_vis:
            ex = {}
            D = bb.depth_channels
            logits = depth_out['x'][cam_front:cam_front + 1, :D].float()
            d_min, _, step = bb.d_bound
            centers = torch.arange(D, device=logits.device) * step + d_min
            prob = logits.softmax(1)
            lss = (prob * centers.view(1, -1, 1, 1)).sum(1)[0]
            if getattr(bb, 'depth_prior_mode', None) == 'lift':
                # Pixels whose prior mass is out of range lift nothing.
                mass = bb.last_depth_prior[cam_front].sum(0)
                lss = torch.where(mass > 0.05, lss,
                                  torch.full_like(lss, float('nan')))
            ex['lss'] = lss.cpu().numpy()
            if getattr(bb, 'last_metric_depth', None) is not None:
                ex['metric3d'] = bb.last_metric_depth[cam_front, 0].cpu() \
                    .numpy()
            info = ds.infos[b]
            ci = info['cam_infos']['CAM_FRONT']
            pts = np.fromfile(osp.join(
                m.data_root, info['lidar_infos']['LIDAR_TOP']['filename']),
                              dtype=np.float32).reshape(-1, 5)[..., :4]
            img = Image.open(osp.join(m.data_root, ci['filename']))
            uvd = ds.get_lidar_depth(pts, img, info['lidar_infos'], ci)
            lid = np.full((img.height, img.width), np.nan, np.float32)
            u, v = uvd[:, 0].astype(int), uvd[:, 1].astype(int)
            ok = (u >= 0) & (u < img.width) & (v >= 0) & (v < img.height)
            for du in range(-2, 3):  # dilate sparse points for visibility
                for dv in range(-2, 3):
                    uu = np.clip(u[ok] + du, 0, img.width - 1)
                    vv = np.clip(v[ok] + dv, 0, img.height - 1)
                    lid[vv, uu] = uvd[ok, 2]
            ex['lidar'] = lid
            extras.append(ex)
        if b % 20 == 0:
            print(f'  infer {b}/{len(loader)}', flush=True)

    # Raw predictions for scripts/compare_boxes.py
    import pickle
    with open(osp.join(args.out, 'raw_results.pkl'), 'wb') as f:
        pickle.dump(dict(results=results, metas=metas,
                         info_path=m.val_info_paths), f)

    # nuScenes evaluation
    ev = DetNuscEvaluator(class_names=m.class_names,
                          data_root=m.data_root,
                          version=args.version,
                          output_dir=args.out)
    res_path = ev._format_bbox(results, metas, args.out)
    ev._evaluate_single(res_path)
    summary = json.load(open(osp.join(args.out, 'metrics_summary.json')))
    print('=' * 60)
    print(f'mAP={summary["mean_ap"]:.4f}  NDS={summary["nd_score"]:.4f}')
    for k, v in summary['tp_errors'].items():
        print(f'  {k:18s} {v:.4f}')
    print('per-class AP:')
    for c, v in summary['mean_dist_aps'].items():
        print(f'  {c:22s} {v:.4f}')

    vis_dir = osp.join(args.out, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    for i, ex in enumerate(extras):
        visualize(i, ds.infos[i], results[i], m.class_names, m.data_root,
                  ex, osp.join(vis_dir, f'{i:03d}.png'), args.score_thr)
    print('vis saved to', vis_dir)


if __name__ == '__main__':
    main()
