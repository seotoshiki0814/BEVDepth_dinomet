"""3D bbox の GT と予測を 1 対 1 マッチングして誤差を定量比較する.

infer_eval.py が出力した raw_results.pkl を入力にする.
マッチング: クラスごとに score 降順で greedy、BEV 中心距離 <= dist_thr [m]
(nuScenes の TP 判定と同じ考え方).

出力 (out_dir):
  matches.csv        GT ごとの行 (マッチした予測と誤差、未検出は空欄)
  false_pos.csv      マッチしなかった予測
  summary.txt        クラス別 / 距離別 / カメラ別 の集計
  bev_match/NNN.png  BEV 上の GT・予測・対応線

usage:
  python scripts/compare_boxes.py --pred outputs/xxx/eval/raw_results.pkl \
      --out outputs/xxx/compare --score_thr 0.3 --dist_thr 2.0
"""
import argparse
import csv
import os
import os.path as osp
import pickle

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import mmengine  # noqa: E402
import numpy as np  # noqa: E402
from nuscenes.utils.data_classes import Box  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from bevdepth.datasets.nusc_det_dataset import \
    map_name_from_general_to_detection  # noqa: E402

CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT']
DIST_BINS = [(0, 20), (20, 30), (30, 40), (40, 52)]


def yaw_of(q):
    return q.yaw_pitch_roll[0]


def ang_diff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def gt_boxes_ego(info, class_names):
    ego = info['cam_infos']['CAM_FRONT']['ego_pose']
    out = []
    for ann in info['ann_infos']:
        name = map_name_from_general_to_detection.get(ann['category_name'])
        if name not in class_names:
            continue
        # Same filter as the dataset: drop boxes without lidar/radar points.
        if ann.get('num_lidar_pts', 1) + ann.get('num_radar_pts', 0) == 0:
            continue
        box = Box(ann['translation'], ann['size'],
                  Quaternion(ann['rotation']))
        box.translate(-np.array(ego['translation']))
        box.rotate(Quaternion(ego['rotation']).inverse)
        out.append(dict(cls=name, center=box.center, wlh=box.wlh,
                        yaw=yaw_of(box.orientation), box=box))
    return out


def pred_boxes_ego(det, class_names, score_thr):
    boxes, scores, labels = det
    out = []
    for b, s, lab in zip(boxes, scores, labels):
        if s < score_thr:
            continue
        box = Box(b[:3], b[[4, 3, 5]], Quaternion(axis=[0, 0, 1],
                                                    radians=b[6]))
        out.append(dict(cls=class_names[lab], center=box.center,
                        wlh=box.wlh, yaw=b[6], score=float(s), box=box))
    return out


def camera_of(center_ego, info):
    """Which camera sees the box center best (smallest angle to axis)."""
    best, best_ang = None, 1e9
    for cam in CAMS:
        cs = info['cam_infos'][cam]['calibrated_sensor']
        p = Quaternion(cs['rotation']).inverse.rotate(
            center_ego - np.array(cs['translation']))
        if p[2] <= 0.5:
            continue
        K = np.array(cs['camera_intrinsic'])
        u = K[0, 0] * p[0] / p[2] + K[0, 2]
        v = K[1, 1] * p[1] / p[2] + K[1, 2]
        if not (0 <= u < 1600 and 0 <= v < 900):
            continue
        ang = abs(np.arctan2(p[0], p[2]))
        if ang < best_ang:
            best, best_ang = cam, ang
    return best or 'none'


def match(gts, preds, dist_thr):
    pairs, used = {}, set()
    order = sorted(range(len(preds)), key=lambda i: -preds[i]['score'])
    for pi in order:
        p = preds[pi]
        best, best_d = None, dist_thr
        for gi, g in enumerate(gts):
            if gi in pairs or g['cls'] != p['cls']:
                continue
            d = np.linalg.norm(p['center'][:2] - g['center'][:2])
            if d <= best_d:
                best, best_d = gi, d
        if best is not None:
            pairs[best] = pi
            used.add(pi)
    return pairs, [i for i in range(len(preds)) if i not in used]


def draw_bev(path, gts, preds, pairs, title):
    fig, ax = plt.subplots(figsize=(9, 9))
    for gi, g in enumerate(gts):
        c = g['box'].bottom_corners()[:2]
        color = 'tab:green' if gi in pairs else 'tab:red'
        ax.fill(c[0], c[1], color=color, alpha=0.35)
        ax.plot(np.append(c[0], c[0, 0]), np.append(c[1], c[1, 0]),
                color=color, lw=1.5)
    matched_preds = set(pairs.values())
    for pi, p in enumerate(preds):
        c = p['box'].bottom_corners()[:2]
        color = 'tab:blue' if pi in matched_preds else 'tab:orange'
        ax.plot(np.append(c[0], c[0, 0]), np.append(c[1], c[1, 0]),
                color=color, lw=1.2, ls='--')
        f = c[:, :2].mean(1)
        ax.plot([p['center'][0], f[0]], [p['center'][1], f[1]], color=color,
                lw=1.0, ls='--')
    for gi, pi in pairs.items():
        g, p = gts[gi]['center'], preds[pi]['center']
        ax.plot([g[0], p[0]], [g[1], p[1]], color='k', lw=0.8)
    ax.plot(0, 0, marker=(3, 0, -90), ms=12, color='k')
    for r in (20, 30, 40):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=':', color='0.7'))
    ax.set_xlim(-52, 52)
    ax.set_ylim(-52, 52)
    ax.set_aspect('equal')
    ax.set_xlabel('x (forward) [m]')
    ax.set_ylabel('y (left) [m]')
    handles = [
        plt.Rectangle((0, 0), 1, 1, color='tab:green', alpha=0.5,
                      label='GT detected (TP)'),
        plt.Rectangle((0, 0), 1, 1, color='tab:red', alpha=0.5,
                      label='GT missed (FN)'),
        plt.Line2D([], [], color='tab:blue', ls='--', label='pred matched'),
        plt.Line2D([], [], color='tab:orange', ls='--',
                   label='pred false positive'),
        plt.Line2D([], [], color='k', label='GT-pred center offset'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fmt_stats(rows):
    tp = [r for r in rows if r['matched']]
    n, ntp = len(rows), len(tp)
    if not ntp:
        return f'GT={n:4d} TP={ntp:4d} recall={0 if n else float("nan"):.3f}'
    a = lambda k: np.mean([r[k] for r in tp])  # noqa: E731
    return (f'GT={n:4d} TP={ntp:4d} recall={ntp / n:.3f} | '
            f'center_err={a("center_err"):.2f}m '
            f'(dx={a("abs_dx"):.2f} dy={a("abs_dy"):.2f} '
            f'dz={a("abs_dz"):.2f}) '
            f'radial_err={a("radial_err"):+.2f}m '
            f'size_rel_err={a("size_rel_err"):.3f} '
            f'yaw_err={a("yaw_err_deg"):.1f}deg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--dist_thr', type=float, default=2.0)
    ap.add_argument('--num_vis', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(osp.join(args.out, 'bev_match'), exist_ok=True)

    raw = pickle.load(open(args.pred, 'rb'))
    infos = mmengine.load(raw['info_path'])
    tok2info = {i['sample_token']: i for i in infos}

    rows, fps = [], []
    for k, (det, meta) in enumerate(zip(raw['results'], raw['metas'])):
        info = tok2info[meta['token']]
        gts = gt_boxes_ego(info, CLASSES)
        preds = pred_boxes_ego(det, CLASSES, args.score_thr)
        pairs, unmatched = match(gts, preds, args.dist_thr)
        for gi, g in enumerate(gts):
            dist = float(np.linalg.norm(g['center'][:2]))
            r = dict(sample=k, token=meta['token'], cls=g['cls'],
                     cam=camera_of(g['center'], info), gt_dist=dist,
                     gt_x=g['center'][0], gt_y=g['center'][1],
                     gt_z=g['center'][2], gt_w=g['wlh'][0], gt_l=g['wlh'][1],
                     gt_h=g['wlh'][2], gt_yaw=g['yaw'], matched=gi in pairs)
            if gi in pairs:
                p = preds[pairs[gi]]
                d = p['center'] - g['center']
                # + : predicted farther than GT (depth over-estimate)
                radial = float(np.linalg.norm(p['center'][:2]) - dist)
                r.update(score=p['score'], pr_x=p['center'][0],
                         pr_y=p['center'][1], pr_z=p['center'][2],
                         pr_w=p['wlh'][0], pr_l=p['wlh'][1],
                         pr_h=p['wlh'][2], pr_yaw=p['yaw'],
                         center_err=float(np.linalg.norm(d[:2])),
                         abs_dx=abs(d[0]), abs_dy=abs(d[1]), abs_dz=abs(d[2]),
                         radial_err=radial,
                         size_rel_err=float(np.mean(
                             np.abs(p['wlh'] - g['wlh']) / g['wlh'])),
                         yaw_err_deg=float(np.degrees(
                             ang_diff(p['yaw'], g['yaw']))))
            rows.append(r)
        for pi in unmatched:
            p = preds[pi]
            fps.append(dict(sample=k, token=meta['token'], cls=p['cls'],
                            score=p['score'], x=p['center'][0],
                            y=p['center'][1], z=p['center'][2],
                            dist=float(np.linalg.norm(p['center'][:2])),
                            cam=camera_of(p['center'], info)))
        if k < args.num_vis:
            draw_bev(osp.join(args.out, 'bev_match', f'{k:03d}.png'), gts,
                     preds, pairs,
                     f'sample {k}  GT={len(gts)} pred={len(preds)} '
                     f'TP={len(pairs)}  (score>={args.score_thr}, '
                     f'match<= {args.dist_thr}m)')

    keys = sorted({kk for r in rows for kk in r})
    with open(osp.join(args.out, 'matches.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if fps:
        with open(osp.join(args.out, 'false_pos.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(fps[0]))
            w.writeheader()
            w.writerows(fps)

    lines = [f'pred={args.pred}',
             f'score_thr={args.score_thr} dist_thr={args.dist_thr}m '
             f'samples={len(raw["results"])}', '',
             f'[ALL] {fmt_stats(rows)}',
             f'      false positives={len(fps)}  precision='
             f'{sum(r["matched"] for r in rows) / max(1, sum(r["matched"] for r in rows) + len(fps)):.3f}',
             '', '[per class]']
    for c in CLASSES:
        rs = [r for r in rows if r['cls'] == c]
        nfp = sum(f['cls'] == c for f in fps)
        if rs or nfp:
            lines.append(f'  {c:20s} {fmt_stats(rs)}  FP={nfp}')
    lines += ['', '[by distance from ego]']
    for lo, hi in DIST_BINS:
        rs = [r for r in rows if lo <= r['gt_dist'] < hi]
        nfp = sum(lo <= f['dist'] < hi for f in fps)
        lines.append(f'  {lo:2d}-{hi:2d}m {fmt_stats(rs)}  FP={nfp}')
    lines += ['', '[by camera (GT center in FOV)]']
    for c in CAMS + ['none']:
        rs = [r for r in rows if r['cam'] == c]
        nfp = sum(f['cam'] == c for f in fps)
        if rs or nfp:
            lines.append(f'  {c:16s} {fmt_stats(rs)}  FP={nfp}')
    text = '\n'.join(lines)
    print(text)
    open(osp.join(args.out, 'summary.txt'), 'w').write(text + '\n')


if __name__ == '__main__':
    main()
