"""DINOv3 / Metric3Dv2 が (A) 凍結されたまま特徴抽出に使われ、
(B) その出力が後段 (Adapter / 深度事前分布 / Lift / BEV / Head) に正しく
受け渡されていることを、エビデンス付きで検証する (lift モード用).

A. 凍結
  A1 requires_grad=False のパラメータ数 / optimizer に含まれていないこと
  A2 model.train() 後も eval モード (Dropout / Norm が推論動作)
  A3 学習済み ckpt の重み == 公式重み (全テンソル一致)
  A4 学習 1 step の backward で凍結側に .grad が付かず、学習側には付く
  A5 凍結モデルの出力に計算グラフが無い (grad_fn=None)
B. 受け渡し (val 1 サンプル、augmentation なし、forward hook で捕捉)
  B1 DINOv3 出力 == 独立に読み込んだ公式 DINOv3 の出力
  B2 Adapter の入力 == DINOv3 出力 (同一テンソル)
  B3 Metric3D 出力 == 独立に読み込んだ公式 Metric3D の出力
  B4 Metric3D の metric depth が LiDAR と整合 (AbsRel)
  B5 context_conv の入力 == concat[Adapter 出力, 深度事前分布]
  B6 LSS に渡る depth = prior / mass、context = context_conv 出力 x mass
  B7 BEV 特徴 (Head の入力) の形状と非ゼロ
C. 後段が実際に依存しているか (アブレーション)
  C1 DINOv3 特徴をゼロにすると BEV / heatmap が変わる
  C2 Metric3D 深度を一定値 (20 m) にすると BEV / heatmap が変わる
  C3 BEV 特徴のエネルギーが LiDAR 点のある格子に集中するか
     (本物の深度 vs 一定深度、偶然の水準 = LiDAR 占有率)

usage (コンテナ内, /workspace/BEVDepth):
  PYTHONPATH=. python scripts/verify_frozen_pipeline.py \
      --ckpt outputs/xxx/lightning_logs/version_0/checkpoints/epoch=1-step=160.ckpt \
      --out outputs/xxx/verify_frozen_pipeline [--sample 0]
"""
import argparse
import hashlib
import json
import os
import os.path as osp

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn  # noqa
from bevdepth.exps.nuscenes.mv.bev_depth_dinov3_metric3d import \
    BEVDepthLightningModel  # noqa: E402

DINOV3_ID = 'vit_base_patch16_dinov3.lvd1689m'
DINOV3_SHA256 = \
    '1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332'
METRIC3D_SHA256 = \
    'b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6'
CAM_FRONT = 1
REPORT = []


def log(msg=''):
    print(msg, flush=True)
    REPORT.append(msg)


def verdict(ok):
    return 'PASS' if ok else 'FAIL'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel_change(a, b):
    return float((a - b).abs().mean() / (b.abs().mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {}

    m = BEVDepthLightningModel(batch_size_per_device=1,
                               depth_prior_mode='lift', lr=1e-4)
    sd = torch.load(args.ckpt, map_location='cpu')['state_dict']
    m.load_state_dict(sd, strict=True)
    m = m.cuda()
    bb = m.model.backbone
    log(f'ckpt: {args.ckpt}')
    log(f'depth_prior_mode={bb.depth_prior_mode} '
        f'depth_prior_type={bb.depth_prior_type}')

    # ================================================================ A
    log('\n' + '=' * 72 + '\nA. 凍結 (weight freeze)\n' + '=' * 72)
    frozen = {'dinov3': bb.dinov3, 'metric3d': bb.metric3d}
    trainable = {'adapter': bb.adapter, 'context_conv': bb.context_conv}

    # A1
    opt = m.configure_optimizers()[0][0]
    opt_ids = {id(p) for g in opt.param_groups for p in g['params']}
    ok = True
    for name, mod in frozen.items():
        ps = list(mod.parameters())
        n_rg = sum(p.requires_grad for p in ps)
        n_opt = sum(id(p) in opt_ids for p in ps)
        ok &= (n_rg == 0 and n_opt == 0)
        log(f'[A1] {name:9s} params={len(ps):4d} '
            f'({sum(p.numel() for p in ps) / 1e6:.1f}M)  '
            f'requires_grad=True: {n_rg}  in optimizer: {n_opt}')
    for name, mod in trainable.items():
        ps = list(mod.parameters())
        log(f'[A1] {name:9s} params={len(ps):4d} (trainable) '
            f'requires_grad=True: {sum(p.requires_grad for p in ps)}  '
            f'in optimizer: {sum(id(p) in opt_ids for p in ps)}')
    results['A1'] = ok
    log(f'[A1] => {verdict(ok)}')

    # A2
    m.train()
    ok = True
    for name, mod in frozen.items():
        n_train = sum(sm.training for sm in mod.modules())
        n_mod = sum(1 for _ in mod.modules())
        ok &= n_train == 0
        log(f'[A2] after model.train(): {name:9s} submodules in training '
            f'mode = {n_train}/{n_mod}')
    log(f'[A2] adapter in training mode = {bb.adapter.training} '
        '(trainable side, expected True)')
    results['A2'] = ok
    log(f'[A2] => {verdict(ok)}')

    # A3: trained ckpt vs official weights
    from huggingface_hub import hf_hub_download
    dino_file = hf_hub_download(f'timm/{DINOV3_ID}', 'model.safetensors')
    m3d_file = osp.join(torch.hub.get_dir(), 'checkpoints',
                        'metric_depth_vit_small_800k.pth')
    sha_d, sha_m = sha256(dino_file), sha256(m3d_file)
    log(f'[A3] official DINOv3 file   {dino_file}\n'
        f'     sha256={sha_d}  matches HF: {sha_d == DINOV3_SHA256}')
    log(f'[A3] official Metric3D file {m3d_file}\n'
        f'     sha256={sha_m}  matches HF: {sha_m == METRIC3D_SHA256}')
    ref_dino = timm.create_model(DINOV3_ID, pretrained=True, num_classes=0)
    ref_dino_sd = ref_dino.state_dict()
    ref_m3d_sd = torch.load(m3d_file, map_location='cpu')['model_state_dict']
    ok = sha_d == DINOV3_SHA256 and sha_m == METRIC3D_SHA256
    for name, ref, prefix in (
            ('dinov3', ref_dino_sd, 'model.backbone.dinov3.'),
            ('metric3d', ref_m3d_sd, 'model.backbone.metric3d.')):
        ck = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        common = [k for k in ref if k in ck]
        equal = sum(torch.equal(ck[k].float(), ref[k].float()) for k in common)
        maxdiff = max(float((ck[k].float() - ref[k].float()).abs().max())
                      for k in common)
        only_ck = sorted(set(ck) - set(ref))
        ok &= equal == len(common) == len(ref)
        log(f'[A3] {name:9s} official tensors={len(ref)}  found in trained '
            f'ckpt={len(common)}  bit-exact equal={equal}  '
            f'max|diff|={maxdiff:.3e}  ckpt-only keys={only_ck}')
    results['A3'] = ok
    log(f'[A3] => {verdict(ok)}  (重みは 160 step 学習後も公式のまま)')

    # A4 / A5: one real training step on a train batch
    batch = next(iter(m.train_dataloader()))
    imgs, mats, _, _, gt_boxes, gt_labels, _ = batch
    mats = {k: v.cuda() for k, v in mats.items()}
    feats_out = {}
    orig_fi = bb.dinov3.forward_intermediates

    def fi_probe(*a, **k):
        out = orig_fi(*a, **k)
        feats_out.setdefault('dino', out[0])
        return out

    bb.dinov3.forward_intermediates = fi_probe
    h = bb.adapter.register_forward_hook(
        lambda mod, i, o: feats_out.setdefault('adapter', o))
    m.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        preds, _ = m(imgs.cuda(), mats)
        loss = m.model.loss(
            m.model.get_targets([g.cuda() for g in gt_boxes],
                                [g.cuda() for g in gt_labels]), preds)
    loss.backward()
    h.remove()
    bb.dinov3.forward_intermediates = orig_fi
    ok = True
    for name, mod in frozen.items():
        n = sum(p.grad is not None for p in mod.parameters())
        ok &= n == 0
        log(f'[A4] {name:9s} params with .grad after backward = {n}')
    for name, mod in trainable.items():
        g = [p.grad.abs().mean().item() for p in mod.parameters()
             if p.grad is not None]
        ok &= len(g) > 0 and max(g) > 0
        log(f'[A4] {name:9s} params with .grad = {len(g)}  '
            f'mean|grad| = {np.mean(g):.3e}')
    results['A4'] = ok
    log(f'[A4] => {verdict(ok)}  (loss={loss.item():.2f})')
    d, a = feats_out['dino'], feats_out['adapter']
    ok = (not d.requires_grad and d.grad_fn is None and a.requires_grad)
    log(f'[A5] DINOv3 output: requires_grad={d.requires_grad} '
        f'grad_fn={d.grad_fn}  |  Adapter output: requires_grad='
        f'{a.requires_grad} grad_fn={type(a.grad_fn).__name__}')
    results['A5'] = ok
    log(f'[A5] => {verdict(ok)}')
    m.zero_grad(set_to_none=True)

    # ================================================================ B
    log('\n' + '=' * 72 + '\nB. 受け渡し (val sample, no augmentation)\n'
        + '=' * 72)
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
    imgs, mats, _, img_metas, _, _, depth_labels = collate_fn(
        [ds[args.sample]], is_return_depth=True)
    mats = {k: v.cuda() for k, v in mats.items()}
    imgs = imgs.cuda()
    if depth_labels.dim() == 5:
        depth_labels = depth_labels[:, 0]
    log(f'sample idx={args.sample} token={img_metas[0]["token"]}')

    cap = {}
    orig_fi = bb.dinov3.forward_intermediates
    orig_inf = bb.metric3d.inference
    orig_fdn = bb._forward_depth_net

    def fi(*a, **k):
        out = orig_fi(*a, **k)
        cap.setdefault('dino', []).append(out[0].detach().clone())
        return out

    def inf(data):
        out = orig_inf(data)
        cap.setdefault('m3d_in', []).append(data['input'].detach().clone())
        cap.setdefault('m3d_out', []).append(out[0].detach().clone())
        return out

    def fdn(feat, mats_dict):
        cap['fdn_feat'] = feat.detach().clone()
        out = orig_fdn(feat, mats_dict)
        cap['fdn_out'] = out.detach().clone()
        return out

    bb.dinov3.forward_intermediates = fi
    bb.metric3d.inference = inf
    bb._forward_depth_net = fdn
    hooks = [
        bb.adapter.register_forward_hook(lambda mod, i, o: cap.update(
            adapter_in=i[0].detach().clone(), adapter_out=o.detach().clone())),
        bb.context_conv.register_forward_hook(lambda mod, i, o: cap.update(
            ctx_in=i[0].detach().clone(), ctx_out=o.detach().clone())),
        m.model.head.register_forward_pre_hook(
            lambda mod, a: cap.update(bev=a[0].detach().clone())),
    ]

    def run():
        with torch.no_grad(), torch.cuda.amp.autocast():
            return m.model(imgs, mats)

    preds = run()
    heat = torch.cat([p[0]['heatmap'].float() for p in preds], 1)
    base = dict(bev=cap['bev'].float(), heat=heat,
                prior=bb.last_depth_prior.clone(),
                metric=bb.last_metric_depth.clone())

    # B1 DINOv3 vs independent official model
    x = imgs[:, 0].reshape(-1, *imgs.shape[-3:])
    ref_dino = ref_dino.cuda().eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        ref = torch.cat([ref_dino.forward_intermediates(
            c, indices=[len(ref_dino.blocks) - 1], norm=True,
            output_fmt='NCHW', intermediates_only=True)[0]
            for c in x.split(bb.camera_chunk_size)])
    dino = torch.cat(cap['dino'])
    diff = float((dino.float() - ref.float()).abs().max())
    ok = diff < 1e-2 and tuple(dino.shape) == (6, 768, 16, 44)
    log(f'[B1] DINOv3 output shape={tuple(dino.shape)} '
        f'mean={dino.float().mean():.4f} std={dino.float().std():.4f}\n'
        f'     vs independently loaded official DINOv3: max|diff|={diff:.3e}')
    results['B1'] = ok
    log(f'[B1] => {verdict(ok)}')
    del ref_dino

    # B2 adapter input is exactly the DINOv3 output
    ok = torch.equal(cap['adapter_in'], dino)
    log(f'[B2] Adapter input == DINOv3 output (bit-exact): {ok}  '
        f'adapter out shape={tuple(cap["adapter_out"].shape)}')
    results['B2'] = ok
    log(f'[B2] => {verdict(ok)}')

    # B3 Metric3D vs independent official model
    repo = os.environ.get('METRIC3D_REPO_DIR')
    ref_m3d = torch.hub.load(repo, 'metric3d_vit_small', source='local',
                             pretrain=True).cuda().eval()
    m3d_in = torch.cat(cap['m3d_in'])
    with torch.no_grad(), torch.cuda.amp.autocast():
        ref_out = torch.cat([ref_m3d.inference({'input': c})[0]
                             for c in m3d_in.split(bb.camera_chunk_size)])
    m3d_out = torch.cat(cap['m3d_out'])
    diff = float((m3d_out.float() - ref_out.float()).abs().max())
    ok = diff < 1e-2
    log(f'[B3] Metric3D input shape={tuple(m3d_in.shape)} (padded 616x1064), '
        f'canonical depth shape={tuple(m3d_out.shape)}\n'
        f'     vs independently loaded official Metric3D: max|diff|='
        f'{diff:.3e}')
    results['B3'] = ok
    log(f'[B3] => {verdict(ok)}')
    del ref_m3d
    torch.cuda.empty_cache()

    # B4 metric depth vs LiDAR
    gt = depth_labels[0].float().unsqueeze(1).cuda()
    pred = F.interpolate(base['metric'], gt.shape[-2:], mode='bilinear',
                         align_corners=False)
    v = (gt > 1) & (gt < 60)
    absrel = float(((pred[v] - gt[v]).abs() / gt[v]).mean())
    d1 = float((torch.max(pred[v] / gt[v], gt[v] / pred[v]) < 1.25)
               .float().mean())
    ok = absrel < 0.2
    log(f'[B4] Metric3D metric depth vs LiDAR (n={int(v.sum())} points): '
        f'AbsRel={absrel:.3f} delta<1.25={d1:.3f}')
    results['B4'] = ok
    log(f'[B4] => {verdict(ok)}')

    # B5 context_conv input = concat[adapter out, prior]
    prior = base['prior']
    ctx_in = cap['ctx_in'].float()
    a_ok = torch.equal(cap['ctx_in'][:, :512], cap['adapter_out'])
    p_diff = float((ctx_in[:, 512:] - prior).abs().max())
    ok = a_ok and p_diff < 1e-3 and ctx_in.shape[1] == 512 + 112
    log(f'[B5] context_conv input shape={tuple(ctx_in.shape)}\n'
        f'     [:, :512] == Adapter output (bit-exact): {a_ok}\n'
        f'     [:, 512:] == depth prior: max|diff|={p_diff:.3e} '
        f'(fp16 cast)')
    results['B5'] = ok
    log(f'[B5] => {verdict(ok)}')

    # B6 what LSS receives
    out = cap['fdn_out'].float()
    D = bb.depth_channels
    mass = prior.sum(1, keepdim=True)
    lifted_depth = out[:, :D].softmax(1)
    sel = (mass > 1e-3).expand_as(prior)
    d_diff = float((lifted_depth - prior / mass.clamp(min=1e-6))[sel]
                   .abs().max())
    c_diff = float((out[:, D:] - cap['ctx_out'].float() * mass).abs().max())
    ok = d_diff < 1e-2 and c_diff < 5e-2
    log(f'[B6] LSS input: softmax(depth logits) vs prior/mass: '
        f'max|diff|={d_diff:.3e}\n'
        f'     LSS context vs context_conv output x mass: '
        f'max|diff|={c_diff:.3e}\n'
        f'     prior mass (in-range probability): mean={mass.mean():.3f} '
        f'min={mass.min():.3f} max={mass.max():.3f}')
    results['B6'] = ok
    log(f'[B6] => {verdict(ok)}')

    # B7 BEV feature
    bev = base['bev']
    nz = float((bev.abs().sum(1) > 0).float().mean())
    ok = tuple(bev.shape) == (1, 80, 128, 128) and nz > 0.05
    log(f'[B7] BEV feature (head input) shape={tuple(bev.shape)} '
        f'non-zero cells={nz * 100:.1f}%  mean|x|={bev.abs().mean():.4f}')
    results['B7'] = ok
    log(f'[B7] => {verdict(ok)}')

    # ================================================================ C
    log('\n' + '=' * 72 + '\nC. 後段が凍結モデルの出力に依存しているか '
        '(アブレーション)\n' + '=' * 72)

    def zero_dino(*a, **k):
        out = orig_fi(*a, **k)
        return [torch.zeros_like(o) for o in out]

    bb.dinov3.forward_intermediates = zero_dino
    preds = run()
    abl_dino = dict(bev=cap['bev'].float(), heat=torch.cat(
        [p[0]['heatmap'].float() for p in preds], 1))
    bb.dinov3.forward_intermediates = fi

    # Constant 20 m depth: replace the canonical depth so metric depth = 20.
    intrin = mats['intrin_mats'][:, 0]
    ida = mats['ida_mats'][:, 0]
    resize = ida[..., :2, :2].float().det().abs().sqrt()
    fx = (intrin[..., 0, 0].float() * resize).reshape(-1, 1, 1, 1)

    def const_m3d(data):
        out = orig_inf(data)
        fxs = fx * bb._metric3d_scale
        n = out[0].shape[0]
        idx = const_m3d.i
        const_m3d.i += n
        c = 20.0 * bb.CANONICAL_FOCAL / fxs[idx:idx + n]
        return (torch.ones_like(out[0]) * c,) + tuple(out[1:])

    const_m3d.i = 0
    bb.metric3d.inference = const_m3d
    preds = run()
    abl_depth = dict(bev=cap['bev'].float(), heat=torch.cat(
        [p[0]['heatmap'].float() for p in preds], 1),
        metric=bb.last_metric_depth.clone())
    bb.metric3d.inference = inf

    for tag, abl in (('C1 DINOv3 features = 0', abl_dino),
                     ('C2 Metric3D depth = 20 m const', abl_depth)):
        rb = rel_change(abl['bev'], base['bev'])
        rh = rel_change(abl['heat'].sigmoid(), base['heat'].sigmoid())
        log(f'[{tag}] relative change: BEV feature {rb * 100:.1f}%  '
            f'heatmap score {rh * 100:.1f}%')
        results[tag.split()[0]] = rb > 0.05 and rh > 0.01
    log(f'[C1] => {verdict(results["C1"])}   [C2] => '
        f'{verdict(results["C2"])}  (後段がそれぞれの出力に依存している)')

    # C3 BEV energy on LiDAR-occupied cells
    info = ds.infos[args.sample]
    pts = np.fromfile(osp.join(m.data_root,
                               info['lidar_infos']['LIDAR_TOP']['filename']),
                      dtype=np.float32).reshape(-1, 5)[:, :3]
    cs = info['lidar_infos']['LIDAR_TOP']['calibrated_sensor']
    pts = pts @ Quaternion(cs['rotation']).rotation_matrix.T + \
        np.array(cs['translation'])
    keep = (np.abs(pts[:, 0]) < 51.2) & (np.abs(pts[:, 1]) < 51.2) & \
        (pts[:, 2] > -5) & (pts[:, 2] < 3)
    ix = ((pts[keep, 0] + 51.2) / 0.8).astype(int)
    iy = ((pts[keep, 1] + 51.2) / 0.8).astype(int)
    occ = torch.zeros(128, 128)
    occ[iy, ix] = 1
    occ = (F.max_pool2d(occ[None, None], 3, 1, 1)[0, 0] > 0).cuda()
    chance = float(occ.float().mean())

    def frac(b):
        e = b[0].pow(2).sum(0)
        return float(e[occ].sum() / e.sum())

    f_real, f_const = frac(base['bev']), frac(abl_depth['bev'])
    ok = f_real > f_const and f_real > chance
    log(f'[C3] BEV feature energy on LiDAR-occupied cells (+-1 cell):\n'
        f'     Metric3D depth: {f_real * 100:.1f}%   constant 20 m: '
        f'{f_const * 100:.1f}%   chance (occupied area): {chance * 100:.1f}%')
    results['C3'] = ok
    log(f'[C3] => {verdict(ok)}  (特徴が LiDAR で物がある位置に置かれている)')

    for hk in hooks:
        hk.remove()

    # ================================================================ figure
    mean = torch.tensor(m.img_conf['img_mean']).view(3, 1, 1)
    std = torch.tensor(m.img_conf['img_std']).view(3, 1, 1)
    img = (imgs[0, 0, CAM_FRONT].cpu() * std + mean).clamp(0, 255)
    img = img.permute(1, 2, 0).numpy().astype(np.uint8)
    f = dino[CAM_FRONT].float().reshape(768, -1).T
    f = f - f.mean(0)
    _, _, vt = torch.linalg.svd(f, full_matrices=False)
    pc = (f @ vt[:3].T).reshape(16, 44, 3).cpu().numpy()
    pc = (pc - pc.min((0, 1))) / (pc.max((0, 1)) - pc.min((0, 1)) + 1e-6)
    centers = (torch.arange(D, device=prior.device) * bb.d_bound[2]
               + bb.d_bound[0]).view(1, -1, 1, 1)
    p = prior[CAM_FRONT:CAM_FRONT + 1]
    ed = ((p / p.sum(1, keepdim=True).clamp(min=1e-6)) * centers).sum(1)[0]
    ed = torch.where(p.sum(1)[0] > 0.05, ed, torch.full_like(ed, np.nan))
    occ_np = occ.cpu().numpy()

    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5), facecolor='#fcfcfb')
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('CAM_FRONT input (256x704, de-normalized)')
    ax[0, 1].imshow(pc, interpolation='nearest')
    ax[0, 1].set_title('DINOv3 frozen feature 768x16x44 (PCA top-3 as RGB)')
    im = ax[0, 2].imshow(base['metric'][CAM_FRONT, 0].cpu(), cmap='magma_r',
                         vmin=0, vmax=60)
    ax[0, 2].set_title(f'Metric3Dv2 frozen metric depth [m]\n'
                       f'AbsRel vs LiDAR (6 cams) = {absrel:.3f}')
    plt.colorbar(im, ax=ax[0, 2], fraction=0.025)
    im = ax[1, 0].imshow(ed.cpu(), cmap='magma_r', vmin=0, vmax=60,
                         interpolation='nearest')
    ax[1, 0].set_title('depth fed to Lift: E[d] of prior (16x44) [m]\n'
                       'white = out of range (sky / >58 m), lifts nothing')
    plt.colorbar(im, ax=ax[1, 0], fraction=0.025)
    for j, (b, t) in enumerate(((base['bev'], f'Metric3D depth: '
                                 f'{f_real * 100:.1f}% energy on LiDAR cells'),
                                (abl_depth['bev'], f'ablation, const 20 m: '
                                 f'{f_const * 100:.1f}% on LiDAR cells'))):
        e = b[0].pow(2).sum(0).cpu().numpy()
        a_ = ax[1, 1 + j]
        a_.imshow(np.log1p(e), origin='lower', cmap='viridis',
                  extent=(-51.2, 51.2, -51.2, 51.2))
        a_.contour(occ_np, levels=[0.5], colors='white', linewidths=0.4,
                   origin='lower', extent=(-51.2, 51.2, -51.2, 51.2))
        a_.plot(0, 0, marker=(3, 0, -90), ms=10, color='red')
        a_.set_title(f'BEV feature energy (head input)\n{t}')
        a_.set_xlabel('x forward [m]')
        a_.set_ylabel('y left [m]')
    for a_ in ax[0]:
        a_.axis('off')
    ax[1, 0].axis('off')
    fig.suptitle('Frozen DINOv3 / Metric3Dv2 -> Adapter / prior -> Lift -> '
                 'BEV  (white contour = LiDAR-occupied cells, red = ego)',
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = osp.join(args.out, 'verify_frozen_pipeline.png')
    fig.savefig(fig_path, dpi=90, facecolor='#fcfcfb')

    log('\n' + '=' * 72 + '\nSUMMARY\n' + '=' * 72)
    for k, v in results.items():
        log(f'  {k:4s} {verdict(v)}')
    log(f'  ALL {verdict(all(results.values()))}')
    open(osp.join(args.out, 'report.txt'), 'w').write('\n'.join(REPORT) + '\n')
    json.dump({k: bool(v) for k, v in results.items()},
              open(osp.join(args.out, 'results.json'), 'w'), indent=1)
    print('saved', fig_path)


if __name__ == '__main__':
    main()
