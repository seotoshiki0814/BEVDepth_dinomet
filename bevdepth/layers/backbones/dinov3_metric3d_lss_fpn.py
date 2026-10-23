import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevdepth.layers.backbones.dinov3_lss_fpn import DINOv3LSSFPN


class DINOv3Metric3DLSSFPN(DINOv3LSSFPN):
    """BEVDepth with frozen DINOv3 image features and a frozen Metric3Dv2
    depth prior.

    Metric3Dv2 predicts depth in a canonical camera space (focal=1000). It is
    converted to metric depth with the actual (ida-augmented) focal length,
    then turned into a distribution over BEVDepth depth bins on the feature
    grid.

    depth_prior_mode:
        'concat':  the prior distribution is concatenated to the image
                   features fed to DepthNet, which learns to refine it.
        'replace': the prior is used directly as the LSS depth distribution;
                   DepthNet only provides context features.
        'lift':    no DepthNet. The prior is the LSS depth distribution and
                   context = Conv1x1(concat[DINOv3 adapter feat, prior]).

    depth_prior_type:
        'hard':  per-cell histogram of 4x4 sub-sampled Metric3D depths.
        'gauss': each of the 4x4 sub-samples contributes a Gaussian
                 N(s * d, sigma(d)), sigma(d) = max(sigma_min, k * s * d),
                 integrated over the bin edges and averaged per cell.
                 Defaults (s=0.958, k=0.18, sigma_min=0.5 m) were calibrated
                 against LiDAR on the nuScenes-mini train split
                 (scripts/calib_metric3d_sigma.py).
    """

    CANONICAL_FOCAL = 1000.0

    def __init__(self,
                 *args,
                 metric3d_model='metric3d_vit_small',
                 metric3d_input_size=(616, 1064),
                 depth_prior_mode='concat',
                 depth_prior_type=None,
                 depth_prior_eps=1e-3,
                 depth_scale=0.958,
                 depth_sigma_k=0.18,
                 depth_sigma_min=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert depth_prior_mode in ('concat', 'replace', 'lift')
        if depth_prior_type is None:
            depth_prior_type = 'gauss' if depth_prior_mode == 'lift' \
                else 'hard'
        assert depth_prior_type in ('hard', 'gauss')
        assert all(s % 14 == 0 for s in metric3d_input_size), \
            'Metric3Dv2 ViT needs input size divisible by 14'

        self.depth_prior_mode = depth_prior_mode
        self.depth_prior_type = depth_prior_type
        self.depth_prior_eps = depth_prior_eps
        self.depth_scale = depth_scale
        self.depth_sigma_k = depth_sigma_k
        self.depth_sigma_min = depth_sigma_min
        self.metric3d_input_size = tuple(metric3d_input_size)

        # METRIC3D_REPO_DIR (set in the Dockerfile) points to a clone pinned
        # to a fixed commit; otherwise fall back to GitHub main via torch.hub.
        repo_dir = os.environ.get('METRIC3D_REPO_DIR')
        if repo_dir and os.path.isfile(os.path.join(repo_dir, 'hubconf.py')):
            self.metric3d = torch.hub.load(repo_dir,
                                           metric3d_model,
                                           source='local',
                                           pretrain=True)
        else:
            self.metric3d = torch.hub.load('yvanyin/metric3d',
                                           metric3d_model,
                                           pretrain=True,
                                           trust_repo=True)
        for param in self.metric3d.parameters():
            param.requires_grad_(False)
        self.metric3d.eval()

        if depth_prior_mode == 'concat':
            old = self.depth_net.reduce_conv[0]
            self.depth_net.reduce_conv[0] = nn.Conv2d(
                old.in_channels + self.depth_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
            )
        elif depth_prior_mode == 'lift':
            in_channels = self.depth_net.reduce_conv[0].in_channels
            del self.depth_net
            self.context_conv = nn.Conv2d(in_channels + self.depth_channels,
                                          self.output_channels,
                                          kernel_size=1)

        # Filled by get_cam_feats, consumed by _forward_depth_net.
        self._canonical_depth = None
        # Last metric depth (B*N, 1, H', W') and prior, kept for analysis.
        self.last_metric_depth = None
        self.last_depth_prior = None

    def train(self, mode=True):
        super().train(mode)
        self.metric3d.eval()
        return self

    @torch.no_grad()
    def _run_metric3d(self, imgs):
        """imgs: (M, 3, H, W) normalized RGB -> canonical depth (M,1,h,w).

        Same preprocessing as the official demo: aspect-preserving resize to
        fit metric3d_input_size, then zero padding (0 == mean color after
        normalization). Plain resizing to a non-canonical size biases the
        scale (AbsRel 0.22 vs 0.14 on nuScenes).
        """
        H, W = imgs.shape[-2:]
        in_h, in_w = self.metric3d_input_size
        scale = min(in_h / H, in_w / W)
        h, w = round(H * scale), round(W * scale)
        pad_h, pad_w = in_h - h, in_w - w
        top, left = pad_h // 2, pad_w // 2
        x = F.interpolate(imgs, size=(h, w), mode='bilinear',
                          align_corners=False)
        x = F.pad(x, (left, pad_w - left, top, pad_h - top))
        depths = []
        for chunk in x.split(self.camera_chunk_size, dim=0):
            pred_depth, _, _ = self.metric3d.inference({'input': chunk})
            depths.append(pred_depth.float()[:, :, top:top + h,
                                             left:left + w])
        self._metric3d_scale = scale
        return torch.cat(depths, dim=0)

    def get_cam_feats(self, imgs):
        B, S, N, C, H, W = imgs.shape
        # Only the key sweep (index 0) feeds DepthNet.
        self._canonical_depth = self._run_metric3d(imgs[:, 0].reshape(
            B * N, C, H, W))
        return super().get_cam_feats(imgs)

    def _depth_prior(self, mats_dict, feat_hw):
        canonical = self._canonical_depth
        intrin = mats_dict['intrin_mats'][:, 0]
        ida = mats_dict['ida_mats'][:, 0]
        # Focal length of the augmented image fed to the network. ida may
        # contain flip (negative [0,0]) and rotation, so take the resize
        # factor from the determinant of its 2x2 linear part.
        resize = ida[..., :2, :2].float().det().abs().sqrt()
        fx = (intrin[..., 0, 0].float() * resize).reshape(-1, 1, 1, 1)
        fx = fx * self._metric3d_scale
        metric = canonical * fx / self.CANONICAL_FOCAL
        self.last_metric_depth = metric.detach()

        # Sample at 4x the feature resolution, then aggregate per cell.
        # Bin i covers [d_min + i*step, d_min + (i+1)*step), the same
        # convention as get_downsampled_gt_depth.
        fh, fw = feat_hw
        depth = F.interpolate(metric, size=(fh * 4, fw * 4), mode='nearest')
        d_min, d_max, d_step = self.d_bound
        if self.depth_prior_type == 'gauss':
            mu = depth * self.depth_scale
            sigma = (mu * self.depth_sigma_k).clamp(min=self.depth_sigma_min)
            edges = d_min + d_step * torch.arange(
                self.depth_channels + 1, device=mu.device,
                dtype=mu.dtype).view(1, -1, 1, 1)
            cdf = torch.special.ndtr((edges - mu) / sigma)
            per_sample = cdf[:, 1:] - cdf[:, :-1]
        else:
            bins = ((depth - d_min) / d_step).floor().long()
            valid = (bins >= 0) & (bins < self.depth_channels)
            bins = bins.clamp(0, self.depth_channels - 1)
            per_sample = F.one_hot(bins[:, 0], self.depth_channels)
            per_sample = per_sample.permute(0, 3, 1, 2).float() * \
                valid.float()
        # Mass outside [d_min, d_max) is dropped, so sum <= 1.
        prior = F.avg_pool2d(per_sample, kernel_size=4)
        self.last_depth_prior = prior.detach()
        return prior

    def _forward_depth_net(self, feat, mats_dict):
        prior = self._depth_prior(mats_dict, feat.shape[-2:])
        if self.depth_prior_mode == 'lift':
            # Downstream LSS does softmax(depth logits) * context. Feed the
            # normalized prior as logits and fold its in-range mass into the
            # context, so the lifted feature is prior * context and pixels
            # whose depth is out of range (sky, >58 m) place nothing.
            mass = prior.sum(dim=1, keepdim=True)
            p = torch.where(mass > 1e-6, prior / mass.clamp(min=1e-6),
                            torch.full_like(prior, 1.0 / prior.shape[1]))
            logits = p.clamp(min=1e-6).log()
            context = self.context_conv(
                torch.cat([feat, prior.to(feat.dtype)], dim=1))
            context = context * mass.to(context.dtype)
            return torch.cat([logits.to(context.dtype), context], dim=1)

        prior = prior.to(feat.dtype)
        if self.depth_prior_mode == 'concat':
            return self.depth_net(torch.cat([feat, prior], dim=1), mats_dict)

        out = self.depth_net(feat, mats_dict)
        # softmax(log(p)) == p, so the downstream softmax recovers the prior.
        prior = prior + self.depth_prior_eps
        prior = prior / prior.sum(dim=1, keepdim=True)
        return torch.cat([prior.log(), out[:, self.depth_channels:]], dim=1)
