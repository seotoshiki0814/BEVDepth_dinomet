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
    """

    CANONICAL_FOCAL = 1000.0

    def __init__(self,
                 *args,
                 metric3d_model='metric3d_vit_small',
                 metric3d_input_size=(616, 1064),
                 depth_prior_mode='concat',
                 depth_prior_eps=1e-3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert depth_prior_mode in ('concat', 'replace')
        assert all(s % 14 == 0 for s in metric3d_input_size), \
            'Metric3Dv2 ViT needs input size divisible by 14'

        self.depth_prior_mode = depth_prior_mode
        self.depth_prior_eps = depth_prior_eps
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

        # Sample at 4x the feature resolution, then histogram per cell.
        fh, fw = feat_hw
        depth = F.interpolate(metric, size=(fh * 4, fw * 4), mode='nearest')
        d_min, d_max, d_step = self.d_bound
        bins = ((depth - d_min) / d_step).floor().long()
        valid = (bins >= 0) & (bins < self.depth_channels)
        bins = bins.clamp(0, self.depth_channels - 1)
        one_hot = F.one_hot(bins[:, 0], self.depth_channels)
        one_hot = one_hot.permute(0, 3, 1, 2).float() * valid.float()
        prior = F.avg_pool2d(one_hot, kernel_size=4)
        self.last_depth_prior = prior.detach()
        return prior

    def _forward_depth_net(self, feat, mats_dict):
        prior = self._depth_prior(mats_dict, feat.shape[-2:]).to(feat.dtype)
        if self.depth_prior_mode == 'concat':
            return self.depth_net(torch.cat([feat, prior], dim=1), mats_dict)

        out = self.depth_net(feat, mats_dict)
        # softmax(log(p)) == p, so the downstream softmax recovers the prior.
        prior = prior + self.depth_prior_eps
        prior = prior / prior.sum(dim=1, keepdim=True)
        return torch.cat([prior.log(), out[:, self.depth_channels:]], dim=1)
