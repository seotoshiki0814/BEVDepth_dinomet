import torch
import torch.nn as nn
import timm

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN


class DINOv3LSSFPN(BaseLSSFPN):
    """BEVDepth with a frozen timm DINOv3 image encoder."""

    def __init__(self, *args, camera_chunk_size=1, **kwargs):
        # Reuse the original geometry, DepthNet and voxel-pooling setup.
        super().__init__(*args, **kwargs)

        # Remove the original ResNet50 and SECONDFPN.
        del self.img_backbone
        del self.img_neck

        self.camera_chunk_size = camera_chunk_size

        # features_only=True prunes the final LayerNorm to Identity, which
        # leaves raw block outputs (std~50, |x|>1000). Keep the full model
        # and take the normalized last block via forward_intermediates.
        self.dinov3 = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
        )
        self.dinov3_out_index = len(self.dinov3.blocks) - 1

        for param in self.dinov3.parameters():
            param.requires_grad_(False)

        self.dinov3.eval()

        # DINOv3: 768 channels -> BEVDepth DepthNet: 512 channels
        self.adapter = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.dinov3.eval()
        return self

    def get_cam_feats(self, imgs):
        B, S, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B * S * N, C, H, W)

        # Frozen backbone: no gradients or activation storage.
        features = []
        with torch.no_grad():
            for chunk in imgs.split(self.camera_chunk_size, dim=0):
                feat = self.dinov3.forward_intermediates(
                    chunk,
                    indices=[self.dinov3_out_index],
                    norm=True,
                    output_fmt="NCHW",
                    intermediates_only=True,
                )[0]
                features.append(feat)

        features = torch.cat(features, dim=0)
        features = self.adapter(features)

        expected_hw = (
            H // self.downsample_factor,
            W // self.downsample_factor,
        )
        if features.shape[-2:] != expected_hw:
            raise RuntimeError(
                f"Feature shape mismatch: {features.shape[-2:]} "
                f"!= {expected_hw}"
            )

        return features.reshape(
            B, S, N,
            features.shape[1],
            features.shape[2],
            features.shape[3],
        )
