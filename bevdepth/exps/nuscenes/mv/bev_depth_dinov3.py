import torch

from bevdepth.exps.nuscenes.base_exp import (
    BEVDepthLightningModel as BaseLightningModel,
)
from bevdepth.exps.base_cli import run_cli
from bevdepth.layers.backbones.dinov3_lss_fpn import DINOv3LSSFPN


class BEVDepthLightningModel(BaseLightningModel):
    validation_epoch_end = None

    def __init__(self, lr=None, max_epochs=24, **kwargs):
        super().__init__(**kwargs)
        # Original lr = 2e-4/64 * batch: 3e-6 at batch 1, far too small.
        self.lr = lr
        self.max_epochs = max_epochs

        # DINOv3はRGB入力を想定する
        self.img_conf = dict(self.img_conf)
        self.img_conf["to_rgb"] = False

        # ResNet50 BackboneをDINOv3に置き換える
        self.model.backbone = DINOv3LSSFPN(
            **self.backbone_conf,
            camera_chunk_size=1,
        )

        print("[DINOv3] Backbone replaced successfully")
        print("[DINOv3] RGB setting:", self.img_conf["to_rgb"])

    def training_step(self, batch):
        loss = super().training_step(batch)

        components = self.model.head.last_loss_components

        self.log(
            "train/loss_heatmap",
            components["loss_heatmap"],
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        self.log(
            "train/loss_bbox",
            components["loss_bbox"],
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        self.log(
            "train/total_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        if self.lr is None:
            return super().configure_optimizers()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=1e-7)
        # Same shape as the original [19, 23] / 24 epoch schedule.
        milestones = [round(self.max_epochs * 19 / 24),
                      round(self.max_epochs * 23 / 24)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones)
        return [[optimizer], [scheduler]]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument('--lr', type=float, default=None)
        return parent_parser


if __name__ == "__main__":
    run_cli(
        BEVDepthLightningModel,
        "bev_depth_dinov3",
    )
