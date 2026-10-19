from bevdepth.exps.nuscenes.base_exp import (
    BEVDepthLightningModel as BaseLightningModel,
)
from bevdepth.exps.base_cli import run_cli
from bevdepth.layers.backbones.dinov3_lss_fpn import DINOv3LSSFPN


class BEVDepthLightningModel(BaseLightningModel):
    validation_epoch_end = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


if __name__ == "__main__":
    run_cli(
        BEVDepthLightningModel,
        "bev_depth_dinov3",
    )
