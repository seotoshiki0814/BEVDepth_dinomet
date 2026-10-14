"""
Camera Layout Generalization Experiment

Phase 1: BEVDepth + ResNet50
Phase 2: BEVDepth + DINOv3 (frozen)
Phase 3: Camera layout variation experiments
"""

import torch

from bevdepth.exps.nuscenes.base_exp import (
    BEVDepthLightningModel as BaseBEVDepthLightningModel,
)
from bevdepth.exps.base_cli import run_cli


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    """Camera-layout research baseline."""

    # Lightning 2.x compatibility
    validation_epoch_end = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch):
        # Keep the original BEVDepth loss calculation.
        loss = super().training_step(batch)

        # Lightning training_step may return a tensor or a dict.
        total_loss = loss["loss"] if isinstance(loss, dict) else loss

        self.log(
            "train/total_loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Record the actual learning rate used by the optimizer.
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        for i, optimizer in enumerate(optimizers):
            self.log(
                f"train/lr_{i}",
                optimizer.param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                logger=True,
            )

        return loss


if __name__ == "__main__":
    run_cli(
        BEVDepthLightningModel,
        "bev_depth_camera_layout",
    )
