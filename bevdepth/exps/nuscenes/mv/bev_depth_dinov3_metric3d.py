"""BEVDepth + frozen DINOv3 (image features) + frozen Metric3Dv2 (depth prior).

python -u bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
  --batch_size_per_device 1 --max_epochs 1 --limit_train_batches 5 \
  --depth_prior_mode concat

--depth_prior_mode lift: no DepthNet. The (Gaussian) Metric3D prior is used
as the LSS depth and depth_loss is logged as a monitor only (not trained).
"""
import torch

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_dinov3 import \
    BEVDepthLightningModel as DINOv3LightningModel
from bevdepth.layers.backbones.dinov3_metric3d_lss_fpn import \
    DINOv3Metric3DLSSFPN


class BEVDepthLightningModel(DINOv3LightningModel):

    def __init__(self,
                 depth_prior_mode='concat',
                 depth_prior_type=None,
                 depth_scale=0.958,
                 depth_sigma_k=0.18,
                 depth_sigma_min=0.5,
                 metric3d_model='metric3d_vit_small',
                 **kwargs):
        super().__init__(**kwargs)
        self.depth_prior_mode = depth_prior_mode
        self.model.backbone = DINOv3Metric3DLSSFPN(
            **self.backbone_conf,
            camera_chunk_size=1,
            metric3d_model=metric3d_model,
            depth_prior_mode=depth_prior_mode,
            depth_prior_type=depth_prior_type,
            depth_scale=depth_scale,
            depth_sigma_k=depth_sigma_k,
            depth_sigma_min=depth_sigma_min,
        )
        bb = self.model.backbone
        print('[Metric3D] depth_prior_mode:', depth_prior_mode,
              '/ depth_prior_type:', bb.depth_prior_type,
              f'/ s={depth_scale} k={depth_sigma_k} '
              f'sigma_min={depth_sigma_min}')

    def training_step(self, batch):
        if self.depth_prior_mode != 'lift':
            return super().training_step(batch)

        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        mats = {k: v.cuda() for k, v in mats.items()}
        gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
        gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds, depth_preds = self(sweep_imgs.cuda(), mats)
        targets = self.model.get_targets(gt_boxes, gt_labels)
        detection_loss = self.model.loss(targets, preds)

        # No trainable depth branch: depth_loss only monitors the Metric3D
        # prior against LiDAR and is not added to the training loss.
        if depth_labels.dim() == 5:
            depth_labels = depth_labels[:, 0]
        with torch.no_grad():
            depth_loss = self.get_depth_loss(depth_labels.cuda(),
                                             depth_preds.float())

        components = self.model.head.last_loss_components
        self.log('detection_loss', detection_loss)
        self.log('depth_loss_monitor', depth_loss)
        self.log('train/loss_heatmap', components['loss_heatmap'])
        self.log('train/loss_bbox', components['loss_bbox'])
        self.log('train/total_loss', detection_loss, prog_bar=True)
        return detection_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = DINOv3LightningModel.add_model_specific_args(parent_parser)
        parser.add_argument('--depth_prior_mode',
                            type=str,
                            default='concat',
                            choices=['concat', 'replace', 'lift'])
        parser.add_argument('--depth_prior_type',
                            type=str,
                            default=None,
                            choices=['hard', 'gauss'],
                            help='default: gauss for lift, hard otherwise')
        parser.add_argument('--depth_scale', type=float, default=0.958)
        parser.add_argument('--depth_sigma_k', type=float, default=0.18)
        parser.add_argument('--depth_sigma_min', type=float, default=0.5)
        parser.add_argument('--metric3d_model',
                            type=str,
                            default='metric3d_vit_small')
        return parser


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_dinov3_metric3d')
