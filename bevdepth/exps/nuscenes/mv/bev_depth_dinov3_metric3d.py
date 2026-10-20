"""BEVDepth + frozen DINOv3 (image features) + frozen Metric3Dv2 (depth prior).

python -u bevdepth/exps/nuscenes/mv/bev_depth_dinov3_metric3d.py \
  --batch_size_per_device 1 --max_epochs 1 --limit_train_batches 5 \
  --depth_prior_mode concat
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_dinov3 import \
    BEVDepthLightningModel as DINOv3LightningModel
from bevdepth.layers.backbones.dinov3_metric3d_lss_fpn import \
    DINOv3Metric3DLSSFPN


class BEVDepthLightningModel(DINOv3LightningModel):

    def __init__(self,
                 depth_prior_mode='concat',
                 metric3d_model='metric3d_vit_small',
                 **kwargs):
        super().__init__(**kwargs)
        self.model.backbone = DINOv3Metric3DLSSFPN(
            **self.backbone_conf,
            camera_chunk_size=1,
            metric3d_model=metric3d_model,
            depth_prior_mode=depth_prior_mode,
        )
        print('[Metric3D] depth_prior_mode:', depth_prior_mode)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = DINOv3LightningModel.add_model_specific_args(parent_parser)
        parser.add_argument('--depth_prior_mode',
                            type=str,
                            default='concat',
                            choices=['concat', 'replace'])
        parser.add_argument('--metric3d_model',
                            type=str,
                            default='metric3d_vit_small')
        return parser


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_dinov3_metric3d')
