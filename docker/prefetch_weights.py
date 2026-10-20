"""Download and verify the frozen foundation-model weights at image build.

DINOv3 / Metric3Dv2 hashes are those published on HuggingFace (X-Linked-ETag)
and verified against the training environment on 2026-10-20.
"""
import hashlib
import os

import timm
import torch

DINOV3_ID = 'vit_base_patch16_dinov3.lvd1689m'
DINOV3_SHA256 = \
    '1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332'
METRIC3D_URL = ('https://huggingface.co/JUGGHM/Metric3D/resolve/main/'
                'metric_depth_vit_small_800k.pth')
METRIC3D_SHA256 = \
    'b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6'
RESNET50_URL = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(name, path, expected):
    got = sha256(path)
    assert got == expected, f'{name}: sha256 mismatch {got} != {expected}'
    print(f'[OK] {name}: {path} sha256={got[:12]}...')


# DINOv3 (timm format, HF hub timm/vit_base_patch16_dinov3.lvd1689m)
timm.create_model(DINOV3_ID, pretrained=True, num_classes=0)
from huggingface_hub import hf_hub_download  # noqa: E402

check('DINOv3', hf_hub_download(f'timm/{DINOV3_ID}', 'model.safetensors'),
      DINOV3_SHA256)

# Metric3Dv2 ViT-Small (same file hubconf.py downloads)
ckpt_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')
torch.hub.load_state_dict_from_url(METRIC3D_URL, model_dir=ckpt_dir,
                                   map_location='cpu')
check('Metric3Dv2', os.path.join(ckpt_dir, os.path.basename(METRIC3D_URL)),
      METRIC3D_SHA256)

# torchvision ResNet50: the base BEVDepth config builds it before the
# backbone is replaced, so it is needed even for the DINOv3 experiments.
torch.hub.load_state_dict_from_url(RESNET50_URL, model_dir=ckpt_dir,
                                   map_location='cpu')
print('[OK] ResNet50 (torchvision)')
