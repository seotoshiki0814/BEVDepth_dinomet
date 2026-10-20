# BEVDepth + DINOv3 (frozen) + Metric3Dv2 (frozen)
#
# Target GPUs: RTX 3060 (Ampere, sm_86) / RTX 6000 Ada (Ada, sm_89)
#   - CUDA 12.1 / cuDNN 8.9 / PyTorch 2.1.2 / Python 3.11
#   - Host NVIDIA driver >= 530 (CUDA 12.1). Both GPUs are supported by it.
#   - Custom CUDA ops (BEVDepth voxel pooling) are compiled for 8.6 and 8.9.
#     The prebuilt mmcv wheel ships sm_86 cubins, which run natively on
#     sm_89 (same major). Set --build-arg MMCV_FROM_SOURCE=1 to compile mmcv
#     for 8.6;8.9 explicitly (adds ~30-60 min).
#
# Build (run at the repository root):
#   docker build -f docker/Dockerfile -t bevdepth-foundation:cu121 .

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG MINIFORGE_VERSION=24.3.0-0
ARG METRIC3D_REF=eb5b6fac0dc155e4e52f576e304fbf11655ff339
ARG MMCV_FROM_SOURCE=0
ARG PREFETCH_WEIGHTS=1
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential cmake ninja-build \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
        libboost-dev libboost-system-dev libboost-filesystem-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- Python env
RUN wget -q "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y -n bevfoundation python=3.11 pip \
    && /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/envs/bevfoundation/bin:/opt/conda/bin:${PATH}

RUN python -m pip install --no-cache-dir 'pip<25' 'setuptools==69.5.1' wheel

RUN python -m pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

COPY docker/requirements-pinned.txt /tmp/requirements-pinned.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-pinned.txt

# ---------------------------------------------------------------- OpenMMLab
RUN if [ "${MMCV_FROM_SOURCE}" = "1" ]; then \
        MMCV_WITH_OPS=1 python -m pip install --no-cache-dir \
            --no-build-isolation 'mmcv==2.1.0'; \
    else \
        python -m pip install --no-cache-dir 'mmcv==2.1.0' \
            -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html; \
    fi \
    && python -m pip install --no-cache-dir 'mmdet==3.2.0' 'mmsegmentation==1.2.2' \
    && python -m pip install --no-cache-dir --no-build-isolation 'mmdet3d==1.4.0' \
    && python -m pip install --no-cache-dir -c /tmp/requirements-pinned.txt \
        -r /tmp/requirements-pinned.txt \
    && python -c "import torch, mmcv, mmcv._ext, mmdet, mmdet3d; \
print('torch', torch.__version__, 'mmcv', mmcv.__version__, \
'mmdet', mmdet.__version__, 'mmdet3d', mmdet3d.__version__)"

# ---------------------------------------------------------------- Metric3Dv2
# Pinned clone, loaded with torch.hub(source='local') via METRIC3D_REPO_DIR
# (see bevdepth/layers/backbones/dinov3_metric3d_lss_fpn.py).
RUN git clone https://github.com/YvanYin/Metric3D.git /opt/Metric3D \
    && git -C /opt/Metric3D checkout "${METRIC3D_REF}"
ENV METRIC3D_REPO_DIR=/opt/Metric3D

# ---------------------------------------------------------------- Weights
# Baked into the image so that runs are offline and reproducible.
ENV HF_HOME=/opt/model_cache/huggingface \
    TORCH_HOME=/opt/model_cache/torch
COPY docker/prefetch_weights.py /opt/prefetch_weights.py
RUN if [ "${PREFETCH_WEIGHTS}" = "1" ]; then python /opt/prefetch_weights.py; fi

# ---------------------------------------------------------------- BEVDepth
WORKDIR /workspace/BEVDepth
COPY . /workspace/BEVDepth
RUN python setup.py develop \
    && python -c "from bevdepth.ops.voxel_pooling_train import voxel_pooling_train; \
from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference; \
print('BEVDepth CUDA ops OK')"
ENV PYTHONPATH=/workspace/BEVDepth

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && echo "conda activate bevfoundation" >> /root/.bashrc

EXPOSE 6006
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
