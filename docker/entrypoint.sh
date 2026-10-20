#!/bin/bash
# Activates the conda env and, when the host source tree is bind-mounted over
# /workspace/BEVDepth, builds the BEVDepth CUDA ops into it if they are
# missing (the .so files baked into the image are hidden by the mount).
source /opt/conda/etc/profile.d/conda.sh
conda activate bevfoundation

cd /workspace/BEVDepth
if ! python -c "from bevdepth.ops.voxel_pooling_train import voxel_pooling_train" \
        >/dev/null 2>&1; then
    echo "[entrypoint] building BEVDepth CUDA ops (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})"
    python setup.py develop >/tmp/bevdepth_build.log 2>&1 \
        || { echo "[entrypoint] build failed, see /tmp/bevdepth_build.log"; }
fi

exec "$@"
