# Training endpoint — depth-supervised splatfacto (abai's recipe) on a cloud GPU.
# Base = official nerfstudio image (nerfstudio + gsplat + CUDA prebuilt), so we
# avoid the torch/gsplat compile hell. We add YOLO (masks), open3d, node (ksplat),
# boto3/runpod, abai's vendored training scripts, AND (new) an inline scene-graph
# stage (Approach A: OWLv2-on-splat) that runs right after training on the same
# warm worker — see train/scenegraph/ and train/handler.py step 10.
#
# Build context = repo ROOT; this Dockerfile lives at train/Dockerfile.
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
USER root

# Fast build-time sanity check (cheap early layer): confirm the base torch/CUDA
# and that nvcc exists — gsplat 1.5.3 (scene-graph render) JIT-compiles its CUDA
# kernel at runtime and needs nvcc. Prints to the build log; fails loudly if gone.
RUN python3 -c "import torch;print('[build] torch',torch.__version__,'cuda',torch.version.cuda)" && \
    { command -v nvcc >/dev/null && nvcc --version | tail -1 && echo '[build] nvcc OK'; } || \
    { echo '[build] ERROR: nvcc missing — gsplat cannot JIT-compile at runtime'; exit 1; }

# Node 20 (NodeSource) — the apt default is v12, too old for the ksplat lib's
# optional-chaining (?.) syntax. ninja/build-essential: gsplat's runtime JIT build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates gnupg ninja-build build-essential && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && node --version

# extra python deps not in the nerfstudio image (GLOBAL env, used by training).
# The nerfstudio image exposes Python as `python3` (no bare `pip` on PATH) — use -m pip.
RUN python3 -m pip install --no-cache-dir \
        ultralytics open3d opencv-python-headless boto3 runpod

# ---- scene-graph venvs (Approach A) layered on the base torch/CUDA ----------
# Two isolated venvs (the pipeline's two envs must NOT be merged), both
# --system-site-packages so they reuse the base image's torch (no second torch,
# matched CUDA/ABI):
#   /opt/venv-splat : gsplat 1.5.3 + transformers/OWLv2  (splat render + detect, GPU)
#   /opt/venv-main  : open3d + open_clip + boto3          (geometry + CLIP + upload)
# TORCH_CUDA_ARCH_LIST covers train's Ada+Ampere GPUs if gsplat AOT-builds; if it
# JIT-builds at runtime instead, it targets the actual GPU and this is ignored.
RUN python3 -m venv --system-site-packages /opt/venv-splat && \
    /opt/venv-splat/bin/pip install --no-cache-dir -U pip && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" /opt/venv-splat/bin/pip install --no-cache-dir \
        "numpy<2" "gsplat==1.5.3" "transformers==4.50.3" "imageio==2.37.3" \
        "plyfile==1.1.3" "opencv-python-headless==4.11.0.86" "scipy==1.15.3" \
        "six==1.17.0" Pillow
RUN python3 -m venv --system-site-packages /opt/venv-main && \
    /opt/venv-main/bin/pip install --no-cache-dir -U pip && \
    /opt/venv-main/bin/pip install --no-cache-dir \
        "numpy<2" scipy scikit-learn Pillow "open3d==0.18.*" \
        "opencv-python-headless==4.11.0.86" "plyfile==1.1.3" open_clip_torch boto3

# vendored training scripts + storage helper + scene-graph pipeline
COPY train/vendor     /app/train/vendor
COPY storage.py       /app/storage.py
COPY train/scenegraph /app/train/scenegraph

# node deps for the .ply -> .ksplat converter
RUN cd /app/train/vendor/node && npm install --omit=dev

# bake OWLv2 (venv-splat) + CLIP ViT-H-14/dfn5b (venv-main) into a fixed cache so
# the scene-graph stage runs offline. run_scenegraph.py points HF_HOME here.
RUN HF_HOME=/opt/sg-models/hf HUGGINGFACE_HUB_CACHE=/opt/sg-models/hf TORCH_HOME=/opt/sg-models/torch \
        /opt/venv-splat/bin/python /app/train/scenegraph/prefetch_weights.py owlv2 && \
    HF_HOME=/opt/sg-models/hf HUGGINGFACE_HUB_CACHE=/opt/sg-models/hf TORCH_HOME=/opt/sg-models/torch \
        /opt/venv-main/bin/python  /app/train/scenegraph/prefetch_weights.py clip

COPY train/handler.py /app/handler.py

# Heavy training scratch + checkpoints go on the CONTAINER DISK (/train), NOT the
# small network volume — splatfacto checkpoints are ~700MB each and filled the
# volume mid-run. Bump the endpoint's Container Disk to ~60GB. Model caches also
# go here (re-downloaded per cold start; small, fine).
# PY_SPLAT/PY_MAIN: the scene-graph venvs (used by train/handler.py step 10).
ENV HF_HOME=/train/hf TORCH_HOME=/train/torch \
    YOLO_CONFIG_DIR=/train/yolo \
    TRAIN_WORKROOT=/train \
    PY_SPLAT=/opt/venv-splat/bin/python \
    PY_MAIN=/opt/venv-main/bin/python
WORKDIR /app
CMD ["python3", "-u", "/app/handler.py"]
