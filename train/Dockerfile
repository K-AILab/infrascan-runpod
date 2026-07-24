# Training endpoint — depth-supervised splatfacto (abai's recipe) on a cloud GPU.
# Base = official nerfstudio image (nerfstudio + gsplat + CUDA prebuilt), so we
# avoid the torch/gsplat compile hell. We add YOLO (masks), open3d, node (ksplat),
# boto3/runpod, and abai's vendored training scripts.
#
# Build context = repo ROOT; this Dockerfile lives at train/Dockerfile.
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs npm curl git && rm -rf /var/lib/apt/lists/*

# extra python deps not in the nerfstudio image.
# The nerfstudio image exposes Python as `python3` (no bare `pip` on PATH) — use -m pip.
RUN python3 -m pip install --no-cache-dir \
        ultralytics open3d opencv-python-headless boto3 runpod

# vendored training scripts + storage helper
COPY train/vendor /app/train/vendor
COPY storage.py   /app/storage.py

# node deps for the .ply -> .ksplat converter
RUN cd /app/train/vendor/node && npm install --omit=dev

COPY train/handler.py /app/handler.py

# Heavy training scratch + checkpoints go on the CONTAINER DISK (/train), NOT the
# small network volume — splatfacto checkpoints are ~700MB each and filled the
# volume mid-run ("PytorchStreamWriter … file write failed"). Bump the endpoint's
# Container Disk to ~60GB. Model caches also go here (re-downloaded per cold start;
# small, fine).
ENV HF_HOME=/train/hf TORCH_HOME=/train/torch \
    YOLO_CONFIG_DIR=/train/yolo \
    TRAIN_WORKROOT=/train
WORKDIR /app
CMD ["python3", "-u", "/app/handler.py"]
