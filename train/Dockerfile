# Training endpoint — depth-supervised splatfacto (abai's recipe) on a cloud GPU.
# Base = official nerfstudio image (nerfstudio + gsplat + CUDA prebuilt), so we
# avoid the torch/gsplat compile hell. We add YOLO (masks), open3d, node (ksplat),
# boto3/runpod, and abai's vendored training scripts.
#
# Build context = repo ROOT; this Dockerfile lives at train/Dockerfile.
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
USER root

# Node 20 (NodeSource) — the apt default is v12, too old for the ksplat lib's
# optional-chaining (?.) syntax ("SyntaxError: Unexpected token '.'").
RUN apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && node --version

# extra python deps not in the nerfstudio image.
# The nerfstudio image exposes Python as `python3` (no bare `pip` on PATH) — use -m pip.
# scipy: reproject_scanner_depth.py's default path (--point-spacing -1, auto-measure)
# lazily imports scipy.spatial.cKDTree. It is very likely already present via
# nerfstudio's own dependency chain (plyfile, imported unconditionally by the SAME
# script, was never explicitly installed here either, and that import already works
# today) -- installed explicitly anyway since pip no-ops if already satisfied and an
# unverified transitive dependency is not something to assume silently.
RUN python3 -m pip install --no-cache-dir \
        ultralytics open3d opencv-python-headless boto3 runpod scipy

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
