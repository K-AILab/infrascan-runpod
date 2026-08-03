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

# Build-time info (cheap early layer): the nerfstudio image is a RUNTIME base —
# it ships a PREBUILT gsplat (training uses it) but has NO nvcc. So the scene-graph
# stage must REUSE that prebuilt gsplat (below, via --system-site-packages), never
# compile its own. Print torch/gsplat/nvcc status; never fail on missing nvcc.
RUN python3 -c "import torch;print('[build] torch',torch.__version__,'cuda',torch.version.cuda)"; \
    python3 -c "import gsplat;print('[build] base gsplat',getattr(gsplat,'__version__','?'))" || echo '[build] WARN base gsplat not importable'; \
    { command -v nvcc >/dev/null && echo "[build] nvcc: $(nvcc --version | tail -1)"; } || echo '[build] no nvcc (expected) — scene-graph reuses the base prebuilt gsplat'

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
# --system-site-packages so they reuse the base image's torch AND its prebuilt
# gsplat (no second torch, no gsplat compile — the base has no nvcc):
#   /opt/venv-splat : (base gsplat) + transformers/OWLv2  (splat render + detect, GPU)
#   /opt/venv-main  : open3d + open_clip + boto3           (geometry + CLIP + upload)
# NOTE: gsplat is intentionally NOT installed here — it comes from the base image.
# The final check confirms venv-splat actually sees gsplat + torch before we go on.
# venv-splat: OWLv2 detector. torch + gsplat + numpy/scipy/opencv/Pillow all come
# from the base image / global pip via --system-site-packages, so install ONLY the
# few extras the splat_analyzer needs (avoids resolver clashes with base pins).
RUN python3 -m venv --system-site-packages /opt/venv-splat && \
    /opt/venv-splat/bin/pip install --no-cache-dir -U pip && \
    /opt/venv-splat/bin/pip install --no-cache-dir \
        "transformers==4.50.3" "plyfile==1.1.3" "imageio==2.37.3" "six==1.17.0"
# gsplat's CUDA backend needs a GPU (absent during build), so its import check is
# INFORMATIONAL only — never fail the build here; it's exercised at runtime.
RUN /opt/venv-splat/bin/python -c "import torch,transformers;print('[build] venv-splat: torch',torch.__version__,'transformers',transformers.__version__)"; \
    /opt/venv-splat/bin/python -c "import gsplat;print('[build] venv-splat sees gsplat',getattr(gsplat,'__version__','?'))" || echo '[build] note: gsplat import deferred to runtime (no GPU at build) — OK'

# venv-main: geometry + CLIP + upload. open3d/opencv/boto3/numpy come from the
# global pip layer above (visible via --system-site-packages); install only the
# extras (CLIP, sklearn, plyfile).
RUN python3 -m venv --system-site-packages /opt/venv-main && \
    /opt/venv-main/bin/pip install --no-cache-dir -U pip && \
    /opt/venv-main/bin/pip install --no-cache-dir \
        open_clip_torch scikit-learn "plyfile==1.1.3"
RUN /opt/venv-main/bin/python -c "import open_clip, sklearn, boto3, open3d, plyfile; print('[build] venv-main: open_clip/sklearn/boto3/open3d/plyfile all import OK')"

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
