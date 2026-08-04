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
# virtualenv: the nerfstudio base's python lacks the `venv`/ensurepip bootstrap,
# so we use virtualenv (bundles its own pip) to create the scene-graph venvs below.
RUN python3 -m pip install --no-cache-dir \
        ultralytics open3d opencv-python-headless boto3 runpod virtualenv

# ---- scene-graph venv (Approach A) layered on the base torch/CUDA -----------
# ONE combined venv, matching upstream: run_space_pipeline.py drives every stage
# — including the OWLv2 detector — with a single `sys.executable`, so the whole
# vendored pipeline expects one environment. --system-site-packages so it reuses
# the base image's torch AND its prebuilt gsplat (no second torch, no gsplat
# compile — the base has no nvcc) plus the global pip layer above
# (ultralytics/open3d/opencv/boto3). The detector uses gsplat's stable
# `rasterization` API, which the base gsplat provides. We add ONLY the extras the
# pipeline needs on top (transformers/OWLv2, open_clip/CLIP, sklearn, plyfile,
# imageio, six) — everything else comes through --system-site-packages.
RUN python3 -m virtualenv --system-site-packages /opt/venv-sg && \
    /opt/venv-sg/bin/pip install --no-cache-dir -U pip && \
    /opt/venv-sg/bin/pip install --no-cache-dir \
        "transformers==4.50.3" open_clip_torch scikit-learn \
        "plyfile==1.1.3" "imageio==2.37.3" "six==1.17.0"
# Confirm the combined env sees every import the pipeline needs. gsplat's CUDA
# backend needs a GPU (absent at build), so its check is INFORMATIONAL only.
RUN /opt/venv-sg/bin/python -c "import torch, transformers, open_clip, sklearn, open3d, cv2, plyfile, boto3, ultralytics; print('[build] venv-sg imports OK; torch', torch.__version__, 'transformers', transformers.__version__)"; \
    /opt/venv-sg/bin/python -c "import gsplat;print('[build] venv-sg sees gsplat',getattr(gsplat,'__version__','?'))" || echo '[build] note: gsplat import deferred to runtime (no GPU at build) — OK'

# vendored training scripts + storage helper + scene-graph pipeline
COPY train/vendor     /app/train/vendor
COPY storage.py       /app/storage.py
COPY train/scenegraph /app/train/scenegraph

# node deps for the .ply -> .ksplat converter
RUN cd /app/train/vendor/node && npm install --omit=dev

# Bake OWLv2 + CLIP ViT-H-14/dfn5b into a fixed HF cache, and SAM (mobile_sam.pt)
# into the vendored repo root, so the scene-graph stage runs fully offline.
# run_scenegraph.py points HF_HOME here at runtime. refit_box_from_masks.py loads
# 'mobile_sam.pt' by its default relative name with cwd=<vendored repo>, so the
# weight must live at that repo root.
RUN HF_HOME=/opt/sg-models/hf HUGGINGFACE_HUB_CACHE=/opt/sg-models/hf TORCH_HOME=/opt/sg-models/torch \
        /opt/venv-sg/bin/python /app/train/scenegraph/prefetch_weights.py owlv2 && \
    HF_HOME=/opt/sg-models/hf HUGGINGFACE_HUB_CACHE=/opt/sg-models/hf TORCH_HOME=/opt/sg-models/torch \
        /opt/venv-sg/bin/python /app/train/scenegraph/prefetch_weights.py clip && \
    cd /app/train/scenegraph/vendor/scene-graph-pipeline && \
    YOLO_CONFIG_DIR=/opt/sg-models/yolo \
        /opt/venv-sg/bin/python /app/train/scenegraph/prefetch_weights.py sam

COPY train/handler.py /app/handler.py

# Heavy training scratch + checkpoints go on the CONTAINER DISK (/train), NOT the
# small network volume — splatfacto checkpoints are ~700MB each and filled the
# volume mid-run. Bump the endpoint's Container Disk to ~60GB. Model caches also
# go here (re-downloaded per cold start; small, fine).
# PY_SG: the combined scene-graph venv (used by train/handler.py step 10).
ENV HF_HOME=/train/hf TORCH_HOME=/train/torch \
    YOLO_CONFIG_DIR=/train/yolo \
    TRAIN_WORKROOT=/train \
    PY_SG=/opt/venv-sg/bin/python
WORKDIR /app
CMD ["python3", "-u", "/app/handler.py"]
