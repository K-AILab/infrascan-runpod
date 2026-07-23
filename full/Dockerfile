# Stage A — full-pipeline GPU serverless worker (amd64, built by RunPod from GitHub).
# Runs the CEO's infrascan-platform pipeline headlessly: DA3 + FastSAM + DINOv2 +
# CLIP + backprojection -> object point cloud + index.

# ---- clone stage: pulls the PRIVATE platform repo using a read token, then the
#      token is discarded (never copied into the final image below). ----
FROM alpine/git AS fetch
ARG PLATFORM_REPO=github.com/ch-ho00/infrascan-platform.git
ARG PLATFORM_REF=fix/da3-serpentine-pose-order
ARG GH_TOKEN                                   # fine-grained PAT with read access (build-arg)
RUN git clone --depth 1 --branch ${PLATFORM_REF} \
      https://x-access-token:${GH_TOKEN}@${PLATFORM_REPO} /src \
    && rm -rf /src/.git                        # drop history + any embedded creds

# ---- runtime image (no token present here) ----
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=fetch /src /app/infrascan-platform

# --- python deps ---
# Platform + GPU extras, DA3 streaming reqs, the 3 vision models, storage + runpod SDK.
# (on amd64 pycolmap has wheels, so depth-anything-3 installs without the aarch64 stub)
RUN pip install --no-cache-dir /app/infrascan-platform[gpu] || \
    pip install --no-cache-dir -e /app/infrascan-platform
RUN pip install --no-cache-dir \
        depth-anything-3 \
        ultralytics \
        "open3d>=0.18" faiss-gpu-cu12 numba pypose einops safetensors \
        pandas prettytable \
        git+https://github.com/openai/CLIP.git \
        boto3 runpod

# --- model weights ---
# FastSAM weight file expected at object_proposals/fastsam/weights/FastSAM-x.pt
RUN mkdir -p /app/infrascan-platform/external/object_proposals/fastsam/weights && \
    curl -fsSL -o /app/infrascan-platform/external/object_proposals/fastsam/weights/FastSAM-x.pt \
      https://huggingface.co/An-619/FastSAM/resolve/main/FastSAM-x.pt || \
    echo "WARN: FastSAM weight fetch failed — set it up at runtime/volume"
# No network volume (using AWS S3 for I/O), so BAKE the big weights into the image
# to avoid re-downloading them on every serverless cold start. Caches live in-image.
ENV HF_HOME=/opt/models/hf TORCH_HOME=/opt/models/torch \
    INFRASCAN_PLATFORM_DIR=/app/infrascan-platform \
    INFRASCAN_WORKROOT=/workspace/runs
RUN mkdir -p /opt/models/hf /opt/models/torch /workspace/runs
COPY prefetch_weights.py /app/prefetch_weights.py
RUN python -u /app/prefetch_weights.py || echo "WARN: prefetch incomplete (will download at first cold start)"

COPY handler.py /app/handler.py
COPY storage.py /app/storage.py

CMD ["python", "-u", "/app/handler.py"]
