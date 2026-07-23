# Full-pipeline GPU serverless worker — builds from THIS (private) repo.
# The infrascan-platform code is vendored in at ./infrascan-platform, so there is
# NO git clone and NO token needed. Build context = repo root.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# --- vendored app (already in the repo; no clone) ---
COPY infrascan-platform /app/infrascan-platform

# --- python deps: platform + GPU extras, DA3, the 3 vision models, storage + runpod SDK ---
RUN pip install --no-cache-dir /app/infrascan-platform[gpu] || \
    pip install --no-cache-dir -e /app/infrascan-platform
RUN pip install --no-cache-dir \
        depth-anything-3 ultralytics \
        "open3d>=0.18" faiss-gpu-cu12 numba pypose einops safetensors \
        pandas prettytable \
        git+https://github.com/openai/CLIP.git \
        boto3 runpod

# --- FastSAM weights (small) ---
RUN mkdir -p /app/infrascan-platform/external/object_proposals/fastsam/weights && \
    curl -fsSL -o /app/infrascan-platform/external/object_proposals/fastsam/weights/FastSAM-x.pt \
      https://huggingface.co/An-619/FastSAM/resolve/main/FastSAM-x.pt || \
    echo "WARN: FastSAM weight fetch failed"

# --- NO weight-baking: keeps the image small enough for the registry to export.
#     DA3 (GIANT) + DINOv2 download ONCE on the first cold start into these caches.
#     (Baking them made a multi-GB layer that failed to commit on push.) ---
ENV HF_HOME=/opt/models/hf TORCH_HOME=/opt/models/torch \
    INFRASCAN_PLATFORM_DIR=/app/infrascan-platform \
    INFRASCAN_WORKROOT=/workspace/runs
RUN mkdir -p /opt/models/hf /opt/models/torch /workspace/runs

COPY handler.py /app/handler.py
COPY storage.py /app/storage.py
CMD ["python", "-u", "/app/handler.py"]
