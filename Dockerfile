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
        "open3d>=0.18" faiss-cpu numba pypose einops safetensors \
        pandas prettytable \
        git+https://github.com/openai/CLIP.git \
        git+https://github.com/cvg/LightGlue.git \
        boto3 runpod

# Some deps above replace torchvision with a wheel that does NOT match the base
# image's torch 2.4.0, which de-registers torchvision's C++ ops and makes any
# `import torchvision` fail with "operator torchvision::nms does not exist".
# Reinstall the matched CUDA-12.4 pair as the LAST step so the ops register.
RUN pip install --no-cache-dir --force-reinstall \
        torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124

# xformers is locked to an EXACT torch build. depth-anything-3's dinov2 imports
# `from xformers.ops import SwiGLU`; the xformers pulled above targets a newer
# torch API (torch.backends.cuda.is_flash_attention_available) absent in 2.4.0.
# Pin the xformers release built against torch 2.4.0.
RUN pip install --no-cache-dir --force-reinstall --no-deps xformers==0.0.27.post2

# --- FastSAM weights (small) ---
RUN mkdir -p /app/infrascan-platform/external/object_proposals/fastsam/weights && \
    curl -fsSL -o /app/infrascan-platform/external/object_proposals/fastsam/weights/FastSAM-x.pt \
      https://huggingface.co/An-619/FastSAM/resolve/main/FastSAM-x.pt || \
    echo "WARN: FastSAM weight fetch failed"

# --- Weights cache + work dir live on the mounted NETWORK VOLUME (/runpod-volume).
#     Image stays small (no baked weights); DA3 (GIANT) + DINOv2 download ONCE onto
#     the volume on the first cold start, then every future worker reuses them. ---
ENV HF_HOME=/runpod-volume/hf TORCH_HOME=/runpod-volume/torch \
    INFRASCAN_PLATFORM_DIR=/app/infrascan-platform \
    INFRASCAN_WORKROOT=/runpod-volume/runs \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY handler.py /app/handler.py
COPY storage.py /app/storage.py
CMD ["python", "-u", "/app/handler.py"]
