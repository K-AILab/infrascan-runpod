# Full-pipeline GPU serverless worker — builds from THIS (private) repo.
# The base pipeline is vendored at ./pipeline (infrascan-platform stripped to
# pipeline-only); our extra stages live in ./pipeline_panoclean. Build ctx = repo root.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# Triton (used by DA3's compute_weighted_mean_triton) JIT-compiles CUDA kernels
# at runtime and needs a host C compiler; the -runtime base ships none.
ENV CC=gcc CXX=g++
WORKDIR /app

# --- base pipeline (vendored infrascan-platform, stripped to pipeline-only: app + pipeline + external) ---
COPY pipeline /app/pipeline

# --- python deps: pipeline pkg + GPU extras, DA3, the 3 vision models, storage + runpod SDK ---
RUN pip install --no-cache-dir /app/pipeline[gpu] || \
    pip install --no-cache-dir -e /app/pipeline
RUN pip install --no-cache-dir \
        depth-anything-3 ultralytics \
        "open3d>=0.18" faiss-cpu numba pypose einops safetensors \
        pandas prettytable \
        simple_lama_inpainting \
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
RUN mkdir -p /app/pipeline/external/object_proposals/fastsam/weights && \
    curl -fsSL -o /app/pipeline/external/object_proposals/fastsam/weights/FastSAM-x.pt \
      https://huggingface.co/An-619/FastSAM/resolve/main/FastSAM-x.pt || \
    echo "WARN: FastSAM weight fetch failed"

# --- operator-removal (pano_clean) weights, BAKED into the image (small: ~320MB total)
#     so the step needs no runtime download and adds nothing to the network volume.
#     YOLO11x-seg (~125MB) loads from an explicit path (PANO_CLEAN_YOLO). LaMa's big-lama.pt
#     (~196MB) is looked up by SimpleLama() at torch.hub.get_dir()/checkpoints/ — we bake it
#     under /app/lama_cache and point the pano_clean subprocess's TORCH_HOME there. ---
RUN mkdir -p /app/weights /app/lama_cache/hub/checkpoints && \
    curl -fsSL -o /app/weights/yolo11x-seg.pt \
      https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-seg.pt && \
    curl -fsSL -o /app/lama_cache/hub/checkpoints/big-lama.pt \
      https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt

# --- Weights cache + work dir live on the mounted NETWORK VOLUME (/runpod-volume).
#     Image stays small (no baked weights); DA3 (GIANT) + DINOv2 download ONCE onto
#     the volume on the first cold start, then every future worker reuses them. ---
ENV HF_HOME=/runpod-volume/hf TORCH_HOME=/runpod-volume/torch \
    INFRASCAN_PLATFORM_DIR=/app/pipeline \
    INFRASCAN_WORKROOT=/runpod-volume/runs \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PANO_CLEAN_YOLO=/app/weights/yolo11x-seg.pt \
    LAMA_TORCH_HOME=/app/lama_cache

COPY handler.py /app/handler.py
COPY storage.py /app/storage.py
COPY pipeline_panoclean /app/pipeline_panoclean
CMD ["python", "-u", "/app/handler.py"]
