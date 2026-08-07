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

# --- DA3 weights (config.json + model.safetensors ~GIANT + dino_salad.ckpt, ~6.6GB
#     total), BAKED into the image — same reasoning as the pano_clean weights above.
#     This used to live on a network volume ("download once, every future worker
#     reuses it"), but that meant every worker had to run in whichever ONE datacenter
#     that volume physically lived in, which repeatedly starved this endpoint of GPU
#     capacity. Baking them in gets the same "no redownload" property WITHOUT pinning
#     to a datacenter — any worker, anywhere, already has them the moment it starts. ---
RUN mkdir -p /app/da3_weights && \
    curl -fsSL -o /app/da3_weights/config.json \
      https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/config.json && \
    curl -fsSL -o /app/da3_weights/model.safetensors \
      https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/model.safetensors && \
    curl -fsSL -o /app/da3_weights/dino_salad.ckpt \
      https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt

# --- Everything else (HF/torch.hub's OWN incidental downloads, e.g. the dinov2 repo
#     code SALAD's backbone pulls in — separate from the explicit weights above and
#     not worth replicating at build time) + the per-job scratch dir now live on the
#     CONTAINER DISK (/ingest), not a network volume — mirrors the train endpoint's
#     already-proven pattern ("scratch+ckpts on container disk, not the small volume").
#     handler.py wipes WORKROOT/<slug> at job start+end anyway, so it never needed to
#     persist across jobs — it just needed to be big enough (see containerDiskInGb on
#     the endpoint template), which a network volume was never actually required for. ---
ENV HF_HOME=/ingest/hf TORCH_HOME=/ingest/torch \
    INFRASCAN_PLATFORM_DIR=/app/pipeline \
    INFRASCAN_WORKROOT=/ingest/runs \
    DA3_WEIGHTS_DIR=/app/da3_weights \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PANO_CLEAN_YOLO=/app/weights/yolo11x-seg.pt \
    LAMA_TORCH_HOME=/app/lama_cache

COPY handler.py /app/handler.py
COPY storage.py /app/storage.py
COPY pipeline_panoclean /app/pipeline_panoclean
CMD ["python", "-u", "/app/handler.py"]
