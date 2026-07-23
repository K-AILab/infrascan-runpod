#!/usr/bin/env bash
# install_pipeline_models.sh — fetch the model weights the pipeline needs.
# Re-runnable; skips work that's already done.
#
# Installs into:
#   external/depth-anything-3/                   DA3 source + editable install
#   external/object_proposals/fastsam/weights/   FastSAM-x.pt (138 MB)
#   ~/.cache/torch/hub/checkpoints/              dinov2_vitl14_pretrain.pth (1.2 GB)
#   ~/.cache/huggingface/hub/                    DA3 weights (DA3NESTED-GIANT-LARGE-1.1)
#
# DA3 weights are usually already in the HF cache from earlier work.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${PY:=/home/chan/miniconda3/envs/infrascan/bin/python}"
: "${PIP:=$PY -m pip}"

EXTERNAL="$REPO_ROOT/external"
DA3_DIR="$EXTERNAL/depth-anything-3"
FASTSAM_DIR="$EXTERNAL/object_proposals/fastsam/weights"

mkdir -p "$EXTERNAL" "$FASTSAM_DIR" "$HOME/.cache/torch/hub/checkpoints"

# 1. Clone + install DA3 (idempotent)
if [ ! -d "$DA3_DIR" ]; then
  echo "==> cloning Depth-Anything-3 …"
  git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_DIR"
fi
if ! $PY -c "import depth_anything_3" 2>/dev/null; then
  echo "==> installing DA3 (--no-deps to avoid pinning conflicts)"
  $PIP install --no-deps -e "$DA3_DIR"
fi

# 2. FastSAM-x weights (138 MB)
if [ ! -f "$FASTSAM_DIR/FastSAM-x.pt" ]; then
  echo "==> downloading FastSAM-x.pt"
  curl -L -sS --max-time 300 -o "$FASTSAM_DIR/FastSAM-x.pt" \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt
fi
ls -lh "$FASTSAM_DIR/FastSAM-x.pt"

# 3. DINOv2 ViT-L/14 weights (~1.2 GB)
DINO_PATH="$HOME/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
if [ ! -f "$DINO_PATH" ]; then
  echo "==> downloading dinov2_vitl14_pretrain.pth"
  curl -L -sS --max-time 600 -o "$DINO_PATH" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
fi
ls -lh "$DINO_PATH"

# 4. DA3 weights — sanity-check the HF cache (usually already populated)
DA3_CACHE="$HOME/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1"
if [ -d "$DA3_CACHE" ]; then
  echo "==> DA3 weights already cached at $DA3_CACHE"
else
  echo "==> downloading DA3 weights via huggingface_hub"
  $PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='depth-anything/DA3NESTED-GIANT-LARGE-1.1')
print('weights at:', p)
"
fi

# 5. Tell load_fastsam where its weights live
export INFRASCAN_TAGGING_MODELS="$EXTERNAL"
echo
echo "✓ Pipeline models ready."
echo "  Set INFRASCAN_TAGGING_MODELS=$EXTERNAL in worker's environment."
