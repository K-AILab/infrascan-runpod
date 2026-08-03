# Self-contained 3DETR (facebookresearch/3detr) vendored for inference,
# with pure-PyTorch PointNet++ ops (ops.py) instead of compiled CUDA extensions.
from .detector import Detr3DDetector, SCANNET_CLASSES  # noqa: F401
