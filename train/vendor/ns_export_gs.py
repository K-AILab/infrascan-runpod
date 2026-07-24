"""Export a splatfacto checkpoint to a 3DGS .ply using nerfstudio's own
ExportGaussianSplat, WITHOUT requiring pymeshlab.

Two gotchas on the abai/GB10 stack:
  1. torch>=2.6 defaults weights_only=True -> rejects nerfstudio checkpoints.
  2. nerfstudio.scripts.exporter imports exporter_utils, which does
     `import pymeshlab` (only needed for mesh/TSDF/Poisson paths, NOT for
     gaussian-splat). pymeshlab has no reliable aarch64 wheel, so we install a
     stub module that returns a dummy for any attribute access. The stub only
     needs to survive import + the `pymeshlab.Mesh` annotation on
     get_mesh_from_pymeshlab_mesh(); the gaussian-splat path never calls it.

Usage: python ns_export_gs.py --load-config <config.yml> --output-dir <dir>
"""
import sys, types, torch

# 1. trusted checkpoint load
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

# 2. stub pymeshlab (only real pymeshlab symbols -> dummy; dunders behave
#    normally so inspect/importlib don't choke, e.g. during torchvision import)
_stub = types.ModuleType("pymeshlab")
_stub.__file__ = "<stub pymeshlab>"
class _Dummy:  # stands in for pymeshlab.Mesh / MeshSet / etc. at import time
    def __init__(self, *a, **k): pass
def _getattr(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Dummy
_stub.__getattr__ = _getattr
_stub.Mesh = _Dummy
_stub.MeshSet = _Dummy
sys.modules["pymeshlab"] = _stub

from nerfstudio.scripts.exporter import entrypoint
sys.argv = ["ns-export", "gaussian-splat"] + sys.argv[1:]
entrypoint()
