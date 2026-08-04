"""Bake the model weights into the image so the first (and every) job runs
offline — no HuggingFace/ultralytics fetch at job time.

Run at build under the one combined venv:
    /opt/venv-sg/bin/python prefetch_weights.py owlv2   # transformers OWLv2
    /opt/venv-sg/bin/python prefetch_weights.py clip    # open_clip ViT-H-14/dfn5b
    cd <vendored repo> && /opt/venv-sg/bin/python prefetch_weights.py sam  # mobile_sam.pt

These are the exact model ids the vendored pipeline loads
(external/splat_analyzer/pipeline.py, pipeline2b/clip_utils.py,
pipeline9/refit_box_from_masks.py). OWLv2/CLIP land in HF_HOME / the open_clip
cache; `sam` downloads mobile_sam.pt into the CURRENT directory (run it from the
vendored repo root, where refit_box_from_masks.py loads it by relative name).
"""
import sys

OWLV2_ID = "google/owlv2-base-patch16-ensemble"
CLIP_MODEL = "ViT-H-14"
CLIP_PRETRAINED = "dfn5b"


def prefetch_owlv2() -> None:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print(f"[prefetch] OWLv2 {OWLV2_ID} ...", flush=True)
    Owlv2Processor.from_pretrained(OWLV2_ID)
    Owlv2ForObjectDetection.from_pretrained(OWLV2_ID)
    print("[prefetch] OWLv2 cached", flush=True)


def prefetch_clip() -> None:
    import open_clip
    print(f"[prefetch] CLIP {CLIP_MODEL}/{CLIP_PRETRAINED} ...", flush=True)
    open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    open_clip.get_tokenizer(CLIP_MODEL)
    print("[prefetch] CLIP cached", flush=True)


def prefetch_sam() -> None:
    # ultralytics downloads mobile_sam.pt into the CWD on first construction.
    from ultralytics import SAM
    print("[prefetch] SAM mobile_sam.pt ...", flush=True)
    SAM("mobile_sam.pt")
    print("[prefetch] SAM cached", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "owlv2":
        prefetch_owlv2()
    elif which == "clip":
        prefetch_clip()
    elif which == "sam":
        prefetch_sam()
    else:
        sys.exit("usage: prefetch_weights.py owlv2|clip|sam")
