"""Bake the model weights into the image so the first (and every) job runs
offline — no HuggingFace fetch at job time.

Run once per venv at build:
    /opt/venv-splat/bin/python prefetch_weights.py owlv2   # transformers OWLv2
    /opt/venv-main/bin/python  prefetch_weights.py clip    # open_clip ViT-H-14/dfn5b

These are the exact model ids the vendored pipeline loads
(external/splat_analyzer/pipeline.py, pipeline2b/clip_utils.py). Downloads land
in HF_HOME / the open_clip cache, both pointed at a baked location by the
Dockerfile.
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


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "owlv2":
        prefetch_owlv2()
    elif which == "clip":
        prefetch_clip()
    else:
        sys.exit("usage: prefetch_weights.py owlv2|clip")
