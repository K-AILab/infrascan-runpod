"""RunPod serverless worker: Insta360 video -> data (infrascan stage 0).

Scope (deliberately minimal for learning): the CPU part of the pipeline only —
stitch (.insv->equirect) -> extract frames -> sample perspective views.
No depth/poses (DA3), no scene reconstruction, no viewer.

Input JSON:  {"input": {"video_url": "https://.../clip.mp4", "every_n": 100}}
Output JSON: {"num_frames":N, "num_views":M, "sample_view_name":..., "sample_view_jpg_base64":...}
The sample view is returned inline (base64) so you can SEE it worked without a bucket.
"""
import base64, glob, os, subprocess, sys, tempfile, urllib.request
import runpod

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "pipeline")


def _run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def handler(job):
    inp = job.get("input", {}) or {}
    video_url = inp.get("video_url")
    every_n = int(inp.get("every_n", 100))
    if not video_url:
        return {"error": "provide input.video_url (a reachable http(s) URL to a .insv or .mp4)"}

    work = tempfile.mkdtemp(prefix="infrascan_")
    ext = os.path.splitext(video_url.split("?")[0])[1].lower()
    if ext not in (".insv", ".insp", ".mp4", ".mov", ".m4v"):
        ext = ".mp4"
    vid = os.path.join(work, "input" + ext)
    frames = os.path.join(work, "frames")
    views = os.path.join(work, "views")
    eq = os.path.join(work, "equirect.mp4")

    try:
        print(f"[dl] {video_url}", flush=True)
        urllib.request.urlretrieve(video_url, vid)

        _run([sys.executable, f"{PIPE}/_00_stitch_insv.py", "--input", vid, "--output", eq])
        _run([sys.executable, f"{PIPE}/00_video_to_img.py", "--video", eq,
              "--output_dir", frames, "--every_n", str(every_n)])
        _run([sys.executable, f"{PIPE}/00a_sample_views.py", "--input_dir", frames,
              "--output_dir", views])

        vs = sorted(glob.glob(f"{views}/*.jpg"))
        sample = vs[len(vs) // 2] if vs else None
        return {
            "num_frames": len(glob.glob(f"{frames}/*.jpg")),
            "num_views": len(vs),
            "sample_view_name": os.path.basename(sample) if sample else None,
            "sample_view_jpg_base64": base64.b64encode(open(sample, "rb").read()).decode() if sample else None,
        }
    except subprocess.CalledProcessError as e:
        return {"error": f"pipeline stage failed: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
