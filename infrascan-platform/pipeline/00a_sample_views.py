"""
Sample perspective views from equirectangular panorama frames.

Output naming convention:  {frame:06d}_pz{pitch:03d}_y{yaw:03d}_normal.jpg

This pattern is required by da3_streaming.py's serpentine ordering, which
ensures that consecutive images in the processing list have overlapping
fields of view — critical for DA3's chunk-based depth/pose alignment.

Perf: the cv2.remap sampling grid depends only on (fov, yaw, pitch, out_size)
and the equirect dimensions — NOT on image content — so each (yaw,pitch) map is
built ONCE and reused across every frame (was recomputed per crop before), and
frames are processed on a thread pool (cv2 releases the GIL). Output is identical.
"""
import argparse
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}
DEFAULT_YAWS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def build_maps(fov_deg, yaw_deg, pitch_deg, out_w, out_h, eqr_h, eqr_w):
    """Precompute the (map_x, map_y) remap grid for one (yaw, pitch). Depends
    only on geometry + equirect size, so it's reused across every frame."""
    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    fx = (out_w / 2.0) / np.tan(fov / 2.0)
    fy = fx
    cx = out_w / 2.0
    cy = out_h / 2.0

    xs = np.arange(out_w, dtype=np.float32)
    ys = np.arange(out_h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    x = (xv - cx) / fx
    y = -(yv - cy) / fy
    z = np.ones_like(x)

    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm; y /= norm; z /= norm

    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    cpit, spit = np.cos(pitch), np.sin(pitch)

    # yaw around Y
    x1 =  cyaw * x + syaw * z
    y1 =  y
    z1 = -syaw * x + cyaw * z

    # pitch around X
    x2 = x1
    y2 = cpit * y1 - spit * z1
    z2 = spit * y1 + cpit * z1

    lon = np.arctan2(x2, z2)
    lat = np.arcsin(np.clip(y2, -1, 1))

    u = (lon / (2 * np.pi) + 0.5) * eqr_w
    v = (0.5 - lat / np.pi) * eqr_h
    return u.astype(np.float32), v.astype(np.float32)


def pano_to_perspective(eqr_bgr, fov_deg, yaw_deg, pitch_deg, out_w, out_h):
    """Backward-compatible single-crop helper (builds its own map each call)."""
    h, w = eqr_bgr.shape[:2]
    map_x, map_y = build_maps(fov_deg, yaw_deg, pitch_deg, out_w, out_h, h, w)
    return cv2.remap(eqr_bgr, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def list_images(d: Path):
    files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Sample perspective views from equirectangular panorama frames",
        epilog="Output files follow the naming convention required by DA3's serpentine ordering."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory of equirectangular panorama images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save perspective views")
    parser.add_argument("--fov", type=int, default=90,
                        help="Field of view in degrees (default: 90)")
    parser.add_argument("--yaws", type=int, nargs="+", default=DEFAULT_YAWS,
                        help="Yaw angles in degrees (default: 0 30 60 ... 330)")
    parser.add_argument("--pitch", type=int, default=0,
                        help="Pitch angle in degrees (default: 0). Ignored if --pitches is set.")
    parser.add_argument("--pitches", type=int, nargs="+", default=None,
                        help="Multiple pitch angles (e.g., --pitches -30 0 30). Overrides --pitch.")
    parser.add_argument("--out_size", type=int, default=504,
                        help="Output image width and height in pixels (default: 504)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel frame workers (default: 0 = auto from CPU count)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = list_images(in_dir)
    if not frames:
        raise RuntimeError(f"No images found in {in_dir}")

    out_w = out_h = args.out_size
    pitch_list = args.pitches if args.pitches is not None else [args.pitch]

    # equirect size from the first frame (all frames share it) → maps built once
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    eqr_h, eqr_w = first.shape[:2]

    specs = []  # (pitch, yaw, map_x, map_y) — computed ONCE, reused per frame
    for pitch in pitch_list:
        for yaw in args.yaws:
            mx, my = build_maps(args.fov, yaw, pitch, out_w, out_h, eqr_h, eqr_w)
            specs.append((pitch, yaw, mx, my))

    # Each Python worker parallelises frames; keep cv2's own pool at 1 to avoid
    # thread oversubscription (16 workers × N internal threads).
    cv2.setNumThreads(1)

    def process_frame(frame_idx, frame_path):
        eqr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if eqr is None:
            print(f"[WARN] Could not read {frame_path}", flush=True)
            return 0
        n = 0
        for pitch, yaw, mx, my in specs:
            view = cv2.remap(eqr, mx, my,
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            pitch_enc = pitch % 360   # -30 -> 330, 0 -> 000, 30 -> 030
            out_name = f"{frame_idx:06d}_pz{pitch_enc:03d}_y{yaw:03d}_normal.jpg"
            cv2.imwrite(str(out_dir / out_name), view)
            n += 1
        return n

    workers = args.workers or min(len(frames), (os.cpu_count() or 8), 16)
    total_views = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_frame, i, f) for i, f in enumerate(frames)]
        for done, fut in enumerate(as_completed(futs)):
            total_views += fut.result()
            if done % 20 == 0:
                print(f"Processed {done}/{len(frames)} frames...", flush=True)

    n_pitches = len(pitch_list)
    n_yaws = len(args.yaws)
    print(f"Done. {len(frames)} frames x {n_pitches} pitches x {n_yaws} yaws "
          f"= {total_views} views saved to: {out_dir} ({workers} workers)")


if __name__ == "__main__":
    main()
