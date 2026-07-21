"""
Sample perspective views from equirectangular panorama frames.

Output naming convention:  {frame:06d}_pz000_y{yaw:03d}_normal.jpg

This pattern is required by da3_streaming.py's serpentine ordering, which
ensures that consecutive images in the processing list have overlapping
fields of view — critical for DA3's chunk-based depth/pose alignment.
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}
DEFAULT_YAWS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


def pano_to_perspective(eqr_bgr, fov_deg, yaw_deg, pitch_deg, out_w, out_h):
    h, w = eqr_bgr.shape[:2]

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

    u = (lon / (2 * np.pi) + 0.5) * w
    v = (0.5 - lat / np.pi) * h

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    out = cv2.remap(
        eqr_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )
    return out


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
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = list_images(in_dir)
    if not frames:
        raise RuntimeError(f"No images found in {in_dir}")

    out_w = out_h = args.out_size
    pitch_list = args.pitches if args.pitches is not None else [args.pitch]

    total_views = 0
    for frame_idx, frame_path in enumerate(frames):
        eqr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if eqr is None:
            print(f"[WARN] Could not read {frame_path}")
            continue

        for pitch in pitch_list:
            for yaw in args.yaws:
                view = pano_to_perspective(
                    eqr_bgr=eqr,
                    fov_deg=args.fov,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    out_w=out_w,
                    out_h=out_h
                )
                # Encode pitch as unsigned 3-digit: -30 -> 330, 0 -> 000, 30 -> 030
                pitch_enc = pitch % 360
                out_name = f"{frame_idx:06d}_pz{pitch_enc:03d}_y{yaw:03d}_normal.jpg"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), view)
                total_views += 1

        if frame_idx % 20 == 0:
            print(f"Processed {frame_idx}/{len(frames)} frames...")

    n_pitches = len(pitch_list)
    n_yaws = len(args.yaws)
    print(f"Done. {len(frames)} frames x {n_pitches} pitches x {n_yaws} yaws "
          f"= {total_views} views saved to: {out_dir}")


if __name__ == "__main__":
    main()
