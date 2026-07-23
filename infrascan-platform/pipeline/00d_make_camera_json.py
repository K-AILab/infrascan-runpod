"""Generate cameras.json from DA3 camera poses and an image directory."""
import argparse
import json
import numpy as np
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Create cameras.json for the 3D web viewer")
    parser.add_argument("--poses_txt", required=True,
                        help="Path to camera_poses.txt (one 4x4 matrix per line)")
    parser.add_argument("--images_dir", required=True,
                        help="Directory of images corresponding to poses")
    parser.add_argument("--output_json", required=True,
                        help="Output cameras.json path")
    args = parser.parse_args()

    poses_path = Path(args.poses_txt)
    images_dir = Path(args.images_dir)

    # Use da3_streaming's recorded processing order (serpentine), which is what
    # camera_poses.txt lines are indexed by — NOT an alphabetical sort.
    order_file = poses_path.parent / "img_list.txt"
    images = [l.strip() for l in order_file.read_text().splitlines() if l.strip()]

    lines = [ln.strip() for ln in poses_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(images), \
        f"poses={len(lines)} images={len(images)} mismatch"

    out = []
    for i, ln in enumerate(lines):
        vals = np.fromstring(ln, sep=" ")
        T = vals.reshape(4, 4)
        t = T[:3, 3]       # camera position (from C2W)
        R = T[:3, :3]

        out.append({
            "id": i,
            "pos": [float(t[0]), float(t[1]), float(t[2])],
            "xy":  [float(t[0]), float(t[2])],
            "pano": f"panos/{images[i]}",
            "R": [[float(x) for x in row] for row in R.tolist()]
        })

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} with {len(out)} entries")


if __name__ == "__main__":
    main()
