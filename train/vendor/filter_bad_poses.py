"""Drop cameras.json entries whose raw pose has a large roll error relative to the
capture's own consensus "up" direction -- automates the ad-hoc diagnostic that was run by
hand on every space so far (see PSNR_PLATEAU_DIAGNOSTIC.md Sec 7): a scanner-rig pose
occasionally has a genuine ~15-100+ degree tilt error (the raw pose itself is wrong, not a
rendering/training artifact), and no loss or optimizer setting can train around it. Globally
rare (~1-2% of frames) but each bad-pose frame renders catastrophically (PSNR ~12-14 vs
~19-23 for clean neighbors) at its own viewpoint, so it's cheaper to drop than to leave in.

Restricted to pz000 (eye-level, pitch=0) crops ONLY, matching the exact method validated by
hand on factory13 (26 dropped) and shinhan (56 dropped, reproduced here bit-for-bit: 56/2604
at the default 15deg threshold). Two reasons this subset, not all pitches:
  1. Rotating in YAW about the vertical axis should leave a camera's "up" vector unchanged in
     world space, so pz000's 12 yaw crops per scanpoint give an (almost) yaw-invariant roll
     reading -- mixing in the +-30deg PITCH crops instead measures mostly the intentional
     pitch tilt, not a pose defect (confirmed empirically: including them inflates the
     median from ~1.4deg to ~29deg and floods the >15deg count to >60% of all frames).
  2. Gaussian TRAINING only ever uses pz000 crops anyway (build_hires_dataset.py's own
     --pz 0 default) -- so this is also exactly the subset that matters for training.
Non-pz000 entries (pz030/pz330) are left untouched in cameras.json; they aren't used
downstream by this pipeline and dropping them isn't this script's job.

Method: a camera's world-space "up" direction is -R[:,1] (R is the OpenCV c2w rotation
stored in cameras.json; OpenCV's local Y axis points down -- see make_transforms.py's own
FLIP convention). The CONSENSUS up direction is the normalized mean of every pz000 entry's
up vector, not a hardcoded world axis -- this self-calibrates to whatever orientation this
particular capture's raw coordinate frame happens to be in (index 1 has been "up" for all 5
spaces trained by hand so far, but that isn't assumed here). Entries whose angle to that
consensus exceeds --max-roll-deg (default 15) are dropped.

Usage:
  python filter_bad_poses.py --cameras <src>/cameras.json [--max-roll-deg 15] [--report out.json]
    (edits cameras.json in place; prints + optionally writes how many entries were dropped)
"""
import argparse, json, re
from pathlib import Path

import numpy as np

PZ000_RE = re.compile(r"_pz000_")


def roll_from_consensus(cams):
    """cams: pz000-only entries. Returns (roll_deg [N], consensus_up [3])."""
    R = np.stack([np.asarray(c["R"], dtype=np.float64) for c in cams])  # [N,3,3]
    up = -R[:, :, 1]                                                    # OpenCV c2w -> world "up"
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    consensus = up.mean(axis=0)
    consensus = consensus / np.linalg.norm(consensus)
    cosang = np.clip(up @ consensus, -1.0, 1.0)
    roll_deg = np.degrees(np.arccos(cosang))
    return roll_deg, consensus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--max-roll-deg", type=float, default=15.0)
    ap.add_argument("--report", type=Path, default=None,
                    help="optional path to write a small JSON summary (counts + dropped ids)")
    args = ap.parse_args()

    cams = json.loads(args.cameras.read_text())
    pz000_idx = [i for i, c in enumerate(cams) if PZ000_RE.search(c["pano"])]
    if not pz000_idx:
        raise RuntimeError(f"no pz000 entries found in {args.cameras} "
                           f"(expected pano filenames like '..._pz000_y000_...')")
    pz000_cams = [cams[i] for i in pz000_idx]
    roll_deg, consensus = roll_from_consensus(pz000_cams)
    bad = roll_deg > args.max_roll_deg
    drop_ids = {pz000_cams[j].get("id", pz000_idx[j]) for j in np.nonzero(bad)[0]}
    dropped = [{"id": pz000_cams[j].get("id", pz000_idx[j]), "pano": pz000_cams[j]["pano"],
                "roll_deg": round(float(roll_deg[j]), 2)} for j in np.nonzero(bad)[0]]
    pct = np.percentile(roll_deg, [50, 90, 95, 99, 100]).round(2).tolist()

    print(f"[filter_bad_poses] pz000 subset: {len(pz000_cams)}/{len(cams)} entries")
    print(f"[filter_bad_poses] consensus up = {consensus.round(4).tolist()}")
    print(f"[filter_bad_poses] roll_deg percentiles [50,90,95,99,100]: {pct}")
    print(f"[filter_bad_poses] dropping {len(dropped)}/{len(pz000_cams)} pz000 frames "
          f"(roll > {args.max_roll_deg} deg): {dropped}")

    kept_cams = [c for c in cams if c.get("id") not in drop_ids or not PZ000_RE.search(c["pano"])]
    args.cameras.write_text(json.dumps(kept_cams))

    if args.report:
        args.report.write_text(json.dumps({
            "pz000_total": len(pz000_cams), "pz000_dropped": len(dropped), "dropped": dropped,
            "max_roll_deg": args.max_roll_deg, "roll_deg_percentiles_50_90_95_99_100": pct,
        }))


if __name__ == "__main__":
    main()
