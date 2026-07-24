#!/usr/bin/env python3
"""Re-render a nerfstudio dataset at higher resolution by projecting the full-res
equirectangular panoramas (frames/NNNNNN.jpg, 7680x3840) into perspective crops.

Reuses the KNOWN-GOOD poses from an existing (504) prepared transforms.json — only
the pixels and intrinsics change. The equirect sampling convention is auto-calibrated
by matching a re-rendered 504 crop against the stored 504 view (so we don't guess).

yaw-only: keep pz000 (eye-level) views, matching abai's ~2600-image recipe.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import cv2

RE = re.compile(r"(?P<sp>\d+)_pz(?P<pz>\d+)_y(?P<yaw>\d+)")

def load_equirect(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    return im

# Validated convention (matches stored 504 views to ~4.5/255): az_off=180, vertical flip.
def render_crop(equirect, yaw_deg, elev_deg, fx, fy, cx, cy, W, H, az_sign, az_off, flipv=True):
    """Project an equirect (H_e x W_e) into a WxH pinhole crop looking at (yaw,elev)."""
    He, We = equirect.shape[:2]
    A = np.deg2rad(az_sign * (yaw_deg + az_off))
    E = np.deg2rad(elev_deg)
    # camera basis in a Y-up world; forward toward (A,E)
    cf = np.cos(E)
    f = np.array([cf*np.sin(A), np.sin(E), cf*np.cos(A)])
    upw = np.array([0.0, 1.0, 0.0])
    r = np.cross(upw, f); r /= (np.linalg.norm(r) or 1)
    d = np.cross(f, r)  # camera "down"
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    xc = (us - cx) / fx
    yc = (vs - cy) / fy
    if flipv: yc = -yc
    # world ray = xc*r + yc*d + 1*f
    wx = xc*r[0] + yc*d[0] + f[0]
    wy = xc*r[1] + yc*d[1] + f[1]
    wz = xc*r[2] + yc*d[2] + f[2]
    n = np.sqrt(wx*wx+wy*wy+wz*wz)
    wx/=n; wy/=n; wz/=n
    az = np.arctan2(wx, wz)           # azimuth around Y
    el = np.arcsin(np.clip(wy,-1,1))  # elevation
    col = ((az/(2*np.pi)) % 1.0) * We
    row = (0.5 - el/np.pi) * He
    crop = cv2.remap(equirect, col.astype(np.float32), row.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return crop

def calibrate(frames_dir, cams, tj_frames, intr, sample_ids):
    """Find (az_sign, az_off, elev) minimizing diff vs the stored 504 views."""
    fx0,fy0,cx0,cy0 = intr["fx"],intr["fy"],intr["cx"],intr["cy"]
    best=None
    for az_sign in (1,-1):
        for az_off in (0,90,180,270):
            for elev in (0.0,):
                err=0; k=0
                for cid in sample_ids:
                    pano=cams[cid]["pano"]; m=RE.search(pano)
                    sp=m.group("sp"); yaw=int(m.group("yaw"))
                    eq=load_equirect(Path(frames_dir)/f"{int(sp):06d}.jpg")
                    if eq is None: continue
                    stored=cv2.imread(str(sample_stored[cid]),cv2.IMREAD_COLOR)
                    if stored is None: continue
                    Hs,Ws=stored.shape[:2]
                    got=render_crop(eq,yaw,elev,fx0,fy0,cx0,cy0,Ws,Hs,az_sign,az_off)
                    err+=np.mean(np.abs(got.astype(float)-stored.astype(float))); k+=1
                if k:
                    e=err/k
                    if best is None or e<best[0]:
                        best=(e,az_sign,az_off,elev)
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="3d-data/<space> (has frames/, cameras.json)")
    ap.add_argument("--ref-data", required=True, help="existing prepared 504 dataset dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--pz", type=int, default=0, help="keep only this pitch level (yaw-only)")
    ap.add_argument("--calibrate-only", action="store_true")
    a=ap.parse_args()

    src=Path(a.src); ref=Path(a.ref_data); out=Path(a.out)
    cams=json.load(open(src/"cameras.json"))
    tj=json.load(open(ref/"transforms.json"))
    intr=json.load(open(src/"intrinsics.json"))
    frames_dir=src/"frames"

    # map transforms frame -> camera_id via the PANO BASENAME (unambiguous — unlike
    # extracting a leading number from file_path, which for our naming convention
    # (e.g. "000000_pz000_y000_normal.jpg") is the shared SCANPOINT id, not a unique
    # per-view id, and silently mis-maps every view to the wrong camera).
    pano_to_cid = {c["pano"].split("/")[-1]: i for i, c in enumerate(cams)}
    global sample_stored; sample_stored={}
    keep=[]  # (tj_frame, camera_id, sp, yaw)
    for fr in tj["frames"]:
        basename = Path(fr["file_path"]).name
        cid = pano_to_cid.get(basename)
        if cid is None: continue
        pano=cams[cid]["pano"]; m=RE.search(pano)
        if m is None or int(m.group("pz"))!=a.pz: continue
        keep.append((fr,cid,m.group("sp"),int(m.group("yaw"))))
        sample_stored[cid]=ref/fr["file_path"]
    print(f"yaw-only (pz{a.pz:03d}) views: {len(keep)} of {len(tj['frames'])}")

    sample_ids=[keep[i][1] for i in range(0,len(keep),max(1,len(keep)//6))][:6]
    err,az_sign,az_off,elev=calibrate(frames_dir,cams,tj,intr,sample_ids)
    print(f"calibration: az_sign={az_sign} az_off={az_off} elev={elev}  mean|diff|/px={err:.2f} (0-255)")
    if a.calibrate_only: return

    scale=a.res/intr["width"]
    fx=intr["fx"]*scale; fy=intr["fy"]*scale; cx=intr["cx"]*scale; cy=intr["cy"]*scale
    (out/"images").mkdir(parents=True, exist_ok=True)

    # equirect size (from first panorama) to precompute grids
    eq0=load_equirect(frames_dir/f"{int(keep[0][2]):06d}.jpg"); He,We=eq0.shape[:2]
    # precompute one (col,row) remap grid per distinct yaw (elev fixed) -> ~12 grids
    def make_grid(yaw):
        A=np.deg2rad(az_sign*(yaw+az_off)); E=np.deg2rad(elev)
        cf=np.cos(E); f=np.array([cf*np.sin(A),np.sin(E),cf*np.cos(A)])
        up=np.array([0.0,1.0,0.0]); r=np.cross(up,f); r/=(np.linalg.norm(r) or 1); d=np.cross(f,r)
        us,vs=np.meshgrid(np.arange(a.res),np.arange(a.res))
        xc=(us-cx)/fx; yc=-(vs-cy)/fy  # flipv=True
        wx=xc*r[0]+yc*d[0]+f[0]; wy=xc*r[1]+yc*d[1]+f[1]; wz=xc*r[2]+yc*d[2]+f[2]
        n=np.sqrt(wx*wx+wy*wy+wz*wz); wx/=n;wy/=n;wz/=n
        az=np.arctan2(wx,wz); el=np.arcsin(np.clip(wy,-1,1))
        col=(((az/(2*np.pi))%1.0)*We).astype(np.float32); row=((0.5-el/np.pi)*He).astype(np.float32)
        return col,row
    grids={y:make_grid(y) for y in sorted({k[3] for k in keep})}
    print(f"precomputed {len(grids)} yaw grids; equirect {We}x{He}")

    # group by scanpoint so each panorama is loaded once
    from collections import defaultdict
    by_sp=defaultdict(list)
    for item in keep: by_sp[item[2]].append(item)
    new_frames=[]; done=0
    for sp in sorted(by_sp):
        eq=load_equirect(frames_dir/f"{int(sp):06d}.jpg")
        for (fr,cid,_sp,yaw) in by_sp[sp]:
            col,row=grids[yaw]
            crop=cv2.remap(eq,col,row,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
            name=f"frame_{cid:06d}.jpg"
            cv2.imwrite(str(out/"images"/name),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
            nf=dict(fr); nf["file_path"]=f"images/{name}"
            nf["fl_x"]=fx; nf["fl_y"]=fy; nf["cx"]=cx; nf["cy"]=cy; nf["w"]=a.res; nf["h"]=a.res
            nf.pop("mask_path",None); nf.pop("depth_file_path",None)
            new_frames.append(nf); done+=1
        if done % 400 < len(by_sp[sp]): print(f"  rendered {done}/{len(keep)}")
    tj_out={k:v for k,v in tj.items() if k!="frames"}
    # if the ref dataset carried SHARED top-level intrinsics (single-camera datasets,
    # e.g. factory13's one intrinsics.json for all views — unlike shinhan's original
    # per-view-only format this script was first written against), those top-level
    # values are now stale (still 504) and nerfstudio prefers them over the correct
    # per-frame ones just written above, causing a width/height mismatch at load time.
    # Keep them in sync with the new resolution.
    for k, v in (("fl_x",fx),("fl_y",fy),("cx",cx),("cy",cy),("w",a.res),("h",a.res)):
        if k in tj_out: tj_out[k]=v
    tj_out["frames"]=new_frames
    # point cloud init: reuse the prepared init ply
    for cand in ("initialization.ply","sparse_pc.ply","pointcloud.ply"):
        if (ref/cand).exists():
            import shutil; shutil.copy(ref/cand, out/cand); tj_out["ply_file_path"]=cand; break
    json.dump(tj_out, open(out/"transforms.json","w"))
    print(f"wrote {len(new_frames)} views + transforms.json -> {out}")

if __name__=="__main__":
    main()
