"""Find-Similar routes — ported from the intern's tagging server.

Mounted at  `/spaces/<slug>/api/...` to match what the vendored
viewer (ui/legacy-viewer/find-similar.js) actually hits.

Endpoints:
    POST /spaces/<slug>/api/query              ← the main click-to-search
    GET  /spaces/<slug>/api/query/status       ← "indexed"/"missing" probe
    GET  /spaces/<slug>/api/warmup/status      ← no-op, returns "ready"
    POST /spaces/<slug>/api/warmup             ← no-op (lazy load on /query)
    GET  /spaces/<slug>/api/mask/<global_id>   ← the FastSAM mask overlay PNG

Notes:
    - FAISS index, metadata, embeddings, object_ids are mmap-loaded on
      the first /query and cached per-space.
    - lightglue post-verification is intentionally NOT ported here; the
      query call's `lightglue_enabled` form field is accepted and ignored.
    - cross-image / predict variants are also out of scope; those buttons
      in the UI will still 404.
"""
from __future__ import annotations

import base64
import io
import json
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .. import spaces as space_repo
from ..auth import require_user


router = APIRouter(tags=["find-similar"])


# ── per-space FAISS store ─────────────────────────────────────────────────
_stores: dict[str, dict] = {}
_store_lock = threading.Lock()

FRONT_YAWS = {"000", "030", "060", "090", "270", "300", "330"}


def _load_store(slug: str) -> Optional[dict]:
    if slug in _stores:
        return _stores[slug]
    with _store_lock:
        if slug in _stores:
            return _stores[slug]
        out = space_repo.out_dir(slug)
        idx_path = out / "index.faiss"
        meta_path = out / "metadata.json"
        emb_path = out / "embeddings.npy"
        if not all(p.exists() for p in (idx_path, meta_path, emb_path)):
            return None

        import faiss
        idx = faiss.read_index(str(idx_path))
        meta = json.loads(meta_path.read_text())
        embs = np.load(str(emb_path)).astype("float32")

        # Optional clustering
        oid_path = out / "object_ids.npy"
        object_ids = None
        n_unique = 0
        object_rep = None
        if oid_path.exists():
            object_ids = np.load(str(oid_path)).astype(np.int64)
            n_unique = int(object_ids.max()) + 1 if len(object_ids) else 0
            from collections import defaultdict
            pts_by_oid = defaultdict(list)
            for ri, mi in enumerate(meta):
                wp = mi.get("world_pos")
                if wp is None:
                    continue
                pts_by_oid[int(object_ids[ri])].append(wp)
            object_rep = np.full((n_unique, 3), np.nan, dtype=np.float32)
            for oid, pts in pts_by_oid.items():
                object_rep[oid] = np.median(np.asarray(pts, dtype=np.float32), axis=0)

        # Pitch / yaw masks for the UI filters
        pz000_mask = np.zeros(len(meta), dtype=bool)
        front_yaw_mask = np.zeros(len(meta), dtype=bool)
        for ri, mi in enumerate(meta):
            p = mi.get("pano", "")
            i = p.find("_pz")
            if i >= 0 and p[i + 3:i + 6] == "000":
                pz000_mask[ri] = True
            j = p.find("_y")
            if j >= 0 and p[j + 2:j + 5] in FRONT_YAWS:
                front_yaw_mask[ri] = True

        _stores[slug] = {
            "index": idx, "meta": meta, "emb": embs,
            "object_ids": object_ids, "n_unique": n_unique,
            "object_rep": object_rep,
            "pz000_mask": pz000_mask,
            "front_yaw_mask": front_yaw_mask,
        }
        print(f"[find-similar:{slug}] FAISS loaded: {idx.ntotal} vectors"
              f"{f' · {n_unique} objects' if object_ids is not None else ''}")
        return _stores[slug]


def _find_proposal_at_click(meta: list, view_id: int, cx: float, cy: float):
    """Smallest enclosing proposal at (cx, cy) for the given view."""
    best_idx, best_m, best_area = None, None, float("inf")
    for i, m in enumerate(meta):
        if m.get("view_id") != view_id:
            continue
        x1, y1, x2, y2 = m["bbox"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area < best_area:
                best_area = area
                best_idx = i
                best_m = m
    return best_idx, best_m


def _resolve_space(slug: str, user) -> dict:
    row = space_repo.by_slug(slug)
    if not row:
        raise HTTPException(404, "Space not found")
    if not space_repo.can_view(row, user["id"], user["role"]):
        raise HTTPException(403, "Forbidden")
    return dict(row)


# ── /query: the main click-to-find-similar ────────────────────────────────
@router.post("/spaces/{slug}/api/query")
async def faiss_query(
    slug: str,
    view_id: int = Form(...),
    click_x: float = Form(...),
    click_y: float = Form(...),
    top_k: int = Form(20),
    world_dedup_m: float = Form(0.5),
    min_score: float = Form(0.50),
    expand_iters: int = Form(0),
    pitch_filter: str = Form("all"),
    yaw_filter: str = Form("all"),
    min_area_frac: float = Form(0.0),
    nms_iou: float = Form(0.0),
    # Accepted for compat; LightGlue post-verification not ported tonight.
    lightglue_enabled: bool = Form(False),
    lightglue_world_dist: float = Form(1.4),
    lightglue_min_total_kpts: int = Form(100),
    lightglue_min_bbox_matches: int = Form(3),
    user=Depends(require_user),
):
    _resolve_space(slug, user)
    store = _load_store(slug)
    if not store:
        raise HTTPException(503, "Search index for this space is not built yet.")

    idx = store["index"]
    meta = store["meta"]
    embeddings = store["emb"]
    object_ids = store["object_ids"]
    pz000_mask = store["pz000_mask"]
    front_yaw_mask = store["front_yaw_mask"]

    prop_idx, prop_meta = _find_proposal_at_click(meta, view_id, click_x, click_y)
    if prop_meta is None:
        n_props_view = sum(1 for m in meta if m.get("view_id") == view_id)
        raise HTTPException(404,
            f"No proposal at ({click_x:.0f},{click_y:.0f}) in view {view_id}. "
            f"This view has {n_props_view} proposals.")

    pool_mult = 30 if object_ids is not None else 60
    raw_pool = min(max(top_k * pool_mult, 600), idx.ntotal)
    apply_pz000 = (pitch_filter == "000")
    apply_front = (yaw_filter == "front")
    apply_area = min_area_frac > 0.0
    VIEW_W = VIEW_H = 504
    min_area_px = float(min_area_frac) * VIEW_W * VIEW_H
    expand_iters = max(0, min(expand_iters, 3))

    def _one_query(vec_i: int) -> list:
        q = embeddings[vec_i:vec_i + 1]
        sc, ids = idx.search(q, raw_pool)
        out = []
        for score, j in zip(sc[0].tolist(), ids[0].tolist()):
            if j < 0:
                continue
            if float(score) < min_score:
                break
            if apply_pz000 and not pz000_mask[j]:
                continue
            if apply_front and not front_yaw_mask[j]:
                continue
            m = meta[j]
            if apply_area:
                bb = m.get("bbox")
                if not bb:
                    continue
                if max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1]) < min_area_px:
                    continue
            mr = re.search(r"(\d+)_pz", m.get("pano", ""))
            sp_id = int(mr.group(1)) if mr else -1
            oid = int(object_ids[j]) if object_ids is not None else -1
            out.append({
                "view_id": m["view_id"],
                "scanpoint_id": sp_id,
                "pano": m["pano"],
                "bbox": m["bbox"],
                "score": float(score),
                "world_pos": m.get("world_pos"),
                "pos": m.get("pos"),
                "global_id": int(j),
                "object_id": oid,
            })
        return out

    all_hits: dict = {}
    explored = {prop_idx}
    for h in _one_query(prop_idx):
        all_hits[h["global_id"]] = h
    for _ in range(expand_iters):
        seeds = sorted(
            [h for h in all_hits.values() if h["global_id"] not in explored],
            key=lambda x: x["score"], reverse=True,
        )[:top_k]
        if not seeds:
            break
        for h in seeds:
            explored.add(h["global_id"])
            for nh in _one_query(h["global_id"]):
                gid = nh["global_id"]
                if gid not in all_hits or nh["score"] > all_hits[gid]["score"]:
                    all_hits[gid] = nh

    hits = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)

    # World-space dedup
    if world_dedup_m and world_dedup_m > 0:
        kept, used = [], []
        for h in hits:
            wp = h.get("world_pos")
            if wp is None:
                kept.append(h)
                continue
            wp_np = np.asarray(wp, dtype=np.float32)
            duped = False
            for uw in used:
                if float(np.linalg.norm(wp_np - uw)) < world_dedup_m:
                    duped = True
                    break
            if not duped:
                kept.append(h)
                used.append(wp_np)
        hits = kept

    hits = hits[:top_k]
    return JSONResponse({
        "query": {
            "view_id": view_id,
            "click_x": click_x, "click_y": click_y,
            "proposal_global_id": int(prop_idx),
            "proposal_bbox": prop_meta["bbox"],
            "proposal_pano": prop_meta.get("pano"),
        },
        "hits": hits,
        "n_hits": len(hits),
    })


# ── status probes ────────────────────────────────────────────────────────
@router.get("/spaces/{slug}/api/query/status")
async def query_status(slug: str, user=Depends(require_user)) -> dict:
    _resolve_space(slug, user)
    out = space_repo.out_dir(slug)
    ok = (out / "index.faiss").exists() and (out / "metadata.json").exists()
    return {"state": "indexed" if ok else "missing"}


@router.get("/spaces/{slug}/api/warmup/status")
async def warmup_status(slug: str, user=Depends(require_user)) -> dict:
    _resolve_space(slug, user)
    loaded = slug in _stores
    return {"state": "ready" if loaded else "cold", "n_done": 0, "n_total": 0}


@router.post("/spaces/{slug}/api/warmup")
async def warmup(slug: str, user=Depends(require_user)) -> dict:
    _resolve_space(slug, user)
    # Trigger the load now so the next /query is fast.
    if _load_store(slug):
        return {"state": "ready"}
    return {"state": "missing"}


# ── mask download (decorative — the click overlay) ───────────────────────
_mask_idx_cache: dict[str, dict] = {}
_mask_lock = threading.Lock()


def _mask_index(slug: str) -> Optional[dict]:
    if slug in _mask_idx_cache:
        return _mask_idx_cache[slug]
    with _mask_lock:
        if slug in _mask_idx_cache:
            return _mask_idx_cache[slug]
        out = space_repo.out_dir(slug)
        proposals_file = out / "proposals.jsonl"
        if not proposals_file.exists():
            _mask_idx_cache[slug] = None
            return None
        offsets: list[int] = []
        proposal_map: dict[int, tuple[int, int]] = {}
        g = 0
        fh = open(proposals_file, "rb")
        while True:
            offset = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            line_idx = len(offsets)
            offsets.append(offset)
            rec = json.loads(raw)
            for pi, prop in enumerate(rec.get("proposals", [])):
                if prop.get("mask_b64"):
                    proposal_map[g] = (line_idx, pi)
                g += 1
        _mask_idx_cache[slug] = {"offsets": offsets, "proposal_map": proposal_map, "fh": fh}
        return _mask_idx_cache[slug]


def _l_mode_to_rgba_b64(b64_l: str) -> str:
    from PIL import Image
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64_l))).convert("L"))
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = arr
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


@router.get("/spaces/{slug}/api/mask/{global_id}")
async def get_mask(slug: str, global_id: int, user=Depends(require_user)):
    _resolve_space(slug, user)
    idx = _mask_index(slug)
    if not idx:
        raise HTTPException(404, "No mask index for this space")
    loc = idx["proposal_map"].get(global_id)
    if loc is None:
        raise HTTPException(404, "No mask for this global_id")
    line_idx, pi = loc
    fh = idx["fh"]
    fh.seek(idx["offsets"][line_idx])
    rec = json.loads(fh.readline())
    raw_b64 = rec["proposals"][pi].get("mask_b64")
    if not raw_b64:
        raise HTTPException(404, "Mask field missing")
    return {"mask_b64": _l_mode_to_rgba_b64(raw_b64)}
