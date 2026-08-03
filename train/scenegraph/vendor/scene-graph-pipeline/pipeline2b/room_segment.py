#!/usr/bin/env python3
"""
(pipeline2b/geo5) Indoor space segmentation into rooms/areas, after

    Tang et al., "Back to geometry: Efficient indoor space segmentation from
    point clouds by 2D-3D geometry constrains", IJAEOG 135 (2024) 104265,
    §3.3 (AdASModule) + §3.4 (HcModule).

Given the interior point cloud and the vertical-structure (wall/partition)
points that bg_removal.py extracted, divide the floor plan into rooms/areas:

  §3.3 AdASModule — project to a 2D occupancy map at R m; mark interior cells
     (any point) white and wall cells black + dilate. Distance-transform the
     free space; a LOCAL adaptive threshold (mean+std in a window) keeps the
     ridge of each room as a "space anchor" core — this survives large
     room-size differences that a global threshold over-filters (paper Fig. 11).

  §3.4 HcModule — the paper organises anchor contours into a hierarchical tree
     and watersheds. On our two-area factory the nesting depth is 1, so we use
     the tree's operational core: connected-component the anchor cores into
     room seeds, then flood each free cell to its geodesically-nearest seed
     WITHOUT crossing wall cells (multi-source BFS). Walls therefore partition
     the plan exactly as the watershed ridges would, and holey photogrammetric
     walls don't leak between rooms as long as the barrier is >1 cell thick
     (the dilation guarantees it).

Returns a RoomMap: a 2D label raster + world→room lookup, so
geo_to_scenegraph.py can tag every object node with the room it stands in.
Uses only numpy + scipy.ndimage + cv2 (all present in the env).
"""
from __future__ import annotations

from collections import deque

import numpy as np


class RoomMap:
    def __init__(self, labels, x0, z0, res, n_rooms):
        self.labels = labels          # (H, W) int; 0 = unassigned/wall, 1..n rooms
        self.x0, self.z0, self.res = x0, z0, res
        self.n_rooms = n_rooms

    def room_of(self, x: float, z: float) -> int:
        gx = int((x - self.x0) / self.res)
        gz = int((z - self.z0) / self.res)
        H, W = self.labels.shape
        if 0 <= gx < H and 0 <= gz < W:
            return int(self.labels[gx, gz])
        return 0


def segment_rooms(interior_xyz: np.ndarray, wall_xyz: np.ndarray,
                  res: float = 0.10, wall_dilate: int = 2,
                  anchor_win: int = 21, verbose: bool = True) -> RoomMap:
    import cv2
    from scipy.ndimage import label as nd_label, uniform_filter
    log = print if verbose else (lambda *a, **k: None)

    x_all = interior_xyz[:, 0]; z_all = interior_xyz[:, 2]
    x0, z0 = float(x_all.min()), float(z_all.min())
    H = int((x_all.max() - x0) / res) + 3
    W = int((z_all.max() - z0) / res) + 3

    def cells(xz):
        gx = np.clip(((xz[:, 0] - x0) / res).astype(np.int64), 0, H - 1)
        gz = np.clip(((xz[:, 2] - z0) / res).astype(np.int64), 0, W - 1)
        return gx, gz

    interior = np.zeros((H, W), np.uint8)
    ix, iz = cells(interior_xyz)
    interior[ix, iz] = 1

    wall = np.zeros((H, W), np.uint8)
    if len(wall_xyz):
        wx, wz = cells(wall_xyz)
        wall[wx, wz] = 1
        if wall_dilate > 0:
            wall = cv2.dilate(wall, np.ones((wall_dilate, wall_dilate), np.uint8))
    free = (interior > 0) & (wall == 0)
    log(f"[room] grid {H}x{W} @ {res}m; interior {int(interior.sum())} cells, "
        f"wall {int(wall.sum())}, free {int(free.sum())}")

    # Rooms = connected components of the free space bounded by walls (the
    # operational core of the paper's "spaces divided by vertical structures":
    # a complete wall separates two components, a gap keeps them one). This
    # avoids the adaptive-anchor over-segmentation that treats every open patch
    # of a large hall as its own room. Small components are noise → merged into
    # the geodesically-nearest real room so no free cell is left unlabelled.
    min_room_cells = max(200, int(0.5 / (res * res)))   # ~0.5 m^2 floor
    cc, n_cc = nd_label(free, structure=np.ones((3, 3), int))
    sizes = np.bincount(cc.ravel())
    big = [i for i in range(1, n_cc + 1) if sizes[i] >= min_room_cells]
    log(f"[room] free-space components: {n_cc} raw -> {len(big)} rooms "
        f"(>= {min_room_cells} cells)")

    labels = np.zeros((H, W), np.int32)
    dq = deque()
    for new_id, cid in enumerate(big, start=1):
        m = cc == cid
        labels[m] = new_id
        gxs, gzs = np.where(m)
        for gx, gz in zip(gxs.tolist(), gzs.tolist()):
            dq.append((gx, gz))
    # flood the leftover free cells (small components) to the nearest room,
    # geodesically over free space (walls block)
    while dq:
        gx, gz = dq.popleft()
        lab = labels[gx, gz]
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, nz = gx + dx, gz + dz
            if 0 <= nx < H and 0 <= nz < W and free[nx, nz] and labels[nx, nz] == 0:
                labels[nx, nz] = lab
                dq.append((nx, nz))

    n_rooms = len(big)
    cell_counts = [int((labels == i).sum()) for i in range(1, n_rooms + 1)]
    log(f"[room] {n_rooms} rooms; cell counts {cell_counts}")
    return RoomMap(labels, x0, z0, res, n_rooms)
