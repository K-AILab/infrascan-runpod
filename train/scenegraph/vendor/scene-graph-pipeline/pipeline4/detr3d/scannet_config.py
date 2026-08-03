# Trimmed ScannetDatasetConfig from 3DETR datasets/scannet.py — detection classes only.
# (Copyright (c) Facebook, Inc. and its affiliates.)
import numpy as np
import torch

from .box_util import (
    flip_axis_to_camera_np,
    flip_axis_to_camera_tensor,
    get_3d_box_batch_np,
    get_3d_box_batch_tensor,
)


class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1
        self.max_num_obj = 64

        self.type2class = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "showercurtrain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "garbagebin": 17,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        return np.zeros(pred_cls.shape[0], dtype=np.float32)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        return get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        return get_3d_box_batch_np(box_size, box_angle, box_center_upright)
